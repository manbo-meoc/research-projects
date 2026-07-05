import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import PeftModel
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer


BASE_MODEL_DIR = Path('models/Qwen/Qwen2.5-0.5B-Instruct')
SFT_ADAPTER_DIR = Path('models/qwen25_05b_gsm8k_lora')
DPO_DATA_PATH = Path('benchmarks/gsm8k/dpo_lora_stable_full1500.jsonl')
OUTPUT_DIR = Path('models/qwen25_05b_gsm8k_sft_dpo_lora_stable_anchor_full1500')


def load_jsonl(path: Path, limit: int | None):
    rows = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def encode(tokenizer, prompt: str, response: str, max_length: int, device):
    text = prompt + response + tokenizer.eos_token
    prompt_len = len(tokenizer(prompt, add_special_tokens=False)['input_ids'])
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
        return_tensors='pt',
    )
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)
    return input_ids, attention_mask, min(prompt_len, input_ids.shape[1])


def response_stats(model, input_ids, attention_mask, prompt_len: int):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]
    labels = input_ids[:, 1:]
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    positions = torch.arange(labels.shape[1], device=labels.device) + 1
    response_mask = (positions >= prompt_len).unsqueeze(0) & attention_mask[:, 1:].bool()
    summed = (token_log_probs * response_mask).sum()
    token_count = response_mask.sum().clamp(min=1)
    return summed, token_count


def load_sft_model(base_model_dir: Path, sft_adapter_dir: Path, dtype, device, trainable: bool):
    base = AutoModelForCausalLM.from_pretrained(
        base_model_dir,
        dtype=dtype,
        device_map=None,
        trust_remote_code=True,
    ).to(device)
    model = PeftModel.from_pretrained(base, sft_adapter_dir, is_trainable=trainable)
    model.config.use_cache = False
    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-model-dir', type=Path, default=BASE_MODEL_DIR)
    parser.add_argument('--sft-adapter-dir', type=Path, default=SFT_ADAPTER_DIR)
    parser.add_argument('--dpo-data-path', type=Path, default=DPO_DATA_PATH)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--max-length', type=int, default=256)
    parser.add_argument('--epochs', type=float, default=1.0)
    parser.add_argument('--learning-rate', type=float, default=2e-6)
    parser.add_argument('--gradient-accumulation-steps', type=int, default=16)
    parser.add_argument('--logging-steps', type=int, default=8)
    parser.add_argument('--beta', type=float, default=0.015)
    parser.add_argument('--sft-loss-weight', type=float, default=0.4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--gradient-checkpointing',
        action='store_true',
        help='Enable gradient checkpointing for the policy model to reduce memory usage.',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float16 if device.type == 'cuda' else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = load_jsonl(args.dpo_data_path, args.limit)
    if not rows:
        raise RuntimeError(f'No DPO pairs found in {args.dpo_data_path}')

    ref = load_sft_model(args.base_model_dir, args.sft_adapter_dir, dtype, device, trainable=False)
    ref.eval()
    ref_margins = []
    with torch.no_grad():
        for idx, row in enumerate(rows, start=1):
            chosen_ids, chosen_mask, chosen_prompt_len = encode(tokenizer, row['prompt'], row['chosen'], args.max_length, device)
            rejected_ids, rejected_mask, rejected_prompt_len = encode(tokenizer, row['prompt'], row['rejected'], args.max_length, device)
            ref_chosen, _ = response_stats(ref, chosen_ids, chosen_mask, chosen_prompt_len)
            ref_rejected, _ = response_stats(ref, rejected_ids, rejected_mask, rejected_prompt_len)
            ref_margins.append(float((ref_chosen - ref_rejected).detach().cpu()))
            if idx % args.logging_steps == 0:
                print(json.dumps({'precompute_ref': idx, 'total': len(rows)}), flush=True)

    del ref
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    policy = load_sft_model(args.base_model_dir, args.sft_adapter_dir, dtype, device, trainable=True)
    if args.gradient_checkpointing:
        policy.gradient_checkpointing_enable()
        if hasattr(policy, 'enable_input_require_grads'):
            policy.enable_input_require_grads()
    policy.train()
    optimizer = AdamW((p for p in policy.parameters() if p.requires_grad), lr=args.learning_rate)

    total_steps = math.ceil(len(rows) * args.epochs)
    optimizer_steps = 0
    policy.zero_grad(set_to_none=True)
    for step_idx in range(total_steps):
        row = rows[step_idx % len(rows)]
        chosen_ids, chosen_mask, chosen_prompt_len = encode(tokenizer, row['prompt'], row['chosen'], args.max_length, device)
        rejected_ids, rejected_mask, rejected_prompt_len = encode(tokenizer, row['prompt'], row['rejected'], args.max_length, device)
        policy_chosen, chosen_tokens = response_stats(policy, chosen_ids, chosen_mask, chosen_prompt_len)
        policy_rejected, _ = response_stats(policy, rejected_ids, rejected_mask, rejected_prompt_len)
        policy_margin = policy_chosen - policy_rejected
        ref_margin = torch.tensor(ref_margins[step_idx % len(rows)], device=device, dtype=policy_margin.dtype)
        dpo_loss = -F.logsigmoid(args.beta * (policy_margin - ref_margin))
        sft_anchor_loss = -(policy_chosen / chosen_tokens.to(policy_chosen.dtype))
        loss = dpo_loss + args.sft_loss_weight * sft_anchor_loss
        (loss / args.gradient_accumulation_steps).backward()

        if (step_idx + 1) % args.gradient_accumulation_steps == 0 or step_idx + 1 == total_steps:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1

        if args.logging_steps and (step_idx + 1) % args.logging_steps == 0:
            print(
                json.dumps(
                    {
                        'step': step_idx + 1,
                        'total_steps': total_steps,
                        'loss': float(loss.detach().cpu()),
                        'dpo_loss': float(dpo_loss.detach().cpu()),
                        'sft_anchor_loss': float(sft_anchor_loss.detach().cpu()),
                        'policy_margin': float(policy_margin.detach().cpu()),
                        'ref_margin': float(ref_margin.detach().cpu()),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    policy.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    summary = {
        'base_model': str(args.base_model_dir),
        'sft_adapter': str(args.sft_adapter_dir),
        'dpo_data_path': str(args.dpo_data_path),
        'output_dir': str(args.output_dir),
        'num_dpo_pairs': len(rows),
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'optimizer_steps': optimizer_steps,
        'beta': args.beta,
        'sft_loss_weight': args.sft_loss_weight,
        'max_length': args.max_length,
        'ref_margin_mode': 'precomputed',
    }
    (args.output_dir / 'dpo_training_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
