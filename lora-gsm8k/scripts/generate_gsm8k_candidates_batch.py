import argparse
import json
import random
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
except ImportError:
    PeftModel = None

BASE_MODEL_DIR = Path('models/Qwen/Qwen2.5-0.5B-Instruct')
SFT_ADAPTER_DIR = Path('models/qwen25_05b_gsm8k_lora')
DATA_PATH = Path('benchmarks/gsm8k/train.jsonl')
RESULT_DIR = Path('results/gsm8k/train')


def extract_gold(answer: str) -> str:
    if '####' in answer:
        answer = answer.split('####')[-1]
    nums = re.findall(r'-?\d+(?:\.\d+)?', answer.replace(',', ''))
    return nums[-1] if nums else ''


def extract_pred(text: str) -> str:
    markers = [r'answer is\s*([-+]?\d+(?:\.\d+)?)', r'答案是\s*([-+]?\d+(?:\.\d+)?)']
    lowered = text.lower().replace(',', '')
    for pattern in markers:
        match = re.search(pattern, lowered)
        if match:
            return match.group(1)
    nums = re.findall(r'-?\d+(?:\.\d+)?', text.replace(',', ''))
    return nums[-1] if nums else ''


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


def build_prompt(tokenizer, question: str) -> str:
    messages = [
        {'role': 'system', 'content': 'You are a careful math assistant. Solve the problem step by step and end with: The answer is <number>.'},
        {'role': 'user', 'content': question},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=Path, default=BASE_MODEL_DIR)
    parser.add_argument('--lora-adapter-dir', type=Path, default=SFT_ADAPTER_DIR)
    parser.add_argument('--data-path', type=Path, default=DATA_PATH)
    parser.add_argument('--limit', type=int, default=1500)
    parser.add_argument('--num-candidates', type=int, default=5)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top-p', type=float, default=0.9)
    parser.add_argument('--max-new-tokens', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path, default=RESULT_DIR / 'qwen25_05b_gsm8k_sft_lora_candidates_k5_full1500_batch.jsonl')
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    data = load_jsonl(args.data_path, args.limit)
    if not data:
        raise RuntimeError(f'No rows found in {args.data_path}')

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    use_cuda = torch.cuda.is_available()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        dtype=torch.float16 if use_cuda else torch.float32,
        device_map='auto' if use_cuda else 'cpu',
        trust_remote_code=True,
    )
    if args.lora_adapter_dir is not None:
        if PeftModel is None:
            raise RuntimeError('peft is required to evaluate a LoRA adapter. Install it with: python -m pip install peft')
        model = PeftModel.from_pretrained(model, args.lora_adapter_dir)

    model.eval()
    infer_device = next(model.parameters()).device

    written = 0
    correct = 0
    total = len(data) * args.num_candidates
    with args.output.open('w', encoding='utf-8') as out:
        for i, row in enumerate(data, start=1):
            prompt = build_prompt(tokenizer, row['question'])
            inputs = tokenizer([prompt], return_tensors='pt')
            inputs = {key: value.to(infer_device) for key, value in inputs.items()}
            gold = extract_gold(row['answer'])
            gen_seed = args.seed + i
            torch.manual_seed(gen_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(gen_seed)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_return_sequences=args.num_candidates,
                    pad_token_id=tokenizer.eos_token_id,
                )
            for cand_idx, seq in enumerate(outputs, start=1):
                generated = seq[inputs['input_ids'].shape[-1]:]
                response = tokenizer.decode(generated, skip_special_tokens=True).strip()
                pred = extract_pred(response)
                is_correct = pred == gold
                correct += int(is_correct)
                out.write(json.dumps({'id': row['id'], 'candidate_rank': cand_idx, 'question': row['question'], 'gold': gold, 'pred': pred, 'correct': is_correct, 'response': response}, ensure_ascii=False) + '\n')
                written += 1
                print(f'[{i}/{len(data)}][{cand_idx}/{args.num_candidates}] correct={is_correct} pred={pred} gold={gold}', flush=True)

    summary = {'model': str(args.model_dir), 'lora_adapter': str(args.lora_adapter_dir) if args.lora_adapter_dir else None, 'benchmark': 'GSM8K train candidate generation', 'data_path': str(args.data_path), 'limit': args.limit, 'num_candidates': args.num_candidates, 'temperature': args.temperature, 'top_p': args.top_p, 'correct': correct, 'total': total, 'accuracy': correct / total if total else 0.0, 'output': str(args.output)}
    args.output.with_suffix('.summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
