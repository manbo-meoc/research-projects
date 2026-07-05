import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

try:
    from trl import DPOTrainer
except ImportError as exc:
    raise RuntimeError(
        "trl is required for DPO training. Install it in the tau2 environment first: "
        "python -m pip install trl"
    ) from exc


BASE_MODEL_DIR = Path("models/Qwen/Qwen2.5-0.5B-Instruct")
SFT_ADAPTER_DIR = Path("models/qwen25_05b_gsm8k_lora")
DPO_DATA_PATH = Path("benchmarks/gsm8k/dpo_lora_wrong_vs_gold.jsonl")
OUTPUT_DIR = Path("models/qwen25_05b_gsm8k_sft_dpo_lora")


def load_jsonl(path: Path, limit: int | None):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append(
                {
                    "prompt": row["prompt"],
                    "chosen": row["chosen"],
                    "rejected": row["rejected"],
                }
            )
            if limit is not None and len(rows) >= limit:
                break
    return rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-dir", type=Path, default=BASE_MODEL_DIR)
    parser.add_argument("--sft-adapter-dir", type=Path, default=SFT_ADAPTER_DIR)
    parser.add_argument("--dpo-data-path", type=Path, default=DPO_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--max-prompt-length", type=int, default=384)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_cuda = torch.cuda.is_available()
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_dir,
        dtype=torch.float16 if use_cuda else torch.float32,
        device_map="auto" if use_cuda else "cpu",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model = PeftModel.from_pretrained(model, args.sft_adapter_dir, is_trainable=True)

    ref_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_dir,
        dtype=torch.float16 if use_cuda else torch.float32,
        device_map="auto" if use_cuda else "cpu",
        trust_remote_code=True,
    )
    ref_model = PeftModel.from_pretrained(ref_model, args.sft_adapter_dir)
    ref_model.eval()

    rows = load_jsonl(args.dpo_data_path, args.limit)
    train_dataset = Dataset.from_list(rows)

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        fp16=use_cuda,
        report_to=[],
        remove_unused_columns=False,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        beta=args.beta,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
    )
    trainer.train()

    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    summary = {
        "base_model": str(args.base_model_dir),
        "sft_adapter": str(args.sft_adapter_dir),
        "dpo_data_path": str(args.dpo_data_path),
        "output_dir": str(args.output_dir),
        "num_dpo_pairs": len(rows),
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "beta": args.beta,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
    }
    (args.output_dir / "dpo_training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
