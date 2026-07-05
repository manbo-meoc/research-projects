import argparse
import json
from pathlib import Path


DATA_PATH = Path("benchmarks/gsm8k/train.jsonl")
EVAL_PATH = Path("results/gsm8k/train/qwen25_05b_gsm8k_lora_eval_train.jsonl")
OUTPUT_PATH = Path("benchmarks/gsm8k/dpo_lora_wrong_vs_gold.jsonl")


SYSTEM_PROMPT = (
    "You are a careful math assistant. Solve the problem step by step "
    "and end with: The answer is <number>."
)


def load_jsonl_by_id(path: Path) -> dict[str, dict]:
    rows = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[row["id"]] = row
    return rows


def normalize_gold_answer(answer: str) -> str:
    if "####" not in answer:
        return answer.strip()
    reasoning, final = answer.split("####", maxsplit=1)
    return f"{reasoning.strip()}\nThe answer is {final.strip()}."


def build_prompt(question: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Build GSM8K DPO pairs with gold answers as chosen and LoRA wrong responses as rejected."
    )
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument("--eval-path", type=Path, default=EVAL_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--prompt-format",
        choices=["chat", "question"],
        default="chat",
        help="Use Qwen chat-template-like text or raw question as the DPO prompt.",
    )
    args = parser.parse_args()

    gold_rows = load_jsonl_by_id(args.data_path)
    eval_rows = load_jsonl_by_id(args.eval_path)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped_correct = 0
    skipped_missing_gold = 0
    skipped_empty_rejected = 0

    with args.output.open("w", encoding="utf-8") as out:
        for row_id, eval_row in eval_rows.items():
            if eval_row.get("correct") is True:
                skipped_correct += 1
                continue

            gold_row = gold_rows.get(row_id)
            if gold_row is None:
                skipped_missing_gold += 1
                continue

            rejected = str(eval_row.get("response", "")).strip()
            if not rejected:
                skipped_empty_rejected += 1
                continue

            question = gold_row["question"]
            record = {
                "id": row_id,
                "prompt": build_prompt(question) if args.prompt_format == "chat" else question,
                "chosen": normalize_gold_answer(gold_row["answer"]),
                "rejected": rejected,
                "gold": str(eval_row.get("gold", "")),
                "pred": str(eval_row.get("pred", "")),
                "source": "gsm8k_gold_vs_sft_lora_wrong",
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

            if args.limit is not None and written >= args.limit:
                break

    summary = {
        "data_path": str(args.data_path),
        "eval_path": str(args.eval_path),
        "output": str(args.output),
        "prompt_format": args.prompt_format,
        "pairs": written,
        "eval_rows": len(eval_rows),
        "gold_rows": len(gold_rows),
        "skipped_correct": skipped_correct,
        "skipped_missing_gold": skipped_missing_gold,
        "skipped_empty_rejected": skipped_empty_rejected,
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
