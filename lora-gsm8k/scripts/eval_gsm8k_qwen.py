import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
except ImportError:
    PeftModel = None

MODEL_DIR = Path('models/Qwen/Qwen2.5-0.5B-Instruct')
DATA_PATH = Path('benchmarks/gsm8k/test.jsonl')
RESULT_DIR = Path('results/gsm8k')


def extract_gold(answer: str) -> str:
    if '####' in answer:
        answer = answer.split('####')[-1]
    nums = re.findall(r'-?\d+(?:\.\d+)?', answer.replace(',', ''))
    return nums[-1] if nums else ''


def extract_pred(text: str) -> str:
    # Prefer explicit final answer markers, then fall back to the last number.
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
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def build_prompt(tokenizer, question: str) -> str:
    messages = [
        {
            'role': 'system',
            'content': 'You are a careful math assistant. Solve the problem step by step and end with: The answer is <number>.',
        },
        {'role': 'user', 'content': question},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=Path, default=MODEL_DIR)
    parser.add_argument('--lora-adapter-dir', type=Path, default=None)
    parser.add_argument('--data-path', type=Path, default=DATA_PATH)
    parser.add_argument('--limit', type=int, default=20)
    parser.add_argument('--max-new-tokens', type=int, default=256)
    parser.add_argument('--output', type=Path, default=RESULT_DIR / 'qwen25_05b_base_gsm8k_eval.jsonl')
    parser.add_argument('--resume', action='store_true', help='Append missing rows if output already has partial results.')
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    data = load_jsonl(args.data_path, args.limit)
    completed = 0
    correct = 0
    if args.resume and args.output.exists():
        with args.output.open('r', encoding='utf-8') as existing:
            for line in existing:
                if not line.strip():
                    continue
                record = json.loads(line)
                completed += 1
                correct += int(bool(record.get('correct')))
        if completed > len(data):
            raise RuntimeError(f'Existing output has {completed} rows, but requested limit only has {len(data)} rows.')

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

    mode = 'a' if args.resume and completed else 'w'
    with args.output.open(mode, encoding='utf-8') as out:
        for i, row in enumerate(data[completed:], start=completed + 1):
            prompt = build_prompt(tokenizer, row['question'])
            inputs = tokenizer([prompt], return_tensors='pt')
            inputs = {key: value.to(infer_device) for key, value in inputs.items()}
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )
            generated = outputs[0][inputs['input_ids'].shape[-1]:]
            response = tokenizer.decode(generated, skip_special_tokens=True).strip()

            gold = extract_gold(row['answer'])
            pred = extract_pred(response)
            is_correct = pred == gold
            correct += int(is_correct)

            record = {
                'id': row['id'],
                'question': row['question'],
                'gold': gold,
                'pred': pred,
                'correct': is_correct,
                'response': response,
            }
            out.write(json.dumps(record, ensure_ascii=False) + '\n')
            print(f'[{i}/{len(data)}] correct={is_correct} pred={pred} gold={gold}')

    total = len(data)
    acc = correct / total if total else 0.0
    summary = {
        'model': str(args.model_dir),
        'lora_adapter': str(args.lora_adapter_dir) if args.lora_adapter_dir else None,
        'benchmark': 'GSM8K test',
        'data_path': str(args.data_path),
        'limit': args.limit,
        'correct': correct,
        'total': total,
        'accuracy': acc,
        'output': str(args.output),
    }
    summary_path = args.output.with_suffix('.summary.json')
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
