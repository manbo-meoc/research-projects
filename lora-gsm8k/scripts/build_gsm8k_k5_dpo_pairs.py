import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

DATA_PATH = Path('benchmarks/gsm8k/train.jsonl')
CANDIDATE_PATH = Path('results/gsm8k/train/qwen25_05b_gsm8k_sft_lora_candidates_k5.jsonl')
OUTPUT_PATH = Path('benchmarks/gsm8k/dpo_lora_k5_hard_medium.jsonl')

SYSTEM_PROMPT = 'You are a careful math assistant. Solve the problem step by step and end with: The answer is <number>.'


def load_jsonl_by_id(path: Path) -> dict[str, dict]:
    rows = {}
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[row['id']] = row
    return rows


def load_candidates(path: Path) -> dict[str, list[dict]]:
    groups = defaultdict(list)
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            groups[row['id']].append(row)
    return groups


def normalize_gold_answer(answer: str) -> str:
    if '####' not in answer:
        return answer.strip()
    reasoning, final = answer.split('####', maxsplit=1)
    return f"{reasoning.strip()}\nThe answer is {final.strip()}."


def build_prompt(question: str) -> str:
    return "<|im_start|>system\n" + SYSTEM_PROMPT + "<|im_end|>\n<|im_start|>user\n" + question + "<|im_end|>\n<|im_start|>assistant\n"


def tokenize(text: str) -> list[str]:
    return re.findall(r'[A-Za-z0-9]+', text.lower())


def cosine_similarity(a: str, b: str) -> float:
    ta = tokenize(a)
    tb = tokenize(b)
    if not ta or not tb:
        return 0.0
    ca = Counter(ta)
    cb = Counter(tb)
    inter = set(ca) & set(cb)
    dot = sum(ca[t] * cb[t] for t in inter)
    na = math.sqrt(sum(v * v for v in ca.values()))
    nb = math.sqrt(sum(v * v for v in cb.values()))
    return dot / (na * nb) if na and nb else 0.0


def jaccard(a: str, b: str) -> float:
    sa = set(tokenize(a))
    sb = set(tokenize(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def hardness_score(candidate: str, gold: str) -> float:
    return 0.7 * cosine_similarity(candidate, gold) + 0.3 * jaccard(candidate, gold)


def choose_rejected(wrongs: list[dict], gold_text: str, row_id: str) -> tuple[dict, str, float]:
    scored = []
    for row in wrongs:
        score = hardness_score(row.get('response', ''), gold_text)
        scored.append((score, row))
    scored.sort(key=lambda x: x[0], reverse=True)
    bucket = int(hashlib.md5(row_id.encode('utf-8')).hexdigest(), 16) % 10
    if bucket < 7:
        selected_score, selected = scored[0]
        return selected, 'hard', selected_score
    if len(scored) == 1:
        selected_score, selected = scored[0]
        return selected, 'medium', selected_score
    mid_idx = min(len(scored) - 1, max(1, len(scored) // 2))
    selected_score, selected = scored[mid_idx]
    return selected, 'medium', selected_score


def main():
    parser = argparse.ArgumentParser(description='Build GSM8K DPO pairs from K=5 candidates with hard/medium negative mixing.')
    parser.add_argument('--data-path', type=Path, default=DATA_PATH)
    parser.add_argument('--candidate-path', type=Path, default=CANDIDATE_PATH)
    parser.add_argument('--output', type=Path, default=OUTPUT_PATH)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--prompt-format', choices=['chat', 'question'], default='chat')
    args = parser.parse_args()

    gold_rows = load_jsonl_by_id(args.data_path)
    candidate_groups = load_candidates(args.candidate_path)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped_missing_gold = 0
    skipped_no_candidates = 0
    skipped_no_wrong = 0
    hard_count = 0
    medium_count = 0

    with args.output.open('w', encoding='utf-8') as out:
        for row_id, gold_row in gold_rows.items():
            cand_rows = candidate_groups.get(row_id)
            if not cand_rows:
                skipped_no_candidates += 1
                continue
            wrongs = [r for r in cand_rows if not r.get('correct')]
            if not wrongs:
                skipped_no_wrong += 1
                continue
            gold_text = normalize_gold_answer(gold_row['answer'])
            rejected, hardness, score = choose_rejected(wrongs, gold_text, row_id)
            prompt = build_prompt(gold_row['question']) if args.prompt_format == 'chat' else gold_row['question']
            record = {
                'id': row_id,
                'prompt': prompt,
                'chosen': gold_text,
                'rejected': str(rejected.get('response', '')).strip(),
                'gold': gold_text,
                'pred': str(rejected.get('pred', '')),
                'hardness': hardness,
                'negative_score': score,
                'candidate_rank': int(rejected.get('candidate_rank', -1)),
                'num_candidates': len(cand_rows),
                'num_wrong_candidates': len(wrongs),
                'source': 'gsm8k_k5_hard_medium_dpo',
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
            if hardness == 'hard':
                hard_count += 1
            else:
                medium_count += 1
            if args.limit is not None and written >= args.limit:
                break

    summary = {
        'data_path': str(args.data_path),
        'candidate_path': str(args.candidate_path),
        'output': str(args.output),
        'prompt_format': args.prompt_format,
        'pairs': written,
        'hard_pairs': hard_count,
        'medium_pairs': medium_count,
        'hard_ratio': hard_count / written if written else 0.0,
        'gold_rows': len(gold_rows),
        'candidate_groups': len(candidate_groups),
        'skipped_missing_gold': skipped_missing_gold,
        'skipped_no_candidates': skipped_no_candidates,
        'skipped_no_wrong': skipped_no_wrong,
    }
    summary_path = args.output.with_suffix('.summary.json')
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
