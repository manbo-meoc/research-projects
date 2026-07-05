import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

DATA_PATH = Path('benchmarks/gsm8k/train.jsonl')
CANDIDATE_PATH = Path('results/gsm8k/train/qwen25_05b_gsm8k_sft_lora_candidates_k5_full1500_batch.jsonl')
OUTPUT_PATH = Path('benchmarks/gsm8k/dpo_lora_stable_full1500.jsonl')

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


def step_overlap(candidate: str, gold: str) -> float:
    return jaccard(candidate, gold)


def numeric_tokens(text: str) -> set[str]:
    return set(re.findall(r'-?\d+(?:\.\d+)?', text.replace(',', '')))


def classify_error(response: str, pred: str, gold_text: str) -> str:
    response = response.strip()
    if not response:
        return 'noise'
    toks = tokenize(response)
    num_count = len(numeric_tokens(response))
    op_count = sum(response.count(op) for op in ['+', '-', '*', '/', '='])
    overlap = step_overlap(response, gold_text)
    lower = response.lower()
    step_markers = sum(lower.count(word) for word in ['first', 'then', 'so', 'therefore', 'because', 'finally', 'altogether', 'which means'])
    truncated = response.rstrip().endswith(('=', '+', '-', '*', '/')) or ('<<' in response and '>>' not in response)
    if len(toks) < 8 or truncated or (not pred and num_count == 0):
        return 'noise'
    if op_count >= 2 and step_markers <= 1:
        return 'arithmetic'
    if num_count >= 2 and step_markers <= 1 and overlap < 0.12:
        return 'arithmetic'
    if step_markers >= 1 or overlap >= 0.06 or len(toks) >= 14:
        return 'reasoning'
    return 'noise'


def score_candidate(row: dict, gold_text: str, error_type: str) -> float:
    response = str(row.get('response', '')).strip()
    pred = str(row.get('pred', '')).strip()
    closeness = 0.5 * (cosine_similarity(response, gold_text) + jaccard(response, gold_text))
    overlap = step_overlap(response, gold_text)
    reasoning_error = {'reasoning': 1.0, 'arithmetic': 0.6, 'noise': 0.2}.get(error_type, 0.5)
    type_bonus = {'reasoning': 0.12, 'arithmetic': 0.06, 'noise': 0.0}.get(error_type, 0.0)
    length_penalty = 0.01 * min(len(tokenize(response)), 80)
    pred_bonus = 0.02 if pred else 0.0
    return 0.6 * reasoning_error + 0.3 * closeness + 0.1 * overlap + type_bonus + pred_bonus - length_penalty


def pick_error_type(row_id: str, available: set[str]) -> str:
    bucket = int(hashlib.md5(row_id.encode('utf-8')).hexdigest(), 16) % 10
    desired = 'reasoning' if bucket < 6 else 'arithmetic' if bucket < 9 else 'noise'
    if desired in available:
        return desired
    for fallback in ['reasoning', 'arithmetic', 'noise']:
        if fallback in available:
            return fallback
    return next(iter(available))


def pick_medium_high(candidates: list[dict], gold_text: str, error_type: str) -> tuple[dict, float]:
    scored = [(score_candidate(row, gold_text, error_type), row) for row in candidates]
    scored.sort(key=lambda x: x[0], reverse=True)
    if len(scored) == 1:
        return scored[0][1], scored[0][0]
    rank = max(0, min(len(scored) - 1, int(round(0.4 * (len(scored) - 1)))))
    return scored[rank][1], scored[rank][0]


def choose_rejected(wrongs: list[dict], gold_text: str, row_id: str) -> tuple[dict, str, float]:
    typed: dict[str, list[dict]] = {'reasoning': [], 'arithmetic': [], 'noise': []}
    for row in wrongs:
        error_type = classify_error(str(row.get('response', '')), str(row.get('pred', '')), gold_text)
        typed.setdefault(error_type, []).append(row)

    desired = pick_error_type(row_id, {kind for kind, rows in typed.items() if rows})
    pool = typed.get(desired) or wrongs
    selected, score = pick_medium_high(pool, gold_text, desired)
    return selected, desired, score


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
    skip_no_candidates = 0
    skip_no_wrong = 0
    error_type_counts = Counter()

    with args.output.open('w', encoding='utf-8') as out:
        for row_id, gold_row in gold_rows.items():
            cand_rows = candidate_groups.get(row_id)
            if not cand_rows:
                skip_no_candidates += 1
                continue
            wrongs = [r for r in cand_rows if not r.get('correct')]
            if not wrongs:
                skip_no_wrong += 1
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
            error_type_counts[hardness] += 1
            if args.limit is not None and written >= args.limit:
                break

    summary = {
        'data_path': str(args.data_path),
        'candidate_path': str(args.candidate_path),
        'output': str(args.output),
        'prompt_format': args.prompt_format,
        'pairs': written,
        'error_type_counts': dict(error_type_counts),
        'gold_rows': len(gold_rows),
        'candidate_groups': len(candidate_groups),
        'skip_no_candidates': skip_no_candidates,
        'skip_no_wrong': skip_no_wrong,
    }
    summary_path = args.output.with_suffix('.summary.json')
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
