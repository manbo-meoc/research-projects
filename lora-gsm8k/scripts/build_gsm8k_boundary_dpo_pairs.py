import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

DATA_PATH = Path('benchmarks/gsm8k/train.jsonl')
CANDIDATE_PATH = Path('results/gsm8k/train/qwen25_05b_gsm8k_sft_lora_candidates_k5_full1500_batch.jsonl')
OUTPUT_PATH = Path('benchmarks/gsm8k/dpo_lora_boundary_full1500.jsonl')

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


def is_complete_wrong(response: str) -> bool:
    response = response.strip()
    if not response:
        return False
    if 'The answer is' not in response and '答案是' not in response:
        return False
    if response.rstrip().endswith(('=', '+', '-', '*', '/')):
        return False
    if '<<' in response and '>>' not in response:
        return False
    return len(tokenize(response)) >= 8


def classify_complete_error(response: str, pred: str, gold_text: str) -> str:
    response = response.strip()
    if not is_complete_wrong(response):
        return 'noise'
    toks = tokenize(response)
    num_count = len(set(re.findall(r'-?\d+(?:\.\d+)?', response.replace(',', ''))))
    op_count = sum(response.count(op) for op in ['+', '-', '*', '/', '='])
    overlap = step_overlap(response, gold_text)
    lower = response.lower()
    step_markers = sum(lower.count(word) for word in ['first', 'then', 'so', 'therefore', 'because', 'finally', 'altogether', 'which means'])
    if op_count >= 2 or (num_count >= 2 and step_markers <= 1 and overlap < 0.12):
        return 'arithmetic'
    if step_markers >= 1 or overlap >= 0.06 or len(toks) >= 14:
        return 'reasoning'
    return 'reasoning'


def score_candidate(row: dict, gold_text: str, error_type: str) -> float:
    response = str(row.get('response', '')).strip()
    closeness = 0.7 * cosine_similarity(response, gold_text) + 0.3 * jaccard(response, gold_text)
    overlap = step_overlap(response, gold_text)
    length_penalty = 0.005 * min(len(tokenize(response)), 120)
    type_bonus = 0.08 if error_type == 'arithmetic' else 0.03
    return closeness + 0.2 * overlap + type_bonus - length_penalty


def choose_rejected(wrongs: list[dict], gold_text: str, row_id: str) -> tuple[dict, str, float]:
    typed: dict[str, list[dict]] = {'arithmetic': [], 'reasoning': []}
    for row in wrongs:
        error_type = classify_complete_error(str(row.get('response', '')), str(row.get('pred', '')), gold_text)
        if error_type in typed:
            typed[error_type].append(row)

    if typed['arithmetic']:
        desired = 'arithmetic'
        pool = typed['arithmetic']
    elif typed['reasoning']:
        desired = 'reasoning'
        pool = typed['reasoning']
    else:
        raise ValueError('no complete wrong candidates available')

    scored = [(score_candidate(row, gold_text, desired), row) for row in pool]
    scored.sort(key=lambda x: x[0], reverse=True)
    if len(scored) == 1:
        return scored[0][1], desired, scored[0][0]
    rank = max(0, min(len(scored) - 1, int(round(0.4 * (len(scored) - 1)))))
    return scored[rank][1], desired, scored[rank][0]


def main():
    parser = argparse.ArgumentParser(description='Build GSM8K DPO pairs from complete wrong candidates only.')
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
    skip_no_complete = 0
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
            complete_wrongs = [r for r in wrongs if is_complete_wrong(str(r.get('response', '')))]
            if not complete_wrongs:
                skip_no_complete += 1
                continue

            gold_text = normalize_gold_answer(gold_row['answer'])
            rejected, hardness, score = choose_rejected(complete_wrongs, gold_text, row_id)
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
                'num_complete_wrong_candidates': len(complete_wrongs),
                'source': 'gsm8k_complete_boundary_dpo',
            }
            out.write(json.dumps(record, ensure_ascii=False) + '\n')
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
        'skip_no_complete': skip_no_complete,
    }
    summary_path = args.output.with_suffix('.summary.json')
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
