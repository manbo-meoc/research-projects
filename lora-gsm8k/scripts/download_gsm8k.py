import json
from pathlib import Path

from datasets import load_dataset

OUT_DIR = Path('benchmarks/gsm8k')
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    ds = load_dataset('openai/gsm8k', 'main')
    for split in ['train', 'test']:
        path = OUT_DIR / f'{split}.jsonl'
        with path.open('w', encoding='utf-8') as f:
            for idx, row in enumerate(ds[split]):
                item = {
                    'id': f'gsm8k-{split}-{idx}',
                    'question': row['question'],
                    'answer': row['answer'],
                }
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f'{split}: {len(ds[split])} -> {path}')


if __name__ == '__main__':
    main()
