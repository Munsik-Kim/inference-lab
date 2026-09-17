"""Prepare smoke/DEV only, offline. Evaluation preparation requires a later freeze."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.storage import digest, write_new
from src.tasks import COUNTS, TASKS, build_prompt, scenario


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New directory; never overwrite inputs')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.snapshot, local_files_only=True, trust_remote_code=False)
    inputs, gold = [], []
    for split in ('smoke', 'dev'):
        for task in TASKS:
            for index in range(COUNTS[split]):
                item = scenario(split, task, index)
                prompt = build_prompt(item, tokenizer, 512 if split == 'smoke' else 4096)
                inputs.append(prompt)
                gold.append(item)
    for field in ('base_id', 'base_facts_hash'):
        if len({r[field] for r in gold}) != len(gold):
            raise ValueError(f'Duplicate {field}')
    if len({x['token_hash'] for x in inputs}) != len(inputs):
        raise ValueError('Duplicate token input')
    manifest = {'status': 'PREPARED_SMOKE_AND_DEV_NOT_MODEL_EVALUATED', 'evaluation_manifest_frozen': False,
                'model_id': 'Qwen/Qwen3-0.6B', 'revision': 'c1899de289a04d12100db370d81485cdf75e47ca',
                'chat_template_sha256': hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
                'enable_thinking': False, 'add_generation_prompt': True, 'gold_in_separate_file': True,
                'counts': {'smoke': 6, 'dev': 96, 'evaluation': 0},
                'inputs_digest': digest(inputs), 'gold_digest': digest(gold),
                'language_scope': 'English task instructions, restricted Python code; no Korean task stratum',
                'label_token_ids': inputs[0]['label_token_ids'],
                'continuation_check': 'Full rendered prompt plus each label equals prompt IDs plus one distinct ID',
                'filler_limit': 'Repeated neutral archive sentence; synthetic contexts, not natural long documents'}
    write_new(args.output/'prompts.json', inputs)
    write_new(args.output/'gold.json', gold)
    write_new(args.output/'manifest.json', manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
