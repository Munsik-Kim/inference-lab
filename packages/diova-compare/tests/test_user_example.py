"""The public tutorial files must remain a runnable, independently graded pair."""
import hashlib
import tempfile
import subprocess
import sys
import json
import unittest
from pathlib import Path
from diova_compare.core import load, compare

EXAMPLES = Path(__file__).resolve().parents[1] / 'examples/compare'


class UserExample(unittest.TestCase):
    def test_tutorial_prompts_gold_and_transitions(self):
        baseline = load(EXAMPLES / 'baseline.jsonl')
        candidate = load(EXAMPLES / 'candidate.jsonl')
        for rows in (baseline, candidate):
            self.assertEqual(len(rows), 4)
            for row, (a, b) in zip(rows, [(2, 2), (3, 5), (6, 1), (4, 5)]):
                prompt = f'What is {a} + {b}? Reply with an integer.'
                self.assertEqual(row['prompt_hash'], hashlib.sha256(prompt.encode()).hexdigest())
                self.assertEqual(row['gold'], [str(a + b)])
                self.assertEqual(row['correct'], row['answer'].strip() == str(a + b))
                self.assertEqual(row['evidence_kind'], 'synthetic_test')
                self.assertNotIn('distribution', row)
        counts = compare(baseline, candidate, replicates=50)['tasks']['integer-addition@1']['counts']
        self.assertEqual([counts[k] for k in ('baseline_correct', 'candidate_correct',
                         'correct_to_wrong', 'wrong_to_correct', 'wrong_to_wrong',
                         'all_answer_disagreement')], [2, 2, 1, 1, 1, 3])

    def test_tutorial_cli_report_and_rejected_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            command = [sys.executable, '-B', '-m', 'diova_compare.cli', 'compare',
                       '--baseline', str(EXAMPLES / 'baseline.jsonl'),
                       '--candidate', str(EXAMPLES / 'candidate.jsonl'),
                       '--output', str(root / 'report')]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual({p.name for p in (root / 'report').iterdir()},
                             {'report.md', 'report.json', 'pairs.csv'})
            rows = load(EXAMPLES / 'candidate.jsonl')
            rows[0]['prompt_hash'] = '0' * 64
            mismatch = root / 'mismatch.jsonl'
            mismatch.write_text(''.join(json.dumps(r) + '\n' for r in rows))
            command[command.index('--candidate') + 1] = str(mismatch)
            command[-1] = str(root / 'rejected')
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn('pair mismatch: prompt_hash', result.stderr)
            self.assertFalse((root / 'rejected').exists())
