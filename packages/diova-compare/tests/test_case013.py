"""Contracts exercised against public historical records; no model imports."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from diova_compare.case013 import Reader, digest, export_case013, score_agent
from diova_compare.core import compare, load

CASE = Path(__file__).resolve().parents[3] / 'cases/013-dualquant-reference-quality'


class Case013Export(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.export = Path(cls.temp.name) / 'export'
        cls.manifest = export_case013(CASE, cls.export)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def pair(self, cohort, method):
        return compare(load(self.export / cohort / 'RTN.jsonl'),
                       load(self.export / cohort / (method + '.jsonl')), replicates=20)

    def test_historical_raw_pair_counts(self):
        for length, expected in ((8192, (9, 2, 1, 0)), (16384, (10, 0, 0, 2))):
            result = self.pair('long/' + str(length), 'D16')
            counts = next(iter(result['tasks'].values()))['counts']
            self.assertEqual(tuple(counts[k] for k in
                                   ('both_correct', 'correct_to_wrong', 'wrong_to_correct', 'both_wrong')), expected)
            self.assertEqual(result['n_pairs'], 12)

    def test_agent_modes_and_extension_denominators(self):
        for cohort, n, correct in (('base', 8, 6), ('extension16k', 4, 3)):
            q = self.pair('agent/' + cohort + '/FREE', 'D16')
            self.assertEqual(q['n_pairs'], n)
            self.assertEqual(next(iter(q['tasks'].values()))['counts']['candidate_correct'], correct)
            d4 = self.pair('agent/' + cohort + '/FREE', 'D4')
            self.assertEqual(next(iter(d4['tasks'].values()))['counts']['candidate_correct'], 0)
            replay = self.pair('agent/' + cohort + '/REPLAY', 'D16')
            self.assertEqual(next(iter(replay['tasks'].values()))['counts']['both_correct'], n)

    def test_same_final_json_without_execution_remains_wrong(self):
        b = {r['sample_id']: r for r in load(self.export / 'agent/base/FREE/RTN.jsonl')}
        c = {r['sample_id']: r for r in load(self.export / 'agent/base/FREE/D16.jsonl')}
        for task in ('AGENT-6', 'AGENT-7'):
            self.assertEqual(b[task]['answer'], c[task]['answer'])
            self.assertTrue(b[task]['correct'])
            self.assertFalse(c[task]['correct'])
        sidecar = [json.loads(x) for x in (self.export / 'agent/base/FREE/D16.sidecar.jsonl').read_text().splitlines()]
        row = next(r for r in sidecar if r['sample_id'] == 'AGENT-6')
        self.assertEqual(row['required_execution'],
                         {'expected_config_matches': False, 'successful_validation_event': False})

    def test_mode_length_gold_and_prompt_mismatch_rejected(self):
        b = load(self.export / 'agent/base/FREE/RTN.jsonl')
        for path in ('agent/base/REPLAY/D16.jsonl', 'agent/extension16k/FREE/D16.jsonl', 'long/8192/D16.jsonl'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                compare(b, load(self.export / path), replicates=10)
        c = copy.deepcopy(b)
        for key, value in (('prompt_hash', 'a' * 64), ('gold_definition', 'different-parser-v2')):
            modified = copy.deepcopy(c)
            modified[0][key] = value
            with self.assertRaises(ValueError):
                compare(b, modified, replicates=10)

    def test_missing_and_duplicate_row_rejected(self):
        b = load(self.export / 'agent/base/FREE/RTN.jsonl')
        for candidate in (b[:-1], b + [b[0]]):
            with self.assertRaises(ValueError):
                compare(b, candidate, replicates=10)

    def test_memory_policy_is_not_zero_score(self):
        self.assertEqual(len(self.manifest['exclusions']), 3)
        self.assertEqual(len(load(self.export / 'long/32768/BF16.jsonl')), 12)
        self.assertFalse((self.export / 'long/32768/D16.jsonl').exists())
        self.assertEqual(self.manifest['independent_units']['agent_unique_base_tasks'], 8)
        self.assertEqual(self.manifest['independent_units']['agent_16k_variants_of_existing_tasks'], 4)

    def test_public_source_changes_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            fake = Path(d) / 'case'
            (fake / 'provenance').mkdir(parents=True)
            (fake / 'provenance/PUBLIC_MANIFEST.json').write_bytes((CASE / 'provenance/PUBLIC_MANIFEST.json').read_bytes())
            (fake / 'records').mkdir()
            (fake / 'records/experiment_manifest.json').write_text('{}\n')
            with self.assertRaisesRegex(ValueError, 'changed source file'):
                Reader(fake).json('records/experiment_manifest.json')

    def test_corrupt_saved_score_or_fixture_is_not_repaired(self):
        reader = Reader(CASE)
        row = next(r for r in reader.rows('records/raw/4b/D16/agent_dev.jsonl')
                   if r['id'] == 'AGENT-6' and r['mode'] == 'FREE')
        task = next(r for r in reader.json('records/inputs/agent_tasks.json') if r['id'] == 'AGENT-6')
        row['success'] = True
        with self.assertRaisesRegex(ValueError, 'source score mismatch'):
            score_agent(row, task, task['fixture']['config'])

    def test_stdlib_cli_help_and_fresh_output(self):
        from diova_compare import __version__
        self.assertEqual(__version__, '0.1.2')
        result = subprocess.run([sys.executable, '-B', '-m', 'diova_compare.cli', 'export-case013', '--help'],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        self.assertIn('--case', result.stdout)
        with self.assertRaisesRegex(ValueError, 'new directory'):
            export_case013(CASE, self.export)
        for name in self.manifest['files']:
            rows = load(self.export / name)
            self.assertTrue(all(r['evidence_kind'] == 'historical' and 'distribution' not in r for r in rows))
            self.assertEqual({r['process_exit_code'] for r in rows}, {0})


if __name__ == '__main__':
    unittest.main()
