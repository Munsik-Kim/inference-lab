"""Publication checks added after measurement; no GPU or model imports."""
import json
import re
import sys
import unittest
from pathlib import Path

CASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CASE))
from scripts.recalculate_posthoc import calculate, paired_units, scalar_error
from scripts.write_report import dev_table


class PublicationTests(unittest.TestCase):
    def test_join_uses_document_and_head_not_position(self):
        a = [('a', 0, .03), ('b', 1, .01)]
        b = [('b', 1, .02), ('a', 0, .02)]
        pairs = paired_units(a, b)
        self.assertGreater(pairs[0]['error_reduction_pp'], 0)
        self.assertLess(pairs[1]['error_reduction_pp'], 0)

    def test_duplicate_pair_is_rejected(self):
        with self.assertRaises(ValueError):
            paired_units([('a', 0, .01)] * 2, [('a', 0, .02)])

    def test_missing_pair_is_rejected(self):
        with self.assertRaises(ValueError):
            paired_units([('a', 0, .01)], [('b', 0, .02)])

    def test_undefined_scalar_is_not_repaired(self):
        for errors, reference in [([1], [0]), ([float('nan')], [1])]:
            with self.assertRaises(ValueError):
                scalar_error(dict(row_error_norms=errors, row_reference_norms=reference))

    def test_posthoc_matches_raw_without_promoting_finalist(self):
        generated = calculate(CASE)
        recorded = json.loads((CASE / 'results/posthoc_review.json').read_text())
        self.assertEqual(generated, recorded)
        self.assertEqual(generated['v4_paired']['lower'], 128)
        self.assertEqual(generated['independent_documents'], 8)
        self.assertFalse(generated['fresh_confirmation'])
        self.assertEqual(generated['sensitivity'][0]['passing_new_configs'], [])
        original = json.loads((CASE / 'results/dev_summary.json').read_text())
        self.assertIsNone(original['finalist_id'])
        self.assertEqual(original['final_decision'], 'STOP_DEV_SCREEN')
        for r in generated['rows']:
            old = next(x for x in original['rows'] if x['config_id'] == r['config_id'])
            self.assertAlmostEqual(r['median_error'], old['numerical']['median'], places=12)
            self.assertAlmostEqual(r['p95_error'], old['numerical']['p95'], places=12)
            self.assertAlmostEqual(r['speedup'], old['speedup'], places=12)

    def test_document_tables_are_measured_results(self):
        expected = dev_table(json.loads((CASE / 'results/dev_summary.json').read_text())).strip()
        for name in ['README.md', 'ANALYSIS.md']:
            text = (CASE / name).read_text()
            match = re.search(r'<!-- BEGIN DEV TABLE -->\n(.*?)\n<!-- END DEV TABLE -->', text, re.S)
            self.assertIsNotNone(match, name)
            self.assertEqual(match.group(1).strip(), expected, name)


if __name__ == '__main__':
    unittest.main()
