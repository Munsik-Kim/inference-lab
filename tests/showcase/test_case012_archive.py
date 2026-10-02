"""Keep Case012 discoverable without implying a deployed result page."""
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/showcase'))
from build import build
from case012 import load_case012
from check import check


class Case012Archive(unittest.TestCase):
    def test_current_pending_scope_and_archive(self):
        self.assertIsNone(load_case012(ROOT)['quality']['primary'])
        with TemporaryDirectory() as t:
            site = Path(t) / 'site'
            build(ROOT, site)
            for lang in ('en', 'ko'):
                html = (site / lang / 'index.html').read_text()
                for n in range(1, 13):
                    self.assertEqual(html.count(f'<span class="archive-number">{n:03}</span>'), 1)
                self.assertIn(f'docs/{lang}/CASEBOOK.md#case-012', html)
                self.assertFalse((site / lang / 'case012.html').exists())
            self.assertFalse((site / 'data/case012.json').exists())
            check(ROOT, site)

    def test_pending_archive_rejects_invented_quality(self):
        with TemporaryDirectory() as t:
            root = Path(t)
            p = root / 'cases/012-looped-dit-inference-budget/analysis/summary.json'
            p.parent.mkdir(parents=True)
            data = json.loads((ROOT / 'cases/012-looped-dit-inference-budget/analysis/summary.json').read_text())
            data['quality']['primary'] = {'estimate': 1.0}
            p.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'pending-quality'):
                load_case012(root)

    def test_pending_archive_requires_every_setting(self):
        with TemporaryDirectory() as t:
            root = Path(t)
            p = root / 'cases/012-looped-dit-inference-budget/analysis/summary.json'
            p.parent.mkdir(parents=True)
            data = json.loads((ROOT / 'cases/012-looped-dit-inference-budget/analysis/summary.json').read_text())
            data['timing']['B_time']['images'] = 63
            p.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'denominator'):
                load_case012(root)
