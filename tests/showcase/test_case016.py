import copy
import sys
import unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/showcase'))
from case016 import load_case016, render_case016, home_case016, SECTIONS, table


class Case016Display(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_case016(ROOT)

    def test_all_conditions_and_original_negative_result_visible(self):
        self.assertEqual(len(self.data['per_round']), 36)
        for lang in ('en', 'ko'):
            html = render_case016(self.data, lang)
            for sid in SECTIONS:
                self.assertIn('id="' + sid + '"', html)
            for token in ('29', '103552', '0/48', '6/48', 'NOT_MET'):
                self.assertIn(token, html)
            self.assertNotIn('<script', html)

    def test_home_capacity_and_speed_are_data_driven(self):
        value = copy.deepcopy(self.data)
        value['capacity_ratio'] = 2.5
        for row in value['summary']:
            row['ratio_median'] = 0.5
        html = home_case016(value, 'en')
        self.assertIn('2.50×', html)
        self.assertIn('0.50×', html)
        self.assertNotEqual(html, home_case016(self.data, 'en'))

    def test_raw_text_escaped(self):
        value = copy.deepcopy(self.data)
        value['fidelity'][0]['F4_text'] = '<script>alert(1)</script>'
        html = render_case016(value, 'en')
        self.assertIn('&lt;script&gt;', html)
        self.assertNotIn('<script>alert', html)

    def test_table_keyboard_scroll_and_escape(self):
        html = table(['<h>'], [['<img src=x>']], 'test')
        self.assertIn('tabindex="0"', html)
        self.assertIn('&lt;img', html)
        self.assertNotIn('<img', html)
