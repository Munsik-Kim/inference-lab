import copy
import importlib.util
from pathlib import Path
import stat
import sys
import tempfile
import unittest
import zipfile
CASE=Path(__file__).resolve().parents[2];sys.path.insert(0,str(CASE/'publication/scripts'))
from restore_p1 import restore
from bundle import validate_names
from derive import derive


class Publication(unittest.TestCase):
    def test_original_view_is_closed_and_byte_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            result=restore(CASE,Path(tmp)/'original');self.assertEqual(result['restored_original_files'],228)
            self.assertFalse((Path(tmp)/'original/followups').exists())
            old=Path(tmp)/'original/publication/PUBLIC_MANIFEST.json';self.assertTrue(old.exists())

    def test_zip_traversal_duplicate_and_symlink_rejected(self):
        for names in [['../outside'],['/absolute'],['C:/outside'],['one','one'],['a\\b']]:
            with self.assertRaises(ValueError):validate_names([zipfile.ZipInfo(n) for n in names])
        link=zipfile.ZipInfo('link');link.external_attr=(stat.S_IFLNK|0o777)<<16
        with self.assertRaises(ValueError):validate_names([link])

    def test_raw_derived_diagnostic_corrections_leave_primary_outcomes(self):
        data=derive(CASE);d=data['A1']['diagnostic_annotations']
        self.assertEqual(d['malformed_early_stop_terminal_annotations'],8);self.assertEqual(d['bad_argument_value_episodes'],1)
        self.assertEqual(d['primary_outcomes_affected'],0)
        failures=[r for r in d['episodes'] if r['terminal_annotation']]
        self.assertTrue(all(r['verified_workflow_success_unchanged'] is False and r['derived_terminal_reason']=='PARSE_ERROR' for r in failures))

    def test_closed_public_v1_restores_without_changing_its_manifest(self):
        from restore_unified_v1 import restore as restore_v1
        with tempfile.TemporaryDirectory() as tmp:
            view=Path(tmp)/'v1';self.assertEqual(restore_v1(CASE,view)['original_v1_files'],423)
            spec=importlib.util.spec_from_file_location('preserved_v1_verify',view/'publication/verify.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
            self.assertEqual(m.verify(view)['status'],'PASS')

    def test_review_renderer_changes_only_the_known_display_separator(self):
        path=CASE/'followups/order-api-regression-v1/postprocess/render_review.py'
        spec=importlib.util.spec_from_file_location('a1_review_renderer',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        html="<script id=records type=application/json>{\"raw\":\"unchanged\"}</script><script>const x=['a','b'].join('\n\n');</script>"
        result,script=m.repair_html(html)
        self.assertIn(".join('\\n\\n')",script);self.assertIn('src="explorer.js"',result);self.assertIn('{"raw":"unchanged"}',result)
        with self.assertRaises(ValueError):m.repair_html('<script>unexpected</script>')

    def test_closed_public_v2_restores_before_explanation_revision(self):
        from restore_unified_v2 import restore as restore_v2
        with tempfile.TemporaryDirectory() as tmp:
            view=Path(tmp)/'v2'
            self.assertEqual(restore_v2(CASE,view)['original_v2_files'],443)
            spec=importlib.util.spec_from_file_location('preserved_v2_verify',view/'publication/verify.py')
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            result=module.verify(view)
            self.assertEqual(result['status'],'PASS');self.assertEqual(result['display_version'],2)


if __name__=='__main__':unittest.main()
