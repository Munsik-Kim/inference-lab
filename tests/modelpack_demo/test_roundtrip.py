"""Random CPU fixtures, excluded from scientific result tables."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('modelpack_roundtrip_demo',ROOT/'tools/modelpack_demo/roundtrip.py')
demo=importlib.util.module_from_spec(spec);spec.loader.exec_module(demo)

class Roundtrip(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory()
        cls.output=Path(cls.tmp.name)/'demo with spaces'
        cls.result=demo.run(cls.output)
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def test_real_two_process_roundtrip_and_source_isolation(self):
        r=self.result
        for key in ('separate_processes','different_directories','build_process_exited_before_reload','original_build_access_blocked','copy_checksum_identity'):
            self.assertTrue(r[key])
        self.assertEqual(r['evidence_kind'],'TINY_RANDOM_CPU_FIXTURE')
        self.assertEqual(r['model_downloads'],0);self.assertEqual(r['gpu_runs'],0)
    def test_same_path_refuses_overwrite(self):
        with self.assertRaises(ValueError):demo.run(self.output)
    def test_absent_or_nonfinite_full_output_evidence_blocks(self):
        good=self.result['observation']
        for key in ('full_outputs_finite','full_hidden_checks','tied_weights','all_parameters_cpu_bf16','all_buffers_materialized'):
            for missing in (True,False):
                bad=copy.deepcopy(good)
                if missing:bad.pop(key)
                else:bad[key]=False
                with self.assertRaises(ValueError):demo.compare(good,bad)
    def test_shape_and_output_change_block(self):
        good=self.result['observation']
        for key,value in [('layer_widths',[64,64]),('logits_sha256','0'*64),('kind','gpu_measurement')]:
            bad={**good,key:value}
            with self.assertRaises(ValueError):demo.compare(good,bad)

if __name__=='__main__':unittest.main()
