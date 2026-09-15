"""Post-measurement protocol/record tests; do not alter frozen experiment code."""
import sys,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import CASE,sha,fingerprint
from src.protocol import verify_phase
from src.anchor import validate_geometry
from src.decision import numerical,choose_dev
class AuditTests(unittest.TestCase):
 def test_actual_freeze(self):self.assertEqual(verify_phase('a')[1],'232be04d7a9275a06664a4403f459b78e61b63060cd1305007d7c9270a220652')
 def test_bad_geometry(self):
  def x(shape,dtype='bf16'):return SimpleNamespace(shape=shape,dtype=dtype,device='cpu')
  q=x((1,16,32,128));k=x((1,8,32,128));validate_geometry(q,k,k)
  with self.assertRaises(ValueError):validate_geometry(x((1,15,32,128)),k,k)
  with self.assertRaises(ValueError):validate_geometry(x((1,16,16,128)),k,k)
  with self.assertRaises(TypeError):validate_geometry(q,k,x(k.shape,'fp32'))
 def test_exact_dev_ids(self):
  docs=json.loads((CASE/'inputs/dev_manifest.json').read_text())['documents'];self.assertEqual(len(docs),8)
  for d in docs:self.assertEqual(sha(CASE/d['text_file']),d['text_sha256']);self.assertEqual(fingerprint(d['token_ids_4096'][:512]),d['prefix_sha256']['512'])
 def test_no_fresh_after_stop(self):
  s=json.loads((CASE/'results/dev_summary.json').read_text());self.assertIsNone(choose_dev(s['rows']));self.assertFalse((CASE/'configs/phase_b.json').exists());self.assertFalse((CASE/'inputs/fresh_manifest.json').exists())
 def test_not_independent_rounds(self):
  for i in range(2):
   r=json.loads((CASE/f'results/dev/round-{i}.json').read_text());self.assertEqual(len({m['document_id'] for m in r['measurements']}),8);self.assertTrue(all(m['split']=='dev' for m in r['measurements']))
 def test_v3_not_benchmarked(self):
  for i in range(2):self.assertNotIn('V3',{m['config_id'] for m in json.loads((CASE/f'results/dev/round-{i}.json').read_text())['measurements']})
 def test_native_and_all_output_finite(self):
  for i in range(2):
   for m in json.loads((CASE/f'results/dev/round-{i}.json').read_text())['measurements']:
    self.assertEqual(m['full_output_nonfinite'],0)
    if m['config_id']=='B':self.assertTrue(m['native_bitwise_equal'])
 def test_protocol_tamper_detected(self):
  import src.protocol as module
  original=module.CASE
  with tempfile.TemporaryDirectory() as tmp:
   c=Path(tmp);(c/'configs').mkdir();(c/'x').write_text('original');spec={'frozen_files':{'x':sha(c/'x')}};p=c/'configs/phase_a.json';p.write_text(json.dumps(spec));(c/'configs/phase_a.sha256').write_text(sha(p));module.CASE=c
   try:
    verify_phase('a');(c/'x').write_text('changed')
    with self.assertRaises(AssertionError):verify_phase('a')
   finally:module.CASE=original
if __name__=='__main__':unittest.main()
