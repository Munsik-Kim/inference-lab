"""Post-run CPU checks; do not modify the frozen experiment's existing tests."""
import hashlib,json,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.data import scenario,render
from src.core import load,digest
from scripts.analyze_partial import summarize
CASE=Path(__file__).resolve().parents[1]
class ArtifactChecks(unittest.TestCase):
 def test_frozen_sources(self):
  for name,h in load(CASE/'configs/protocol.json')['code_hashes'].items():self.assertEqual(hashlib.sha256((CASE/name).read_bytes()).hexdigest(),h,name)
 def test_inputs_unique_frozen(self):
  spec=load(CASE/'configs/protocol.json');allrows=[]
  for split,h in spec['inputs'].items():
   p=CASE/f'inputs/{split}.json';self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),h);rows=load(p);allrows+=rows
   for r in rows:self.assertEqual(len(r['token_ids']),512);self.assertFalse(r['future_answer_tokens_present']);self.assertEqual(r['token_hash'],digest(r['token_ids']));self.assertEqual(len(set(r['label_ids'])),4)
  for key in ['id','token_hash','facts_hash']:self.assertEqual(len({r[key] for r in allrows}),342)
 def test_historical_partial_and_completion(self):
  self.assertEqual(load(CASE/'results/partial_summary.json')['status'],'BLOCKED_RESOURCE')
  self.assertEqual(len(list((CASE/'results/raw/heldout').glob('c007-*'))),960)
  self.assertEqual(load(CASE/'results/raw/heldout/summary.json')['status'],'PASS')
  self.assertEqual(load(CASE/'RUN_STATE.json')['status'],load(CASE/'results/derived/summary.json')['status'])
 def test_actual_alias(self):
  x=load(CASE/'provenance/structural_sizes.json')['modules'];self.assertEqual(x['PAIRWISE_8']['weights_sha256'],x['INDEPENDENT_8']['weights_sha256']);self.assertNotEqual(x['PAIRWISE_4']['weights_sha256'],x['INDEPENDENT_4']['weights_sha256'])
 def test_report_rejects_wrong_status(self):
  import subprocess,tempfile
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'mock.json';p.write_text('{"status":"MOCK"}')
   r=subprocess.run([sys.executable,'-B',str(CASE/'scripts/report_partial.py'),'--case',str(CASE),'--summary',str(p),'--output',str(Path(tmp)/'out')],capture_output=True);self.assertNotEqual(r.returncode,0);self.assertFalse((Path(tmp)/'out').exists())
 def test_no_missing_full_output_evidence(self):
  for stage in ['calibrate','development','heldout']:
   for p in (CASE/'results/raw'/stage).glob('c007-*'):
    r=load(p);self.assertEqual(r['full_blocks_checked'],28);self.assertTrue(r['full_final_norm_checked']);self.assertTrue(r['full_logits_finite']);self.assertTrue(r['valid'])
 def test_completion_rejects_missing_full_validity(self):
  from scripts.verify_completion import require_validity
  r={'method':'B','valid':True,'full_blocks_checked':28,'full_final_norm_checked':True,'full_logits_finite':True,'selection_sha256':'test'}
  require_validity(r,'test')
  for key in ['valid','full_blocks_checked','full_final_norm_checked','full_logits_finite']:
   bad=dict(r);del bad[key]
   with self.assertRaises(ValueError):require_validity(bad,'test')
 def test_completion_rejects_invalid_unsampled_output(self):
  from scripts.verify_completion import require_validity
  r={'method':'B','valid':False,'full_blocks_checked':28,'full_final_norm_checked':False,'full_logits_finite':True,'selection_sha256':'test'}
  with self.assertRaises(ValueError):require_validity(r,'test')
 def test_completion_rejects_undefined_equivalence(self):
  from scripts.verify_completion import require_validity
  r={'method':'PAIRWISE_4','valid':True,'full_blocks_checked':28,'full_final_norm_checked':True,'full_logits_finite':True,'selection_sha256':'test'}
  for value in [None,float('nan'),float('inf'),.0101]:
   r['integrated_vs_standalone']={'relative_error':value}
   with self.assertRaises(ValueError):require_validity(r,'test')
if __name__=='__main__':unittest.main()
