"""Mutation fixtures prove the CPU audit refuses changed evidence."""
import pathlib,sys,copy,tempfile,unittest,json,hashlib,importlib.util
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import audit
class TamperTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.protocol=audit.read(ROOT/'configs/protocol.json');cls.cell=audit.read(ROOT/'provenance/planned_cells.json')[0];cls.raw=audit.read(ROOT/'results/20261026-1215'/f"{cls.cell['cell_id']}.json")
 def test_changed_raw_sample_hash_is_rejected(self):
  raw=copy.deepcopy(self.raw);raw['timing']['host_us'][0]+=1
  with tempfile.TemporaryDirectory() as t:
   p=pathlib.Path(t)/'mutated.json';p.write_text(json.dumps(raw))
   original=ROOT/'results/20261026-1215'/f"{self.cell['cell_id']}.json"
   with self.assertRaisesRegex(ValueError,'hash mismatch'):audit.check_hash(p,audit.digest(original))
 def test_changed_input_manifest_U_is_rejected(self):
  changed=copy.deepcopy(self.cell);changed['U']+=1
  with self.assertRaisesRegex(ValueError,'cell identity mismatch U'):audit.validate_raw(self.raw,changed,self.protocol)
 def test_dtype_payload_units_are_recomputed(self):
  raw=copy.deepcopy(self.raw);raw['resident_inventory'][0]['payload_bytes']*=1024
  with self.assertRaisesRegex(ValueError,'payload mismatch'):audit.validate_raw(raw,self.cell,self.protocol)
 def test_positive_fixture_and_no_torch_import(self):
  self.assertGreater(audit.validate_raw(self.raw,self.cell,self.protocol),0);self.assertNotIn('torch',sys.modules)
if __name__=='__main__':unittest.main()
