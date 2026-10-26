import pathlib,sys,copy,unittest,tempfile,json
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'src'))
import audit
class AuditFixtures(unittest.TestCase):
 def test_manifest_mutation_and_unlisted_file_refused(self):
  with tempfile.TemporaryDirectory() as d:
   p=pathlib.Path(d);raw=p/'raw.json';raw.write_text('{"sample":1}\n');original=raw.read_bytes()
   (p/'PUBLIC_MANIFEST_P4.json').write_text(json.dumps({'files':{'raw.json':{'bytes':len(original),'sha256':audit.h(raw)}}}));audit.public_manifest(p)
   raw.write_text('{"sample":9}\n')
   with self.assertRaisesRegex(ValueError,'file changed'):audit.public_manifest(p)
   raw.write_bytes(original);(p/'unlisted.txt').write_text('new')
   with self.assertRaisesRegex(ValueError,'coverage'):audit.public_manifest(p)
 def test_host_unit_error_refused(self):
  row={'K':1,'start_ns':1000,'end_ns':11000,'host_us':10.,'event_us':12.,'fence':'end_event_synchronize'};audit.clock_row(row,'D')
  row['host_us']*=1000
  with self.assertRaisesRegex(ValueError,'units'):audit.clock_row(row,'D')
 def test_event_K_conversion_refused(self):
  row={'K':16,'host_us':None,'event_us':2.,'bracket_stream_us':32.,'fence':'end_event_synchronize'};audit.clock_row(row,'E');row['event_us']=32
  with self.assertRaisesRegex(ValueError,'bracket/K'):audit.clock_row(row,'E')
 def test_same_call_timers_need_not_match(self):
  row={'K':1,'start_ns':0,'end_ns':10000,'host_us':10.,'event_us':11.,'fence':'end_event_synchronize'};audit.clock_row(row,'D');self.assertNotIn('torch',sys.modules)
 def test_numeric_saved_PASS_not_trusted(self):
  er={'error_sumsq':[0.]*16,'reference_sumsq':[1.]*16,'max_abs':[0.]*16,'finite':[True]*16,'pass':True};audit.numerical(er);er['error_sumsq'][3]=1.
  with self.assertRaisesRegex(ValueError,'numeric'):audit.numerical(er)
if __name__=='__main__':unittest.main()
