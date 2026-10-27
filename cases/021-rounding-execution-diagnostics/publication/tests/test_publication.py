"""Mutation and independent formula checks on the actual public package."""
import importlib.util,json,os,subprocess,sys,tempfile,unittest,shutil
from pathlib import Path
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P))
from verify import verify,CASE,safe,sha
from derive import projection,numerical
class PublicContracts(unittest.TestCase):
 def test_source_input_raw_mutations_rejected(self):
  with tempfile.TemporaryDirectory() as tmp:
   c=Path(tmp)/'case';shutil.copytree(CASE,c)
   for name in ['publication/derive.py','inputs/synthetic/dev_code_00.txt','evidence/p0p3/primary.csv']:
    target=c/name
    if not target.exists() and name.startswith('inputs/'):target=next((c/'inputs/synthetic').glob('*.txt'))
    old=target.read_bytes();target.write_bytes(old+b'\n')
    with self.assertRaisesRegex(ValueError,'PUBLIC_BYTES'):verify(c)
    target.write_bytes(old)
   (c/'unexpected.json').write_text('{}')
   with self.assertRaisesRegex(ValueError,'PUBLIC_COVERAGE'):verify(c)
 def test_statistic_mutation_rejected_without_manifest(self):
  with tempfile.TemporaryDirectory() as tmp:
   c=Path(tmp)/'case';shutil.copytree(CASE,c)
   p=c/'evidence/p0p3/primary.csv';s=p.read_text();p.write_text(s.replace('0.000', '0.100',1))
   with self.assertRaises((ValueError,AssertionError)):projection(c)
 def test_safe_paths_and_output_no_overwrite(self):
  for path in ['../private','/absolute','a//b','a/./b','a\\b']:
   with self.assertRaises(ValueError):safe(path)
  with tempfile.TemporaryDirectory() as tmp:
   marker=Path(tmp)/'sentinel';marker.write_bytes(b'keep')
   p=subprocess.run([sys.executable,'-I','-B',str(P/'run_checks.py'),'--output',tmp],capture_output=True,text=True)
   self.assertNotEqual(p.returncode,0);self.assertEqual(marker.read_bytes(),b'keep');self.assertFalse((Path(tmp)/'receipt.json').exists())
 def test_no_gpu_import_and_help(self):
  p=subprocess.run([sys.executable,'-I','-B',str(P/'run_checks.py'),'--help'],capture_output=True,text=True)
  self.assertEqual(p.returncode,0);self.assertNotIn('torch',sys.modules)
 def test_current_documents_use_derived_facts(self):
  f=projection(CASE)
  for lang in ['','.ko']:
   report=(CASE/('REPORT'+lang+'.md')).read_text();readme=(CASE/('README'+lang+'.md')).read_text()
   for value in [f"{f['E0']:.13f}",f"{f['E1']:.13f}",f"{f['delta']:.13f}",f['original_primary_verdict']]:self.assertIn(value,report)
   self.assertIn(f['original_primary_verdict'],readme)
   for row in f['sparse']:self.assertIn(row['document_id'],report);self.assertIn(f"{row['K1024_relative_RMSE']:.10g}",report)
   self.assertIn('406,626',report);self.assertIn('4,102',report);self.assertIn('1,018',report)
 def test_synthetic_math_independent_integer(self):
  # Exact dyadic fixture: RNE to 8 significant binary bits, no float32 intermediary.
  from fractions import Fraction
  def rounded(x):
   if x==0:return x
   sign=-1 if x<0 else 1;x=abs(x);e=x.numerator.bit_length()-x.denominator.bit_length()
   if x<Fraction(2)**e:e-=1
   unit=Fraction(2)**(e-7);q=x/unit;n,r=divmod(q.numerator,q.denominator)
   if 2*r>q.denominator or (2*r==q.denominator and n%2):n+=1
   return sign*n*unit
  s=Fraction(129,128)
  expected=[(Fraction(1),Fraction(97,32),Fraction(97,32)),(Fraction(1,8),Fraction(97,256),Fraction(97,256)),(Fraction(3,2),Fraction(73,16),Fraction(145,32))]
  for a,A,B in expected:self.assertEqual(rounded(a*rounded(3*s)),A);self.assertEqual(rounded(a*3*s),B)
  # Cross terms may be negative; both decreasing individual errors and increasing gap are possible.
  d=[1.,-2.];n0=[-.5,1.];n1=[-.1,.2]
  energy=lambda a:sum(x*x for x in a)
  G=lambda n:energy([x+y for x,y in zip(d,n)])
  C=lambda n:2*sum(x*y for x,y in zip(d,n))
  self.assertAlmostEqual(G(n1)-G(n0),energy(n1)-energy(n0)+C(n1)-C(n0))
if __name__=='__main__':unittest.main()
