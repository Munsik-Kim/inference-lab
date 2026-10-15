"""Publication contracts with hand-calculated scalars; no model inference."""
import copy,importlib.util,json,math,sys,tempfile,unittest
from pathlib import Path
CASE=Path(__file__).resolve().parents[2];sys.path.insert(0,str(CASE/'publication/scripts'))
from derive import p2_metrics,unique,derive
from restore_stage import restore

class Publication(unittest.TestCase):
    def data(self):
        docs=[{'split':s,'document_id_hash':s+str(i),'token_hash':s+str(i)} for s in ['EVAL-ID','SHIFT'] for i in [0,1]];rows=[]
        for method,loss in [('BF16',2),('RTN',3),('COMP_QUARTER',2.5),('COMP_HALF',2.4)]:
            for d in docs:rows.append({**d,'method':method,'NLL_sum':255*loss,'valid_token_count':255,'KL_position_count':16,'sampled_full_vocab_KL_sum':0,'flip_count':0,'projection_core_bytes':10,'status':'COMPLETE'})
        return rows,docs
    def test_ppl_uses_token_nll_and_paired_loss(self):
        rows,docs=self.data();summary,_=p2_metrics(rows,docs);q=next(x for x in summary if x['split']=='EVAL-ID' and x['method']=='COMP_QUARTER')
        self.assertAlmostEqual(q['PPL'],math.exp(2.5));self.assertAlmostEqual(q['NLL_minus_RTN'],-.5);self.assertAlmostEqual(q['PPL_relative_change'],math.expm1(-.5));self.assertEqual(q['lower_documents'],2)
    def test_missing_duplicate_or_changed_prompt_rejected(self):
        rows,docs=self.data()
        for changed in [rows[:-1],rows+[rows[0]],copy.deepcopy(rows)]:
            if len(changed)==len(rows):changed[0]['token_hash']='wrong'
            with self.assertRaises((ValueError,KeyError)):p2_metrics(changed,docs)
    def test_invalid_denominator_not_silently_accepted(self):
        rows,docs=self.data();rows[0]['valid_token_count']=254
        with self.assertRaises(ValueError):p2_metrics(rows,docs)
    def test_historical_view_rejects_existing_output(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):restore(CASE,'P2',Path(d))
    def test_all_stages_close_and_manifest_paths_are_safe(self):
        with tempfile.TemporaryDirectory() as d:
            for stage in ['P0','P1','P1b','P2']:
                r=restore(CASE,stage,Path(d)/stage);self.assertTrue(r['closed_set']);self.assertEqual(r['status'],'PASS')
    def test_saved_display_can_only_match_actual_raw_projection(self):
        actual=derive(CASE);saved=json.loads((CASE/'publication/summary/project.json').read_text());self.assertEqual(actual,saved)
        changed=copy.deepcopy(saved);changed['p2']['methods'][0]['PPL']+=1;self.assertNotEqual(actual,changed)

if __name__=='__main__':unittest.main()
