import copy
import importlib.util
from pathlib import Path
import sys
import unittest
P=Path(__file__).resolve().parents[1]/'scripts';sys.path.insert(0,str(P))
from analyze_serving import aggregate_cell,round_ratio
from data_v2 import generate,audit,oracle


def fixture():
    return dict(elapsed_seconds=10,records=[dict(request_id=str(i),status='OK',input_tokens=128,output_tokens=256,finish_reason='length',ttft_seconds=.1,latency_seconds=2.65,tpot_seconds=.01) for i in range(64)])


class Contracts(unittest.TestCase):
    def test_tpot_seconds_to_ms(self):
        r=aggregate_cell(fixture(),(128,1),list(map(str,range(64))))
        self.assertAlmostEqual(r['tpot_p50_ms'],10);self.assertAlmostEqual(r['output_tokens_per_second'],64*256/10)
    def test_wrong_tpot_denominator(self):
        x=fixture();x['records'][0]['tpot_seconds']=2.65/256
        with self.assertRaisesRegex(ValueError,'TPOT'):aggregate_cell(x,(128,1),list(map(str,range(64))))
    def test_duplicate_missing(self):
        x=fixture();x['records'].pop()
        with self.assertRaises(ValueError):aggregate_cell(x,(128,1),list(map(str,range(64))))
        x=fixture();x['records'][1]['request_id']='0'
        with self.assertRaises(ValueError):aggregate_cell(x,(128,1),list(map(str,range(64))))
    def test_failures_not_dropped(self):
        x=fixture();x['records'][0]['status']='OOM'
        r=aggregate_cell(x,(128,1),list(map(str,range(64))))
        self.assertEqual(r['status'],'INVALID');self.assertEqual(r['failures'],1);self.assertNotIn('tpot_p50_ms',r)
    def test_nan_length(self):
        for key,val in [('latency_seconds',float('nan')),('input_tokens',127),('output_tokens',255)]:
            x=fixture();x['records'][0][key]=val
            with self.assertRaises(ValueError):aggregate_cell(x,(128,1),list(map(str,range(64))))
    def test_rounds_not_requests(self):
        b={i:{'m':2} for i in range(3)};c={i:{'m':1} for i in range(3)}
        r=round_ratio(b,c,'m');self.assertEqual(r['n_independent_rounds'],3);self.assertEqual(r['pointwise_95_interval'],[2,2])
        c.pop(2)
        with self.assertRaises(ValueError):round_ratio(b,c,'m')
    def test_data_oracle(self):
        rows=generate();r=audit(rows)
        self.assertEqual(r['heldout']['n'],192);self.assertEqual(set(r['heldout']['label_counts'].values()),{48})
        self.assertEqual(rows,generate())
    def test_old_count_label_coupling_detected(self):
        rows=generate()
        for i,r in enumerate(rows):
            f=r['facts'];f['count']=r['gold']+2;f['modulus']=23;f['start']=(5-f['step']*f['count']*(f['count']-1)//2)%23+23*(i+10)
            r['answer']=oracle(f);r['options']=[4,6,7];r['options'].insert(r['gold'],r['answer'])
        with self.assertRaisesRegex(ValueError,'count-label'):audit(rows)
    def test_foils_both_sides_and_no_duplicates(self):
        rows=generate();rows[0]['options']=[rows[0]['answer']]*4
        with self.assertRaises(ValueError):audit(rows)
    def test_out_of_domain_foil_rejected(self):
        rows=generate();r=rows[0];r['options'][(r['gold']+1)%4]=-1
        with self.assertRaisesRegex(ValueError,'domain'):audit(rows)
    def test_no_repeated_program_facts_across_splits(self):
        rows=generate();self.assertEqual(len(rows),len({tuple(sorted(r['facts'].items())) for r in rows}))
    def test_missing_fixture_rejected(self):
        with self.assertRaisesRegex(ValueError,'missing/extra'):audit(generate()[:-1])

if __name__=='__main__':unittest.main()
