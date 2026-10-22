"""Synthetic contract boundaries; never measured model episodes."""
import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from analysis.quantity import diagnose, strict_map
from verify import safe

def sample():
    task={'public_request':{'record_ids':['a','b'],'scope_id':'s'}}
    state={'scope_id':'s','revision':0,'records':{'a':{'quantity':4},'b':{'quantity':6}}}
    event={'event_id':'e0','tool':'read_records','arguments':{'record_ids':['a','b']},'generated_in_turn':0,
           'callable_executed':True,'status':'SUCCESS','result':{'records':[{'record_id':'a','quantity':4},{'record_id':'b','quantity':6}]},
           'state_before':copy.deepcopy(state),'state_after':copy.deepcopy(state)}
    action={'tool':'final','arguments':{'quantities':[{'record_id':'a','quantity':4},{'record_id':'b','quantity':6}]}}
    row={'requests':[{'available_result_event_ids':[],'decision':'TOOL'},
                     {'available_result_event_ids':['e0'],'decision':'FINAL_ACCEPTED','parse':{'action':action}}],
         'events':[event],'terminal_snapshot':copy.deepcopy(state),'final_action':action}
    return task,row

def mutation(row,a,b,metadata=False):
    prev=copy.deepcopy(row['terminal_snapshot']);after=copy.deepcopy(prev)
    after['revision']+=1;after['records']['a']['quantity']=a;after['records']['b']['quantity']=b
    n=len(row['events']);eid='e'+str(n)
    row['events'].append({'event_id':eid,'tool':'release_operation' if metadata else 'compensate_stock','generated_in_turn':n,
                         'state_before':prev,'state_after':after,'callable_executed':True,'status':'SUCCESS','result':{'revision':after['revision']}})
    row['terminal_snapshot']=after
    row['requests'].insert(-1,{'available_result_event_ids':[e['event_id'] for e in row['events'][:-1]],'decision':'TOOL'})
    row['requests'][-1]['available_result_event_ids'].append(eid)

class QuantityBoundaries(unittest.TestCase):
    def test_current_observed_read(self):
        t,r=sample();self.assertEqual(diagnose(t,r)['current_quantity_read_observed'],1)
    def test_request_not_observed(self):
        t,r=sample();r['requests'][-1]['available_result_event_ids']=[]
        self.assertEqual(diagnose(t,r)['current_quantity_read_observed'],0)
    def test_failed_read(self):
        t,r=sample();r['events'][0]['status']='ERROR';self.assertEqual(diagnose(t,r)['current_quantity_read_observed'],0)
    def test_wrong_scope(self):
        t,r=sample();r['events'][0]['state_before']['scope_id']='other';self.assertEqual(diagnose(t,r)['current_quantity_read_observed'],0)
    def test_partial_read(self):
        t,r=sample();r['events'][0]['result']['records'].pop();r['events'][0]['arguments']['record_ids']=['a']
        d=diagnose(t,r);self.assertEqual(d['current_read_coverage'],1);self.assertEqual(d['current_quantity_read_observed'],0);self.assertIsNone(d['final_matches_first_read'])
    def test_quantity_change_expires(self):
        t,r=sample();mutation(r,3,7);self.assertEqual(diagnose(t,r)['current_quantity_read_observed'],0)
    def test_metadata_only_does_not_expire_quantity(self):
        t,r=sample();mutation(r,4,6,True);d=diagnose(t,r)
        self.assertEqual(d['current_quantity_read_observed'],1);self.assertEqual(d['read_after_all_state_mutations_observed'],0)
    def test_change_then_restore_still_expires(self):
        t,r=sample();mutation(r,3,7);mutation(r,4,6);self.assertEqual(diagnose(t,r)['current_quantity_read_observed'],0)
    def test_CPU_continuation_read_not_counted(self):
        t,r=sample();r['requests'][-1]['available_result_event_ids']=[];r['continuations']={'CPU_reads':[copy.deepcopy(r['events'][0])]}
        self.assertEqual(diagnose(t,r)['current_quantity_read_observed'],0)
    def test_no_accepted_final_is_NA(self):
        t,r=sample();r['requests'][-1]['decision']='GUARD_LIMIT';r['final_action']=None;d=diagnose(t,r)
        self.assertEqual(d['has_accepted_final'],0);self.assertIsNone(d['final_quantity_matches_terminal']);self.assertEqual(d['anchor_kind'],'LAST_GENERATION_DIAGNOSTIC')
    def test_final_incorrect_does_not_change_observation_flag(self):
        t,r=sample();r['final_action']['arguments']['quantities'][0]['quantity']=999;d=diagnose(t,r)
        self.assertEqual(d['current_quantity_read_observed'],1);self.assertFalse(d['final_quantity_matches_terminal'])
    def test_missing_or_extra_final_key(self):
        for change in ('missing','extra'):
            t,r=sample();a=r['final_action']['arguments']['quantities']
            a.pop() if change=='missing' else a.append({'record_id':'c','quantity':1})
            self.assertFalse(diagnose(t,r)['final_quantity_matches_terminal'])
    def test_bool_is_not_integer(self):
        t,r=sample();r['final_action']['arguments']['quantities'][0]['quantity']=True
        self.assertFalse(diagnose(t,r)['final_quantity_matches_terminal'])
    def test_duplicate_key_rejected(self):
        self.assertEqual(strict_map([{'record_id':'a','quantity':1},{'record_id':'a','quantity':1}],['a'])[1],'duplicate_record')
    def test_unknown_or_future_event_rejected(self):
        for eid in ['missing','e0']:
            t,r=sample();r['requests'][-1]['available_result_event_ids']=[eid];r['events'][0]['generated_in_turn']=1
            with self.assertRaises(ValueError):diagnose(t,r)
    def test_archive_unsafe_names(self):
        for name in ['/tmp/x','a/../x','a//x','a/./x','a\\x','C:x']:
            with self.assertRaises(ValueError):safe(name)
    def test_union_of_partial_reads(self):
        t,r=sample();first=r['events'][0];second=copy.deepcopy(first)
        first['arguments']['record_ids']=['a'];first['result']['records']=first['result']['records'][:1]
        second['event_id']='e1';second['generated_in_turn']=1;second['arguments']['record_ids']=['b'];second['result']['records']=second['result']['records'][1:]
        r['events'].append(second);r['requests'].insert(1,{'available_result_event_ids':['e0'],'decision':'TOOL'});r['requests'][-1]['available_result_event_ids'].append('e1')
        self.assertEqual(diagnose(t,r)['current_quantity_read_observed'],1)

if __name__=='__main__':unittest.main()
