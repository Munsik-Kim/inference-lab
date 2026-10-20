"""Post-run CPU review properties; not new model measurements or task selection."""
import copy
import random
import unittest
from kv_agent_bench.fixture_generator import make_tasks,oracle

class EvidenceProperties(unittest.TestCase):
    def test_late_arrival_changes_eligibility_when_moved_before_cutoff(self):
        task=next(t for t in make_tasks('CPU_REVIEW',1,6101901)if t['family']=='C');before=oracle(task)
        late=next(s for s in task['visible']['sources']if s['arrival_time']>task['visible']['cutoff'])
        late['arrival_time']='2026-10-18T23:59:59Z';after=oracle(task)
        self.assertNotEqual(before['source_id'],after['source_id']);self.assertNotEqual(before['total_minor'],after['total_minor'])
    def test_cancel_precedence_and_uncancel_change_state(self):
        task=next(t for t in make_tasks('CPU_REVIEW',1,6101901)if t['family']=='D');before=oracle(task)
        self.assertEqual(before['status'],'cancelled');self.assertEqual(before['total_minor'],0)
        cancel=next(s for s in task['visible']['sources']if s['kind']=='cancel');rival=copy.deepcopy(cancel);rival['source_id']='INDEPENDENT-ID';rival['kind']='correction';task['visible']['sources'].insert(0,rival)
        self.assertEqual(oracle(task)['source_id'],cancel['source_id'])
        cancel['kind']='correction';after=oracle(task);self.assertEqual(after['status'],'active');self.assertGreater(after['total_minor'],0)
    def test_id_renaming_and_source_order_do_not_reveal_gold(self):
        for i,task in enumerate(make_tasks('CPU_REVIEW',3,6101901)):
            before=oracle(task);mapping={s['source_id']:f'RENAMED-{i}-{j}'for j,s in enumerate(task['visible']['sources'])}
            for row in task['visible']['sources']:row['source_id']=mapping[row['source_id']]
            random.Random(i).shuffle(task['visible']['sources']);after=oracle(task)
            self.assertEqual(after['source_id'],mapping[before['source_id']])
            for key in ('quantity','unit_price_minor','discount_percent','total_minor','source_version','status'):self.assertEqual(before[key],after[key])

if __name__=='__main__':unittest.main()
