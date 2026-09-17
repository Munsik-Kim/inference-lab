"""CPU resource-policy regression tests; no GPU queries."""
import unittest
from scripts.run_case006 import classify_resource

class ResourceTests(unittest.TestCase):
    def samples(self,util,registered=0,free=14900):
        return [dict(utilization_percent=x,active_compute_process_count=registered,free_mib=free) for x in util]
    def test_registered_idle_process_is_not_busy(self):
        self.assertEqual(classify_resource(self.samples([1]*5,1)),'RESOURCE_GATE_PASSED')
    def test_single_transient_is_not_sustained_load(self):
        self.assertEqual(classify_resource(self.samples([2,18,3,2,1])),'RESOURCE_GATE_PASSED')
    def test_sustained_load_blocks_even_without_compute_pid(self):
        self.assertEqual(classify_resource(self.samples([22]*5)),'GPU_BUSY')
    def test_low_memory_and_missing_probe_block(self):
        self.assertEqual(classify_resource(self.samples([0]*5,free=12000)),'INSUFFICIENT_FREE_MEMORY')
        self.assertEqual(classify_resource([]),'GPU_ACCESS_BLOCKED')
