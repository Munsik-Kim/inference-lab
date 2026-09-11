"""Run the small numerical suite and write the freeze prerequisite."""
import json
import sys
import unittest
from prepare import CASE, write_json

sys.path.insert(0, str(CASE/'tests'))
import test_numerics

result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test_numerics))
write_json(CASE/'provenance/numerical_tests.json', {'tests_run': result.testsRun, 'success': result.wasSuccessful(),
    'skipped': len(result.skipped), 'failures': len(result.failures), 'errors': len(result.errors),
    'scope': 'FP64 scalar and JS analytic references, CPU/GPU match, real Qwen fixture, masks, boundaries, shared operand, underflow.'})
raise SystemExit(0 if result.wasSuccessful() and not result.skipped else 1)
