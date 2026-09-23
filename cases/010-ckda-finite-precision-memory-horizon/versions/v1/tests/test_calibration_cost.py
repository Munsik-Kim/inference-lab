"""Cost-accounting checks without executing calibration, inference, or timing."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import measure_calibration_cost as cost


class CalibrationCostTests(unittest.TestCase):
    def test_every_array_compared_without_mutation(self):
        original = {"basis":np.arange(8,dtype=np.float32).reshape(2,4),"bits":np.array([4,8],dtype=np.uint8)}
        actual = {name:array.copy() for name,array in original.items()}
        exact = cost.compare_arrays(actual,original)
        self.assertTrue(exact["all_arrays_byte_exact"])
        self.assertEqual(exact["returned_stats_array_nbytes"],34)
        actual["basis"][0,0] += .25
        changed = cost.compare_arrays(actual,original)
        self.assertFalse(changed["all_arrays_byte_exact"])
        self.assertEqual(changed["arrays"]["basis"]["maximum_absolute_difference"],.25)
        self.assertEqual(changed["arrays"]["basis"]["different_value_count"],1)
        self.assertEqual(original["basis"][0,0],0)
        self.assertTrue(changed["arrays"]["bits"]["byte_exact"])

    def test_missing_shape_and_nonfinite_differences_explicit(self):
        reference = {"a":np.zeros(2,dtype=np.float32),"b":np.ones(2)}
        result = cost.compare_arrays({"a":np.zeros(3,dtype=np.float32),"c":np.array([np.nan])},reference)
        self.assertFalse(result["array_keys_equal"])
        self.assertFalse(result["all_arrays_byte_exact"])
        self.assertIsNone(result["arrays"]["a"]["maximum_absolute_difference"])
        self.assertFalse(result["arrays"]["b"]["actual_present"])
        self.assertEqual(result["arrays"]["c"]["actual"]["nonfinite_elements"],1)

    def test_timer_wraps_exactly_one_fit_and_excludes_comparison(self):
        events=[]
        times=iter([10.,10.25])
        reference={"covariance":np.zeros((2,2),dtype=np.float64)}
        table,tokens=object(),object()
        def clock():
            events.append("clock")
            return next(times)
        def calibrator(received_table,received_tokens):
            events.append("fit")
            self.assertIs(received_table,table)
            self.assertIs(received_tokens,tokens)
            return reference.copy()
        original=cost.compare_arrays
        def comparison(left,right):
            events.append("compare")
            return original(left,right)
        with patch.object(cost,"compare_arrays",side_effect=comparison):
            result=cost.measure_once(calibrator,table,tokens,reference,clock=clock)
        self.assertEqual(events,["clock","fit","clock","compare"])
        self.assertEqual(result["wall_seconds"],.25)
        self.assertEqual(result["timed_calls"],1)
        self.assertEqual(result["returned_stats_array_nbytes"],32)
        self.assertFalse(result["stats_storage_is_peak_memory"])

    def test_frozen_config_keeps_original_cal_and_one_call(self):
        config,_=cost.provenance.frozen_json(cost.ROOT/"configs/calibration_cost.json")
        inspections=[dict(manifest=dict(model_seed=seed,protocol_sha256=config["original_protocol_sha256"]),
                          protocol=dict(splits=dict(CAL=dict(sequences=64,length=32,seed=1001)))) for seed in range(3)]
        cost.validate_config(config,inspections)
        changed=copy.deepcopy(config)
        changed["timed_calls_per_model_seed"]=2
        with self.assertRaisesRegex(ValueError,"one calibration call"):
            cost.validate_config(changed,inspections)
        changed=copy.deepcopy(config)
        changed["seed"]=3001
        with self.assertRaisesRegex(ValueError,"original CAL"):
            cost.validate_config(changed,inspections)

    def test_isolated_completion_requires_all_matching_checkpoints(self):
        with tempfile.TemporaryDirectory(prefix="ckda-cal-cost-tests-") as temporary:
            directory=Path(temporary)
            inspections=[dict(manifest=dict(model_seed=seed,checkpoint_sha256=f"checkpoint{seed}")) for seed in range(3)]
            index=dict(status="COMPLETE",serial=True,seeds={})
            for seed in range(3):
                path=directory/f"seed{seed}.json"
                cost.provenance.write_json(path,dict(model_seed=seed,checkpoint_sha256=f"checkpoint{seed}"))
                index["seeds"][str(seed)]=dict(file=path.name,sha256=cost.provenance.sha256(path))
            path=directory/"index.json"
            cost.provenance.write_json(path,index)
            path.with_suffix(".sha256").write_text(cost.provenance.sha256(path)+"\n")
            self.assertEqual(cost.verify_isolated_complete(directory,inspections),cost.provenance.sha256(path))
            inspections[1]["manifest"]["checkpoint_sha256"]="different"
            with self.assertRaisesRegex(ValueError,"different checkpoint"):
                cost.verify_isolated_complete(directory,inspections)
            path.unlink()
            with self.assertRaises(FileNotFoundError):
                cost.verify_isolated_complete(directory,inspections)


if __name__ == "__main__":
    unittest.main()
