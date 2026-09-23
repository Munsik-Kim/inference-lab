"""Runner boundary checks with model/input/checkpoint execution fully mocked."""
from contextlib import ExitStack
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from source import run_fresh


class RunnerContractTests(unittest.TestCase):
    def test_bos_only_resume_does_not_prepend_second_bos(self):
        adapter = SimpleNamespace(config_bytes=b"config", basis_bytes=b"")
        resumed_state = object()
        tokens = np.zeros((2, 2048), np.int64)
        gold = np.zeros((2, 2048), np.uint8)
        counts = {"active_update_attempts": 2, "terminal_noop_steps": 0, "invalid_readout_counts": [0, 0]}
        restored = (adapter, resumed_state, np.zeros((2, 1), np.int8), gold[:, :0],
                    {"scored_offset": 0, "counts": counts})
        calls = []

        class StopAfterBoundaryCheck(Exception):
            pass

        def check_call(*args, **kwargs):
            calls.append(kwargs)
            raise StopAfterBoundaryCheck()

        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            stack.enter_context(patch.object(run_fresh, "verify_protocol", return_value={}))
            stack.enter_context(patch.object(run_fresh, "load_cohort", return_value=(tokens, gold, ["a", "b"], {"sample_ids_sha256": "0" * 64, "tokens_file_sha256": "1" * 64})))
            stack.enter_context(patch.object(run_fresh, "load_model", return_value=(None, {})))
            stack.enter_context(patch.object(run_fresh, "load_arms", return_value=({"NATIVE_FP32": adapter}, {})))
            stack.enter_context(patch.object(run_fresh, "file_sha", return_value="2" * 64))
            stack.enter_context(patch.object(run_fresh, "load_checkpoint", return_value=restored))
            stack.enter_context(patch.object(run_fresh, "sequence_call", side_effect=check_call))
            args = SimpleNamespace(seed=0, arm="NATIVE_FP32", upstream="unused", checkpoint="unused",
                                   output=Path(directory) / "out", private_output=Path(directory) / "private",
                                   resume="BOS-only-checkpoint", deadline_epoch=None)
            with self.assertRaises(StopAfterBoundaryCheck):
                run_fresh.run(args)
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0]["initial"], resumed_state)
        self.assertFalse(calls[0]["include_bos"], "a restored checkpoint has already consumed BOS, including scored_offset=0")


if __name__ == "__main__":
    unittest.main()
