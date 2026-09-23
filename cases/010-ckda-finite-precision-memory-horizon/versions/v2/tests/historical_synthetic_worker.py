"""Synthetic numeric test helper only: no checkpoint/model inference is loaded."""
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

CASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CASE))
from source import run_historical as runner


def synthetic_sequence_call(model, table, tokens, adapter, seeds=None, initial=None,
                            include_bos=True, diagnostics=False):
    if initial is not None and include_bos:
        raise ValueError("Synthetic resume must not add BOS")
    batch = len(tokens)
    state = adapter.initial(np.zeros((batch,) + adapter.state_codec.shape, np.float32), seeds=seeds) if initial is None else initial
    is_v2 = hasattr(adapter, "terminal_info")
    output = []
    counts = runner.empty_counts(batch)
    writes = ([np.zeros(batch, np.int64)] if include_bos else []) + list(tokens.T)
    for ids in writes:
        active = adapter.terminal_info(state)["active"] if is_v2 else np.ones(batch, bool)
        counts["active_update_attempts"] += int(active.sum())
        counts["terminal_noop_steps"] += int((~active).sum())

        def update(value):
            result = value * np.float32(.89) + ((ids + 1) / 13).astype(np.float32)[:, None, None, None]
            result[ids == 5] = np.inf  # Marked synthetic numerical fault.
            return result

        state, represented, _ = adapter.step(state, update, diagnostics=False)
        prediction = (np.rint(represented.sum(axis=(1, 2, 3))) % 6).astype(np.int8)
        if is_v2:
            prediction[~adapter.terminal_info(state)["active"]] = -1
        output.append(prediction)
    counts.update(status="SYNTHETIC_TEST_ONLY", completed_writes=len(writes),
                  first_invalid_write_position=[None] * batch)
    return np.stack(output, axis=1), state, counts


if __name__ == "__main__":
    # The production worker still verifies source hashes, protocol file hash,
    # input bytes and every persisted checkpoint. Only model execution and the
    # costly whole-study frozen inventory scan are replaced for this unit test.
    with patch.object(runner, "load_runtime", return_value=(None, {}, None, synthetic_sequence_call)), \
         patch.object(runner.common, "verify_protocol", return_value={}):
        if '--failure-boundary' in sys.argv:
            from source import run_failure_boundaries as boundary_runner
            with patch.object(boundary_runner, 'verify_appendix', return_value={}):
                boundary_runner.worker(sys.argv[-1])
        else:
            runner.worker(sys.argv[-1])
