"""Small CPU checks only; no learned TEST or long precision cohort is executed."""

import hashlib
import json
import os
import unittest
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from source.precision import (DTYPES, MODES, PrecisionDiagnostic,
                              explicit_readout, original_readout, summarize_step,
                              verify_fp32_readout)
from source.reference import learned, v1


@lru_cache(maxsize=1)
def model():
    torch.set_num_threads(2)
    torch.manual_seed(1901)
    upstream_location = os.environ.get("CKDA_UPSTREAM")
    if not upstream_location:
        raise unittest.SkipTest("Set CKDA_UPSTREAM to the pinned upstream checkout")
    upstream_path = Path(upstream_location)
    upstream = learned.load_upstream(upstream_path)
    return learned.create_model(upstream, backend="naive_recurrent", device="cpu").eval()


def identity(table):
    # Synthetic file identities are only for this untrained unit-test model.
    return {"model_seed": 0, "checkpoint_sha256": "0" * 64,
            "verified_checkpoint_sha256": {str(seed): str(seed) * 64 for seed in range(3)},
            "token_table_sha256": hashlib.sha256(v1.table_bytes(table)[1]).hexdigest()}


@lru_cache(maxsize=1)
def diagnostic():
    reference_model = model()
    table = v1.token_table(reference_model)
    return PrecisionDiagnostic(reference_model, table, identity(table))


class PrecisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = model()
        cls.diagnostic = diagnostic()

    def test_explicit_fp32_matches_actual_original_readout(self):
        model, diagnostic = self.model, self.diagnostic
        coefficients = {name: value[[0, 3]] for name, value in diagnostic.table.items()}
        generator = torch.Generator().manual_seed(1902)
        for scale in (0., .1, 10., 10000.):
            state = torch.randn((2, 12, 16, 16), generator=generator) * scale
            receipt = verify_fp32_readout(model, diagnostic.weights, state, coefficients)
            assert receipt["passed"]
            assert receipt["atol"] == 2e-5 and receipt["rtol"] == 2e-4
            assert set(receipt["dtype_trace"].values()) == {"torch.float32"}


    def test_four_modes_use_fixed_coefficients_and_separate_boundaries(self):
        model, diagnostic = self.model, self.diagnostic
        states = diagnostic.initial_states(2)
        ref32, ref64 = states["D00"].clone(), states["D10"].clone()
        for position in range(9):
            ids = np.array([6, 6]) if position == 0 else np.array([position % 6, (position + 3) % 6])
            coeff32 = {name: value[ids] for name, value in diagnostic.table.items()}
            coeff64 = {name: value.double() for name, value in coeff32.items()}
            ref32 = learned.transition(ref32, coeff32)
            ref64 = learned.transition(ref64, coeff64)
            states, outputs, traces = diagnostic.step(states, ids)
            assert torch.equal(states["D00"], ref32)
            assert torch.equal(states["D01"], ref32)
            assert torch.equal(states["D10"], ref64)
            assert torch.equal(states["D11"], ref64)
            assert torch.equal(outputs["D00"], original_readout(model, ref32, coeff32))
            assert torch.equal(outputs["D10"], original_readout(model, ref64.float(), coeff32))
            for mode in MODES:
                assert states[mode].dtype == DTYPES[mode][0]
                assert outputs[mode].dtype == DTYPES[mode][1]
                assert bool(torch.isfinite(outputs[mode]).all())
            for mode in ("D01", "D11"):
                assert set(traces[mode]["readout"].values()) == {"torch.float64"}


    def test_fp64_norm_does_not_silently_downcast(self):
        model, diagnostic = self.model, self.diagnostic
        coefficients = {name: value[[0, 1]] for name, value in diagnostic.table.items()}
        generator = torch.Generator().manual_seed(1903)
        state = torch.randn((2, 12, 16, 16), generator=generator) * 1e25
        original_parameters = {name: value.detach().clone() for name, value in model.named_parameters()}
        double, trace = explicit_readout(diagnostic.weights, state, coefficients, torch.float64)
        original = original_readout(model, state, coefficients)
        assert bool(torch.isfinite(double).all())
        assert set(trace.values()) == {"torch.float64"}
        assert float((double-original.double()).abs().max()) > 1e-4
        for name, parameter in model.named_parameters():
            assert parameter.dtype == torch.float32
            assert torch.equal(parameter, original_parameters[name])
        for name, weight in diagnostic.weights.fp32.items():
            if weight is not None:
                assert torch.equal(diagnostic.weights.fp64[name], weight.double())


    def test_nonzero_handoff_and_no_input_mutation(self):
        model, diagnostic = self.model, self.diagnostic
        generator = torch.Generator().manual_seed(1904)
        initial = torch.randn((2, 12, 16, 16), generator=generator)
        states = diagnostic.initial_states(2, initial)
        before = {name: value.clone() for name, value in states.items()}
        first, _, _ = diagnostic.step(states, [0, 2])
        for name in MODES:
            assert torch.equal(states[name], before[name])
        uninterrupted, outputs, _ = diagnostic.step(first, [3, 5])
        copied = {name: value.clone() for name, value in first.items()}
        resumed, resumed_outputs, _ = diagnostic.step(copied, [3, 5])
        for name in MODES:
            assert torch.equal(uninterrupted[name], resumed[name])
            assert torch.equal(outputs[name], resumed_outputs[name])


    def test_scalar_summary_flags_nonfinite_and_uses_gold_margin(self):
        model, diagnostic = self.model, self.diagnostic
        states = diagnostic.initial_states(2)
        logits = {name: torch.tensor([[1., 3., 2., 0., -1., -2.],
                                     [1., 2., 0., -1., -2., -3.]], dtype=DTYPES[name][1])
                  for name in MODES}
        logits["D01"][0, 0] = torch.nan
        states["D10"][1, 0, 0, 0] = torch.inf
        result = summarize_step(states, logits, [1, 1], position=1)
        assert result["scored"]
        assert result["modes"]["D00"]["gold_margin"] == [1., 1.]
        assert result["modes"]["D00"]["top_two_margin"] == [1., 1.]
        assert result["modes"]["D01"]["prediction"][0] == -1
        assert not result["modes"]["D10"]["correct"][1]
        assert result["modes"]["D10"]["state_norm"][1] is None
        json.dumps(result, allow_nan=False)
        assert not summarize_step(states, logits, [1, 1], position=0)["scored"]


    def test_metadata_and_identity_require_all_three_checkpoints(self):
        model, diagnostic = self.model, self.diagnostic
        metadata = diagnostic.metadata()
        assert not metadata["coefficients_reprojected"]
        assert len(metadata["fp32_readout_parity"]) == 2
        json.dumps(metadata, allow_nan=False)
        table = v1.token_table(model)
        wrong = identity(table)
        del wrong["verified_checkpoint_sha256"]["2"]
        with self.assertRaisesRegex(ValueError, "all three"):
            PrecisionDiagnostic(model, table, wrong)
        wrong = identity(table)
        wrong["token_table_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "table identity"):
            PrecisionDiagnostic(model, table, wrong)
        wrong_table = dict(table, alpha=table["alpha"].astype(np.float64))
        with self.assertRaisesRegex(ValueError, "FP32"):
            PrecisionDiagnostic(model, wrong_table, identity(table))


    def test_transition_dtype_is_not_silently_coerced(self):
        model, diagnostic = self.model, self.diagnostic
        states = diagnostic.initial_states(1)
        states["D10"] = states["D10"].float()
        with self.assertRaisesRegex(ValueError, "D10"):
            diagnostic.step(states, [0])


if __name__ == "__main__":
    unittest.main()
