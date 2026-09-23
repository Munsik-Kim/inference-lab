"""CPU parity gates at fixed tolerances, including real nonzero state handoff."""

import os
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from codec.learned import (
    ATOL, RTOL, coefficients, create_model, load_upstream, physical_forward,
    read_logits, select_step, sequence_metrics, transition,
)


class LearnedAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = os.environ.get("CKDA_UPSTREAM")
        if root is None:
            raise RuntimeError("Set CKDA_UPSTREAM to the explicit pinned checkout")
        torch.set_num_threads(1)
        cls.upstream = load_upstream(root)

    def setUp(self):
        torch.manual_seed(72)
        self.model = create_model(self.upstream).eval()
        self.tokens = torch.cat([torch.full((3, 1), 6), torch.randint(6, (3, 32))], 1)

    def assertClose(self, actual, expected):
        torch.testing.assert_close(actual, expected, atol=ATOL, rtol=RTOL)

    def native(self, tokens, initial=None):
        from fla.models.utils import Cache
        cache = Cache()
        if initial is not None:
            cache.update(recurrent_state=initial.clone(), conv_state=None, layer_idx=0, offset=0)
        e = self.model.emb(tokens)
        out, _, cache = self.model.layer(e, past_key_values=cache, use_cache=True)
        logits = self.model.mlp(self.model.norm(e + out))
        return logits, cache[0]["recurrent_state"]

    @torch.inference_mode()
    def test_zero_state_parity_and_shapes(self):
        actual, state = physical_forward(self.model, self.tokens)
        expected, native_state = self.native(self.tokens)
        self.assertClose(actual, self.model(self.tokens))
        self.assertClose(actual, expected)
        self.assertClose(state, native_state)
        self.assertEqual(actual.shape, (3, 33, 6))
        self.assertEqual(state.shape, (3, 12, 16, 16))
        self.assertEqual(state.dtype, torch.float32)
        self.assertTrue(torch.isfinite(state).all())
        self.assertTrue(torch.isfinite(actual).all())

    @torch.inference_mode()
    def test_nonzero_initial_state_and_chunk_handoff(self):
        initial = torch.randn(3, 12, 16, 16) * .1
        whole, whole_state = physical_forward(self.model, self.tokens, initial)
        native_whole, native_state = self.native(self.tokens, initial)
        self.assertClose(whole, native_whole)
        self.assertClose(whole_state, native_state)
        left, middle = physical_forward(self.model, self.tokens[:, :11], initial)
        right, end = physical_forward(self.model, self.tokens[:, 11:], middle)
        native_left, native_middle = self.native(self.tokens[:, :11], initial)
        native_right, native_end = self.native(self.tokens[:, 11:], native_middle)
        self.assertTrue(middle.abs().sum() > 0)
        self.assertClose(torch.cat([left, right], 1), whole)
        self.assertClose(torch.cat([native_left, native_right], 1), whole)
        self.assertClose(end, whole_state)
        self.assertClose(end, native_end)

    @torch.inference_mode()
    def test_readout_uses_supplied_state(self):
        coeff = coefficients(self.model, self.tokens)
        self.assertEqual(set(coeff), {"q", "k", "v", "alpha", "beta", "g", "e"})
        self.assertTrue((coeff["alpha"] < 0).any())
        step = select_step(coeff, 0)
        state = transition(torch.zeros(3, 12, 16, 16), step)
        ordinary = read_logits(self.model, state, step)
        altered = read_logits(self.model, torch.zeros_like(state), step)
        self.assertGreater(float((ordinary - altered).abs().max()), 1e-5)

    def test_bos_excluded_from_survival_and_first_failure(self):
        labels = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]])
        predictions = torch.tensor([[5, 1, 2, 3], [0, 1, 0, 3]])
        logits = torch.nn.functional.one_hot(predictions, 6).float()
        result = sequence_metrics(logits, labels)
        self.assertEqual(result["gold_length"], 3)
        self.assertEqual(result["bos_accuracy"], .5)
        self.assertEqual(result["bos_predictions"], [5, 0])
        self.assertEqual(result["all_prefix_survival"], .5)
        self.assertEqual(result["final_quarter_accuracy"], 1.)
        self.assertEqual(result["first_failure_position"], [None, 2])
        self.assertEqual(result["survival_by_length"], [1., .5, .5])

    def test_nonfinite_logits_are_failures_even_if_argmax_matches(self):
        labels = torch.zeros((2, 3), dtype=torch.long)
        logits = torch.zeros(2, 3, 6)
        logits[..., 0] = 1
        logits[0, 0, 0] = float("nan")
        logits[1, 1, 0] = float("inf")
        result = sequence_metrics(logits, labels)
        self.assertEqual(result["bos_accuracy"], .5)
        self.assertEqual(result["all_prefix_survival"], .5)
        self.assertEqual(result["first_failure_position"], [None, 1])
        self.assertEqual(result["invalid_logit_token_count"], 2)
        self.assertEqual(result["invalid_scored_logit_token_count"], 1)
        self.assertEqual(result["invalid_bos_logit_count"], 1)


if __name__ == "__main__":
    unittest.main()
