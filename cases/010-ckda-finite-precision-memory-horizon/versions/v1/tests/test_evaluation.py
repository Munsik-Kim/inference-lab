"""DEV-only CPU parity, calibration and evaluator isolation checks."""

import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from codec.groups import frozen_sequences
from codec.learned import create_model, load_upstream, transition
from codec.online import OnlineAdapter, OnlineState
from codec.packed import NativeFloatCodec
from scripts.evaluate_learned import (
    ARM_NAMES, SHAPE, calibrate, coefficient_metadata, column_norm_squared, gather, ledger_with_table,
    make_arms, torch_transition, sequence_call, stream_seeds, table_bytes,
    summarize_arm, token_table, verify_route_parity,
)


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        upstream = load_upstream(os.environ["CKDA_UPSTREAM"])
        torch.manual_seed(17)
        cls.model = create_model(upstream).eval()
        cls.table = token_table(cls.model)
        cls.batch = frozen_sequences("S3", "DEV", 2, 32, 2001)

    @torch.inference_mode()
    def test_trained_path_parity_contract(self):
        result = verify_route_parity(self.model, self.table, self.batch.tokens)
        self.assertTrue(result["passed"])
        state = np.random.default_rng(91).normal(size=(2,)+SHAPE).astype(np.float32)
        coeff = gather(self.table, np.array([0, 6]))
        reference = transition(torch.from_numpy(state), {k:torch.from_numpy(v) for k,v in coeff.items()})
        np.testing.assert_allclose(torch_transition(state, coeff), reference.numpy(), atol=2e-5, rtol=2e-4)

    def test_full_transition_column_norm(self):
        coeff = gather(self.table, np.arange(7))
        key, beta, alpha = coeff["k"], coeff["beta"], coeff["alpha"]
        matrix = (np.eye(16, dtype=np.float32)-beta[...,None,None]*key[..., :,None]*key[...,None,:])*alpha[...,None,:]
        np.testing.assert_allclose(column_norm_squared(coeff), (matrix*matrix).sum(-2), atol=2e-6, rtol=2e-5)

    def test_calibration_and_full_declared_family(self):
        stats = calibrate(self.table, self.batch.tokens)
        basis = stats["basis"]
        np.testing.assert_allclose(np.einsum("hkr,hks->hrs",basis,basis),
                                   np.broadcast_to(np.eye(4),(12,4,4)), atol=2e-5)
        self.assertTrue(np.all((stats["mixed4_bits"] == 8).sum(1) == 4))
        self.assertTrue(np.all((stats["mixed6_bits"] == 8).sum(1) == 8))
        arms = make_arms(stats)
        self.assertEqual(list(arms), ARM_NAMES)
        self.assertEqual(len(arms), 26)
        config, raw = table_bytes(self.table)
        for name in ("NATIVE_FP32", "STOCHASTIC_4", "LOWRANK_4_8_R2", "FULL_RESIDUAL_4_FP32", "MIXED_4_8"):
            pred, state, execution = sequence_call(self.model, self.table, self.batch.tokens[:, :4], arms[name], stream_seeds(2001,2))
            self.assertEqual(pred.shape, (2,5))
            self.assertEqual(state.payload.dtype, np.uint8)
            self.assertEqual(execution["status"], "COMPLETE")
            ledger = ledger_with_table(arms[name],state,len(config)+len(raw))
            self.assertEqual(ledger["total_bytes"]["16"], ledger["shared_bytes"]+16*arms[name].bytes_per_stream)
            self.assertEqual(ledger["shared_token_coefficient_bytes"],len(config)+len(raw))

    def test_byte_only_restart_and_future_prefix_invariance(self):
        arms = make_arms(calibrate(self.table, self.batch.tokens))
        for name in ("STOCHASTIC_4", "LOWRANK_4_8_R2", "FULL_RESIDUAL_4_FP32"):
            adapter = arms[name]
            seeds = stream_seeds(2001, 2)
            whole, final, _ = sequence_call(self.model, self.table, self.batch.tokens, adapter, seeds)
            left, middle, _ = sequence_call(self.model, self.table, self.batch.tokens[:, :8], adapter, seeds)
            restored_adapter = OnlineAdapter.from_shared(adapter.config_bytes, adapter.basis_bytes)
            restored = restored_adapter.from_bytes(middle.payload.tobytes(), batch_size=2)
            right, end, _ = sequence_call(self.model, self.table, self.batch.tokens[:, 8:], restored_adapter,
                                          initial=restored, include_bos=False)
            np.testing.assert_array_equal(np.concatenate((left,right),1),whole)
            np.testing.assert_array_equal(end.payload,final.payload)
            changed = self.batch.tokens.copy()
            changed[:, 8:] = (changed[:, 8:] + 1) % 6
            other, _, _ = sequence_call(self.model, self.table, changed, adapter, seeds)
            np.testing.assert_array_equal(other[:, :9], whole[:, :9])

    def test_nonfinite_readout_is_explicit_failure(self):
        adapter = make_arms(calibrate(self.table, self.batch.tokens))["NATIVE_FP32"]
        with patch("scripts.evaluate_learned.logits_from_numpy", return_value=torch.full((2,6),float("nan"))):
            predictions, _, execution = sequence_call(self.model,self.table,self.batch.tokens[:, :2],adapter)
        self.assertTrue(np.all(predictions == -1))
        self.assertEqual(execution["nonfinite_output_sequence_steps"],6)
        self.assertEqual(execution["status"],"NONFINITE_LOGITS_WITH_FINITE_STATE")
        self.assertEqual(execution["absorbed_sequences"],0)
        metadata = coefficient_metadata(self.model,self.table)
        self.assertEqual(metadata["parameter_count"],56530)
        self.assertEqual(metadata["native_cache_elements"],3072)

    def test_unscored_bos_readout_does_not_absorb_finite_state(self):
        import scripts.evaluate_learned as evaluation
        adapter=make_arms(calibrate(self.table,self.batch.tokens))["NATIVE_FP32"]
        baseline,_,_=sequence_call(self.model,self.table,self.batch.tokens[:,:4],adapter)
        original=evaluation.logits_from_numpy
        calls=0
        def bad_bos(model,state,coeff):
            nonlocal calls
            result=original(model,state,coeff)
            if calls==0:
                result[:]=torch.nan
            calls+=1
            return result
        with patch("scripts.evaluate_learned.logits_from_numpy",side_effect=bad_bos):
            predictions,_,execution=sequence_call(self.model,self.table,self.batch.tokens[:,:4],adapter)
        self.assertTrue(np.all(predictions[:,0]==-1))
        np.testing.assert_array_equal(predictions[:,1:],baseline[:,1:])
        self.assertEqual(execution["absorbed_sequences"],0)

    def test_one_nonfinite_transition_does_not_fail_other_sequences(self):
        adapter = make_arms(calibrate(self.table, self.batch.tokens))["NATIVE_FP32"]
        original = torch_transition
        calls = 0
        def corrupted(state, coeff):
            nonlocal calls
            value = original(state,coeff)
            if calls == 1:
                value[0] = np.nan
            calls += 1
            return value
        baseline,_,_ = sequence_call(self.model,self.table,self.batch.tokens[:, :4],adapter)
        with patch("scripts.evaluate_learned.torch_transition",side_effect=corrupted):
            pred,_,execution = sequence_call(self.model,self.table,self.batch.tokens[:, :4],adapter)
        np.testing.assert_array_equal(pred[1],baseline[1])
        self.assertTrue(np.all(pred[0,1:] == -1))
        self.assertEqual(execution["absorbed_sequences"],1)
        self.assertEqual(execution["first_invalid_write_position"],[1,None])

    def test_constant_local_error_has_zero_centered_covariance(self):
        value = np.broadcast_to(np.linspace(0,1,16,dtype=np.float32)[None,None,:,None],(2,)+SHAPE).copy()
        with patch("scripts.evaluate_learned.torch_transition",return_value=value):
            stats=calibrate(self.table,self.batch.tokens)
        np.testing.assert_allclose(stats["covariance"],0.,atol=1e-15)

    def test_post_transition_residual_overflow_isolated_by_row(self):
        native = NativeFloatCodec(SHAPE)
        adapter = OnlineAdapter(native, NativeFloatCodec(SHAPE), mode="untransported")
        initial = adapter.initial(np.zeros((2,)+SHAPE,dtype=np.float32))
        residual = np.zeros((2,)+SHAPE,dtype=np.float32)
        residual[0] = 3e38
        raw = initial.payload.copy()
        raw[:, 8+native.bytes_per_stream:] = native.encode(residual).payload
        original = torch_transition
        def large_finite(state, coeff):
            value = original(state, coeff)
            value[0] = 3e38
            return value
        baseline,_,_ = sequence_call(self.model,self.table,self.batch.tokens[:, :3],adapter,
                                      initial=initial,include_bos=False)
        with patch("scripts.evaluate_learned.torch_transition",side_effect=large_finite):
            pred,_,execution = sequence_call(self.model,self.table,self.batch.tokens[:, :3],adapter,
                                              initial=OnlineState(raw),include_bos=False)
        np.testing.assert_array_equal(pred[1],baseline[1])
        self.assertTrue(np.all(pred[0] == -1))
        self.assertEqual(execution["absorbed_sequences"],1)

    def test_persisted_truth_bits_recompute_counts_and_failure_times(self):
        adapter = make_arms(calibrate(self.table,self.batch.tokens))["UNIFORM_4"]
        config,raw=table_bytes(self.table)
        with tempfile.TemporaryDirectory(prefix="ckda-correctness-") as directory:
            path=Path(directory)/"correctness.npz"
            summary=summarize_arm(self.model,self.table,adapter,self.batch,[32],546,0,.05,(.05,.01),
                                  len(config)+len(raw),correctness_output=path)
            with np.load(path,allow_pickle=False) as stored:
                shape=stored["shape"]
                correct=np.unpackbits(stored["packed"],axis=1,count=int(shape[1]),bitorder="little").astype(bool)
            np.testing.assert_array_equal(correct.sum(0),summary["step_correct_counts"])
            recomputed=[]
            for row in correct:
                bad=np.flatnonzero(~row)
                recomputed.append(None if not len(bad) else int(bad[0])+1)
            self.assertEqual(recomputed,summary["tau"])
            self.assertEqual(summary["correctness_artifact"]["shape"],[2,32])


if __name__ == "__main__":
    unittest.main()
