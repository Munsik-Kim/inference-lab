"""Causal residual, byte-ledger, and actual fresh-process restart checks (CPU)."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

import numpy as np

CASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CASE_DIR))
from codec import NativeFloatCodec, PackedCodec, PackedBatch
from codec.online import NonfiniteStateError, OnlineAdapter, OnlineState, change_basis

AUDIT_OUTPUT = os.environ.get("CKDA_AUDIT_OUTPUT")


def canonical(config):
    return json.dumps(config, sort_keys=True, separators=(",", ":")).encode()


def affine(matrix, bias):
    return lambda x: np.einsum("hij,bhjv->bhiv", matrix, x) + bias


FRESH_PROCESS = r'''
import hashlib, json, pathlib, sys
import numpy as np
sys.path.insert(0, sys.argv[1])
from codec.online import OnlineAdapter
folder = pathlib.Path(sys.argv[2])
request = json.loads((folder / "request.json").read_bytes())
adapter = OnlineAdapter.from_shared((folder / "shared.json").read_bytes(),
                                    (folder / "basis.bin").read_bytes())
state = adapter.from_bytes((folder / "prefix.bin").read_bytes(), request["batch_size"])
# The independent audit inputs are public suffix operators, not a hidden cache.
inputs = np.load(folder / "audit_inputs.npz", allow_pickle=False)
reads = []
digests = []
for matrix, bias in zip(inputs["matrices"], inputs["biases"]):
    state, read, diagnostic = adapter.step(
        state, lambda x: np.einsum("hij,bhjv->bhiv", matrix, x) + bias)
    reads.append(read)
    digests.append(hashlib.sha256(state.payload.tobytes()).hexdigest())
(folder / "restarted_suffix.bin").write_bytes(state.payload.tobytes())
np.save(folder / "restarted_reads.npy", np.asarray(reads), allow_pickle=False)
(folder / "child_result.json").write_text(json.dumps({
    "pid": __import__("os").getpid(), "parent_pid": __import__("os").getppid(),
    "suffix_steps": len(reads), "per_step_payload_sha256": digests,
    "final_payload_sha256": digests[-1], "ledger": adapter.ledger(state)
}, indent=2, sort_keys=True) + "\n")
'''


class OnlineAdapterTests(unittest.TestCase):
    def test_diagnostics_optional_fp64_finite_and_bytes_unchanged(self):
        adapter = OnlineAdapter(PackedCodec((1, 2, 1), bits=4), NativeFloatCodec((1, 2, 1)))
        source = np.array([[[[1e30], [0.7e30]]], [[[2.0], [1.0]]]], dtype=np.float32)
        state = adapter.initial(source)
        with np.errstate(over="raise", invalid="raise"):
            with_diagnostics, read, diagnostic = adapter.step(state, lambda x: x * np.float32(0.9))
            without_diagnostics, other, omitted = adapter.step(state, lambda x: x * np.float32(0.9), diagnostics=False)
        np.testing.assert_array_equal(with_diagnostics.payload, without_diagnostics.payload)
        np.testing.assert_array_equal(read, other)
        self.assertEqual(omitted, {})
        self.assertTrue(all(np.isfinite(value) for value in diagnostic.values()))
        json.dumps(diagnostic, allow_nan=False)
        self.assertGreater(diagnostic["state_norm_mean"], 1e29)

    def test_nonfinite_rows_retry_healthy_bytes_and_rng_unchanged(self):
        adapter = OnlineAdapter(PackedCodec((1, 2, 1), bits=3, stochastic=True))
        source = np.array([[[[1], [0.4]]], [[[2], [-0.3]]], [[[3], [0.2]]]], dtype=np.float32)
        state = adapter.initial(source, seeds=[10, 20, 30])
        before = state.payload.copy()
        def broken(x):
            y = x * np.float32(0.9)
            y[1] = np.inf
            return y
        with np.errstate(over="raise", invalid="raise"):
            with self.assertRaises(NonfiniteStateError) as context:
                adapter.step(state, broken)
        failure = context.exception
        np.testing.assert_array_equal(failure.bad_rows, [False, True, False])
        self.assertEqual(failure.stage, "transition_output")
        np.testing.assert_array_equal(state.payload, before)
        healthy = OnlineState(state.payload[[0, 2]].copy())
        retried, _, _ = adapter.step(healthy, lambda x: x * np.float32(0.9))
        baseline, _, _ = adapter.step(state, lambda x: x * np.float32(0.9))
        np.testing.assert_array_equal(retried.payload, baseline.payload[[0, 2]])

    def test_untransported_post_callback_overflow_is_row_local(self):
        adapter = OnlineAdapter(NativeFloatCodec((1, 1, 1)), NativeFloatCodec((1, 1, 1)), mode="untransported")
        z = adapter.state_codec.encode(np.zeros((2, 1, 1, 1), dtype=np.float32))
        residual = adapter.residual_codec.encode(np.array([[[[2e38]]], [[[1.0]]]], dtype=np.float32))
        state = adapter._join(np.zeros(2, dtype=np.uint64), z, residual)
        with np.errstate(over="raise", invalid="raise"):
            with self.assertRaises(NonfiniteStateError) as context:
                adapter.step(state, lambda x: np.full_like(x, 2e38))
        np.testing.assert_array_equal(context.exception.bad_rows, [True, False])
        self.assertEqual(context.exception.stage, "transition_output")
        with self.assertRaises(ValueError) as shape_error:
            adapter.step(state, lambda x: x[0])
        self.assertNotIsInstance(shape_error.exception, NonfiniteStateError)

    def test_reconstructed_overflow_and_finite_scale_range_failure(self):
        adapter = OnlineAdapter(NativeFloatCodec((1, 1, 1)), NativeFloatCodec((1, 1, 1)))
        values = np.array([[[[2e38]]], [[[1.0]]]], dtype=np.float32)
        state = adapter._join(np.zeros(2, dtype=np.uint64), adapter.state_codec.encode(values), adapter.residual_codec.encode(values))
        with self.assertRaises(NonfiniteStateError) as context:
            adapter.represented(state)
        np.testing.assert_array_equal(context.exception.bad_rows, [True, False])
        self.assertEqual(context.exception.stage, "represented_state")
        with self.assertRaises(NonfiniteStateError) as context:
            adapter.step(state, lambda x: x)
        self.assertEqual(context.exception.stage, "reconstructed_state")
        packed = OnlineAdapter(PackedCodec((1, 1, 1)))
        initial = packed.initial(np.zeros((2, 1, 1, 1), dtype=np.float32))
        with self.assertRaises(NonfiniteStateError) as context:
            packed.step(initial, lambda x: np.array([[[[1e100]]], [[[1.0]]]], dtype=np.float64))
        np.testing.assert_array_equal(context.exception.bad_rows, [True, False])
        self.assertEqual(context.exception.stage, "state_encode")

    def test_finite_deterministic_payloads_match_archived_v1(self):
        archived = CASE_DIR / "provenance" / "implementation_v1" / "online.py"
        if not archived.exists():
            self.fail("committed v1 online source fixture is unavailable")
        spec = importlib.util.spec_from_file_location("codec._online_v1_audit", archived)
        old_module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = old_module
        spec.loader.exec_module(old_module)
        rng = np.random.default_rng(171)
        shape = (1, 3, 2)
        x = rng.normal(size=(3,) + shape).astype(np.float32)
        basis = np.eye(3, dtype=np.float32)[None, :, :2]
        configurations = [
            (PackedCodec(shape, bits=3), None, None, "transported"),
            (PackedCodec(shape, mixed_bits=[4, 8, 4]), None, None, "transported"),
            (PackedCodec(shape), NativeFloatCodec(shape), None, "transported"),
            (PackedCodec(shape), PackedCodec(shape), None, "untransported"),
            (PackedCodec(shape), PackedCodec((1, 2, 2), bits=8), basis, "transported"),
        ]
        for args in configurations:
            old, new = old_module.OnlineAdapter(*args), OnlineAdapter(*args)
            self.assertEqual(old.config_bytes, new.config_bytes)
            old_state, new_state = old.initial(x), new.initial(x)
            np.testing.assert_array_equal(old_state.payload, new_state.payload)
            for _ in range(12):
                matrix = np.eye(3, dtype=np.float32)[None] * 0.9 + rng.normal(size=(1, 3, 3)).astype(np.float32) * 0.02
                bias = rng.normal(size=x.shape).astype(np.float32) * 0.03
                transition = affine(matrix, bias)
                old_state, old_read, _ = old.step(old_state, transition)
                new_state, new_read, _ = new.step(new_state, transition)
                np.testing.assert_array_equal(old_state.payload, new_state.payload)
                np.testing.assert_array_equal(old_read, new_read)

    def test_full_fp64_residual_noncommuting_affine_identity(self):
        shape = (1, 3, 2)
        adapter = OnlineAdapter(PackedCodec(shape, bits=2), NativeFloatCodec(shape, "float64"))
        rng = np.random.default_rng(992)
        initial = rng.normal(size=(4,) + shape)
        state = adapter.initial(initial)
        reference = initial.copy()
        np.testing.assert_allclose(adapter.represented(state), reference, rtol=0, atol=2e-16)
        a = np.array([[[0.91, 0.17, -0.04], [0.0, 0.87, 0.13], [0.09, 0.0, 0.93]]])
        b = np.array([[[0.88, 0.0, 0.11], [-0.14, 0.92, 0.0], [0.0, 0.06, 0.9]]])
        self.assertGreater(np.linalg.norm(a @ b - b @ a), 0.01)
        max_error = 0.0
        for t in range(40):
            matrix = a if t % 2 == 0 else b
            bias = rng.normal(size=(4,) + shape) * 0.03
            transition = affine(matrix, bias)
            reference = transition(reference)
            state, read, diagnostics = adapter.step(state, transition)
            max_error = max(max_error, float(np.max(np.abs(read - reference))))
            np.testing.assert_allclose(read, reference, rtol=0, atol=3e-15)
            self.assertLess(diagnostics["writeback_mse"], 1e-29)
        self.assertLess(max_error, 3e-15)  # FP64 arithmetic tolerance, not bitwise algebra.
        np.testing.assert_array_equal(adapter._parts(state)[0], np.full(4, 40, dtype=np.uint64))

    def test_full_fp32_residual_compute_tolerance(self):
        shape = (1, 3, 2)
        adapter = OnlineAdapter(PackedCodec(shape, bits=2), NativeFloatCodec(shape, "float32"))
        rng = np.random.default_rng(122)
        reference = rng.normal(size=(3,) + shape).astype(np.float32)
        state = adapter.initial(reference)
        matrix = np.array([[[0.91, 0.09, 0], [-0.08, 0.9, 0.12], [0.07, 0, 0.88]]], dtype=np.float32)
        for _ in range(32):
            bias = (rng.normal(size=reference.shape) * 0.05).astype(np.float32)
            transition = affine(matrix, bias)
            reference = transition(reference)
            state, read, _ = adapter.step(state, transition)
            self.assertEqual(read.dtype, np.float32)
            np.testing.assert_allclose(read, reference, rtol=0, atol=5e-7)

    def test_untransported_residual_is_a_negative_control(self):
        shape = (1, 2, 1)
        x = np.array([[[[0.25], [0.75]]]], dtype=np.float64)
        matrix = np.array([[[0, 1], [-1, 0]]], dtype=np.float64)
        transported = OnlineAdapter(PackedCodec(shape, bits=2, max_abs=1), NativeFloatCodec(shape, "float64"))
        untransported = OnlineAdapter(PackedCodec(shape, bits=2, max_abs=1), NativeFloatCodec(shape, "float64"), mode="untransported")
        transition = affine(matrix, np.zeros_like(x))
        _, correct, _ = transported.step(transported.initial(x), transition)
        _, negative, diagnostics = untransported.step(untransported.initial(x), transition)
        np.testing.assert_array_equal(correct, transition(x))
        self.assertGreater(float(np.max(np.abs(negative - transition(x)))), 0.49)
        # Its own writeback can be perfect while its recurrence is wrong.
        self.assertEqual(diagnostics["writeback_mse"], 0)

    def test_lowrank_error_outside_projector_remains(self):
        basis = np.array([[[1], [0], [0]]], dtype=np.float32)
        adapter = OnlineAdapter(PackedCodec((1, 3, 1), bits=2, max_abs=1),
                                NativeFloatCodec((1, 1, 1), "float64"), basis)
        target = np.array([[[[0.25], [0.25], [0.5]]]], dtype=np.float64)
        state = adapter.initial(np.zeros_like(target))
        _, read, diagnostic = adapter.step(state, lambda _: target.copy())
        np.testing.assert_array_equal(read.ravel(), [0.25, 0, 0])
        expected_mse = (0.25**2 + 0.5**2) / 3
        self.assertAlmostEqual(diagnostic["projection_leakage_mse"], expected_mse)
        self.assertAlmostEqual(diagnostic["writeback_mse"], expected_mse)
        np.testing.assert_array_equal(adapter.project(target - read), np.zeros((1, 1, 1, 1)))

    def test_change_basis_same_span_full_span_and_lost_direction(self):
        old = np.eye(3, dtype=np.float64)[None, :, :2]
        coefficients = np.array([[[[1.0], [2.0]]], [[[3.0], [4.0]]]])
        same, same_loss = change_basis(coefficients, old, old)
        np.testing.assert_array_equal(same, coefficients)
        self.assertEqual(same_loss, 0)
        rotated = old @ np.array([[0.6, -0.8], [0.8, 0.6]])
        converted, rotation_loss = change_basis(coefficients, old, rotated)
        np.testing.assert_allclose(np.einsum("hkr,bhrv->bhkv", rotated, converted),
                                   np.einsum("hkr,bhrv->bhkv", old, coefficients), atol=1e-15, rtol=0)
        self.assertLess(rotation_loss, 1e-30)
        full = np.eye(3, dtype=np.float64)[None]
        enlarged, full_loss = change_basis(coefficients, old, full)
        np.testing.assert_array_equal(enlarged[:, :, :2], coefficients)
        self.assertEqual(full_loss, 0)
        smaller = full[:, :, :1]
        reduced, lost = change_basis(coefficients, old, smaller)
        np.testing.assert_array_equal(reduced, coefficients[:, :, :1])
        self.assertAlmostEqual(lost, (2**2 + 4**2) / 6)

    def test_shared_roundtrip_native_packed_full_and_lowrank(self):
        shape = (2, 3, 2)
        basis = np.broadcast_to(np.eye(3, dtype=np.float32)[:, :2], (2, 3, 2)).copy()
        adapters = [
            OnlineAdapter(NativeFloatCodec(shape, "float64")),
            OnlineAdapter(PackedCodec(shape, bits=5)),
            OnlineAdapter(PackedCodec(shape), NativeFloatCodec(shape, "float64")),
            OnlineAdapter(PackedCodec(shape, stochastic=True), PackedCodec((2, 2, 2), bits=3, stochastic=True), basis),
        ]
        x = np.random.default_rng(83).normal(size=(3,) + shape)
        for adapter in adapters:
            with self.subTest(residual=type(adapter.residual_codec).__name__):
                state = adapter.initial(x, seeds=[7, 11, 13])
                restored = OnlineAdapter.from_shared(adapter.config_bytes, adapter.basis_bytes)
                self.assertEqual(restored.config_bytes, adapter.config_bytes)
                self.assertEqual(restored.basis_bytes, adapter.basis_bytes)
                imported = restored.from_bytes(state.payload.tobytes(), batch_size=3)
                np.testing.assert_array_equal(restored.represented(imported), adapter.represented(state))

    def test_actual_ledger_shared_metadata_and_payload_only_state(self):
        basis = np.array([[[1], [0], [0]]], dtype=np.float32)
        adapter = OnlineAdapter(PackedCodec((1, 3, 3), mixed_bits=[4, 8, 4], stochastic=True),
                                PackedCodec((1, 1, 3), bits=3, stochastic=True), basis)
        source = np.arange(45, dtype=np.float32).reshape(5, 1, 3, 3)
        state = adapter.initial(source)
        ledger = adapter.ledger(state)
        # Main codes=48bits=6B; residual=9bits=2B; scales=8B, RNG=32B, cursor=8B.
        self.assertEqual(adapter.bytes_per_stream, 56)
        self.assertEqual(ledger["state_code_bytes"], 6)
        self.assertEqual(ledger["residual_code_bytes"], 2)
        self.assertEqual(ledger["scale_bytes"], 8)
        self.assertEqual(ledger["rng_bytes"], 32)
        self.assertEqual(ledger["cursor_bytes"], 8)
        self.assertEqual(ledger["payload_tensor_storage_bytes"], state.payload.nbytes)
        self.assertEqual(state.payload.nbytes, len(state.payload.tobytes()))
        self.assertEqual(ledger["shared_bytes"], len(adapter.config_bytes) + basis.nbytes)
        for count in [1, 16, 128]:
            self.assertEqual(ledger["total_bytes"][str(count)], 56 * count + adapter.shared_bytes)
        metadata = json.loads(adapter.config_bytes)
        self.assertEqual(metadata["state"]["mixed_bits"], [[4, 8, 4]])
        self.assertEqual(metadata["basis_shape"], [1, 3, 1])
        self.assertEqual(metadata["cursor"], "<u8")
        self.assertEqual(metadata["gauge_bytes_per_stream"], 0)
        self.assertEqual(OnlineState.__slots__, ("payload",))
        self.assertEqual(state.payload.dtype, np.uint8)
        self.assertTrue(state.payload.flags.c_contiguous)
        self.assertTrue(state.payload.flags.owndata)
        self.assertFalse(np.shares_memory(state.payload, source))
        represented = adapter.represented(state)
        represented.fill(-999)
        self.assertFalse(np.any(adapter.represented(state) == -999))
        _, z, r = adapter._parts(state)
        self.assertEqual(PackedBatch.__slots__, ("payload",))
        self.assertEqual(z.payload.dtype, np.uint8)
        self.assertEqual(r.payload.dtype, np.uint8)

    def test_invalid_basis_shape_nonfinite_and_unknown_metadata(self):
        state_codec = PackedCodec((1, 3, 2))
        residual_codec = PackedCodec((1, 1, 2))
        for basis in [np.zeros((1, 3)), np.zeros((1, 2, 1)), np.ones((1, 3, 1)),
                      np.full((1, 3, 1), np.nan), np.full((1, 3, 1), np.inf)]:
            with self.assertRaises(ValueError):
                OnlineAdapter(state_codec, residual_codec, basis)
        with self.assertRaises(ValueError):
            OnlineAdapter(state_codec, residual_codec)
        with self.assertRaises(ValueError):
            OnlineAdapter(state_codec, mode="invented")
        adapter = OnlineAdapter(state_codec)
        for config in [b"not json", b"[]", b"{}", canonical({"format": "ckda-online-v1"})]:
            with self.assertRaises(ValueError):
                OnlineAdapter.from_shared(config)
        for key, value in [("format", "unknown"), ("cursor", ">u8"), ("unknown", True),
                           ("basis_shape", [-1, 3, 1]), ("basis_shape", [1.5, 3, 1])]:
            metadata = json.loads(adapter.config_bytes)
            metadata[key] = value
            with self.assertRaises(ValueError):
                OnlineAdapter.from_shared(canonical(metadata))
        metadata = json.loads(adapter.config_bytes)
        metadata["state"]["format"] = "invented"
        with self.assertRaisesRegex(ValueError, "unknown nested"):
            OnlineAdapter.from_shared(canonical(metadata))
        with self.assertRaises(ValueError):
            OnlineAdapter.from_shared(adapter.config_bytes, b"unexpected")
        lowrank = OnlineAdapter(state_codec, residual_codec, np.array([[[1], [0], [0]]]))
        with self.assertRaises(ValueError):
            OnlineAdapter.from_shared(lowrank.config_bytes, lowrank.basis_bytes[:-1])
        corrupt_basis = np.array([[[np.nan], [0], [0]]], dtype="<f4").tobytes()
        with self.assertRaises(ValueError):
            OnlineAdapter.from_shared(lowrank.config_bytes, corrupt_basis)

    def test_malformed_state_transition_length_and_cursor_overflow(self):
        adapter = OnlineAdapter(PackedCodec((1, 2, 2), stochastic=True))
        x = np.ones((2, 1, 2, 2), dtype=np.float32)
        state = adapter.initial(x)
        original = state.payload.copy()
        for transition in [lambda x: x[0], lambda x: np.full_like(x, np.nan),
                           lambda x: np.full_like(x, np.inf)]:
            with self.assertRaises(ValueError):
                adapter.step(state, transition)
            np.testing.assert_array_equal(state.payload, original)
        for malformed in [state.payload[:, :-1].copy(), state.payload.astype(np.int16),
                          state.payload[:, ::-1]]:
            with self.assertRaises(ValueError):
                adapter.represented(OnlineState(malformed))
        for raw in [state.to_bytes()[:-1], state.to_bytes() + b"\x00"]:
            with self.assertRaises(ValueError):
                adapter.from_bytes(raw)
        overflow = original.copy()
        overflow[0, :8] = np.frombuffer(struct.pack("<Q", 2**64 - 1), dtype=np.uint8)
        reached_transition = []
        with self.assertRaisesRegex(ValueError, "cursor overflow"):
            adapter.step(OnlineState(overflow), lambda x: reached_transition.append(True) or x)
        self.assertEqual(reached_transition, [])
        np.testing.assert_array_equal(state.payload, original)

    def test_fresh_process_restart_payload_rng_cursor_and_suffix_reads(self):
        if AUDIT_OUTPUT:
            audit_dir = Path(AUDIT_OUTPUT)
        else:
            temporary = tempfile.TemporaryDirectory(prefix="case010-restart-audit-")
            self.addCleanup(temporary.cleanup)
            audit_dir = Path(temporary.name)
        shape = (2, 3, 2)
        rng = np.random.default_rng(901)
        batch = 4
        initial = rng.normal(size=(batch,) + shape) * 0.3
        basis = np.broadcast_to(np.eye(3, dtype=np.float32)[:, :2], (2, 3, 2)).copy()
        adapters = {
            "packed-sr-lowrank": OnlineAdapter(
                PackedCodec(shape, mixed_bits=[[2, 4, 8], [4, 2, 8]], stochastic=True),
                PackedCodec((2, 2, 2), bits=3, stochastic=True), basis),
            "packed-fp64-fullres": OnlineAdapter(PackedCodec(shape, bits=3), NativeFloatCodec(shape, "float64")),
        }
        steps, prefix_steps = 13, 5
        matrices = np.eye(3)[None, None] * 0.96 + rng.normal(size=(steps, 2, 3, 3)) * 0.02
        biases = rng.normal(size=(steps, batch) + shape) * 0.04
        for name, adapter in adapters.items():
            with self.subTest(case=name):
                folder = audit_dir / name
                folder.mkdir(parents=True, exist_ok=True)
                state = adapter.initial(initial, seeds=[3, 7, 11, 19])
                for matrix, bias in zip(matrices[:prefix_steps], biases[:prefix_steps]):
                    state, _, _ = adapter.step(state, affine(matrix, bias))
                (folder / "shared.json").write_bytes(adapter.config_bytes)
                (folder / "basis.bin").write_bytes(adapter.basis_bytes)
                (folder / "prefix.bin").write_bytes(state.payload.tobytes())
                (folder / "request.json").write_bytes(canonical({"batch_size": batch, "prefix_steps": prefix_steps, "total_steps": steps}))
                np.savez(folder / "audit_inputs.npz", matrices=matrices[prefix_steps:], biases=biases[prefix_steps:])
                reads, digests = [], []
                for matrix, bias in zip(matrices[prefix_steps:], biases[prefix_steps:]):
                    state, read, _ = adapter.step(state, affine(matrix, bias))
                    reads.append(read)
                    digests.append(hashlib.sha256(state.payload.tobytes()).hexdigest())
                (folder / "uninterrupted_suffix.bin").write_bytes(state.payload.tobytes())
                np.save(folder / "uninterrupted_reads.npy", np.asarray(reads), allow_pickle=False)
                environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
                completed = subprocess.run([sys.executable, "-c", FRESH_PROCESS, str(CASE_DIR), str(folder)],
                                           check=True, capture_output=True, text=True, env=environment, timeout=30)
                self.assertEqual(completed.stdout, "")
                self.assertEqual(completed.stderr, "")
                child = json.loads((folder / "child_result.json").read_bytes())
                self.assertNotEqual(child["pid"], os.getpid())
                self.assertEqual(child["parent_pid"], os.getpid())
                self.assertEqual(child["per_step_payload_sha256"], digests)
                child_payload = (folder / "restarted_suffix.bin").read_bytes()
                self.assertEqual(child_payload, state.payload.tobytes())
                np.testing.assert_array_equal(np.load(folder / "restarted_reads.npy", allow_pickle=False), np.asarray(reads))
                restored = adapter.from_bytes(child_payload, batch_size=batch)
                cursor, z, r = adapter._parts(restored)
                np.testing.assert_array_equal(cursor, np.full(batch, steps, dtype=np.uint64))
                if adapter.state_codec.stochastic:
                    seeds, counters = adapter.state_codec.rng_state(z)
                    np.testing.assert_array_equal(seeds, [3, 7, 11, 19])
                    np.testing.assert_array_equal(counters, np.full(batch, (steps + 1) * 12, dtype=np.uint64))
                    residual_seeds, residual_counters = adapter.residual_codec.rng_state(r)
                    np.testing.assert_array_equal(residual_seeds, seeds)
                    np.testing.assert_array_equal(residual_counters, np.full(batch, (steps + 1) * 8, dtype=np.uint64))
                manifest = dict(case=name, status="pass", prefix_steps=prefix_steps, suffix_steps=steps-prefix_steps,
                                separate_process=True, per_step_payload_identical=True, suffix_reads_identical=True,
                                cursor_restored=True, state_and_residual_rng_restored=name == "packed-sr-lowrank",
                                final_payload_sha256=digests[-1], ledger=adapter.ledger(state),
                                codec_config_sha256=hashlib.sha256(adapter.config_bytes).hexdigest(),
                                source_sha256={filename: hashlib.sha256((CASE_DIR / "codec" / filename).read_bytes()).hexdigest()
                                               for filename in ("online.py", "packed.py")})
                (folder / "audit_result.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    unittest.main()
