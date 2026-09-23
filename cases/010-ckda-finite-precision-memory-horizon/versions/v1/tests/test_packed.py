"""Independent byte-layout, numerical, restart, and exact-budget checks (CPU)."""

import json
from pathlib import Path
import struct
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from codec.packed import NativeFloatCodec, PackedBatch, PackedCodec


def reference_bytes(codes, widths):
    """Small independent scalar implementation, intentionally unlike production."""
    accumulator = 0
    cursor = 0
    for code, width in zip(codes, widths):
        accumulator |= int(code) << cursor
        cursor += int(width)
    return accumulator.to_bytes((cursor + 7) // 8, byteorder="little")


class PackedCodecTests(unittest.TestCase):
    def test_all_codes_all_required_bits(self):
        for bits in range(2, 9):
            with self.subTest(bits=bits):
                qmax = (1 << (bits - 1)) - 1
                q = np.arange(-qmax, qmax + 1, dtype=np.float32)
                codec = PackedCodec((1, 1, len(q)), bits=bits, max_abs=qmax)
                state = codec.encode(q.reshape(1, 1, 1, -1))
                np.testing.assert_array_equal(codec.decode(state).ravel(), q)
                expected = struct.pack("<f", qmax) + reference_bytes(q.astype(int) + qmax + 1, [bits] * len(q))
                self.assertEqual(state.to_bytes(), expected)
                self.assertEqual(state.nbytes, 4 + (len(q) * bits + 7) // 8)

    def test_extended_uniform_bits_and_three_byte_straddle(self):
        for bits in range(9, 17):
            qmax = (1 << (bits - 1)) - 1
            # More than eight entries exercise every reachable starting offset.
            q = np.array([-qmax, -qmax + 1, -128, -2, -1, 0, 1, 2, 127, qmax - 1, qmax], dtype=np.float32)
            codec = PackedCodec((1, 1, q.size), bits=bits, max_abs=qmax)
            state = codec.encode(q.reshape(1, 1, 1, -1))
            np.testing.assert_array_equal(codec.decode(state).ravel(), q)
            self.assertEqual(state.to_bytes()[4:], reference_bytes(q.astype(int) + qmax + 1, [bits] * q.size))

    def test_exact_little_endian_example_and_padding(self):
        codec = PackedCodec((1, 1, 3), bits=3, max_abs=3)
        state = codec.encode(np.array([[[[-3, 0, 3]]]], dtype=np.float32))
        # 001,100,111 LSB-first => 11100001 00000001.
        self.assertEqual(state.to_bytes(), struct.pack("<f", 3) + bytes([0xE1, 0x01]))
        corrupt = state.payload.copy()
        corrupt[0, -1] |= 0x80
        with self.assertRaisesRegex(ValueError, "padding"):
            codec.from_payload(corrupt)

    def test_zero_constant_extremes_and_per_head_scales(self):
        codec = PackedCodec((2, 2, 3), bits=4)
        x = np.zeros((3, 2, 2, 3), dtype=np.float32)
        x[1, 0] = -7
        x[1, 1] = 31
        x[2, 0, :, :] = np.finfo(np.float32).tiny
        x[2, 1, :, :] = np.nextafter(np.float32(0), np.float32(1))
        state = codec.encode(x)
        np.testing.assert_array_equal(codec.decode(state), x)
        self.assertEqual(state.to_bytes(1)[:8], struct.pack("<ff", 7, 31))
        self.assertEqual(state.to_bytes(0)[8:], bytes([0x88] * 6))
        bad = state.payload.copy()
        bad[0, 8] = 0x89
        with self.assertRaisesRegex(ValueError, "zero scale"):
            codec.from_payload(bad)

    def test_clipping_and_signed_ties_even(self):
        codec = PackedCodec((1, 1, 10), bits=4, max_abs=7)
        x = np.array([-100, -6.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 6.5, 100]).reshape(1, 1, 1, 10)
        np.testing.assert_array_equal(codec.decode(codec.encode(x)).ravel(), [-7, -6, -2, -2, 0, 0, 2, 2, 6, 7])
        tiny = float(np.nextafter(np.float32(0), np.float32(1)))
        narrow = PackedCodec((1, 1, 1), max_abs=tiny)
        with np.errstate(all="raise"):
            decoded = narrow.decode(narrow.encode(np.array([[[[1e300]]]])))
        self.assertEqual(float(decoded.item()), tiny)

    def test_payload_only_ownership_noncontiguous_input_and_empty_batch(self):
        x = np.arange(24, dtype=np.float32).reshape(2, 1, 3, 4)[:, :, :, ::-1]
        for codec in [PackedCodec((1, 3, 4)), NativeFloatCodec((1, 3, 4))]:
            state = codec.encode(x)
            self.assertEqual(PackedBatch.__slots__, ("payload",))
            self.assertTrue(state.payload.flags.c_contiguous)
            self.assertTrue(state.payload.flags.owndata)
            self.assertFalse(np.shares_memory(x, state.payload))
            decoded = codec.decode(state)
            self.assertFalse(np.shares_memory(decoded, state.payload))
            np.testing.assert_array_equal(codec.decode(codec.from_payload(state.payload)), decoded)
            empty = codec.encode(np.empty((0, 1, 3, 4), dtype=np.float32))
            self.assertEqual(empty.nbytes, 0)
            self.assertEqual(codec.decode(codec.from_bytes(b"", batch_size=0)).shape, (0, 1, 3, 4))

    def test_random_error_bound_multiple_heads(self):
        rng = np.random.default_rng(31)
        x = rng.normal(size=(17, 3, 5, 7)).astype(np.float32)
        for bits in range(2, 9):
            codec = PackedCodec((3, 5, 7), bits=bits)
            decoded = codec.decode(codec.encode(x))
            bound = np.max(np.abs(x), axis=(2, 3), keepdims=True) / ((1 << (bits - 1)) - 1) / 2
            self.assertTrue(np.all(np.abs(decoded - x) <= bound + 4e-7))
            self.assertEqual(decoded.dtype, np.float32)

    def test_mixed_precision_mask_exact_budget_and_reference(self):
        bits = np.array([[4, 8, 4], [8, 4, 8]])
        codec = PackedCodec((2, 3, 5), mixed_bits=bits, stochastic=True, max_abs=[7, 127])
        x = np.zeros((2, 2, 3, 5), dtype=np.float32)
        state = codec.encode(x, seeds=[1, 2], counters=[3, 4])
        widths = np.repeat(bits.ravel(), 5)
        codes = 1 << (widths - 1)
        self.assertEqual(state.to_bytes()[24:], reference_bytes(codes, widths))
        ledger = codec.ledger(state)
        self.assertEqual(codec.data_bytes, 23)  # 180 bits, exactly one partial byte.
        self.assertEqual(codec.bytes_per_stream, 8 + 16 + 23)
        self.assertEqual(ledger["payload_array_nbytes"], len(state.payload.tobytes()))
        self.assertEqual(ledger["serialized_payload_bytes"], 2 * 47)
        self.assertEqual(ledger["total_serialized_bytes"], 94 + len(codec.config_bytes))
        self.assertEqual(ledger["shared_config_bytes"], len(codec.config_bytes))
        np.testing.assert_array_equal(codec.decode(state), x)
        mask_codec = PackedCodec((2, 3, 5), bits=4, mixed_bits=bits == 8)
        self.assertEqual(mask_codec.data_bytes, codec.data_bytes)
        one_dimensional = PackedCodec((2, 3, 5), mixed_bits=[4, 8, 4])
        self.assertEqual(one_dimensional.data_bytes, 20)

    def test_mixed_extended_widths(self):
        widths = [3, 16, 9, 2, 15]
        codec = PackedCodec((1, 5, 1), mixed_bits=widths, max_abs=1)
        x = np.array([-1, 1, -1, 0, 1], dtype=np.float32).reshape(1, 1, 5, 1)
        state = codec.encode(x)
        expected_codes = [1, 65535, 1, 2, 32767]
        self.assertEqual(state.to_bytes()[4:], reference_bytes(expected_codes, widths))
        np.testing.assert_array_equal(codec.decode(state), x)

    def test_stochastic_determinism_restart_and_rng_endianness(self):
        codec = PackedCodec((2, 3, 5), bits=3, stochastic=True)
        x = np.linspace(-1, 1, 90).reshape(3, 2, 3, 5)
        seeds = np.array([0x0123456789ABCDEF, 17, 29], dtype=np.uint64)
        counters = np.array([9, 10, 11], dtype=np.uint64)
        first = codec.encode(x, seeds=seeds, counters=counters)
        self.assertEqual(first.to_bytes()[8:24], struct.pack("<QQ", int(seeds[0]), 39))
        np.testing.assert_array_equal(first.payload, codec.encode(x, seeds=seeds, counters=counters).payload)
        restored_codec = PackedCodec.from_config(codec.config_bytes)
        restored = restored_codec.from_bytes(first.payload.tobytes(), batch_size=3)
        next_seeds, next_counters = restored_codec.rng_state(restored)
        np.testing.assert_array_equal(next_seeds, seeds)
        np.testing.assert_array_equal(next_counters, counters + 30)
        uninterrupted = codec.encode(codec.decode(first) + 0.02, *codec.rng_state(first))
        restarted = restored_codec.encode(restored_codec.decode(restored) + 0.02, next_seeds, next_counters)
        np.testing.assert_array_equal(restarted.payload, uninterrupted.payload)
        # A different rounding epoch changes at least one quantized data byte.
        shifted = codec.encode(x, seeds=seeds, counters=counters + 30)
        self.assertFalse(np.array_equal(first.payload[:, 24:], shifted.payload[:, 24:]))
        overflow = codec.encode(x[:1], seeds=3, counters=np.uint64(2**64 - 1))
        self.assertEqual(int(codec.rng_state(overflow)[1][0]), 29)

    def test_stochastic_fraction_and_splitmix_known_output(self):
        codec = PackedCodec((1, 1, 1), bits=2, max_abs=1, stochastic=True)
        def independent_mix(value):
            mask = (1 << 64) - 1
            value = (value + 0x9E3779B97F4A7C15) & mask
            value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
            value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
            return value ^ (value >> 31)
        self.assertEqual(independent_mix(0), 0xE220A8397B1DCDAF)
        expected = (independent_mix(independent_mix(0)) >> 11) / 2**53
        self.assertEqual(float(codec._uniforms(np.array([0], dtype=np.uint64), np.array([0], dtype=np.uint64))[0, 0]), expected)
        x = np.full((10000, 1, 1, 1), 0.25, dtype=np.float32)
        mean = float(codec.decode(codec.encode(x, seeds=np.arange(10000))).mean())
        self.assertLess(abs(mean - 0.25), 0.02)

    def test_seedkey_v2_separates_adjacent_streams_and_versions_config(self):
        codec = PackedCodec((1, 8, 1), stochastic=True)
        draws = codec._uniforms(np.array([100, 101], dtype=np.uint64), np.zeros(2, dtype=np.uint64))
        self.assertFalse(np.array_equal(draws[0, 1:], draws[1, :-1]))
        self.assertEqual(json.loads(codec.config_bytes)["rng"], "splitmix64_seedkey_v2_next_counter_le_u64")
        old = json.loads(codec.config_bytes)
        old["rng"] = "splitmix64_seed_next_counter_le_u64"
        with self.assertRaises(ValueError):
            PackedCodec.from_config(json.dumps(old, sort_keys=True, separators=(",", ":")).encode())

    def test_config_roundtrip_and_strict_validation(self):
        codec = PackedCodec((2, 3, 5), mixed_bits=[[4, 8, 4], [8, 4, 8]], stochastic=True)
        restored = PackedCodec.from_config(codec.config_bytes)
        self.assertEqual(restored.config_bytes, codec.config_bytes)
        config = json.loads(codec.config_bytes)
        self.assertEqual(config["scale_dtype"], "<f4")
        for mutation in [b"{}", b"[]", b"not json", codec.config_bytes + b" "]:
            with self.assertRaises(ValueError):
                PackedCodec.from_config(mutation)
        config["bit_order"] = "msb_first"
        with self.assertRaises(ValueError):
            PackedCodec.from_config(json.dumps(config, sort_keys=True, separators=(",", ":")).encode())

    def test_invalid_inputs_and_truncation(self):
        for bits in [-1, 0, 1, 17, 2.0, True]:
            with self.assertRaises(ValueError):
                PackedCodec((1, 2, 3), bits=bits)
        for shape in [(1, 2), (1, 0, 3), (1, 2, 3.0)]:
            with self.assertRaises(ValueError):
                PackedCodec(shape)
        for mixed in [[1, 4], [4.0, 8.0], [4], [[4, 8], [8, 4]]]:
            with self.assertRaises(ValueError):
                PackedCodec((1, 2, 3), mixed_bits=mixed)
        for maximum in [0, -1, np.nan, np.inf, 1e100, 1e-100, [1, 2]]:
            with self.assertRaises(ValueError):
                PackedCodec((1, 2, 3), max_abs=maximum)
        codec = PackedCodec((1, 2, 3))
        for x in [np.zeros((1, 2, 3)), np.zeros((1, 1, 3, 2)), np.full((1, 1, 2, 3), np.nan),
                  np.full((1, 1, 2, 3), np.inf), np.full((1, 1, 2, 3), -np.inf),
                  np.full((1, 1, 2, 3), 1e100), np.full((1, 1, 2, 3), 1e-100)]:
            with self.assertRaises(ValueError):
                codec.encode(x)
        state = codec.encode(np.ones((2, 1, 2, 3)))
        for raw in [state.to_bytes()[:-1], state.to_bytes() + b"\x00"]:
            with self.assertRaises(ValueError):
                codec.from_bytes(raw)
        for raw in [state.payload.astype(np.int16), state.payload[:, :-1], [1, 2]]:
            with self.assertRaises(ValueError):
                codec.from_payload(raw)
        with self.assertRaises(ValueError):
            PackedBatch(state.payload[:, ::-1])
        with self.assertRaises(ValueError):
            codec.encode(np.zeros((1, 1, 2, 3)), seeds=1)

    def test_corrupt_reserved_codes_scales_and_rng_arguments(self):
        codec = PackedCodec((1, 1, 1), bits=4)
        state = codec.encode(np.zeros((1, 1, 1, 1)))
        bad = state.payload.copy()
        bad[0, -1] = 0
        with self.assertRaisesRegex(ValueError, "reserved"):
            codec.from_payload(bad)
        for scale in [-1.0, np.inf, np.nan]:
            bad = state.payload.copy()
            bad[0, :4] = np.frombuffer(struct.pack("<f", scale), dtype=np.uint8)
            with self.assertRaises(ValueError):
                codec.from_payload(bad)
        stochastic = PackedCodec((1, 1, 1), stochastic=True)
        for seed in [-1, 1.5, [1, 2], 2**65]:
            with self.assertRaises(ValueError):
                stochastic.encode(np.zeros((1, 1, 1, 1)), seeds=seed)
        with self.assertRaises(ValueError):
            codec.rng_state(state)

    def test_native_controls_and_serialization(self):
        for dtype, format_code in [("float32", "f"), ("float64", "d")]:
            codec = NativeFloatCodec((1, 1, 3), dtype=dtype)
            x = np.array([1, -2, 0.1], dtype=dtype).reshape(1, 1, 1, 3)
            state = codec.encode(x)
            self.assertEqual(state.to_bytes(), struct.pack("<" + format_code * 3, *x.ravel()))
            np.testing.assert_array_equal(codec.decode(state), x)
            self.assertEqual(codec.decode(state).dtype, np.dtype(dtype))
            clone = NativeFloatCodec.from_config(codec.config_bytes)
            np.testing.assert_array_equal(clone.decode(clone.from_bytes(state.to_bytes())), x)
            self.assertEqual(codec.ledger(state)["scale_bytes_per_stream"], 0)
            self.assertEqual(codec.ledger(state)["rng_bytes_per_stream"], 0)
        with self.assertRaises(ValueError):
            NativeFloatCodec((1, 1, 3), dtype="float16")
        with self.assertRaises(ValueError):
            NativeFloatCodec((1, 1, 1)).encode(np.array([[[[1e100]]]]))
        with self.assertRaises(ValueError):
            NativeFloatCodec((1, 1, 1)).from_bytes(struct.pack("<f", float("nan")))


if __name__ == "__main__":
    unittest.main()
