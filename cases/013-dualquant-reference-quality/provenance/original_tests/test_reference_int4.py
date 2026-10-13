"""CPU tests of REFERENCE_PORT contracts, not packed GPU or full-model tests.

The source-parity tests compile only named pure function AST nodes from the
official source file. They execute no vendor module imports, private blockfmt
code or ebias computation. This checks element-only scale equations on finite
fixtures; it deliberately does not certify the official default configuration
or the port-specific final standard group-RTN casting.
"""

import ast
from pathlib import Path
import struct
import unittest

import torch
import torch.nn.functional as F

from source.reference_int4 import (
    DEFAULT_NUM_ITER,
    EPS,
    DegenerateScaleError,
    QuantizedWeight,
    compare_methods,
    pack_nibbles,
    quantize_element_int4,
    quantize_weight,
    unpack_nibbles,
)


def fixture(rows=16, cols=128, dtype=torch.float64, seed=317):
    generator = torch.Generator().manual_seed(seed)
    weight = torch.randn(rows, cols, generator=generator, dtype=dtype)
    # Vary column and row magnitudes to exercise both scales.
    weight *= torch.linspace(0.4, 2.2, cols, dtype=dtype).unsqueeze(0)
    weight *= torch.linspace(0.6, 1.7, rows, dtype=dtype).unsqueeze(1)
    return weight


def extracted_element_functions():
    """Extract only math functions; import statements never enter the AST."""
    path = Path(__file__).resolve().parents[1] / "vendor/dualquant/layer_wrapper_dualquant.py"
    wanted = {
        "_init_beta", "_init_alpha", "_quant_max_val", "_quant_element_int",
        "_inner_quant_element", "_inner_quant", "_quantize_alpha_inv",
        "dualquant_with_alpha_and_beta", "dualquant_with_alpha_only",
    }
    tree = ast.parse(path.read_text())
    definitions = [node for node in tree.body
                   if isinstance(node, ast.FunctionDef) and node.name in wanted]
    if {node.name for node in definitions} != wanted:
        raise AssertionError("official math function set changed; review extraction")
    module = ast.Module(body=definitions, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "torch": torch,
        "EPS": EPS,
        # ebias is unused by int4 + scale_format='none'; private code is skipped.
        "find_ebias": lambda matrix: None,
    }
    exec(compile(module, str(path) + ":pure_function_extraction", "exec"), namespace)
    return namespace


class RoundingAndPackingTests(unittest.TestCase):
    def test_signed_int4_ties_even_and_saturation(self):
        values = torch.tensor(
            [-100, -8.5, -8, -7.5, -6.5, -1.5, -0.5, 0, 0.5, 1.5, 2.5, 6.5, 7, 7.5, 100],
            dtype=torch.float64,
        )
        expected = torch.tensor([-8, -8, -8, -8, -6, -2, 0, 0, 0, 2, 2, 6, 7, 7, 7], dtype=torch.int8)
        self.assertTrue(torch.equal(quantize_element_int4(values), expected))

    def test_nextafter_rounding_boundaries(self):
        center = torch.tensor([0.5, -0.5, 6.5, -6.5], dtype=torch.float64)
        lower = torch.nextafter(center, torch.full_like(center, -torch.inf))
        upper = torch.nextafter(center, torch.full_like(center, torch.inf))
        self.assertTrue(torch.equal(quantize_element_int4(lower), torch.tensor([0, -1, 6, -7], dtype=torch.int8)))
        self.assertTrue(torch.equal(quantize_element_int4(upper), torch.tensor([1, 0, 7, -6], dtype=torch.int8)))

    def test_element_quantizer_rejects_nonfinite(self):
        for bad in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "nonfinite"):
                quantize_element_int4(torch.tensor([bad]))

    def test_all_codes_have_known_nibble_layout(self):
        codes = torch.arange(-8, 8, dtype=torch.int8)
        packed = pack_nibbles(codes)
        self.assertEqual(packed.tolist(), [0x98, 0xBA, 0xDC, 0xFE, 0x10, 0x32, 0x54, 0x76])
        self.assertTrue(torch.equal(unpack_nibbles(packed, (4, 4)), codes.reshape(4, 4)))

    def test_nibble_roundtrip_noncontiguous(self):
        codes = torch.arange(128, dtype=torch.int16).remainder(16).sub(8).reshape(8, 16).T
        packed = pack_nibbles(codes)
        self.assertEqual(packed.numel(), codes.numel() // 2)
        self.assertTrue(torch.equal(unpack_nibbles(packed, codes.shape), codes.to(torch.int8)))

    def test_unsigned_nonnegative_codes_pack_without_wrapped_range_checks(self):
        codes = torch.arange(8, dtype=torch.uint8)
        self.assertTrue(torch.equal(unpack_nibbles(pack_nibbles(codes), (8,)), codes.to(torch.int8)))

    def test_nibble_validation(self):
        for invalid in (
            torch.tensor([0.0, 1.0]), torch.tensor([-9, 0]),
            torch.tensor([0, 8]), torch.tensor([0]), torch.tensor([], dtype=torch.int8),
            torch.tensor([False, True]),
        ):
            with self.subTest(invalid=invalid), self.assertRaises((ValueError, TypeError)):
                pack_nibbles(invalid)
        for packed, shape in (
            (torch.tensor([0], dtype=torch.uint8), (3,)),
            (torch.tensor([0], dtype=torch.uint8), (2, 2)),
            (torch.tensor([0], dtype=torch.int8), (2,)),
            (torch.tensor([[0]], dtype=torch.uint8), (2,)),
            (torch.tensor([0], dtype=torch.uint8), (0,)),
        ):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                unpack_nibbles(packed, shape)


class ValidationAndZeroTests(unittest.TestCase):
    def test_nonfinite_weight_is_strictly_rejected(self):
        for method in ("rtn", "row_only", "row_column"):
            for bad in (float("nan"), float("inf"), -float("inf")):
                W = torch.ones(4, 64)
                W[0, 0] = bad
                with self.subTest(method=method, bad=bad), self.assertRaisesRegex(ValueError, "nonfinite"):
                    quantize_weight(W, method)

    def test_invalid_weight_shapes_and_dtypes(self):
        for W in (
            torch.ones(64), torch.ones(2, 65), torch.ones(2, 32),
            torch.ones(2, 2, 64), torch.empty(0, 64), torch.empty(2, 0),
            torch.ones(2, 64, dtype=torch.int32), torch.ones(2, 64, dtype=torch.complex64),
        ):
            with self.subTest(shape=W.shape, dtype=W.dtype), self.assertRaises((ValueError, TypeError)):
                quantize_weight(W)

    def test_invalid_configuration_is_rejected(self):
        W = fixture(4, 64)
        for kwargs in (
            {"method": "only_column"}, {"group_size": 32}, {"group_size": 64.0},
            {"num_iter": -1}, {"num_iter": 1.5}, {"num_iter": True},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                quantize_weight(W, **kwargs)

    def test_all_zero_weight_keeps_exact_zeros_and_unit_scales(self):
        for method in ("rtn", "row_only", "row_column"):
            with self.subTest(method=method):
                result = quantize_weight(torch.zeros(5, 128, dtype=torch.float64), method)
                self.assertEqual(torch.count_nonzero(result.weight_hat()).item(), 0)
                self.assertEqual(torch.count_nonzero(result.codes).item(), 0)
                self.assertTrue(bool((result.gamma == 1).all()))
                self.assertTrue(bool((result.alpha_inv == 1).all()))
                self.assertTrue(bool((result.beta_inv == 1).all()))
                for record in result.iteration_records:
                    self.assertEqual(record["mse_requantized"], 0.0)

    def test_zero_rows_columns_and_group_are_exactly_preserved(self):
        W = fixture(16, 192)
        W[3, :] = 0
        W[:, 8] = 0
        W[:, 64:128] = 0
        W[4, 128:] = 0
        for method in ("rtn", "row_only", "row_column"):
            with self.subTest(method=method):
                result = quantize_weight(W, method)
                self.assertEqual(torch.count_nonzero(result.weight_hat()[W == 0]).item(), 0)
                self.assertTrue(bool((result.gamma[:, 1] == 1).all()))
                self.assertTrue(bool((result.alpha_inv[:, 1] == 1).all()))
                self.assertTrue(bool((result.beta_inv[64:128] == 1).all()))

    def test_degenerate_nonzero_column_is_not_silently_repaired(self):
        # The dense first column has normalized magnitude 1/64 in every row;
        # sparse remaining columns set row maxima >=1/2, making its codes zero.
        W = torch.zeros(64, 64, dtype=torch.float64)
        W[1:, 1:] = torch.eye(63, dtype=torch.float64)
        W[0, 1] = 1
        W[:, 0] = 1
        with self.assertRaisesRegex(DegenerateScaleError, "updated beta_inv"):
            quantize_weight(W, "row_column", num_iter=1)

    def test_finite_input_with_nonfinite_intermediate_is_rejected(self):
        W = torch.full((4, 64), torch.finfo(torch.float32).max / 2, dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "column L1 initialization.*nonfinite"):
            quantize_weight(W, "row_column")

    def test_noncontiguous_inputs_and_no_input_mutation(self):
        W = fixture(16, 128).T.contiguous().T.requires_grad_(True)
        original = W.detach().clone()
        result = quantize_weight(W)
        contiguous = quantize_weight(W.detach().contiguous())
        self.assertTrue(torch.equal(W, original))
        self.assertTrue(torch.equal(result.weight_hat(), contiguous.weight_hat()))
        self.assertFalse(result.alpha_inv.requires_grad)

    def test_half_inputs_use_finite_fp32_working_scales(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                result = quantize_weight(fixture(16, 64, torch.float32).to(dtype))
                self.assertEqual(result.dtype, torch.float32)
                self.assertTrue(bool(torch.isfinite(result.weight_hat()).all()))


class SourceEquationAndOptimizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.official = extracted_element_functions()

    def test_element_only_alpha_beta_match_extracted_source(self):
        W = fixture(16, 128)
        for count in (0, 1, DEFAULT_NUM_ITER):
            result = quantize_weight(W, "row_column", num_iter=count)
            for group in range(2):
                config = {
                    "num_iter": count, "block_size": 64, "row_init": "max_abs",
                    "col_init": "l1_norm", "scale_format": "none",
                    "dualquant_to_element_tensor": True,
                }
                alpha, beta, _ = self.official["dualquant_with_alpha_and_beta"](
                    W[:, group*64:(group+1)*64], config, "rtn_int4",
                )
                with self.subTest(count=count, group=group):
                    torch.testing.assert_close(result.alpha_inv[:, group], alpha.reciprocal(), rtol=2e-12, atol=2e-14)
                    torch.testing.assert_close(result.beta_inv[group*64:(group+1)*64], beta.reciprocal(), rtol=2e-12, atol=2e-12)

    def test_row_only_alpha_matches_extracted_source(self):
        W = fixture(16, 128)
        result = quantize_weight(W, "row_only")
        for group in range(2):
            config = {
                "num_iter": DEFAULT_NUM_ITER, "block_size": 64,
                "row_init": "max_abs", "scale_format": "none",
                "dualquant_to_element_tensor": True,
            }
            alpha, _ = self.official["dualquant_with_alpha_only"](
                W[:, group*64:(group+1)*64], config, "rtn_int4",
            )
            torch.testing.assert_close(result.alpha_inv[:, group], alpha.reciprocal(), rtol=2e-12, atol=2e-13)
        self.assertTrue(bool((result.beta_inv == 1).all()))

    def test_beta_uses_fresh_codes_after_alpha(self):
        W = fixture(16, 64)
        beta_init = W.abs().sum(dim=0).clamp_min(1e-8)
        alpha_init = (W / beta_init).abs().amax(dim=1) / 7
        q0 = quantize_element_int4(alpha_init.reciprocal().unsqueeze(1) * W * beta_init.reciprocal()).to(W.dtype)
        alpha = (W*q0*beta_init).sum(dim=1) / ((q0*beta_init).square().sum(dim=1)+EPS)
        q1 = quantize_element_int4(alpha.reciprocal().unsqueeze(1) * W * beta_init.reciprocal()).to(W.dtype)
        beta_fresh = (W*q1*alpha.unsqueeze(1)).sum(dim=0) / ((q1*alpha.unsqueeze(1)).square().sum(dim=0)+EPS)
        beta_stale = (W*q0*alpha.unsqueeze(1)).sum(dim=0) / ((q0*alpha.unsqueeze(1)).square().sum(dim=0)+EPS)
        result = quantize_weight(W, "row_column", num_iter=1)
        torch.testing.assert_close(result.alpha_inv[:, 0], alpha, rtol=2e-12, atol=2e-14)
        torch.testing.assert_close(result.beta_inv, beta_fresh, rtol=2e-12, atol=2e-12)
        self.assertGreater((beta_fresh-beta_stale).abs().max().item(), 1e-4)
        self.assertEqual(result.iteration_records[0]["alpha_code_changes"], int((q1 != q0).sum()))

    def test_fixed_code_regularized_ls_objective_never_increases(self):
        for method in ("row_only", "row_column"):
            result = quantize_weight(fixture(), method)
            self.assertEqual(len(result.iteration_records), DEFAULT_NUM_ITER)
            for record in result.iteration_records:
                with self.subTest(method=method, iteration=record["iteration"]):
                    for scale in ("alpha", "beta"):
                        before = record[f"{scale}_objective_before"]
                        after = record[f"{scale}_objective_after"]
                        self.assertLessEqual(after, before + 1e-10 * max(1.0, before))
                        self.assertLessEqual(record[f"mse_after_{scale}"], record[f"mse_before_{scale}"] + 1e-12)
                    self.assertGreaterEqual(record["alpha_code_changes"], 0)
                    self.assertGreaterEqual(record["beta_code_changes"], 0)
                    self.assertFalse(record["nonfinite_repaired"])

    def test_bounded_default_and_comparison_methods(self):
        self.assertEqual(DEFAULT_NUM_ITER, 15)
        results = compare_methods(fixture(16, 64))
        self.assertEqual(set(results), {"rtn", "row_only", "row_column"})
        self.assertEqual(len(results["rtn"].iteration_records), 0)
        self.assertEqual(results["rtn"].num_iter, 0)
        for method in ("row_only", "row_column"):
            self.assertEqual(len(results[method].iteration_records), 15)
            self.assertIn("REFERENCE_PORT", results[method].provenance)

    def test_final_cast_uses_explicit_standard_inner_gamma(self):
        W = fixture(16, 128)
        result = quantize_weight(W)
        blocks = W.reshape(16, 2, 64)
        scaled = result.alpha_inv.reciprocal().unsqueeze(-1) * blocks * result.beta_inv.reshape(2, 64).reciprocal()
        expected_gamma = scaled.abs().amax(dim=-1) / 7
        expected_codes = quantize_element_int4(scaled / expected_gamma.unsqueeze(-1))
        self.assertTrue(torch.equal(result.gamma, expected_gamma))
        self.assertTrue(torch.equal(result.codes.reshape(16, 2, 64), expected_codes))

    def test_toy_exactly_representable_rtn_fixture(self):
        # Include positive 7 so group gamma=1 and signed -8 isn't the absmax.
        # Standard absmax/7 uses gamma=8/7 when -8 occurs, so [-7,7] is the
        # exactly representable symmetric toy; -8 is covered by packing tests.
        pattern = torch.arange(-7, 8, dtype=torch.float64)
        W = pattern.repeat(5)[:64].reshape(1, 64)
        result = quantize_weight(W, "rtn")
        self.assertTrue(torch.equal(result.gamma, torch.ones(1, 1, dtype=torch.float64)))
        self.assertTrue(torch.equal(result.weight_hat(), W))


class TransportAndSerializationTests(unittest.TestCase):
    def test_fp64_transport_matches_dense_for_batch_and_bias(self):
        result = quantize_weight(fixture())
        generator = torch.Generator().manual_seed(718)
        x = torch.randn(2, 3, 128, generator=generator, dtype=torch.float64)
        bias = torch.randn(16, generator=generator, dtype=torch.float64)
        transported = result.linear(x, bias)
        dense = F.linear(x, result.weight_hat(), bias)
        torch.testing.assert_close(transported, dense, rtol=2e-13, atol=2e-12)
        torch.testing.assert_close(result.weight_hat(), result.weight_pre()*result.beta_inv, rtol=0, atol=0)
        vector = result.linear(x[0, 0])
        torch.testing.assert_close(vector, F.linear(x[0, 0], result.weight_hat()), rtol=2e-13, atol=2e-12)

    def test_beta_omission_is_detected_by_negative_control(self):
        result = quantize_weight(fixture())
        x = torch.ones(3, 128, dtype=torch.float64)
        dense = F.linear(x, result.weight_hat())
        missing_beta = F.linear(x, result.weight_pre())
        relative_error = (missing_beta-dense).norm()/dense.norm()
        self.assertGreater(relative_error.item(), 0.8)

    def test_transport_rejects_bad_shape_dtype_and_nonfinite(self):
        result = quantize_weight(fixture())
        for x in (torch.ones(127, dtype=torch.float64), torch.ones(128), torch.full((128,), float("nan"), dtype=torch.float64)):
            with self.subTest(shape=x.shape, dtype=x.dtype), self.assertRaises(ValueError):
                result.linear(x)
        x = torch.ones(128, dtype=torch.float64)
        for bias in (torch.ones(17, dtype=torch.float64), torch.ones(16), torch.full((16,), float("inf"), dtype=torch.float64)):
            with self.subTest(bias=bias.shape), self.assertRaises(ValueError):
                result.linear(x, bias)

    def test_serialization_exact_roundtrip_all_methods_and_scale_dtypes(self):
        for dtype in (torch.float32, torch.float64):
            for method in ("rtn", "row_only", "row_column"):
                with self.subTest(dtype=dtype, method=method):
                    result = quantize_weight(fixture(dtype=dtype), method)
                    serialized = result.serialize()
                    restored = QuantizedWeight.deserialize(serialized)
                    self.assertEqual(restored.method, result.method)
                    self.assertEqual(restored.num_iter, result.num_iter)
                    self.assertEqual(restored.serialize(), serialized)
                    for name in ("codes", "gamma", "alpha_inv", "beta_inv"):
                        self.assertTrue(torch.equal(getattr(result, name), getattr(restored, name)))
                    self.assertTrue(torch.equal(result.weight_hat(), restored.weight_hat()))

    def test_storage_budget_counts_alpha_beta_and_inner_scales(self):
        for method in ("rtn", "row_only", "row_column"):
            result = quantize_weight(fixture(), method)
            ledger = result.storage_ledger()
            expected_code = 16*128//2
            expected_gamma = 16*2*8
            expected_alpha = 0 if method == "rtn" else 16*2*8
            expected_beta = 128*8 if method == "row_column" else 0
            self.assertEqual(ledger["code_bytes"], expected_code)
            self.assertEqual(ledger["gamma_bytes"], expected_gamma)
            self.assertEqual(ledger["alpha_bytes"], expected_alpha)
            self.assertEqual(ledger["beta_bytes"], expected_beta)
            self.assertEqual(ledger["payload_bytes"], expected_code+expected_gamma+expected_alpha+expected_beta)
            self.assertEqual(ledger["serialized_bytes"], len(result.serialize()))
            self.assertFalse(ledger["packed_gpu_execution"])
            self.assertFalse(ledger["scale_folding_applied"])

    def test_serialization_rejects_corruption_and_nonfinite_scales(self):
        data = quantize_weight(fixture(16, 64), "row_column").serialize()
        for bad in (b"", data[:-1], data+b"x", b"bad!"+data[4:], data[:4]+b"\x02"+data[5:]):
            with self.subTest(length=len(bad)), self.assertRaises(ValueError):
                QuantizedWeight.deserialize(bad)
        altered = bytearray(data)
        gamma_offset = 24 + 16*64//2
        altered[gamma_offset:gamma_offset+8] = struct.pack("<d", float("nan"))
        with self.assertRaisesRegex(ValueError, "gamma.*nonfinite"):
            QuantizedWeight.deserialize(bytes(altered))


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
