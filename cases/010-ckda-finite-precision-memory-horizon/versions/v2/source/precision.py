"""Factor transition/readout arithmetic while holding v1 FP32 coefficients fixed.

Dab uses FP64 transition when a=1 and explicit FP64 readout when b=1.
The caller verifies all checkpoint identities before constructing this diagnostic.
This module performs no model training, coefficient reprojection, or tensor logging.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from types import MappingProxyType

import numpy as np
import torch
from torch.nn import functional as F

from .reference import learned, v1

VERSION = "case010-v2-fixed-coefficient-precision-v1"
MODES = ("D00", "D10", "D01", "D11")
DTYPES = {"D00": (torch.float32, torch.float32),
          "D10": (torch.float64, torch.float32),
          "D01": (torch.float32, torch.float64),
          "D11": (torch.float64, torch.float64)}
LIMITATIONS = (
    "All normalized q/k, v, alpha, beta, output gates and embeddings are the exact "
    "stored FP32 values; FP64 promotion does not recompute projections or gates.",
    "FP64 readout promotes the fixed FP32 readout parameters and explicitly uses "
    "FP64 RMS normalization, sigmoid, projection, LayerNorm, GELU and classifier.",
    "F.linear/F.layer_norm/F.gelu and einsum retain the same mathematical order "
    "and epsilon/activation settings; dtype-dependent backend reduction algorithms "
    "need not have the same rounding order. This is not arbitrary-precision truth.",
    "D10 casts its FP64 state to FP32 before the complete original readout; D01 "
    "promotes its already-rounded FP32 state before the complete FP64 readout.",
    "State and logit differences are diagnostic comparisons, not causal proof "
    "of why the first label failed. Gold is used only by the external scalar summary.",
)


def _cpu_float32(value, name):
    tensor = value if isinstance(value, torch.Tensor) else torch.from_numpy(np.array(value, copy=True))
    if tensor.device.type != "cpu" or tensor.dtype != torch.float32:
        raise ValueError(f"{name} must contain CPU FP32 values")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must be finite")
    return tensor.detach()


@dataclass(frozen=True)
class ReadoutWeights:
    """Independent frozen values, never a .double() mutation of FLA modules."""

    fp32: object
    fp64: object
    rms_eps: float
    layer_eps: float
    normalized_shape: tuple
    gelu_approximate: str
    head_key_dim: int

    @classmethod
    def from_model(cls, model):
        layer = model.layer
        if layer.o_norm.activation != "sigmoid":
            raise ValueError("The pinned CKDA output gate must be sigmoid")
        if not (isinstance(model.norm, torch.nn.LayerNorm)
                and isinstance(model.mlp, torch.nn.Sequential)
                and len(model.mlp) == 3
                and isinstance(model.mlp[0], torch.nn.Linear)
                and isinstance(model.mlp[1], torch.nn.GELU)
                and isinstance(model.mlp[2], torch.nn.Linear)
                and isinstance(layer.o_proj, torch.nn.Linear)):
            raise ValueError("Expected the pinned LayerNorm/Linear/GELU/Linear readout")
        values = {
            "rms_weight": layer.o_norm.weight,
            "projection_weight": layer.o_proj.weight,
            "projection_bias": layer.o_proj.bias,
            "layer_weight": model.norm.weight,
            "layer_bias": model.norm.bias,
            "hidden_weight": model.mlp[0].weight,
            "hidden_bias": model.mlp[0].bias,
            "classifier_weight": model.mlp[2].weight,
            "classifier_bias": model.mlp[2].bias,
        }
        values = {name: None if value is None else _cpu_float32(value, name).clone()
                  for name, value in values.items()}
        doubles = {name: None if value is None else value.to(torch.float64)
                   for name, value in values.items()}
        return cls(MappingProxyType(values), MappingProxyType(doubles),
                   float(layer.o_norm.eps), float(model.norm.eps),
                   tuple(model.norm.normalized_shape), model.mlp[1].approximate,
                   int(layer.head_k_dim))


@torch.inference_mode()
def explicit_readout(weights, state, coefficients, dtype):
    """Return logits and actual intermediate dtypes; no forced .float() norm."""
    if dtype not in (torch.float32, torch.float64) or state.device.type != "cpu":
        raise ValueError("Readout requires CPU float32 or float64 arithmetic")
    trace = {}

    def record(name, tensor):
        if tensor.dtype != dtype:
            raise AssertionError(f"{name} unexpectedly computed in {tensor.dtype}")
        trace[name] = str(tensor.dtype)
        return tensor

    parameters = weights.fp32 if dtype == torch.float32 else weights.fp64
    x = record("state_at_read_boundary", state.to(dtype))
    q, gate, embedding = [record(name, coefficients[name].to(dtype))
                          for name in ("q", "g", "e")]
    q = record("scaled_query", q * weights.head_key_dim**-0.5)
    out = record("state_read", torch.einsum("b h k, b h k v -> b h v", q, x))
    square = record("rms_square", out.pow(2))
    mean_square = record("rms_mean_square", square.mean(-1, keepdim=True))
    inverse = record("rms_inverse", torch.rsqrt(mean_square + weights.rms_eps))
    out = record("rms_normalized", out * inverse)
    if parameters["rms_weight"] is not None:
        out = record("rms_affine", out * parameters["rms_weight"])
    gate = record("sigmoid_gate", torch.sigmoid(gate))
    out = record("gated_output", out * gate)
    out = record("output_projection", F.linear(out.flatten(-2),
                 parameters["projection_weight"], parameters["projection_bias"]))
    out = record("embedding_residual", embedding + out)
    out = record("layer_norm", F.layer_norm(out, weights.normalized_shape,
                 parameters["layer_weight"], parameters["layer_bias"], weights.layer_eps))
    out = record("hidden_linear", F.linear(out, parameters["hidden_weight"],
                 parameters["hidden_bias"]))
    out = record("gelu", F.gelu(out, approximate=weights.gelu_approximate))
    out = record("classifier", F.linear(out, parameters["classifier_weight"],
                 parameters["classifier_bias"]))
    return out, trace


@torch.inference_mode()
def original_readout(model, state, coefficients):
    """The unchanged v1 original modules, with an explicit FP32 read boundary."""
    state32 = state.to(torch.float32)
    coeff = {name: value.detach().numpy() for name, value in coefficients.items()}
    return v1.logits_from_numpy(model, state32.numpy(), coeff)


@torch.inference_mode()
def verify_fp32_readout(model, weights, state, coefficients):
    """Fixed v1 tolerances; failure must block the diagnostic, never relax them."""
    actual, trace = explicit_readout(weights, state, coefficients, torch.float32)
    expected = original_readout(model, state, coefficients)
    torch.testing.assert_close(actual, expected, atol=learned.ATOL, rtol=learned.RTOL)
    return {"passed": True, "atol": learned.ATOL, "rtol": learned.RTOL,
            "maximum_absolute_difference": float((actual - expected).abs().max()),
            "bitwise_equal": bool(torch.equal(actual, expected)), "dtype_trace": trace}


def _identity(table, identity):
    result = dict(identity)
    verified = {str(key): value for key, value in result["verified_checkpoint_sha256"].items()}
    if set(verified) != {"0", "1", "2"}:
        raise ValueError("External identity verification must cover all three checkpoints")
    if any(not isinstance(value, str) or len(value) != 64
           or any(c not in "0123456789abcdef" for c in value) for value in verified.values()):
        raise ValueError("Expected SHA256 checkpoint identities")
    if result["checkpoint_sha256"] != verified[str(result["model_seed"])]:
        raise ValueError("Selected checkpoint identity does not match its model seed")
    table_sha = hashlib.sha256(v1.table_bytes(table)[1]).hexdigest()
    if result["token_table_sha256"] != table_sha:
        raise ValueError("FP32 token-table identity mismatch")
    result["verified_checkpoint_sha256"] = verified
    return result


class PrecisionDiagnostic:
    """CPU precision intervention using externally verified model/table inputs.

    `identity` must carry model_seed, checkpoint_sha256, token_table_sha256 and
    verified_checkpoint_sha256={"0": sha0, "1": sha1, "2": sha2}. The caller
    verifies file/source identities and loaded checkpoint weights before use.
    """

    def __init__(self, model, fp32_table, identity):
        if any(p.device.type != "cpu" or p.dtype != torch.float32 for p in model.parameters()):
            raise ValueError("Keep all original model parameters on CPU in FP32")
        self.model = model
        self.table = {name: _cpu_float32(value, name).clone() for name, value in fp32_table.items()}
        if set(self.table) != {"q", "k", "v", "alpha", "beta", "g", "e"}:
            raise ValueError("Expected the complete seven-coefficient v1 table")
        if any(value.shape[0] != 7 for value in self.table.values()):
            raise ValueError("Expected all six group tokens plus BOS")
        self.identity = _identity(fp32_table, identity)
        self.weights = ReadoutWeights.from_model(model)
        self.shape = (model.layer.num_heads, model.layer.head_k_dim, model.layer.head_v_dim)
        pattern = torch.arange(7 * math.prod(self.shape), dtype=torch.float32)
        pattern = (pattern.reshape((7,) + self.shape).remainder(31) - 15) / 100
        self.fp32_parity = [verify_fp32_readout(model, self.weights, state, self.table)
                            for state in (torch.zeros_like(pattern), pattern)]

    def initial_states(self, batch_size, initial_fp32=None):
        shape = (batch_size,) + self.shape
        initial = (torch.zeros(shape, dtype=torch.float32) if initial_fp32 is None
                   else _cpu_float32(initial_fp32, "initial_fp32"))
        if tuple(initial.shape) != shape:
            raise ValueError("Initial state shape mismatch")
        return {mode: initial.to(DTYPES[mode][0]).clone() for mode in MODES}

    @torch.inference_mode()
    def step(self, states, token_ids):
        ids = torch.as_tensor(token_ids, device="cpu")
        if ids.ndim != 1 or ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("token_ids must be a one-dimensional integer array")
        if bool(((ids < 0) | (ids > 6)).any()):
            raise ValueError("Token ID outside the frozen table")
        coeff32 = {name: value[ids] for name, value in self.table.items()}
        coeff64 = {name: value.to(torch.float64) for name, value in coeff32.items()}
        next_states, outputs, traces = {}, {}, {}
        for mode in MODES:
            transition_dtype, read_dtype = DTYPES[mode]
            state = states[mode]
            if (state.device.type != "cpu" or state.dtype != transition_dtype
                    or tuple(state.shape) != (len(ids),) + self.shape):
                raise ValueError(f"Invalid state shape/device/dtype for {mode}")
            coeff = coeff32 if transition_dtype == torch.float32 else coeff64
            updated = learned.transition(state, coeff)
            if read_dtype == torch.float32:
                output = original_readout(self.model, updated, coeff32)
                read_trace = {"route": "unchanged_v1_original_modules",
                              "state_at_read_boundary": "torch.float32",
                              "parameters": "torch.float32", "output": str(output.dtype)}
            else:
                output, read_trace = explicit_readout(self.weights, updated, coeff32, torch.float64)
            next_states[mode], outputs[mode] = updated, output
            traces[mode] = {"transition_state": str(updated.dtype),
                            "transition_coefficients": str(coeff["alpha"].dtype),
                            "coefficient_source": "fixed_FP32_table", "readout": read_trace}
        return next_states, outputs, traces

    def metadata(self):
        return {"version": VERSION, "identity": self.identity, "limitations": list(LIMITATIONS),
                "shape": list(self.shape), "rms_eps": self.weights.rms_eps,
                "layer_norm_eps": self.weights.layer_eps,
                "gelu_approximate": self.weights.gelu_approximate,
                "parameter_source_dtype": "torch.float32", "coefficients_reprojected": False,
                "fp32_readout_parity": self.fp32_parity,
                "parity_scope": "all seven tokens with zero and patterned nonzero states; original checkpoint readout modules",
                "formula_order": ["q_scaled_einsum_updated_state", "RMSNorm", "RMS_weight",
                                  "sigmoid_gating", "output_projection", "embedding_add",
                                  "LayerNorm", "Linear", "GELU", "Linear"]}


def _scalar_list(tensor):
    return [float(value) if math.isfinite(float(value)) else None for value in tensor.tolist()]


@torch.inference_mode()
def summarize_step(states, logits, gold, position):
    """Return scalar vectors only; position0 is unscored BOS, later positions score.

    Norms/differences are accumulated in FP64 for diagnostics only. Nonfinite
    values become JSON null with explicit finite flags, never a correct prediction.
    """
    gold = torch.as_tensor(gold, dtype=torch.int64, device="cpu")
    reference = states["D00"].to(torch.float64).flatten(1)
    reference_logits = logits["D00"].to(torch.float64)
    result = {"position": int(position), "scored": position > 0, "modes": {}}
    for mode in MODES:
        state = states[mode].to(torch.float64).flatten(1)
        output = logits[mode].to(torch.float64)
        if gold.shape != output.shape[:1] or bool(((gold < 0) | (gold >= output.shape[-1])).any()):
            raise ValueError("Gold shape or class is invalid")
        state_finite = torch.isfinite(state).all(-1)
        logit_finite = torch.isfinite(output).all(-1)
        finite = state_finite & logit_finite
        predictions = torch.where(finite, output.argmax(-1), -1)
        alternatives = output.clone()
        alternatives.scatter_(1, gold[:, None], -torch.inf)
        margin = output.gather(1, gold[:, None]).squeeze(1) - alternatives.amax(-1)
        largest = output.topk(2, dim=-1).values
        delta = state - reference
        norm = torch.linalg.vector_norm(state, dim=1)
        result["modes"][mode] = {
            "state_finite": state_finite.tolist(), "logits_finite": logit_finite.tolist(),
            "prediction": predictions.tolist(), "correct": (predictions == gold).tolist(),
            "gold_margin": _scalar_list(margin),
            "top_two_margin": _scalar_list(largest[:, 0] - largest[:, 1]),
            "state_norm": _scalar_list(norm),
            "state_difference_l2_from_D00": _scalar_list(torch.linalg.vector_norm(delta, dim=1)),
            "state_difference_mse_from_D00": _scalar_list(delta.square().mean(-1)),
            "logit_max_difference_from_D00": _scalar_list((output-reference_logits).abs().amax(-1)),
        }
    return result
