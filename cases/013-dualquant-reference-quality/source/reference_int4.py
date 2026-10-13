"""Bounded CPU REFERENCE_PORT of DualQuant's element-only INT4 equations.

Provenance: ``vendor/dualquant/layer_wrapper_dualquant.py`` is read as source,
never imported here. Its ``_init_beta(l1_norm)``, ``_init_alpha(max_abs)``,
``dualquant_with_alpha_and_beta`` and ``dualquant_with_alpha_only`` ground the
initialization and alternating updates below. The explicit variant is
``dualquant_to_element_tensor=True, scale_format='none', num_splits=1``.
It is not the official default configuration or a reproduction of private
``blockfmt.Format`` casting. In particular, the final cast is our standard
per-row, per-group INT4 RTN with an explicit inner scale gamma.

Each 64-column block is optimized independently, as in the source wrapper.
For a block, Q = clamp(round(alpha * W * beta), -8, 7), with torch's ties-even
rounding. One iteration is Q -> alpha_inv LS -> fresh Q -> beta_inv LS:
  alpha_inv[r] = sum_c(W[r,c]*Q[r,c]*beta_inv[c]) /
                 (sum_c((Q[r,c]*beta_inv[c])**2) + 1e-12)
  beta_inv[c] = sum_r(W[r,c]*Q[r,c]*alpha_inv[r]) /
                (sum_r((Q[r,c]*alpha_inv[r])**2) + 1e-12).
The fresh Q between updates is essential. EPS makes these fixed-code updates
minimizers of the corresponding least-squares objective plus EPS*scale**2.

The source lacks an INT4 zero-scale guard. Our explicit exceptional rule uses
unit scales for structurally zero rows/columns/groups, preserving exact zeros.
A zero LS scale for a *nonzero* row/column raises DegenerateScaleError; NaN,
Inf and intermediate nonfinite values raise rather than being repaired.

Stored codes, gamma, alpha_inv and beta_inv define
  W_pre = alpha_inv * gamma * codes,
  W_hat = W_pre * beta_inv,
  y = F.linear(x * beta_inv, W_pre, bias).
Nibble packing and serialization below are real CPU byte representations.
All linear operations reconstruct dense weights; no packed GPU execution,
official packed reproduction, model quality or speed claim is implied.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import struct
import sys
from typing import Any, Sequence

import torch
import torch.nn.functional as F

EPS = 1e-12
GROUP_SIZE = 64
DEFAULT_NUM_ITER = 15
PROVENANCE = "REFERENCE_PORT;element_only=True;scale_format=none;final_cast=standard_group_rtn"
_METHODS = ("rtn", "row_only", "row_column")
_HEADER = struct.Struct("<4sBBBBIIII")
_MAGIC = b"RPI4"
_VERSION = 1
_INTEGER_DTYPES = (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)


class DegenerateScaleError(ValueError):
    """A source-equation update yielded zero scale for nonzero source data."""


def _finite(tensor: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} contains a nonfinite value; no silent repair is allowed")


def _floating_cpu(tensor: torch.Tensor, name: str) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.device.type != "cpu":
        raise ValueError(f"{name} must be on CPU for this bounded reference port")
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must have a floating dtype")
    if tensor.layout != torch.strided:
        raise ValueError(f"{name} must have strided layout")
    _finite(tensor, name)


def _positive_scales(tensor: torch.Tensor, name: str) -> None:
    _finite(tensor, name)
    if bool((tensor <= 0).any()):
        raise DegenerateScaleError(f"{name} must be positive; nonzero data yielded a zero scale")


def quantize_element_int4(values: torch.Tensor) -> torch.Tensor:
    """Standard signed INT4: torch.round ties-even, then clamp to [-8, 7]."""
    _floating_cpu(values, "values")
    return torch.round(values).clamp(-8, 7).to(torch.int8)


def pack_nibbles(codes: torch.Tensor) -> torch.Tensor:
    """Serialize two signed two's-complement codes per uint8, low nibble first."""
    if not isinstance(codes, torch.Tensor):
        raise TypeError("codes must be a torch.Tensor")
    if codes.device.type != "cpu" or codes.layout != torch.strided:
        raise ValueError("codes must be a strided CPU tensor")
    if codes.dtype not in _INTEGER_DTYPES:
        raise TypeError("codes must have an integer dtype")
    if codes.numel() == 0 or codes.numel() % 2:
        raise ValueError("nibble packing requires a positive even number of codes")
    signed_codes = codes.to(torch.int16)
    if bool(((signed_codes < -8) | (signed_codes > 7)).any()):
        raise ValueError("codes must lie in [-8, 7]")
    nibbles = (codes.reshape(-1).to(torch.int16) & 15).to(torch.uint8)
    return (nibbles[0::2] | (nibbles[1::2] << 4)).contiguous()


def unpack_nibbles(packed: torch.Tensor, shape: Sequence[int]) -> torch.Tensor:
    """Invert pack_nibbles, rejecting missing bytes and odd or empty shapes."""
    if not isinstance(packed, torch.Tensor):
        raise TypeError("packed must be a torch.Tensor")
    if packed.device.type != "cpu" or packed.dtype != torch.uint8 or packed.ndim != 1:
        raise ValueError("packed must be a 1-D CPU uint8 tensor")
    shape = tuple(shape)
    if not shape or any(type(size) is not int or size <= 0 for size in shape):
        raise ValueError("shape must contain positive integers")
    count = math.prod(shape)
    if count % 2 or packed.numel() != count // 2:
        raise ValueError("packed byte count does not match an even output shape")
    raw = torch.stack((packed & 15, packed >> 4), dim=1).reshape(-1).to(torch.int16)
    signed = torch.where(raw >= 8, raw - 16, raw).to(torch.int8)
    return signed.reshape(shape)


def _raw_bytes(tensor: torch.Tensor) -> bytes:
    # Clone prevents an offset view from exposing unrelated storage bytes.
    if sys.byteorder != "little":
        raise RuntimeError("this CPU serialization is implemented for little-endian hosts")
    owned = tensor.detach().contiguous().clone()
    return bytes(owned.untyped_storage())


@dataclass(frozen=True)
class QuantizedWeight:
    """Explicit unmerged CPU representation; reconstruction is always dense."""

    codes: torch.Tensor
    gamma: torch.Tensor
    alpha_inv: torch.Tensor
    beta_inv: torch.Tensor
    method: str
    num_iter: int
    iteration_records: tuple[dict[str, Any], ...] = ()
    group_size: int = GROUP_SIZE
    provenance: str = PROVENANCE

    def __post_init__(self) -> None:
        self.validate()

    @property
    def dtype(self) -> torch.dtype:
        return self.gamma.dtype

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.codes.shape)

    def validate(self) -> None:
        if self.method not in _METHODS:
            raise ValueError(f"method must be one of {_METHODS}")
        if self.group_size != GROUP_SIZE or type(self.group_size) is not int:
            raise ValueError("this audit is fixed to group_size=64")
        if type(self.num_iter) is not int or self.num_iter < 0:
            raise ValueError("num_iter must be a nonnegative integer")
        if not isinstance(self.codes, torch.Tensor) or self.codes.dtype != torch.int8:
            raise TypeError("stored codes must be a torch.int8 tensor")
        if self.codes.device.type != "cpu" or self.codes.ndim != 2 or self.codes.layout != torch.strided:
            raise ValueError("stored codes must be a 2-D CPU tensor")
        rows, cols = self.codes.shape
        if rows <= 0 or cols <= 0 or cols % GROUP_SIZE:
            raise ValueError("weight must have positive dimensions and columns divisible by 64")
        if bool(((self.codes < -8) | (self.codes > 7)).any()):
            raise ValueError("stored codes must lie in [-8, 7]")
        for name, scale, expected in (
            ("gamma", self.gamma, (rows, cols // GROUP_SIZE)),
            ("alpha_inv", self.alpha_inv, (rows, cols // GROUP_SIZE)),
            ("beta_inv", self.beta_inv, (cols,)),
        ):
            _floating_cpu(scale, name)
            if scale.dtype not in (torch.float32, torch.float64) or scale.dtype != self.gamma.dtype:
                raise TypeError("all stored scales must share float32 or float64 dtype")
            if tuple(scale.shape) != expected:
                raise ValueError(f"{name} shape must be {expected}")
            _positive_scales(scale, name)
        if self.method == "rtn" and not bool((self.alpha_inv == 1).all()):
            raise ValueError("RTN alpha_inv must be identity because it is not serialized")
        if self.method != "row_column" and not bool((self.beta_inv == 1).all()):
            raise ValueError("RTN/row-only beta_inv must be identity because it is not serialized")

    def weight_pre(self) -> torch.Tensor:
        """W_pre = alpha_inv * gamma * codes, without column transport scales."""
        self.validate()
        rows, cols = self.shape
        blocks = self.codes.to(self.dtype).reshape(rows, cols // GROUP_SIZE, GROUP_SIZE)
        pre = blocks * self.gamma.unsqueeze(-1) * self.alpha_inv.unsqueeze(-1)
        _finite(pre, "W_pre")
        return pre.reshape(rows, cols)

    def weight_hat(self) -> torch.Tensor:
        """Equivalent dense W_hat = W_pre * beta_inv."""
        weight = self.weight_pre() * self.beta_inv.unsqueeze(0)
        _finite(weight, "W_hat")
        return weight

    def linear(self, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        """Dense CPU transport check, F.linear(x*beta_inv, W_pre, bias)."""
        _floating_cpu(x, "x")
        if x.ndim < 1 or x.shape[-1] != self.shape[1] or x.dtype != self.dtype:
            raise ValueError("x must match the weight input dimension and stored scale dtype")
        if bias is not None:
            _floating_cpu(bias, "bias")
            if bias.shape != (self.shape[0],) or bias.dtype != self.dtype:
                raise ValueError("bias must match output rows and stored scale dtype")
        scaled_x = x * self.beta_inv
        _finite(scaled_x, "scaled x")
        output = F.linear(scaled_x, self.weight_pre(), bias)
        _finite(output, "linear output")
        return output

    def serialize(self) -> bytes:
        """Actual nibble bytes plus unmerged scale tensors; excludes trace records."""
        self.validate()
        rows, cols = self.shape
        header = _HEADER.pack(
            _MAGIC, _VERSION, 1 if self.dtype == torch.float32 else 2,
            _METHODS.index(self.method), 0, rows, cols, self.group_size, self.num_iter,
        )
        payload = [_raw_bytes(pack_nibbles(self.codes)), _raw_bytes(self.gamma)]
        if self.method != "rtn":
            payload.append(_raw_bytes(self.alpha_inv))
        if self.method == "row_column":
            payload.append(_raw_bytes(self.beta_inv))
        return header + b"".join(payload)

    @classmethod
    def deserialize(cls, data: bytes) -> QuantizedWeight:
        """Load this port's CPU representation; reject malformed/nonfinite payloads."""
        if not isinstance(data, bytes):
            raise TypeError("serialized data must be bytes")
        if len(data) < _HEADER.size:
            raise ValueError("serialized header is truncated")
        magic, version, dtype_code, method_code, reserved, rows, cols, group, iters = _HEADER.unpack_from(data)
        if magic != _MAGIC or version != _VERSION or reserved != 0:
            raise ValueError("invalid serialization magic, version or reserved byte")
        if dtype_code not in (1, 2) or method_code >= len(_METHODS):
            raise ValueError("invalid serialized dtype or method")
        if group != GROUP_SIZE or rows <= 0 or cols <= 0 or cols % GROUP_SIZE:
            raise ValueError("invalid serialized weight dimensions or group size")
        dtype = torch.float32 if dtype_code == 1 else torch.float64
        method = _METHODS[method_code]
        scale_bytes = 4 if dtype_code == 1 else 8
        code_count = rows * cols // 2
        inner_count = rows * (cols // GROUP_SIZE)
        total = _HEADER.size + code_count + inner_count * scale_bytes
        if method != "rtn":
            total += inner_count * scale_bytes
        if method == "row_column":
            total += cols * scale_bytes
        if len(data) != total:
            raise ValueError("serialized byte count does not match header")
        cursor = _HEADER.size

        def take(count: int, item_dtype: torch.dtype, shape: tuple[int, ...]) -> torch.Tensor:
            nonlocal cursor
            segment = bytearray(data[cursor:cursor + count])
            cursor += count
            return torch.frombuffer(segment, dtype=item_dtype).clone().reshape(shape)

        packed = take(code_count, torch.uint8, (code_count,))
        codes = unpack_nibbles(packed, (rows, cols))
        gamma = take(inner_count * scale_bytes, dtype, (rows, cols // GROUP_SIZE))
        alpha = (take(inner_count * scale_bytes, dtype, gamma.shape)
                 if method != "rtn" else torch.ones_like(gamma))
        beta = (take(cols * scale_bytes, dtype, (cols,))
                if method == "row_column" else torch.ones(cols, dtype=dtype))
        return cls(codes, gamma, alpha, beta, method, iters)

    def storage_ledger(self) -> dict[str, Any]:
        """Count actual CPU serialized bytes, including every nonidentity scale."""
        self.validate()
        code_bytes = self.codes.numel() // 2
        gamma_bytes = self.gamma.numel() * self.gamma.element_size()
        alpha_bytes = 0 if self.method == "rtn" else self.alpha_inv.numel() * self.alpha_inv.element_size()
        beta_bytes = self.beta_inv.numel() * self.beta_inv.element_size() if self.method == "row_column" else 0
        payload_bytes = code_bytes + gamma_bytes + alpha_bytes + beta_bytes
        actual_bytes = len(self.serialize())
        if actual_bytes != payload_bytes + _HEADER.size:
            raise RuntimeError("serialization and storage ledger disagree")
        dense_fp16_bytes = self.codes.numel() * 2
        return {
            "representation": "REFERENCE_PORT_CPU_NIBBLE_SERIALIZATION",
            "packed_gpu_execution": False,
            "method": self.method,
            "group_size": self.group_size,
            "scale_dtype": str(self.dtype),
            "code_bytes": code_bytes,
            "gamma_bytes": gamma_bytes,
            "alpha_bytes": alpha_bytes,
            "beta_bytes": beta_bytes,
            "scale_bytes": gamma_bytes + alpha_bytes + beta_bytes,
            "payload_bytes": payload_bytes,
            "header_bytes": _HEADER.size,
            "serialized_bytes": actual_bytes,
            "dense_fp16_bytes": dense_fp16_bytes,
            "dense_fp32_bytes": self.codes.numel() * 4,
            "compression_vs_dense_fp16": dense_fp16_bytes / actual_bytes,
            "trace_records_included": False,
            "scale_folding_applied": False,
        }


def _mse(weight: torch.Tensor, estimate: torch.Tensor, name: str) -> float:
    _finite(estimate, name)
    squared_error = (weight - estimate).square()
    _finite(squared_error, f"{name} squared error")
    value = squared_error.mean()
    _finite(value, f"{name} MSE")
    return float(value.item())


def _fixed_code_objective(weight: torch.Tensor, estimate: torch.Tensor,
                          scale: torch.Tensor, name: str) -> float:
    # FP64 reporting prevents a large finite FP32 SSE from overflowing in the
    # diagnostic reduction. The optimizer itself still uses its working dtype.
    error = (weight.to(torch.float64) - estimate.to(torch.float64)).square().sum()
    objective = error + EPS * scale.to(torch.float64).square().sum()
    _finite(objective, name)
    return float(objective.item())


@torch.no_grad()
def quantize_weight(weight: torch.Tensor, method: str = "row_column", *,
                    group_size: int = GROUP_SIZE,
                    num_iter: int = DEFAULT_NUM_ITER) -> QuantizedWeight:
    """Run a bounded group64 CPU comparison variant, rejecting invalid numerics.

    FP64 stays FP64 for algebra audits; other floating inputs use FP32. RTN has
    no alternating loop. Row-only fixes beta_inv=1; row-column initializes
    beta_inv=max(sum_rows(abs(W)), 1e-8), then applies the source-order updates.
    Zero exceptions and the standard gamma final cast are port-specific.
    """
    _floating_cpu(weight, "weight")
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}")
    if group_size != GROUP_SIZE or type(group_size) is not int:
        raise ValueError("this audit is fixed to integer group_size=64")
    if type(num_iter) is not int or num_iter < 0:
        raise ValueError("num_iter must be a nonnegative integer")
    if weight.ndim != 2 or min(weight.shape) <= 0 or weight.shape[1] % GROUP_SIZE:
        raise ValueError("weight must be a nonempty matrix with columns divisible by 64")
    dtype = torch.float64 if weight.dtype == torch.float64 else torch.float32
    W = weight.detach().to(dtype).contiguous().clone().reshape(weight.shape[0], -1, GROUP_SIZE)
    _finite(W, "working weight")
    zero_rows = (W == 0).all(dim=-1)
    zero_columns = (W == 0).all(dim=0)
    alpha_inv = torch.ones(W.shape[:2], dtype=dtype)
    beta_inv = torch.ones(W.shape[1:], dtype=dtype)
    records: list[dict[str, Any]] = []
    if method == "row_column":
        beta_stat = W.abs().sum(dim=0)
        _finite(beta_stat, "column L1 initialization")
        beta_inv = torch.where(zero_columns, torch.ones_like(beta_stat), beta_stat.clamp_min(1e-8))
    if method != "rtn":
        normalized = W * beta_inv.reciprocal().unsqueeze(0)
        _finite(normalized, "beta-normalized initialization")
        alpha_stat = normalized.abs().amax(dim=-1) / 7.0
        alpha_inv = torch.where(zero_rows, torch.ones_like(alpha_stat), alpha_stat)
        _positive_scales(alpha_inv, "initial alpha_inv")

    def scaled_codes(a_inv: torch.Tensor, b_inv: torch.Tensor) -> torch.Tensor:
        _positive_scales(a_inv, "alpha_inv")
        _positive_scales(b_inv, "beta_inv")
        scaled = a_inv.reciprocal().unsqueeze(-1) * W * b_inv.reciprocal().unsqueeze(0)
        _finite(scaled, "element-only scaled weight")
        return quantize_element_int4(scaled)

    def estimate(codes: torch.Tensor, a_inv: torch.Tensor, b_inv: torch.Tensor) -> torch.Tensor:
        value = codes.to(dtype) * a_inv.unsqueeze(-1) * b_inv.unsqueeze(0)
        _finite(value, "element reconstruction")
        return value

    for index in range(num_iter if method != "rtn" else 0):
        q_alpha = scaled_codes(alpha_inv, beta_inv)
        before_alpha = estimate(q_alpha, alpha_inv, beta_inv)
        record: dict[str, Any] = {
            "iteration": index + 1,
            "mse_before_alpha": _mse(W, before_alpha, "before alpha"),
            "alpha_objective_before": _fixed_code_objective(W, before_alpha, alpha_inv, "alpha objective before"),
        }
        # Source equation: sum_c(W*M*beta_inv)/(sum_c((M*beta_inv)^2)+EPS).
        m_alpha = q_alpha.to(dtype)
        alpha_numerator = (W * m_alpha * beta_inv.unsqueeze(0)).sum(dim=-1)
        alpha_denominator = (m_alpha * beta_inv.unsqueeze(0)).square().sum(dim=-1) + EPS
        _finite(alpha_numerator, "alpha LS numerator")
        _finite(alpha_denominator, "alpha LS denominator")
        new_alpha = alpha_numerator / alpha_denominator
        alpha_inv = torch.where(zero_rows, torch.ones_like(new_alpha), new_alpha)
        _positive_scales(alpha_inv, "updated alpha_inv")
        after_alpha = estimate(q_alpha, alpha_inv, beta_inv)
        record["mse_after_alpha"] = _mse(W, after_alpha, "after alpha")
        record["alpha_objective_after"] = _fixed_code_objective(W, after_alpha, alpha_inv, "alpha objective after")

        # Fresh quantization after alpha; beta must not use stale q_alpha.
        q_beta = scaled_codes(alpha_inv, beta_inv)
        record["alpha_code_changes"] = int((q_beta != q_alpha).sum().item())
        before_beta = estimate(q_beta, alpha_inv, beta_inv)
        record["mse_before_beta"] = _mse(W, before_beta, "before beta")
        record["beta_objective_before"] = _fixed_code_objective(W, before_beta, beta_inv, "beta objective before")
        if method == "row_column":
            m_beta = q_beta.to(dtype)
            beta_numerator = (W * m_beta * alpha_inv.unsqueeze(-1)).sum(dim=0)
            beta_denominator = (m_beta * alpha_inv.unsqueeze(-1)).square().sum(dim=0) + EPS
            _finite(beta_numerator, "beta LS numerator")
            _finite(beta_denominator, "beta LS denominator")
            new_beta = beta_numerator / beta_denominator
            beta_inv = torch.where(zero_columns, torch.ones_like(new_beta), new_beta)
            _positive_scales(beta_inv, "updated beta_inv")
        after_beta = estimate(q_beta, alpha_inv, beta_inv)
        record["mse_after_beta"] = _mse(W, after_beta, "after beta")
        record["beta_objective_after"] = _fixed_code_objective(W, after_beta, beta_inv, "beta objective after")
        q_after = scaled_codes(alpha_inv, beta_inv)
        record["beta_code_changes"] = int((q_after != q_beta).sum().item())
        record["mse_requantized"] = _mse(W, estimate(q_after, alpha_inv, beta_inv), "after requantization")
        record["nonfinite_repaired"] = False
        records.append(record)

    _positive_scales(alpha_inv, "final alpha_inv")
    _positive_scales(beta_inv, "final beta_inv")
    scaled = alpha_inv.reciprocal().unsqueeze(-1) * W * beta_inv.reciprocal().unsqueeze(0)
    _finite(scaled, "final scaled weight")
    gamma_stat = scaled.abs().amax(dim=-1) / 7.0
    zero_groups = (scaled == 0).all(dim=-1)
    gamma = torch.where(zero_groups, torch.ones_like(gamma_stat), gamma_stat)
    _positive_scales(gamma, "final gamma")
    codes = quantize_element_int4(scaled / gamma.unsqueeze(-1)).reshape(weight.shape)
    result = QuantizedWeight(codes, gamma, alpha_inv, beta_inv.reshape(-1),
                             method, num_iter if method != "rtn" else 0, tuple(records))
    _finite(result.weight_hat(), "final dense reconstruction")
    return result


def compare_methods(weight: torch.Tensor, *, num_iter: int = DEFAULT_NUM_ITER) -> dict[str, QuantizedWeight]:
    """Return RTN, row-only and row-column CPU results without quality claims."""
    return {method: quantize_weight(weight, method, num_iter=num_iter) for method in _METHODS}
