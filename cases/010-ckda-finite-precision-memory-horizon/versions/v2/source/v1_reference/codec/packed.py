"""Actual byte-backed batched state, with separately serialized configuration.

``PackedCodec((H, K, V), bits=4).encode(x)`` accepts ``[B,H,K,V]`` and
returns a PackedBatch whose only field is a C-contiguous uint8 payload
``[B, bytes_per_stream]``. Decode returns float32 workspace, never persistent
state. A residual ``[B,H,R,V]`` uses exactly the same API with K=R.

Each stream contains H little-endian float32 max-absolute scales, optionally
one little-endian uint64 seed and next counter, then contiguous LSB-first
codes in C order. For bit width b, Q=2**(b-1)-1, q is in [-Q,Q], and code
is q+Q+1. Code zero is reserved and rejected. The reconstruction step is
scale[head]/Q[key]. Unused high bits of the last byte must be zero. There
is no alignment padding. Zero scales require the canonical q=0 codes.

Dynamic scales use per-head max(abs(x)), rounded to float32. Fixed max_abs
scales explicitly clip before quantization. Deterministic rounding is
ties-to-even; stochastic rounding uses SplitMix64(SplitMix64(seed) + counter + index)
so adjacent seeds do not create one-element-shifted copies of the same stream,
and stores the counter for the NEXT encode, advancing by H*K*V modulo 2**64.
Restart with ``seeds, counters = codec.rng_state(state)``. Configuration is
canonical JSON bytes, separate from stream payloads and counted once in
``ledger``. Python object/runtime workspace overhead is outside this ledger.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import operator
from typing import Any

import numpy as np


def _shape3(shape: Any) -> tuple[int, int, int]:
    try:
        values = tuple(operator.index(n) for n in shape)
    except (TypeError, ValueError) as exc:
        raise ValueError("shape must contain three positive integers") from exc
    if len(values) != 3 or any(n <= 0 for n in values):
        raise ValueError("shape must contain three positive integers")
    return values


def _bit_width(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("bit widths must be integers from 2 through 16")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError("bit widths must be integers from 2 through 16") from exc
    if not 2 <= result <= 16:
        raise ValueError("bit widths must be integers from 2 through 16")
    return result


def _canonical_config(config: dict[str, Any]) -> bytes:
    return json.dumps(config, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _read_config(data: bytes) -> dict[str, Any]:
    if not isinstance(data, bytes):
        raise ValueError("configuration must be bytes")
    try:
        config = json.loads(data)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError("invalid configuration JSON") from exc
    if not isinstance(config, dict):
        raise ValueError("configuration must be a JSON object")
    return config


@dataclass(frozen=True, slots=True)
class PackedBatch:
    """Persistent stream state: exactly one uint8 matrix, including all scales/RNG."""

    payload: np.ndarray

    def __post_init__(self) -> None:
        if (not isinstance(self.payload, np.ndarray) or
                self.payload.dtype != np.uint8 or self.payload.ndim != 2 or
                not self.payload.flags.c_contiguous):
            raise ValueError("payload must be a C-contiguous uint8 matrix")

    @property
    def nbytes(self) -> int:
        return self.payload.nbytes

    @property
    def batch_size(self) -> int:
        return self.payload.shape[0]

    def to_bytes(self, stream: int = 0) -> bytes:
        """Serialize one stream. Use payload.tobytes() for an entire batch."""
        if not 0 <= stream < self.batch_size:
            raise IndexError("stream index out of range")
        return self.payload[stream].tobytes()


class _PayloadCodec:
    def _check_payload(self, state: PackedBatch) -> np.ndarray:
        if not isinstance(state, PackedBatch):
            raise ValueError("expected PackedBatch")
        payload = state.payload
        if (payload.dtype != np.uint8 or payload.ndim != 2 or
                not payload.flags.c_contiguous or
                payload.shape[1] != self.bytes_per_stream):
            raise ValueError("invalid payload dtype, layout, or stream byte length")
        return payload

    def from_payload(self, payload: np.ndarray, *, copy: bool = True) -> PackedBatch:
        """Import [B,bytes_per_stream], validating contents; copy by default."""
        if not isinstance(payload, np.ndarray):
            raise ValueError("payload must be a numpy array")
        state = PackedBatch(payload.copy(order="C") if copy else payload)
        self.decode(state)
        return state

    def from_bytes(self, data: bytes, *, batch_size: int = 1) -> PackedBatch:
        """Import one stream or explicit batch count; reject truncation/trailing bytes."""
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise ValueError("data must be a bytes-like object")
        try:
            count = operator.index(batch_size)
        except TypeError as exc:
            raise ValueError("batch_size must be a nonnegative integer") from exc
        if count < 0 or len(data) != count * self.bytes_per_stream:
            raise ValueError("serialized payload length does not match batch_size")
        payload = np.frombuffer(data, dtype=np.uint8).copy().reshape(count, self.bytes_per_stream)
        return self.from_payload(payload, copy=False)

    def ledger(self, state: PackedBatch) -> dict[str, int]:
        """Exact serialized byte ledger; shared config is counted once per codec."""
        payload = self._check_payload(state)
        count = payload.shape[0]
        return {
            "batch_size": count,
            "bytes_per_stream": self.bytes_per_stream,
            "data_bytes_per_stream": self.data_bytes,
            "scale_bytes_per_stream": self.scale_bytes,
            "rng_bytes_per_stream": self.rng_bytes,
            "payload_array_nbytes": payload.nbytes,
            "serialized_payload_bytes": len(payload.tobytes()),
            "shared_config_bytes": len(self.config_bytes),
            "total_serialized_bytes": payload.nbytes + len(self.config_bytes),
        }

    def _input(self, x: Any) -> np.ndarray:
        result = np.asarray(x)
        if result.ndim != 4 or result.shape[1:] != self.shape:
            raise ValueError(f"expected [batch,{','.join(map(str, self.shape))}]")
        if result.dtype.kind not in "fiu":
            raise ValueError("state must contain real numeric values")
        if not np.all(np.isfinite(result)):
            raise ValueError("state must contain only finite values")
        return result


class PackedCodec(_PayloadCodec):
    """Symmetric 2..16-bit storage with uniform or per-key-channel precision.

    mixed_bits is an integer [K] or [H,K] array. A bool mask of either shape
    selects 8 bits for True and ``bits`` for False. max_abs is None (dynamic),
    a finite positive scalar, or [H] fixed clipping bounds. No statistics or
    calibration samples are retained by the codec.
    """

    def __init__(self, shape: Any, bits: int = 4, mixed_bits: Any = None,
                 stochastic: bool = False, max_abs: Any = None):
        self.shape = _shape3(shape)
        self.bits = _bit_width(bits)
        if not isinstance(stochastic, (bool, np.bool_)):
            raise ValueError("stochastic must be boolean")
        self.stochastic = bool(stochastic)
        heads, keys, values = self.shape
        if mixed_bits is None:
            bit_map = np.full((heads, keys), self.bits, dtype=np.uint8)
            mixed_config = None
        else:
            raw = np.asarray(mixed_bits)
            if raw.shape == (keys,):
                raw = np.broadcast_to(raw, (heads, keys))
            if raw.shape != (heads, keys):
                raise ValueError("mixed_bits must have shape [key] or [head,key]")
            if raw.dtype.kind == "b":
                raw = np.where(raw, 8, self.bits)
            if raw.dtype.kind not in "iu" or np.any((raw < 2) | (raw > 16)):
                raise ValueError("mixed bit widths must be integers from 2 through 16")
            bit_map = raw.astype(np.uint8)
            mixed_config = bit_map.tolist()
        self._bit_map = bit_map
        self._flat_bits = np.repeat(bit_map.reshape(-1), values).astype(np.uint16)
        self._qmax = ((1 << (bit_map.astype(np.int32) - 1)) - 1)[None, :, :, None]
        self.num_values = int(np.prod(self.shape))
        offsets = np.cumsum(np.r_[0, self._flat_bits[:-1]], dtype=np.int64)
        self._byte_index = offsets // 8
        self._shift = (offsets % 8).astype(np.uint16)
        self._code_mask = ((np.uint32(1) << self._flat_bits.astype(np.uint32)) - 1)
        self._groups = []
        for shift in range(8):
            indices = np.flatnonzero(self._shift == shift)
            if indices.size:
                cross = indices[self._flat_bits[indices] + shift > 8]
                cross16 = indices[self._flat_bits[indices] + shift > 16]
                self._groups.append((shift, indices, cross, cross16))
        self.total_bits = int(self._flat_bits.sum())
        self.data_bytes = (self.total_bits + 7) // 8
        self.scale_bytes = heads * 4
        self.rng_bytes = 16 if self.stochastic else 0
        self._data_start = self.scale_bytes + self.rng_bytes
        self.bytes_per_stream = self._data_start + self.data_bytes
        self._fixed_scales = None
        if max_abs is not None:
            bounds = np.asarray(max_abs, dtype=np.float64)
            if bounds.ndim == 0:
                bounds = np.full(heads, bounds.item(), dtype=np.float64)
            if (bounds.shape != (heads,) or not np.all(np.isfinite(bounds)) or
                    np.any(bounds <= 0) or np.any(bounds > np.finfo(np.float32).max)):
                raise ValueError("max_abs must be finite positive scalar or [head] FP32 bounds")
            self._fixed_scales = bounds.astype(np.float32)
            if np.any(self._fixed_scales == 0):
                raise ValueError("max_abs underflows FP32")
        self.config_bytes = _canonical_config({
            "format": "ckda-packed-v1", "shape": list(self.shape), "bits": self.bits,
            "mixed_bits": mixed_config, "stochastic": self.stochastic,
            "max_abs": None if self._fixed_scales is None else self._fixed_scales.tolist(),
            "scale_policy": "per_head_max_abs", "scale_dtype": "<f4",
            "decoded_dtype": "<f4", "payload_dtype": "uint8",
            "code_policy": "offset_signed_zero_reserved", "bit_order": "lsb_first",
            "rng": "splitmix64_seedkey_v2_next_counter_le_u64" if self.stochastic else None,
            "rounding": "stochastic_floor_bernoulli" if self.stochastic else "ties_to_even",
        })

    @classmethod
    def from_config(cls, data: bytes) -> PackedCodec:
        config = _read_config(data)
        try:
            codec = cls(config["shape"], bits=config["bits"],
                        mixed_bits=config["mixed_bits"], stochastic=config["stochastic"],
                        max_abs=config["max_abs"])
        except (KeyError, TypeError, OverflowError) as exc:
            raise ValueError("invalid packed codec configuration") from exc
        if codec.config_bytes != data:
            raise ValueError("unsupported, altered, or noncanonical codec configuration")
        return codec

    @staticmethod
    def _u64_vector(value: Any, batch: int, name: str) -> np.ndarray:
        if value is None:
            return np.zeros(batch, dtype=np.uint64)
        raw = np.asarray(value)
        if raw.ndim == 0:
            raw = np.broadcast_to(raw, (batch,))
        if raw.shape != (batch,) or raw.dtype.kind not in "iu" or np.any(raw < 0):
            raise ValueError(f"{name} must be nonnegative uint64-compatible scalar or [batch]")
        return raw.astype(np.uint64)

    @staticmethod
    def _splitmix64(values: np.ndarray) -> np.ndarray:
        with np.errstate(over="ignore"):
            z = values + np.uint64(0x9E3779B97F4A7C15)
            z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
            z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
            z = z ^ (z >> np.uint64(31))
        return z

    def _uniforms(self, seeds: np.ndarray, counters: np.ndarray) -> np.ndarray:
        keys = self._splitmix64(seeds)
        with np.errstate(over="ignore"):
            positions = keys[:, None] + counters[:, None] + np.arange(self.num_values, dtype=np.uint64)
        z = self._splitmix64(positions)
        return (z >> np.uint64(11)).astype(np.float64) * (1.0 / (1 << 53))

    def encode(self, x: Any, seeds: Any = None, counters: Any = None) -> PackedBatch:
        source = self._input(x)
        batch = source.shape[0]
        source = source.astype(np.float64, copy=False)
        if self._fixed_scales is None:
            maxima = np.max(np.abs(source), axis=(2, 3))
            if np.any(maxima > np.finfo(np.float32).max):
                raise ValueError("per-head scale exceeds finite FP32 range")
            scales = maxima.astype(np.float32)
            if np.any((scales == 0) & (maxima != 0)):
                raise ValueError("nonzero per-head scale underflows FP32")
        else:
            scales = np.broadcast_to(self._fixed_scales, (batch, self.shape[0]))
        # Clip before division to avoid overflow for a finite input and tiny bound.
        # Divide by scales first: step=scale/Q could underflow for FP32 subnormals.
        clipped = np.clip(source, -scales[:, :, None, None], scales[:, :, None, None])
        normalized = np.divide(clipped, scales[:, :, None, None],
                               out=np.zeros_like(source), where=scales[:, :, None, None] != 0)
        normalized = np.clip(normalized, -1.0, 1.0) * self._qmax
        seed_values = counter_values = None
        if self.stochastic:
            seed_values = self._u64_vector(seeds, batch, "seeds")
            counter_values = self._u64_vector(counters, batch, "counters")
            lower = np.floor(normalized)
            uniforms = self._uniforms(seed_values, counter_values).reshape(source.shape)
            quantized = lower + (uniforms < normalized - lower)
        else:
            if seeds is not None or counters is not None:
                raise ValueError("RNG arguments require stochastic=True")
            quantized = np.rint(normalized)
        codes = (quantized.astype(np.int32) + self._qmax + 1).reshape(batch, self.num_values).astype(np.uint32)
        payload = np.zeros((batch, self.bytes_per_stream), dtype=np.uint8)
        payload[:, :self.scale_bytes] = np.ascontiguousarray(scales, dtype="<f4").view(np.uint8).reshape(batch, self.scale_bytes)
        if self.stochastic:
            with np.errstate(over="ignore"):
                next_counters = counter_values + np.uint64(self.num_values)
            rng = np.column_stack((seed_values, next_counters)).astype("<u8")
            payload[:, self.scale_bytes:self._data_start] = rng.view(np.uint8).reshape(batch, 16)
        data = payload[:, self._data_start:]
        for shift, indices, cross, cross16 in self._groups:
            data[:, self._byte_index[indices]] |= ((codes[:, indices] << shift) & 255).astype(np.uint8)
            if cross.size:
                data[:, self._byte_index[cross] + 1] |= (codes[:, cross] >> (8 - shift)).astype(np.uint8)
            if cross16.size:
                data[:, self._byte_index[cross16] + 2] |= (codes[:, cross16] >> (16 - shift)).astype(np.uint8)
        return PackedBatch(payload)

    def _parts(self, state: PackedBatch) -> tuple[np.ndarray, np.ndarray]:
        payload = self._check_payload(state)
        batch = payload.shape[0]
        scales = payload[:, :self.scale_bytes].copy().view("<f4").reshape(batch, self.shape[0])
        if not np.all(np.isfinite(scales)) or np.any(scales < 0):
            raise ValueError("payload has nonfinite or negative scales")
        if self._fixed_scales is not None and not np.all(scales == self._fixed_scales):
            raise ValueError("payload scales do not match fixed-scale configuration")
        data = payload[:, self._data_start:]
        remainder = self.total_bits % 8
        if remainder and np.any(data[:, -1] >> remainder):
            raise ValueError("nonzero final-byte padding")
        codes = data[:, self._byte_index].astype(np.uint32) >> self._shift
        crossing = self._shift + self._flat_bits > 8
        if np.any(crossing):
            codes[:, crossing] |= data[:, self._byte_index[crossing] + 1].astype(np.uint32) << (8 - self._shift[crossing])
        crossing16 = self._shift + self._flat_bits > 16
        if np.any(crossing16):
            codes[:, crossing16] |= data[:, self._byte_index[crossing16] + 2].astype(np.uint32) << (16 - self._shift[crossing16])
        codes &= self._code_mask
        if np.any(codes == 0):
            raise ValueError("reserved all-zero quantized code")
        quantized = codes.astype(np.int32).reshape((batch,) + self.shape) - self._qmax - 1
        if np.any((scales[:, :, None, None] == 0) & (quantized != 0)):
            raise ValueError("zero scale requires canonical zero codes")
        return scales, quantized

    def decode(self, state: PackedBatch) -> np.ndarray:
        scales, quantized = self._parts(state)
        # Float64 intermediates avoid spurious underflow; returned workspace is FP32.
        result = quantized.astype(np.float64) / self._qmax * scales[:, :, None, None]
        return result.astype(np.float32)

    def rng_state(self, state: PackedBatch) -> tuple[np.ndarray, np.ndarray]:
        """Return copies of stored seed and NEXT counter, one uint64 per stream."""
        payload = self._check_payload(state)
        if not self.stochastic:
            raise ValueError("codec has no stochastic RNG state")
        rng = payload[:, self.scale_bytes:self._data_start].copy().view("<u8").reshape(-1, 2)
        return rng[:, 0].copy(), rng[:, 1].copy()


class NativeFloatCodec(_PayloadCodec):
    """Lossless finite FP32/FP64 controls using the identical payload-only API."""

    def __init__(self, shape: Any, dtype: Any = "float32"):
        self.shape = _shape3(shape)
        parsed = np.dtype(dtype)
        if parsed.kind != "f" or parsed.itemsize not in (4, 8):
            raise ValueError("native dtype must be float32 or float64")
        self.dtype = np.dtype("<f4" if parsed.itemsize == 4 else "<f8")
        self.data_bytes = int(np.prod(self.shape)) * self.dtype.itemsize
        self.scale_bytes = self.rng_bytes = 0
        self.bytes_per_stream = self.data_bytes
        self.config_bytes = _canonical_config({
            "format": "ckda-native-v1", "shape": list(self.shape),
            "dtype": self.dtype.str, "payload_dtype": "uint8", "order": "C",
        })

    @classmethod
    def from_config(cls, data: bytes) -> NativeFloatCodec:
        config = _read_config(data)
        try:
            codec = cls(config["shape"], config["dtype"])
        except (KeyError, TypeError) as exc:
            raise ValueError("invalid native codec configuration") from exc
        if codec.config_bytes != data:
            raise ValueError("unsupported, altered, or noncanonical codec configuration")
        return codec

    def encode(self, x: Any, seeds: Any = None, counters: Any = None) -> PackedBatch:
        if seeds is not None or counters is not None:
            raise ValueError("native codec has no RNG state")
        source = self._input(x)
        with np.errstate(over="ignore"):
            data = np.array(source, dtype=self.dtype, order="C", copy=True)
        if not np.all(np.isfinite(data)):
            raise ValueError("state exceeds native dtype finite range")
        payload = data.view(np.uint8).reshape(source.shape[0], self.bytes_per_stream).copy()
        return PackedBatch(payload)

    def decode(self, state: PackedBatch) -> np.ndarray:
        payload = self._check_payload(state)
        data = payload.copy().view(self.dtype).reshape((payload.shape[0],) + self.shape)
        if not np.all(np.isfinite(data)):
            raise ValueError("payload has nonfinite native values")
        return data
