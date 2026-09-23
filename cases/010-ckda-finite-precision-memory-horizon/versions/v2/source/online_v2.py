"""Failure-aware, byte-only online cache; the unchanged v1 codecs are the body.

Each row is cursor<u8, terminal_code<u1, first_terminal_write<u8, then the
unchanged state/residual codec bytes. Cursor is the next zero-based write.
Failed writes consume a cursor but commit no body or RNG bytes. Terminal rows
remain terminal; their represented output is zero scratch, never a prediction.
Only detected numeric failures become terminal. Callback exceptions, malformed
arrays, corrupt codec bytes, and unsupported shared metadata remain errors.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import hashlib
import json
import operator
from pathlib import Path
import sys

import numpy as np

# Use the same package identity as source.reference, without its eager Torch
# imports. The file check prevents silently using another checkout's codec.
_REFERENCE = Path(__file__).resolve().parent / "v1_reference"
if str(_REFERENCE) not in sys.path:
    sys.path.insert(0, str(_REFERENCE))
from codec.packed import PackedCodec, NativeFloatCodec, PackedBatch
from codec.online import OnlineAdapter as _V1Adapter
if Path(sys.modules[PackedCodec.__module__].__file__).resolve() != (_REFERENCE / "codec/packed.py").resolve():
    raise RuntimeError("online_v2 requires this study's unchanged v1 codec copy")


class TerminalCode(IntEnum):
    ACTIVE = 0
    RECONSTRUCTION_NONFINITE = 1
    TRANSITION_NONFINITE = 2
    STATE_RANGE = 3
    RESIDUAL_NONFINITE = 4
    RESIDUAL_RANGE = 5
    POST_WRITE_NONFINITE = 6


ACTIVE_FIRST_WRITE = np.uint64(np.iinfo(np.uint64).max)
HEADER_BYTES = 17
_CODES = {item.name: int(item) for item in TerminalCode}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


@dataclass(frozen=True, slots=True)
class OnlineState:
    payload: np.ndarray

    def __post_init__(self):
        if (not isinstance(self.payload, np.ndarray) or self.payload.dtype != np.uint8 or
                self.payload.ndim != 2 or not self.payload.flags.c_contiguous):
            raise ValueError("online state must own a C-contiguous uint8 matrix")

    @property
    def nbytes(self):
        return self.payload.nbytes

    @property
    def batch_size(self):
        return self.payload.shape[0]

    def to_bytes(self, stream=0):
        if isinstance(stream, (bool, np.bool_)):
            raise IndexError("invalid stream index")
        index = operator.index(stream)
        if not 0 <= index < self.batch_size:
            raise IndexError("stream index out of range")
        return self.payload[index].tobytes()


def _numeric_array(value, shape, label):
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind not in "fiu":
        raise ValueError(f"{label} must have shape {shape} and real numeric dtype")
    return array


def _nonfinite_rows(values):
    return ~np.isfinite(values).all(axis=(1, 2, 3))


def _range_rows(codec, values):
    """Input is finite; only a declared representation-range failure is caught."""
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        absolute = np.abs(values.astype(np.float64))
        if isinstance(codec, PackedCodec):
            if codec._fixed_scales is not None:
                return np.zeros(len(values), dtype=bool)  # Explicit fixed-bound clipping.
            maxima = absolute.max(axis=(2, 3))
            scales = maxima.astype(np.float32)
            # The reference encoder rejects values above FP32 max even when a
            # nearby FP64 value rounds DOWN to a finite FP32 scale.
            return ((maxima > np.finfo(np.float32).max) | (~np.isfinite(scales)) |
                    ((scales == 0) & (maxima != 0))).any(axis=1)
        return (absolute > np.finfo(codec.dtype).max).any(axis=(1, 2, 3))


class OnlineAdapter(_V1Adapter):
    def __init__(self, state_codec, residual_codec=None, basis=None, mode="transported", *, bos_policy="present_write0"):
        if not isinstance(state_codec, (PackedCodec, NativeFloatCodec)) or (
                residual_codec is not None and not isinstance(residual_codec, (PackedCodec, NativeFloatCodec))):
            raise ValueError("only the unchanged reference packed/native codecs are supported")
        if bos_policy not in ("present_write0", "absent"):
            raise ValueError("unknown BOS policy")
        super().__init__(state_codec, residual_codec, basis, mode)
        self.bos_policy = bos_policy
        self.bytes_per_stream += 9
        if self.basis is not None:
            self.basis.setflags(write=False)
        cfg = json.loads(self.config_bytes)
        cfg.update(format="ckda-online-failure-aware-v2", bos_policy=bos_policy,
            terminal_code="u1", first_terminal_write="<u8", header_bytes=HEADER_BYTES,
            active_first_terminal_write=int(ACTIVE_FIRST_WRITE), terminal_codes=_CODES,
            body_layout="unchanged_v1_state_then_residual", cursor_semantics="next_zero_based_write",
            terminal_commit="retain_last_body_and_rng_consume_cursor", basis_sha256=hashlib.sha256(self.basis_bytes).hexdigest())
        self.config_bytes = _canonical(cfg)
        self.shared_bytes = len(self.config_bytes) + len(self.basis_bytes)
        state_dtype = np.float32 if isinstance(state_codec, PackedCodec) else state_codec.dtype
        residual_dtype = state_dtype if residual_codec is None or isinstance(residual_codec, PackedCodec) else residual_codec.dtype
        self._state_dtype = np.dtype(state_dtype)
        self._output_dtype = np.result_type(state_dtype, residual_dtype)

    def _parts(self, state):
        if not isinstance(state, OnlineState):
            raise ValueError("expected a v2 OnlineState; legacy states are not migrated")
        p = state.payload
        if p.dtype != np.uint8 or p.ndim != 2 or not p.flags.c_contiguous or p.shape[1] != self.bytes_per_stream:
            raise ValueError("invalid v2 payload dtype, layout, or row length")
        cursor = p[:, :8].copy().view("<u8").reshape(-1)
        code = p[:, 8].copy()
        first = p[:, 9:17].copy().view("<u8").reshape(-1)
        if np.any(code > max(_CODES.values())):
            raise ValueError("unknown terminal code")
        active = code == TerminalCode.ACTIVE
        if np.any(first[active] != ACTIVE_FIRST_WRITE) or np.any(first[~active] >= cursor[~active]):
            raise ValueError("terminal flag/sentinel/cursor mismatch")
        end = HEADER_BYTES + self.state_codec.bytes_per_stream
        z = PackedBatch(p[:, HEADER_BYTES:end].copy(order="C"))
        r = None if self.residual_codec is None else PackedBatch(p[:, end:].copy(order="C"))
        return cursor, code, first, z, r

    @staticmethod
    def _headers(payload, cursor, code, first):
        payload[:, :8] = np.asarray(cursor, dtype="<u8").view(np.uint8).reshape(-1, 8)
        payload[:, 8] = code
        payload[:, 9:17] = np.asarray(first, dtype="<u8").view(np.uint8).reshape(-1, 8)

    def terminal_info(self, state):
        cursor, code, first, _, _ = self._parts(state)
        return dict(cursor=cursor, terminal_code=code, first_terminal_write=first, active=code == TerminalCode.ACTIVE)

    def initial(self, x, seeds=None):
        source = np.asarray(x)
        if source.ndim != 4:
            raise ValueError("initial state must have four axes")
        source = _numeric_array(source, (len(source),) + self.shape, "initial state")
        if np.any(_nonfinite_rows(source)) or np.any(_range_rows(self.state_codec, source)):
            raise ValueError("initial state must be finite and representable")
        z = self._encode(self.state_codec, source, None, seeds)
        r = None
        if self.residual_codec is not None:
            with np.errstate(over="ignore", invalid="ignore", under="ignore"):
                projected = self.project(source - self.state_codec.decode(z))
            if np.any(_nonfinite_rows(projected)) or np.any(_range_rows(self.residual_codec, projected)):
                raise ValueError("initial residual must be finite and representable")
            r = self._encode(self.residual_codec, projected, None, seeds)
        body = z.payload if r is None else np.concatenate((z.payload, r.payload), axis=1)
        payload = np.empty((len(source), self.bytes_per_stream), dtype=np.uint8)
        payload[:, HEADER_BYTES:] = body
        self._headers(payload, np.zeros(len(source), dtype="<u8"), np.zeros(len(source), dtype=np.uint8),
                      np.full(len(source), ACTIVE_FIRST_WRITE, dtype="<u8"))
        state = OnlineState(payload)
        self.represented(state)  # Initial states must have a finite representation.
        return state

    def represented(self, state):
        _, code, _, z, r = self._parts(state)
        active = code == TerminalCode.ACTIVE
        result = np.zeros((len(code),) + self.shape, dtype=self._output_dtype)
        if not np.any(active):
            return result
        decoded = self.state_codec.decode(z)
        decoded[~active] = 0
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            if r is not None:
                residual = self.residual_codec.decode(r)
                residual[~active] = 0
                decoded = decoded + self.expand(residual)
        if np.any(active & _nonfinite_rows(decoded)):
            raise ValueError("nonfinite ACTIVE reconstruction; step records its numerical failure")
        result[active] = decoded[active]
        return result

    @staticmethod
    def _encode_rows(codec, values, old, rows):
        # Row-independent codec work may use a subset; transition shape never does.
        selected = values if len(rows) == len(values) else values[rows]
        prior = old if len(rows) == len(values) else PackedBatch(old.payload[rows].copy(order="C"))
        return _V1Adapter._encode(codec, selected, prior)

    def step(self, state, transition_callable, *, diagnostics=True):
        if not isinstance(diagnostics, (bool, np.bool_)):
            raise ValueError("diagnostics must be boolean")
        cursor, code, first, z, r = self._parts(state)
        if np.any(cursor == ACTIVE_FIRST_WRITE):
            raise ValueError("cursor overflow")
        live = code == TerminalCode.ACTIVE
        active_at_start = int(live.sum())
        output = state.payload.copy()
        represented = np.zeros((len(code),) + self.shape, dtype=self._output_dtype)
        if not np.any(live):
            # Absorbing cursor-only no-op, including a throwing callback.
            self._headers(output, cursor + np.uint64(1), code, first)
            diagnostic = self._diagnostics(None, None, None, None, 0, 0, len(code)) if diagnostics else {}
            return OnlineState(output), represented, diagnostic

        def fail(mask, reason):
            failed = live & mask
            code[failed] = int(reason)
            first[failed] = cursor[failed]
            live[failed] = False

        decoded = self.state_codec.decode(z)
        decoded[~live] = 0
        residual = None
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            if r is not None:
                residual_coefficients = self.residual_codec.decode(r)
                residual_coefficients[~live] = 0
                residual = self.expand(residual_coefficients)
                fail(_nonfinite_rows(residual), TerminalCode.RECONSTRUCTION_NONFINITE)
            working = decoded if residual is None or self.mode == "untransported" else decoded + residual
            fail(_nonfinite_rows(working), TerminalCode.RECONSTRUCTION_NONFINITE)
            working[~live] = 0
            # Callback errors are programming/execution errors, never terminal codes.
            updated = transition_callable(working)
        updated = _numeric_array(updated, (len(code),) + self.shape, "transition output").copy()
        updated[~live] = 0
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            if residual is not None and self.mode == "untransported":
                residual[~live] = 0
                updated = updated + residual
        fail(_nonfinite_rows(updated), TerminalCode.TRANSITION_NONFINITE)
        updated[~live] = 0
        fail(_range_rows(self.state_codec, updated), TerminalCode.STATE_RANGE)
        updated[~live] = 0
        z_rows = np.flatnonzero(live)
        next_z = None
        next_decoded = np.zeros((len(code),) + self.shape, dtype=self._state_dtype)
        if z_rows.size:
            next_z = self._encode_rows(self.state_codec, updated, z, z_rows)
            next_decoded[z_rows] = self.state_codec.decode(next_z)
        error, projected, next_r, r_rows = None, None, None, np.empty(0, dtype=np.int64)
        if r is not None:
            with np.errstate(over="ignore", invalid="ignore", under="ignore"):
                error = updated - next_decoded
                projected = self.project(error)
            fail(_nonfinite_rows(projected), TerminalCode.RESIDUAL_NONFINITE)
            projected[~live] = 0
            fail(_range_rows(self.residual_codec, projected), TerminalCode.RESIDUAL_RANGE)
            projected[~live] = 0
            r_rows = np.flatnonzero(live)
            next_coefficients = np.zeros((len(code),) + self.residual_codec.shape,
                                         dtype=np.float32 if isinstance(self.residual_codec, PackedCodec) else self.residual_codec.dtype)
            if r_rows.size:
                next_r = self._encode_rows(self.residual_codec, projected, r, r_rows)
                next_coefficients[r_rows] = self.residual_codec.decode(next_r)
            with np.errstate(over="ignore", invalid="ignore", under="ignore"):
                represented = next_decoded + self.expand(next_coefficients)
        else:
            represented = next_decoded
        fail(_nonfinite_rows(represented), TerminalCode.POST_WRITE_NONFINITE)
        represented[~live] = 0
        end = HEADER_BYTES + self.state_codec.bytes_per_stream
        if next_z is not None:
            commit = live[z_rows]
            output[z_rows[commit], HEADER_BYTES:end] = next_z.payload[commit]
        if next_r is not None:
            commit = live[r_rows]
            output[r_rows[commit], end:] = next_r.payload[commit]
        self._headers(output, cursor + np.uint64(1), code, first)
        diagnostic = self._diagnostics(updated[live], represented[live],
            None if error is None else error[live], None if projected is None else projected[live],
            active_at_start, int(live.sum()), len(code)) if diagnostics else {}
        return OnlineState(output), represented, diagnostic

    def _diagnostics(self, updated, represented, error, projected, active_start, committed, batch):
        """Stable FP64 diagnostics; an unrepresentable diagnostic is JSON null."""
        result = dict(active_at_start=active_start, committed_rows=committed,
            newly_terminal_rows=active_start-committed, terminal_rows=batch-committed,
            writeback_mse=None, projection_leakage_mse=None, state_norm_mean=None,
            diagnostic_overflow_fields=[])
        if not committed:
            return result
        def mse(values):
            scale = np.max(np.abs(values))
            if scale == 0:
                return 0.0
            return float(scale * (scale * np.mean((values / scale) ** 2)))
        with np.errstate(over="ignore", invalid="ignore", under="ignore", divide="ignore"):
            update64, output64 = updated.astype(np.float64), represented.astype(np.float64)
            result["writeback_mse"] = mse(update64-output64)
            if self.basis is None or error is None:
                result["projection_leakage_mse"] = 0.0
            else:
                result["projection_leakage_mse"] = mse(error.astype(np.float64)-self.expand(projected.astype(np.float64)))
            scale = np.max(np.abs(output64))
            result["state_norm_mean"] = 0.0 if scale == 0 else float(scale * np.sqrt(
                np.sum((output64.reshape(committed, -1) / scale) ** 2, axis=1)).mean())
        for key in ("writeback_mse", "projection_leakage_mse", "state_norm_mean"):
            if not np.isfinite(result[key]):
                result[key] = None
                result["diagnostic_overflow_fields"].append(key)
        return result

    def from_payload(self, payload, *, copy=True):
        if not isinstance(payload, np.ndarray) or payload.dtype != np.uint8 or payload.ndim != 2:
            raise ValueError("expected uint8 payload matrix")
        state = OnlineState(payload.copy(order="C") if copy else payload)
        _, _, _, z, r = self._parts(state)
        # Validate even terminal bodies as bytes; composing finite components is
        # a numerical transition boundary, not a malformed-serialization check.
        self.state_codec.decode(z)
        if r is not None:
            self.residual_codec.decode(r)
        return state

    def from_bytes(self, raw, batch_size=1):
        if not isinstance(raw, (bytes, bytearray, memoryview)) or isinstance(batch_size, (bool, np.bool_)):
            raise ValueError("invalid byte payload or batch size")
        try:
            count = operator.index(batch_size)
        except TypeError as exc:
            raise ValueError("batch size must be a nonnegative integer") from exc
        if count < 0 or len(raw) != count*self.bytes_per_stream:
            raise ValueError("truncated or trailing v2 state bytes")
        payload = np.frombuffer(raw, dtype=np.uint8).copy().reshape(count, self.bytes_per_stream)
        return self.from_payload(payload, copy=False)

    @classmethod
    def from_shared(cls, config, basis_bytes=b""):
        if not isinstance(config, bytes) or not isinstance(basis_bytes, bytes):
            raise ValueError("shared metadata and basis must be bytes")
        try:
            cfg = json.loads(config)
            if not isinstance(cfg, dict) or cfg.get("format") != "ckda-online-failure-aware-v2":
                raise ValueError("unknown v2 schema; legacy state migration is forbidden")
            if hashlib.sha256(basis_bytes).hexdigest() != cfg["basis_sha256"]:
                raise ValueError("basis hash mismatch")
            def codec(value):
                if not isinstance(value, dict):
                    raise ValueError("nested codec metadata must be an object")
                raw = _canonical(value)
                if value.get("format") == "ckda-packed-v1":
                    return PackedCodec.from_config(raw)
                if value.get("format") == "ckda-native-v1":
                    return NativeFloatCodec.from_config(raw)
                raise ValueError("unknown codec schema")
            shape = cfg["basis_shape"]
            if shape is None:
                if basis_bytes:
                    raise ValueError("unexpected basis bytes")
                basis = None
            else:
                if not isinstance(shape, list) or len(shape) != 3 or any(type(n) is not int or n <= 0 for n in shape):
                    raise ValueError("invalid basis shape")
                if len(basis_bytes) != int(np.prod(shape))*4:
                    raise ValueError("basis length mismatch")
                basis = np.frombuffer(basis_bytes, dtype="<f4").reshape(shape)
            instance = cls(codec(cfg["state"]), None if cfg["residual"] is None else codec(cfg["residual"]),
                           basis, cfg["mode"], bos_policy=cfg["bos_policy"])
        except (KeyError, TypeError, UnicodeDecodeError, OverflowError) as exc:
            raise ValueError("invalid v2 shared metadata") from exc
        if instance.config_bytes != config:
            raise ValueError("unsupported, inconsistent, or noncanonical v2 metadata")
        return instance

    def ledger(self, state=None):
        if state is not None:
            self._parts(state)
        codecs = [self.state_codec] + ([] if self.residual_codec is None else [self.residual_codec])
        payload_bytes = None if state is None else state.payload.nbytes
        return dict(per_stream_persistent_bytes=self.bytes_per_stream, state_code_bytes=self.state_codec.data_bytes,
            residual_code_bytes=0 if self.residual_codec is None else self.residual_codec.data_bytes,
            scale_bytes=sum(c.scale_bytes for c in codecs), rng_bytes=sum(c.rng_bytes for c in codecs),
            cursor_bytes=8, terminal_code_bytes=1, first_terminal_write_bytes=8, header_bytes=HEADER_BYTES,
            gauge_bytes=0, padding_bytes=0, shared_bytes=self.shared_bytes, shared_config_bytes=len(self.config_bytes),
            shared_basis_bytes=len(self.basis_bytes), total_bytes={str(n):self.shared_bytes+n*self.bytes_per_stream for n in (1,16,128,512,1024)},
            payload_tensor_storage_bytes=payload_bytes, serialized_payload_bytes=payload_bytes)
