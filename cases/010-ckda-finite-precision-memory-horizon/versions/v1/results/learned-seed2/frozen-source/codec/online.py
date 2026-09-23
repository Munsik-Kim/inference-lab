"""Causal write-back adapter; the only per-stream cache is a byte matrix.

Temporary decoded states leave step() only as its read-after-write output.
The caller must not feed that output back as a hidden native-state cache.
Fixed bases are serialized shared metadata, never supplied by an evaluator.
"""
from dataclasses import dataclass
import json
import numpy as np
from .packed import PackedCodec, NativeFloatCodec, PackedBatch


@dataclass(frozen=True, slots=True)
class OnlineState:
    payload: np.ndarray

    def to_bytes(self, stream=0):
        return self.payload[stream].tobytes()


class NonfiniteStateError(ValueError):
    """Row-local failed state write; the incoming byte payload is unchanged.

    bad_rows is a boolean mask for the current batch, not integer indices.
    stage identifies reconstruction, transition, encoding, or read-after-write.
    An unrepresentable finite scale is also a failed encoding, never clipped.
    Callers may record failures and retry healthy rows from the original bytes.
    """

    def __init__(self, bad_rows, stage, reason="nonfinite state"):
        self.bad_rows = np.asarray(bad_rows, dtype=bool).copy()
        self.stage = stage
        self.reason = reason
        super().__init__(f"{reason} at {stage}; failed rows {np.flatnonzero(self.bad_rows).tolist()}")


def _finite_rows(values, stage):
    bad = ~np.isfinite(values).all(axis=tuple(range(1, values.ndim)))
    if np.any(bad):
        raise NonfiniteStateError(bad, stage)


def _encodable_rows(codec, values, stage):
    """Expose row-local range failures before a codec's batch-wide validation."""
    _finite_rows(values, stage)
    if isinstance(codec, PackedCodec):
        if codec._fixed_scales is not None:
            return  # This codec explicitly declares clipping to its fixed bounds.
        maxima = np.max(np.abs(values.astype(np.float64)), axis=(2, 3))
        with np.errstate(over="ignore", under="ignore"):
            scales = maxima.astype(np.float32)
        bad = ((~np.isfinite(scales)) | ((scales == 0) & (maxima != 0))).any(axis=1)
    else:
        bad = (np.abs(values.astype(np.float64)) > np.finfo(codec.dtype).max).any(axis=(1, 2, 3))
    if np.any(bad):
        raise NonfiniteStateError(bad, stage, reason="state outside codec finite representable range")


class OnlineAdapter:
    def __init__(self, state_codec, residual_codec=None, basis=None, mode="transported"):
        if mode not in ("transported", "untransported"):
            raise ValueError("unknown residual transport mode")
        self.state_codec, self.residual_codec = state_codec, residual_codec
        self.mode = mode
        self.shape = state_codec.shape
        self.basis = None if basis is None else np.asarray(basis, dtype="<f4").copy()
        if self.basis is not None:
            h, k, v = self.shape
            if (self.basis.ndim != 3 or self.basis.shape[:2] != (h, k)
                    or residual_codec is None
                    or residual_codec.shape != (h, self.basis.shape[2], v)
                    or not np.isfinite(self.basis).all()):
                raise ValueError("basis/residual shape or finite contract")
            gram = np.einsum("hkr,hks->hrs", self.basis, self.basis)
            if not np.allclose(gram, np.eye(self.basis.shape[2]), atol=2e-5, rtol=2e-5):
                raise ValueError("basis must have orthonormal columns")
        elif residual_codec is not None and residual_codec.shape != self.shape:
            raise ValueError("full residual must match state shape")
        self.bytes_per_stream = 8 + state_codec.bytes_per_stream + (0 if residual_codec is None else residual_codec.bytes_per_stream)
        cfg = dict(format="ckda-online-v1", cursor="<u8", mode=mode,
                   state=json.loads(state_codec.config_bytes),
                   residual=None if residual_codec is None else json.loads(residual_codec.config_bytes),
                   basis_shape=None if self.basis is None else list(self.basis.shape),
                   basis_dtype=None if self.basis is None else "<f4",
                   coordinate_system="physical", read_boundary="after_write",
                   gauge_bytes_per_stream=0)
        self.config_bytes = json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()
        self.basis_bytes = b"" if self.basis is None else self.basis.tobytes()
        self.shared_bytes = len(self.config_bytes) + len(self.basis_bytes)

    def _parts(self, state):
        p = state.payload
        if (p.dtype != np.uint8 or p.ndim != 2 or not p.flags.c_contiguous
                or p.shape[1] != self.bytes_per_stream):
            raise ValueError("invalid online state payload")
        cursor = np.frombuffer(p[:, :8].copy().tobytes(), dtype="<u8")
        end = 8 + self.state_codec.bytes_per_stream
        z = PackedBatch(p[:, 8:end].copy(order="C"))
        r = None if self.residual_codec is None else PackedBatch(p[:, end:].copy(order="C"))
        return cursor, z, r

    def _join(self, cursor, z, r):
        header = np.asarray(cursor, dtype="<u8").view(np.uint8).reshape(-1, 8)
        parts = [header, z.payload] + ([] if r is None else [r.payload])
        return OnlineState(np.concatenate(parts, axis=1))

    def initial(self, x, seeds=None):
        x = np.asarray(x)
        z = self._encode(self.state_codec, x, None, seeds)
        error = x - self.state_codec.decode(z)
        r = None if self.residual_codec is None else self._encode(self.residual_codec, self.project(error), None, seeds)
        return self._join(np.zeros(x.shape[0], dtype="<u8"), z, r)

    @staticmethod
    def _encode(codec, x, old, seeds=None):
        if isinstance(codec, PackedCodec) and codec.stochastic:
            if old is not None:
                seeds, counters = codec.rng_state(old)
            else:
                counters = np.zeros(x.shape[0], dtype=np.uint64)
                if seeds is None:
                    seeds = np.arange(x.shape[0], dtype=np.uint64)
            return codec.encode(x, seeds=seeds, counters=counters)
        return codec.encode(x)

    def project(self, error):
        return error if self.basis is None else np.einsum("hkr,bhkv->bhrv", self.basis, error)

    def expand(self, residual):
        return residual if self.basis is None else np.einsum("hkr,bhrv->bhkv", self.basis, residual)

    def represented(self, state):
        _, z, r = self._parts(state)
        value = self.state_codec.decode(z)
        if r is not None:
            with np.errstate(over="ignore", invalid="ignore"):
                value = value + self.expand(self.residual_codec.decode(r))
        _finite_rows(value, "represented_state")
        return value

    def step(self, state, transition_callable, *, diagnostics=True, nonfinite="raise"):
        """Write once, or expose failed rows without modifying the incoming bytes.

        diagnostics=False changes only diagnostic work and returns an empty dict.
        nonfinite='raise' is the sole policy: this adapter never repairs/clips an
        invalid state. A caller may separately terminate failed stream records.
        """
        if not isinstance(diagnostics, (bool, np.bool_)) or nonfinite != "raise":
            raise ValueError("diagnostics must be boolean and nonfinite policy must be 'raise'")
        cursor, z, r = self._parts(state)
        if np.any(cursor == np.iinfo(np.uint64).max):
            raise ValueError("cursor overflow")
        decoded = self.state_codec.decode(z)
        # Inspect complete row outputs even when the caller requests NumPy errors.
        # This allows healthy rows to be retried; no invalid value is sanitized here.
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            residual = None if r is None else self.expand(self.residual_codec.decode(r))
            if residual is None:
                updated = transition_callable(decoded)
            elif self.mode == "transported":
                reconstructed = decoded + residual
                _finite_rows(reconstructed, "reconstructed_state")
                updated = transition_callable(reconstructed)
            else:
                updated = transition_callable(decoded)
        updated = np.asarray(updated)
        if updated.shape != decoded.shape or updated.dtype.kind not in "fiu":
            raise ValueError("entire transition output must have expected shape and real numeric dtype")
        if residual is not None and self.mode == "untransported":
            with np.errstate(over="ignore", invalid="ignore"):
                updated = updated + residual
        _finite_rows(updated, "transition_output")
        _encodable_rows(self.state_codec, updated, "state_encode")
        next_z = self._encode(self.state_codec, updated, z)
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            error = updated - self.state_codec.decode(next_z)
            projected = self.project(error)
        if r is not None:
            _encodable_rows(self.residual_codec, projected, "residual_encode")
        next_r = None if r is None else self._encode(self.residual_codec, projected, r)
        new_state = self._join(cursor + 1, next_z, next_r)
        represented = self.represented(new_state)
        if not diagnostics:
            return new_state, represented, {}
        # Diagnostics describe this candidate's own rounding, not native error.
        # Promote BEFORE subtraction, squaring, norms, and projector diagnostics.
        working64, represented64 = updated.astype(np.float64), represented.astype(np.float64)
        error64, projected64 = error.astype(np.float64), projected.astype(np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            leakage = error64 if self.basis is None else error64 - self.expand(projected64)
            diagnostic = dict(writeback_mse=float(np.mean((working64-represented64)**2)),
                              projection_leakage_mse=0.0 if self.basis is None else float(np.mean(leakage**2)),
                              state_norm_mean=float(np.linalg.norm(represented64.reshape(len(represented), -1), axis=1).mean()))
        if not all(np.isfinite(value) for value in diagnostic.values()):
            raise ValueError("nonfinite diagnostic in FP64; finite state is not a failed stream; diagnostics=False remains available")
        return new_state, represented, diagnostic

    def from_bytes(self, raw, batch_size=1):
        if len(raw) != batch_size * self.bytes_per_stream:
            raise ValueError("truncated or trailing online state bytes")
        state = OnlineState(np.frombuffer(raw, dtype=np.uint8).copy().reshape(batch_size, self.bytes_per_stream))
        self.represented(state)
        return state

    @classmethod
    def from_shared(cls, config, basis_bytes=b""):
        if not isinstance(config, bytes) or not isinstance(basis_bytes, bytes):
            raise ValueError("shared configuration and basis must be bytes")
        try:
            cfg = json.loads(config)
        except (ValueError, UnicodeDecodeError) as exc:
            raise ValueError("invalid shared configuration JSON") from exc
        if not isinstance(cfg, dict) or cfg.get("format") != "ckda-online-v1":
            raise ValueError("unknown online format")
        def decode(c):
            if not isinstance(c, dict):
                raise ValueError("nested codec configuration must be an object")
            data = json.dumps(c, sort_keys=True, separators=(",", ":")).encode()
            if c.get("format") == "ckda-native-v1":
                return NativeFloatCodec.from_config(data)
            if c.get("format") == "ckda-packed-v1":
                return PackedCodec.from_config(data)
            raise ValueError("unknown nested codec format")
        required = {"state", "residual", "basis_shape", "mode"}
        if not required.issubset(cfg):
            raise ValueError("missing shared configuration fields")
        shape = cfg["basis_shape"]
        if shape is None:
            if basis_bytes:
                raise ValueError("unexpected basis bytes")
            basis = None
        else:
            if (not isinstance(shape, list) or len(shape) != 3 or
                    any(type(n) is not int or n <= 0 for n in shape)):
                raise ValueError("invalid basis shape metadata")
            if len(basis_bytes) != int(np.prod(shape))*4:
                raise ValueError("basis length mismatch")
            basis = np.frombuffer(basis_bytes, dtype="<f4").reshape(shape)
        instance = cls(decode(cfg["state"]), None if cfg["residual"] is None else decode(cfg["residual"]), basis, cfg["mode"])
        if json.loads(instance.config_bytes) != cfg:
            raise ValueError("unsupported or inconsistent shared metadata")
        return instance

    def ledger(self, state=None):
        if state is not None:
            self._parts(state)
            assert state.payload.nbytes == state.payload.shape[0]*self.bytes_per_stream
        codecs = [self.state_codec] + ([] if self.residual_codec is None else [self.residual_codec])
        return dict(per_stream_persistent_bytes=self.bytes_per_stream,
                    state_code_bytes=self.state_codec.data_bytes,
                    residual_code_bytes=0 if self.residual_codec is None else self.residual_codec.data_bytes,
                    scale_bytes=sum(c.scale_bytes for c in codecs), rng_bytes=sum(c.rng_bytes for c in codecs),
                    cursor_bytes=8, gauge_bytes=0, padding_bytes=0,
                    shared_bytes=self.shared_bytes, shared_config_bytes=len(self.config_bytes),
                    shared_basis_bytes=len(self.basis_bytes),
                    total_bytes={str(n):self.shared_bytes+n*self.bytes_per_stream for n in (1,16,128)},
                    payload_tensor_storage_bytes=None if state is None else state.payload.nbytes)


def change_basis(coefficients, old_basis, new_basis):
    """Causal coordinate change, retaining the projection onto the new span."""
    full = np.einsum("hkr,bhrv->bhkv", old_basis, coefficients)
    converted = np.einsum("hks,bhkv->bhsv", new_basis, full)
    restored = np.einsum("hks,bhsv->bhkv", new_basis, converted)
    return converted, float(np.mean((full-restored)**2))
