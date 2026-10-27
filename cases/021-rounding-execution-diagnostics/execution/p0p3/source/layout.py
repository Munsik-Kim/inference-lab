"""Independent NumPy RPI4 reader, encoding checks and reference arithmetic.

The RPI4 byte schema is reused from Case013 reference_int4.py. This module
does not quantize weights, import that module, or execute a GPU kernel.
"""
import struct
from dataclasses import dataclass
from pathlib import Path
import numpy as np

HEADER = struct.Struct('<4sBBBBIIII')

@dataclass
class Artifact:
    codes: np.ndarray
    gamma: np.ndarray
    alpha_inv: np.ndarray
    beta_inv: np.ndarray
    method: str
    iterations: int

    @property
    def shape(self):
        return self.codes.shape

def unpack_signed(data, shape):
    raw = np.frombuffer(data, dtype=np.uint8)
    if raw.size * 2 != int(np.prod(shape)):
        raise ValueError('NIBBLE_LENGTH')
    n = np.empty(raw.size * 2, dtype=np.int16)
    n[::2], n[1::2] = raw & 15, raw >> 4
    n[n >= 8] -= 16
    return n.astype(np.int8).reshape(shape)

def pack_signed(codes):
    c = np.asarray(codes)
    if c.dtype.kind not in 'iu' or c.size % 2 or not c.size or np.any((c < -8) | (c > 7)):
        raise ValueError('SIGNED_CODE_CONTRACT')
    n = (c.reshape(-1).astype(np.int16) & 15).astype(np.uint8)
    return (n[::2] | (n[1::2] << 4)).tobytes()

def load_rpi4(path):
    b = Path(path).read_bytes()
    if len(b) < HEADER.size:
        raise ValueError('TRUNCATED_RPI4')
    magic, version, dtype_code, method, reserved, rows, cols, group, iterations = HEADER.unpack_from(b)
    if magic != b'RPI4' or version != 1 or reserved or dtype_code not in (1, 2) or method not in (0, 1, 2):
        raise ValueError('RPI4_HEADER')
    if rows < 1 or cols < 1 or cols % 64 or group != 64:
        raise ValueError('RPI4_SHAPE_GROUP64')
    dtype = np.dtype('<f4' if dtype_code == 1 else '<f8')
    code_bytes, groups = rows * cols // 2, rows * (cols // 64)
    expected = 24 + code_bytes + dtype.itemsize * (groups * (1 if method == 0 else 2) + (cols if method == 2 else 0))
    if len(b) != expected:
        raise ValueError('RPI4_BYTE_COUNT')
    codes = unpack_signed(b[24:24 + code_bytes], (rows, cols))
    offset = 24 + code_bytes
    def take(count, shape):
        nonlocal offset
        x = np.frombuffer(b, dtype=dtype, count=count, offset=offset).copy().reshape(shape)
        offset += count * dtype.itemsize
        if not np.isfinite(x).all() or not (x > 0).all():
            raise ValueError('RPI4_INVALID_SCALE')
        return x
    gamma = take(groups, (rows, cols // 64))
    alpha = take(groups, gamma.shape) if method else np.ones_like(gamma)
    beta = take(cols, (cols,)) if method == 2 else np.ones(cols, dtype=dtype)
    return Artifact(codes, gamma, alpha, beta, ['rtn', 'row_only', 'row_column'][method], iterations)

def bf16_bits(x):
    x = np.asarray(x, dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError('NONFINITE_BF16_INPUT')
    u = x.view(np.uint32)
    return ((u + np.uint32(0x7fff) + ((u >> 16) & 1)) >> 16).astype(np.uint16)

def bf16_float(bits):
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)

def bf16(x):
    return bf16_float(bf16_bits(x))

def reconstruct(q, path):
    blocks = q.codes.astype(q.gamma.dtype).reshape(q.shape[0], -1, 64)
    if path == 'D0':
        pre = (blocks * q.gamma[..., None]) * q.alpha_inv[..., None]
    elif path == 'D1':
        scales = bf16(q.alpha_inv.astype(np.float32) * q.gamma.astype(np.float32))
        pre = q.codes.astype(np.float32).reshape(q.shape[0], -1, 64) * scales[..., None]
    else:
        raise ValueError('DENSE_PATH')
    return bf16(pre.reshape(q.shape))

def pack_marlin_input(codes):
    c = np.asarray(codes)
    if c.ndim != 2 or c.shape[1] % 8 or c.dtype.kind not in 'iu' or np.any((c < -8) | (c > 7)):
        raise ValueError('MARLIN_CODES')
    transposed = (c.astype(np.int32) + 8).T.astype(np.uint32)
    packed = np.zeros((c.shape[1] // 8, c.shape[0]), dtype=np.uint32)
    for nibble in range(8):
        packed |= transposed[nibble::8] << (4 * nibble)
    return packed.view(np.int32)

def marlin_scale_permute(scales_GO):
    s = np.asarray(scales_GO)
    if s.ndim != 2 or s.shape[0] < 2 or s.size % 64:
        raise ValueError('GROUPED_SCALE_ALIGNMENT')
    # Independent coordinate definition: each contiguous8x8 tile is transposed.
    return s.reshape(-1, 8, 8).transpose(0, 2, 1).reshape(s.shape).copy()

def metrics(reference, actual):
    a, b = np.asarray(reference, dtype=np.float64), np.asarray(actual, dtype=np.float64)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('METRIC_SHAPE_OR_FINITE')
    d = b - a
    sse, energy = float(np.sum(d * d)), float(np.sum(a * a))
    return {'SSE': sse, 'reference_energy': energy, 'relative_RMSE': None if energy == 0 else float(np.sqrt(sse / energy)),
            'energy_status': 'UNDEFINED' if energy == 0 else 'DEFINED', 'MAE_sum': float(np.abs(d).sum()),
            'elements': a.size, 'max_abs': float(np.max(np.abs(d), initial=0)), 'finite': True}
