"""Exact finite-group targets, independent of any learned or floating state.

Permutation tuples map a position j to p[j].  Products are function composition
``(a * b)[j] = a[b[j]]``.  The primary target at step t >= 1 is therefore
``y[t] = x[t] * y[t-1]`` with y[0] equal to the identity.  BOS is optional and is
never implicitly counted as a predicted token.
"""

from dataclasses import dataclass
from itertools import permutations
from numbers import Integral
from typing import Union

import numpy as np


SPLIT_SEEDS = {"CAL": 23092301, "DEV": 23092302, "TEST": 23092303}
_SPLIT_TAGS = {"CAL": 101, "DEV": 202, "TEST": 303}


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return int(value)


def _readonly(array):
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class FiniteGroup:
    name: str
    elements: tuple
    multiplication: np.ndarray
    inverses: np.ndarray
    matrices: np.ndarray
    identity: int = 0

    @property
    def order(self):
        return len(self.elements)

    @property
    def dimension(self):
        return self.matrices.shape[-1]

    def trace(self, tokens, initial=None, include_bos=False):
        """Return integer group IDs, preserving batch axes; time is last.

        ``initial`` is either one group ID or one per batch element.  A supplied
        initial state is a known condition, not an additional scored step.
        """
        tokens = np.asarray(tokens)
        if tokens.ndim < 1 or tokens.dtype.kind not in "iu":
            raise ValueError("tokens must be an integer array with a time axis")
        if np.any(tokens < 0) or np.any(tokens >= self.order):
            raise ValueError("token IDs are outside the group")
        state = np.asarray(self.identity if initial is None else initial)
        if state.dtype.kind not in "iu" or np.any(state < 0) or np.any(state >= self.order):
            raise ValueError("initial state must contain valid integer group IDs")
        try:
            state = np.broadcast_to(state, tokens.shape[:-1]).copy()
        except ValueError as exc:
            raise ValueError("initial state must broadcast to the batch axes") from exc
        offset = int(bool(include_bos))
        targets = np.empty(tokens.shape[:-1] + (tokens.shape[-1] + offset,), dtype=np.int64)
        if include_bos:
            targets[..., 0] = state
        for step in range(tokens.shape[-1]):
            state = self.multiplication[tokens[..., step], state]
            targets[..., step + offset] = state
        return targets


def permutation_group(degree):
    """Build S3 or S4 with exact integer composition and natural matrices."""
    degree = _integer(degree, "degree", 1)
    if degree not in (3, 4):
        raise ValueError("this case prespecifies only S3 and S4")
    elements = tuple(permutations(range(degree)))
    index = {element: i for i, element in enumerate(elements)}
    size = len(elements)
    table = np.empty((size, size), dtype=np.int64)
    matrices = np.zeros((size, degree, degree), dtype=np.int64)
    for a_id, a in enumerate(elements):
        matrices[a_id, np.asarray(a), np.arange(degree)] = 1
        for b_id, b in enumerate(elements):
            table[a_id, b_id] = index[tuple(a[b[j]] for j in range(degree))]
    inverses = np.asarray([np.flatnonzero(table[i] == 0)[0] for i in range(size)])
    return FiniteGroup(f"S{degree}", elements, _readonly(table), _readonly(inverses), _readonly(matrices))


def cyclic_group(order=31):
    """Build C31 (or a caller-specified Cn) as addition modulo n.

    Regular permutation matrices keep the test representation exact over the
    integers; they are not floating approximations to planar rotations.
    """
    order = _integer(order, "order", 2)
    elements = tuple(range(order))
    ids = np.arange(order, dtype=np.int64)
    table = (ids[:, None] + ids[None, :]) % order
    matrices = np.zeros((order, order, order), dtype=np.int64)
    for element in elements:
        matrices[element, (ids + element) % order, ids] = 1
    return FiniteGroup(f"C{order}", elements, _readonly(table), _readonly((-ids) % order), _readonly(matrices))


def make_group(name):
    if name == "S3":
        return permutation_group(3)
    if name == "S4":
        return permutation_group(4)
    if name == "C31":
        return cyclic_group(31)
    raise ValueError("group must be one of S3, S4, C31")


@dataclass(frozen=True)
class SequenceBatch:
    tokens: np.ndarray
    gold: np.ndarray
    sequence_ids: tuple[str, ...]
    split: str
    seed: int
    group_name: str


def frozen_sequences(group: Union[FiniteGroup, str], split, n_sequences, horizon, seed=None):
    """Generate uniform full-group elements with disjoint split seed streams.

    Each sequence has its own PCG64 stream.  Increasing either the batch size or
    horizon preserves every existing prefix.  Even an explicit seed is mixed
    with the split tag, so CAL/DEV/TEST cannot reuse the same RNG stream.
    Learned S3 inputs consequently include all six elements, not only a chosen
    generator set.  Caller-created structured prefixes should use group.trace.
    """
    if isinstance(group, str):
        group = make_group(group)
    if not isinstance(group, FiniteGroup):
        raise TypeError("group must be a FiniteGroup or supported group name")
    if split not in SPLIT_SEEDS:
        raise ValueError("split must be CAL, DEV, or TEST")
    n_sequences = _integer(n_sequences, "n_sequences", 1)
    horizon = _integer(horizon, "horizon", 1)
    seed = SPLIT_SEEDS[split] if seed is None else _integer(seed, "seed")
    tokens = np.empty((n_sequences, horizon), dtype=np.int64)
    for sequence in range(n_sequences):
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, _SPLIT_TAGS[split], sequence])))
        tokens[sequence] = rng.integers(group.order, size=horizon, dtype=np.int64)
    ids = tuple(f"{group.name}:{split}:{seed}:{i:06d}" for i in range(n_sequences))
    return SequenceBatch(_readonly(tokens), _readonly(group.trace(tokens)), ids, split, seed, group.name)
