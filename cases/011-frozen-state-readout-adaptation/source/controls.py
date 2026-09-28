"""Predeclared controls; neither control uses TEST gold or future features."""
from __future__ import annotations

import numpy as np

CONTROL_DEFINITION = {
    "input_only": {
        "features": ["current_group_token", "fixed_position_band"],
        "fit": "FIT labels only; add-one class frequencies in token/band cells",
        "bands": [[1, 32], [33, 64], [65, 128], [129, 256]],
        "extrapolation": "all positions >256 reuse the final FIT band",
        "tie_rule": "lowest S3 label ID",
        "classes": 6,
    },
    "state_shuffle": {
        "bucket": "same scored group-token position and current token",
        "rule": "uniform random ordering followed by cyclic shift; bucket size>=2 has no fixed points",
        "singletons": "unchanged and counted; no claim of destroyed state dependence for these rows",
        "seed": 61401,
        "generator": "numpy PCG64 via SeedSequence([seed, position_1based, current_token])",
        "labels_used": False,
    },
}


def position_bands(positions):
    p = np.asarray(positions)
    if p.dtype.kind not in "iu" or np.any(p < 1):
        raise ValueError("positions must be positive integer group-token indices, excluding BOS")
    return np.searchsorted(np.asarray([32, 64, 128]), p, side="left")


def _tokens(value, ndim=None):
    t = np.asarray(value)
    if (ndim is not None and t.ndim != ndim) or t.dtype.kind not in "iu" or np.any(t < 0) or np.any(t > 5):
        raise ValueError("current tokens must be integer S3 IDs 0..5")
    return t


class InputOnlyPredictor:
    """A fixed 4-band by 6-token categorical lookup, not a recurrent model."""

    def __init__(self, counts):
        counts = np.asarray(counts)
        if counts.shape != (4, 6, 6) or counts.dtype.kind not in "iu" or np.any(counts < 0):
            raise ValueError("counts must be nonnegative integers with shape [4,6,6]")
        self.counts = counts.astype(np.int64, copy=True)
        smoothed = self.counts.astype(np.float64) + 1.0
        self.probabilities = smoothed / smoothed.sum(axis=-1, keepdims=True)

    @classmethod
    def fit(cls, tokens, gold, positions=None):
        tokens = _tokens(tokens, 2)
        gold = _tokens(gold, 2)
        if tokens.shape != gold.shape or not tokens.size:
            raise ValueError("FIT tokens/gold must have the same nonempty shape")
        if positions is None:
            positions = np.arange(1, tokens.shape[1] + 1)
        positions = np.asarray(positions)
        if positions.shape != (tokens.shape[1],) or np.any(positions > 256):
            raise ValueError("input-only fit is limited to the predeclared FIT positions 1..256")
        band = np.broadcast_to(position_bands(positions), tokens.shape)
        counts = np.zeros((4, 6, 6), dtype=np.int64)
        np.add.at(counts, (band.ravel(), tokens.ravel(), gold.ravel()), 1)
        return cls(counts)

    def predict(self, tokens, positions=None):
        tokens = _tokens(tokens)
        if positions is None:
            if tokens.ndim != 2:
                raise ValueError("positions are required except for a [sequence,time] token matrix")
            positions = np.arange(1, tokens.shape[1] + 1)
        band = position_bands(positions)
        try:
            probabilities = self.probabilities[band, tokens]
        except (IndexError, ValueError) as exc:
            raise ValueError("positions do not broadcast to current tokens") from exc
        return probabilities.argmax(axis=-1).astype(np.int8)

    def to_dict(self):
        return {"definition": CONTROL_DEFINITION["input_only"],
                "counts": self.counts.tolist(), "fit_rows": int(self.counts.sum()),
                "probabilities": self.probabilities.tolist()}


def shuffle_permutation(tokens, position: int, rngseed: int = 61401):
    """Return source indices for phi[permutation], plus auditable bucket counts.

    The order of base sequence IDs is frozen by the input manifest. The mapping
    is deterministic for that order and never reads gold, predictions, or phi.
    """
    current = _tokens(tokens, 1)
    if not len(current) or isinstance(position, bool) or int(position) != position or position < 1:
        raise ValueError("shuffle requires nonempty tokens and a 1-based integer position")
    if int(rngseed) != rngseed or rngseed < 0:
        raise ValueError("rngseed must be a nonnegative integer")
    permutation = np.arange(len(current), dtype=np.int64)
    sizes = []
    for token in range(6):
        bucket = np.flatnonzero(current == token)
        sizes.append(int(len(bucket)))
        if len(bucket) < 2:
            continue
        rng = np.random.default_rng(np.random.SeedSequence([int(rngseed), int(position), token]))
        order = rng.permutation(bucket)
        permutation[order] = np.roll(order, 1)
    fixed = int(np.count_nonzero(permutation == np.arange(len(current))))
    singleton = int(sum(size == 1 for size in sizes))
    if fixed != singleton or not np.array_equal(current, current[permutation]):
        raise AssertionError("shuffle construction violated its bucket/derangement contract")
    return permutation, {"position": int(position), "rng_seed": int(rngseed),
                         "n_sequences": len(current), "bucket_sizes": sizes,
                         "singleton_unchanged_rows": singleton,
                         "shuffled_rows": len(current) - singleton,
                         "fixed_point_rows": fixed}
