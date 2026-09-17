"""Bounded token-history diagnostics; matching tokens do not restore caches."""
from __future__ import annotations

import json
from collections.abc import Iterator


def reference_prefixes(prompt: list[int], continuation: list[int]) -> Iterator[tuple[list[int], int]]:
    for i, token in enumerate(continuation):
        yield prompt + continuation[:i], token


def alignment(base: list[int], other: list[int], eos_ids: set[int], limit: int = 64) -> dict:
    def observed(x: list[int]) -> list[int]:
        if len(x) > limit:
            raise ValueError("Token budget exceeded")
        for i, token in enumerate(x):
            if token in eos_ids:
                if i + 1 != len(x):
                    raise ValueError("Tokens after EOS are not observations")
                break
        return x
    a, b = observed(base), observed(other)
    common = min(len(a), len(b))
    differing = [i for i in range(common) if a[i] != b[i]]
    first = differing[0] if differing else common if len(a) != len(b) else None
    match = None
    streak = None
    if first is not None:
        matches = [i for i in range(first+1, common) if a[i] == b[i]]
        match = matches[0] if matches else None
        for i in range(first+1, common-7):
            if a[i:i+8] == b[i:i+8]:
                streak = i
                break
    return {"first_difference_zero_based": first, "first_aligned_match_after_difference": match,
            "eight_token_realignment_start": streak,
            "realignment_status": "NOT_APPLICABLE" if first is None or common-first-1 < 8 else "OBSERVED" if streak is not None else "NOT_OBSERVED_WITHIN_BUDGET",
            "right_censored": (not a or a[-1] not in eos_ids) or (not b or b[-1] not in eos_ids),
            "base_length": len(a), "candidate_length": len(b), "state_recovery_claim": False}


def parse_structured(text: str, gold: str, support: list[str]) -> dict:
    def unique(pairs: list[tuple]) -> dict:
        out = {}
        for key, value in pairs:
            if key in out:
                raise ValueError("Duplicate key")
            out[key] = value
        return out
    try:
        value = json.loads(text, object_pairs_hook=unique)
        valid = (isinstance(value, dict) and set(value) == {"choice", "evidence"}
                 and value["choice"] in ("A", "B", "C", "D")
                 and isinstance(value["evidence"], list)
                 and all(isinstance(x, str) for x in value["evidence"])
                 and len(set(value["evidence"])) == len(value["evidence"]))
    except (ValueError, TypeError):
        value, valid = None, False
    return {"schema_valid": valid, "parsed": value,
            "answer_correct": valid and value["choice"] == gold,
            "evidence_correct": valid and sorted(value["evidence"]) == sorted(support)}
