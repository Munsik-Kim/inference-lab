"""Pinned CKDA loading and explicit physical-coordinate recurrent state access."""

from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
import subprocess
import sys

import torch
from torch.nn import functional as F

UPSTREAM_COMMIT = "ef9d108d1692387cae37f5b2d539a71826a127c1"
ADAPTER_VERSION = "physical-ckda-v2-finite-metrics"
ATOL = 2e-5
RTOL = 2e-4
REQUIRED_SOURCES = (
    "group_word_problems/train_wordproblem.py",
    "fla/layers/complex_kda_layer.py",
    "fla/layers/utils.py",
    "fla/ops/kda/naive.py",
    "fla/models/utils.py",
)


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_upstream(path):
    """Import only an explicit clean checkout at the declared revision."""
    root = Path(path).resolve(strict=True)
    head = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != UPSTREAM_COMMIT:
        raise ValueError(f"Unexpected upstream revision: {head}")
    hashes = {}
    for relative in REQUIRED_SOURCES:
        actual = (root / relative).read_bytes()
        expected = subprocess.check_output(
            ["git", "-C", str(root), "show", f"{UPSTREAM_COMMIT}:{relative}"]
        )
        if actual != expected:
            raise ValueError(f"Modified required upstream source: {relative}")
        hashes[relative] = hashlib.sha256(actual).hexdigest()
    for name in ("fla", "group_word_problems"):
        existing = sys.modules.get(name)
        if existing is not None and getattr(existing, "__file__", None):
            if not Path(existing.__file__).resolve().is_relative_to(root):
                raise RuntimeError(f"{name} was already imported from another checkout")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root))
    upstream = importlib.import_module("group_word_problems.train_wordproblem")
    if not Path(upstream.__file__).resolve().is_relative_to(root):
        raise RuntimeError("Resolved training module is outside the pinned checkout")
    upstream.case010_provenance = {
        "upstream_path": str(root), "upstream_commit": head,
        "required_source_sha256": hashes,
    }
    return upstream


def create_model(upstream, backend="naive_recurrent", device="cpu", checkpoint=None):
    """Construct the original S3 model; checkpoints are evaluation-only payloads."""
    if backend not in ("naive_recurrent", "kernel"):
        raise ValueError("Backend must be chosen explicitly: naive_recurrent or kernel")
    model = upstream.Model(
        6, 48, 12, 16, "signed_sigmoid2", True, backend,
        "spread", "standard", vocab_in=7, readout="mlp", drop_silu=True,
    ).to(device=device, dtype=torch.float32)
    model.case010_provenance = dict(upstream.case010_provenance)
    if checkpoint is not None:
        payload = torch.load(checkpoint, map_location=device, weights_only=True)
        if payload["provenance"]["upstream_commit"] != UPSTREAM_COMMIT:
            raise ValueError("Checkpoint upstream revision mismatch")
        if payload["provenance"]["required_source_sha256"] != upstream.case010_provenance["required_source_sha256"]:
            raise ValueError("Checkpoint source hashes mismatch")
        model.load_state_dict(payload["model_state_dict"], strict=True)
    return model


def coefficients(model, tokens):
    """Return q/k/v/alpha/beta and original output gate g/embedding e, [B,T,...]."""
    if tokens.ndim != 2 or tokens.shape[1] == 0:
        raise ValueError("tokens must be nonempty [batch, time]")
    layer = model.layer
    if layer.use_short_conv or layer.num_heads != layer.num_v_heads:
        raise ValueError("Adapter requires the declared no-convolution, equal-head model")
    if next(model.parameters()).dtype != torch.float32:
        raise ValueError("Model coefficients and weights must remain FP32")
    gate_module = importlib.import_module("fla.layers.complex_kda_layer")
    e = model.emb(tokens)
    batch, time = tokens.shape
    shape = (batch, time, layer.num_heads, layer.head_k_dim)
    value_shape = (batch, time, layer.num_v_heads, layer.head_v_dim)
    q = F.normalize(layer.act(layer.q_proj(e)).reshape(shape).float(), dim=-1, eps=1e-6)
    k = F.normalize(layer.k_act(layer.k_proj(e)).reshape(shape).float(), dim=-1, eps=1e-6)
    v = layer.act(layer.v_proj(e)).reshape(value_shape).float()
    sign, logmag = gate_module.compute_gate(
        layer.gate, layer.f_proj(e).reshape(shape), layer.A_log,
        layer.dt_bias, layer.lower_bound,
    )
    alpha = logmag.exp() if sign is None else sign.float() * logmag.exp()
    beta = 2.0 * layer.b_proj(e).float().sigmoid()
    g = layer.g_proj(e).reshape(value_shape)
    return dict(q=q, k=k, v=v, alpha=alpha, beta=beta, g=g, e=e)


def select_step(coeff, position):
    return {name: value[:, position] for name, value in coeff.items()}


def transition(state, coeff_t):
    """Apply the full physical CKDA affine transition to the supplied state."""
    decayed = state * coeff_t["alpha"].unsqueeze(-1)
    key = coeff_t["k"]
    innovation = coeff_t["v"] - (key.unsqueeze(-1) * decayed).sum(-2)
    return decayed + (coeff_t["beta"].unsqueeze(-1) * key).unsqueeze(-1) * innovation.unsqueeze(-2)


def read_logits(model, state, coeff_t):
    """Read this supplied updated state through the original norm/projection/MLP."""
    layer = model.layer
    out = ((coeff_t["q"] * layer.head_k_dim**-0.5).unsqueeze(-1) * state).sum(-2)
    out = out.to(coeff_t["e"].dtype)
    out = layer.o_norm(out.unsqueeze(1), coeff_t["g"].unsqueeze(1)).squeeze(1)
    out = layer.o_proj(out.flatten(-2))
    return model.mlp(model.norm(coeff_t["e"] + out))


def physical_forward(model, tokens, initial_state=None):
    """Codec-disabled physical recurrence, with no native candidate or gauge cache."""
    coeff = coefficients(model, tokens)
    expected = (tokens.shape[0], model.layer.num_heads, model.layer.head_k_dim, model.layer.head_v_dim)
    state = coeff["v"].new_zeros(expected) if initial_state is None else initial_state
    if tuple(state.shape) != expected:
        raise ValueError(f"State shape {tuple(state.shape)} does not match {expected}")
    outputs = []
    for position in range(tokens.shape[1]):
        coeff_t = select_step(coeff, position)
        state = transition(state, coeff_t)
        outputs.append(read_logits(model, state, coeff_t))
    return torch.stack(outputs, dim=1), state


def sequence_metrics(logits, labels, bos=True):
    """Token and all-prefix metrics; BOS is reported separately and never in L."""
    if logits.shape[:2] != labels.shape:
        raise ValueError("Logit and label sequence dimensions differ")
    predictions = logits.argmax(-1)
    finite = torch.isfinite(logits).all(-1)
    correct = predictions.eq(labels) & finite
    bos_accuracy = float(correct[:, 0].float().mean()) if bos else None
    correct = correct[:, 1:] if bos else correct
    if correct.shape[1] == 0:
        raise ValueError("At least one scored group token is required")
    survives = correct.to(torch.int64).cumprod(1).bool()
    length = correct.shape[1]
    first_fail = (~correct).to(torch.int64).argmax(1) + 1
    first_fail = torch.where(correct.all(1), length + 1, first_fail)
    quarter = max(length // 4, 1)
    return {
        "gold_length": length, "batch_size": correct.shape[0],
        "bos_accuracy": bos_accuracy,
        "bos_predictions": predictions[:, 0].tolist() if bos else None,
        "bos_targets": labels[:, 0].tolist() if bos else None,
        "invalid_logit_token_count": int((~finite).sum()),
        "invalid_scored_logit_token_count": int((~(finite[:, 1:] if bos else finite)).sum()),
        "invalid_bos_logit_count": int((~finite[:, 0]).sum()) if bos else None,
        "token_accuracy": float(correct.float().mean()),
        "final_quarter_accuracy": float(correct[:, -quarter:].float().mean()),
        "all_prefix_survival": float(survives[:, -1].float().mean()),
        "survival_by_length": survives.float().mean(0).tolist(),
        "first_failure_position": [None if v == length + 1 else v for v in first_fail.tolist()],
        "first_failure_censor_length": length,
    }


def implementation_provenance():
    path = Path(__file__).resolve()
    return {"adapter_version": ADAPTER_VERSION, "adapter_sha256": file_sha256(path),
            "parity_atol": ATOL, "parity_rtol": RTOL}
