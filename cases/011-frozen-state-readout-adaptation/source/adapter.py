"""Case011 feature boundary over the unchanged physical FP32 Case010 recurrence.

A rollout has one persistent byte cache. Readout heads see the same temporary
GELU features and have no route back into transition coefficients or cache.
Gold, position and sequence identity are deliberately absent from this API.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np
import torch

from .references import load_references, load_storage

FEATURE_BOUNDARY = "case011-original-modules-post-gelu-pre-mlp2-fp32-v1"
HEAD_TENSORS = ("mlp.2.weight", "mlp.2.bias")
SHAPE = (12, 16, 16)


def parameter_manifest(model):
    return {name: {"shape": list(value.shape), "dtype": str(value.dtype),
                   "sha256": hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()}
            for name, value in model.named_parameters()}


def freeze_model(model):
    """Validate the predeclared architecture, then disable all autograd paths."""
    if (not isinstance(model.mlp, torch.nn.Sequential) or len(model.mlp) != 3 or
        not isinstance(model.mlp[0], torch.nn.Linear) or
        not isinstance(model.mlp[1], torch.nn.GELU) or
        not isinstance(model.mlp[2], torch.nn.Linear) or
        tuple(model.mlp[0].weight.shape) != (192, 48) or
        tuple(model.mlp[2].weight.shape) != (6, 192) or
        model.mlp[2].bias is None or tuple(model.mlp[2].bias.shape) != (6,)):
        raise ValueError("Expected the frozen 48 -> 192 -> 6 MLP with final weight and bias")
    if tuple((model.layer.num_heads, model.layer.head_k_dim, model.layer.head_v_dim)) != SHAPE:
        raise ValueError("Unexpected CKDA recurrent shape")
    if any(p.device.type != "cpu" or p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("All original model parameters must stay CPU FP32")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
        p.grad = None
    return parameter_manifest(model)


def assert_frozen(model, expected):
    if any(p.requires_grad or p.grad is not None for p in model.parameters()):
        raise ValueError("Frozen model acquired a gradient or a trainable parameter")
    actual = parameter_manifest(model)
    if actual != expected:
        changed = sorted(set(actual) ^ set(expected) | {name for name in actual.keys() & expected.keys()
                                                       if actual[name] != expected[name]})
        raise ValueError("Frozen model parameters changed: " + ", ".join(changed))


@torch.inference_mode()
def read_features(model, state, coefficients):
    """Exact original module order; split only the final Sequential Linear."""
    if not isinstance(state, np.ndarray) or state.dtype != np.float32 or state.ndim != 4:
        raise ValueError("Feature read requires the original FP32 represented state")
    layer = model.layer
    q, gate, embedding = (torch.from_numpy(coefficients[name]) for name in ("q", "g", "e"))
    out = torch.einsum("b h k, b h k v -> b h v", q * layer.head_k_dim**-0.5,
                       torch.from_numpy(state))
    out = layer.o_norm(out.unsqueeze(1), gate.unsqueeze(1)).squeeze(1)
    out = layer.o_proj(out.flatten(-2))
    hidden = model.norm(embedding + out)
    phi = model.mlp[1](model.mlp[0](hidden))
    logits = model.mlp[2](phi)
    if tuple(phi.shape) != (len(state), 192) or phi.dtype != torch.float32:
        raise ValueError("Unexpected feature boundary shape or dtype")
    return phi, logits


@dataclass(frozen=True)
class StepOutput:
    phi: torch.Tensor
    logits_original: torch.Tensor
    predictions_original: np.ndarray
    status: dict


class FrozenRollout:
    def __init__(self, model, table, storage, *, model_seed=0,
                 case010_root=None, batch_size=None):
        self.reference = load_references(case010_root)
        self.model = model
        self.parameter_identity = freeze_model(model)
        if set(table) != {"q", "k", "v", "alpha", "beta", "g", "e"}:
            raise ValueError("Expected the seven frozen FP32 coefficient tables")
        self.table = {}
        expected = {"q": (7, 12, 16), "k": (7, 12, 16), "v": (7, 12, 16),
                    "alpha": (7, 12, 16), "beta": (7, 12),
                    "g": (7, 12, 16), "e": (7, 48)}
        for name, value in table.items():
            array = np.asarray(value)
            if array.dtype != np.float32 or array.shape != expected[name] or not np.isfinite(array).all():
                raise ValueError("Invalid fixed coefficient table: " + name)
            array = array.copy()
            array.setflags(write=False)
            self.table[name] = array
        self.table_sha256 = hashlib.sha256(self.reference.v1.table_bytes(self.table)[1]).hexdigest()
        self.storage = storage
        self.model_seed = model_seed
        self.adapter = load_storage(storage, model_seed, case010_root)
        self.state = None
        if batch_size is not None:
            self.initialize(batch_size)

    def initialize(self, batch_size):
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        self.state = self.adapter.initial(np.zeros((batch_size,) + SHAPE, dtype=np.float32))
        return self

    def export_bytes(self, stream=None):
        if self.state is None:
            raise ValueError("Initialize the rollout before exporting")
        return self.state.payload.tobytes() if stream is None else self.state.to_bytes(stream)

    def import_bytes(self, raw, batch_size):
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.state = self.adapter.from_bytes(raw, batch_size=batch_size)
        return self

    def assert_frozen(self):
        assert_frozen(self.model, self.parameter_identity)

    @torch.inference_mode()
    def step(self, token_ids):
        ids = np.asarray(token_ids)
        if ids.ndim != 1 or ids.dtype.kind not in "iu" or np.any(ids < 0) or np.any(ids > 6):
            raise ValueError("Expected one integer group/BOS token per stream")
        if self.state is None:
            self.initialize(len(ids))
        if len(ids) != self.state.batch_size:
            raise ValueError("Changing batch shape on resume is forbidden")
        coeff = self.reference.v1.gather(self.table, ids)
        before = self.adapter.terminal_info(self.state)
        self.state, represented, _ = self.adapter.step(
            self.state, lambda value: self.reference.v1.torch_transition(value, coeff), diagnostics=False)
        status = self.adapter.terminal_info(self.state)
        if np.any(status["active"]):
            phi, logits = read_features(self.model, represented, coeff)
        else:
            phi = torch.zeros((len(ids), 192), dtype=torch.float32)
            logits = torch.zeros((len(ids), 6), dtype=torch.float32)
        finite = torch.isfinite(logits).all(-1).numpy()
        predictions = logits.argmax(-1).numpy().astype(np.int16)
        predictions[~status["active"] | ~finite] = -1
        status.update(active_before=before["active"], write_index=before["cursor"],
                      finite_logits=finite, finite_features=torch.isfinite(phi).all(-1).numpy(),
                      valid_readout=status["active"] & finite)
        return StepOutput(phi, logits, predictions, status)
