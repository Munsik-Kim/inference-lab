"""Train only the frozen Case010 schedule; final checkpoint selection uses no TEST."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.nn import functional as F

from codec.learned import (
    create_model, file_sha256, implementation_provenance, load_upstream, sequence_metrics,
)


def validate_training(protocol):
    cfg = dict(protocol["training"])
    for key in ("backend", "max_updates", "batch_size", "walltime_seconds", "dev_batch_size", "dev_seed", "log_every"):
        if key not in cfg:
            raise ValueError(f"Missing protocol training.{key}")
    for key in ("max_updates", "batch_size", "dev_batch_size", "log_every"):
        if not isinstance(cfg[key], int) or cfg[key] < 1:
            raise ValueError(f"training.{key} must be a positive integer")
    if cfg["max_updates"] == 10:
        raise ValueError("The original OneCycleLR pct_start=.1 is undefined for exactly ten updates")
    if not isinstance(cfg["walltime_seconds"], (int, float)) or cfg["walltime_seconds"] <= 0:
        raise ValueError("walltime_seconds must be positive")
    if cfg["backend"] not in ("naive_recurrent", "kernel"):
        raise ValueError("Protocol must choose an explicit backend")
    cfg.setdefault("curriculum", [4, 6, 8, 16, 32])
    if cfg["curriculum"] != [4, 6, 8, 16, 32]:
        raise ValueError("The declared curriculum is fixed at 4,6,8,16,32")
    defaults = dict(lr=.005, muon_lr=.02, weight_decay=1e-12, muon_momentum=.95, muon_ns_steps=5)
    for key, value in defaults.items():
        cfg.setdefault(key, value)
    return cfg


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args(argv)
    protocol = json.loads(args.protocol.read_text())
    cfg = validate_training(protocol)
    train_seed = args.seed + 1234
    if train_seed == cfg["dev_seed"]:
        raise ValueError("TRAIN and DEV generator seeds must differ")
    args.output.mkdir(parents=True, exist_ok=False)
    upstream = load_upstream(args.upstream)
    torch.manual_seed(args.seed)
    model = create_model(upstream, backend=cfg["backend"], device=args.device)
    optimizer_cfg = dict(cfg, optimizer="muon", muon_scope="hidden")
    optimizer, max_lr = upstream.build_optimizer(model, optimizer_cfg)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=max_lr, total_steps=cfg["max_updates"], pct_start=.1,
    )
    _, table, identity = upstream.perm_group("s3")
    rng = np.random.default_rng(train_seed)
    provenance = dict(model.case010_provenance, **implementation_provenance())
    provenance.update(protocol_sha256=file_sha256(args.protocol),
                      runner_sha256=file_sha256(__file__))
    manifest = dict(schema_version="case010-train-v1", provenance=provenance,
                    training=cfg, seed=args.seed, train_generator_seed=train_seed,
                    dev_generator_seed=cfg["dev_seed"], device=args.device,
                    parameter_dtype="float32", native_recurrent_dtype="float32",
                    checkpoint_selection="final_scheduled_update_only", test_accessed=False)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    stopped = []

    def on_signal(signum, _frame):
        stopped.append(f"signal_{signum}")

    previous_handlers = {sig: signal.signal(sig, on_signal) for sig in (signal.SIGTERM, signal.SIGINT)}
    start = time.monotonic()
    updates = 0
    reason = None
    model.train()
    try:
        with (args.output / "train_log.jsonl").open("x") as log:
            for index in range(cfg["max_updates"]):
                if stopped or time.monotonic() - start >= cfg["walltime_seconds"]:
                    reason = stopped[0] if stopped else "walltime_limit"
                    break
                length = upstream.curriculum_len(index, cfg["max_updates"], cfg["curriculum"])
                x, y = upstream.make_batch(table, identity, cfg["batch_size"], length, rng, args.device, bos_id=6)
                logits = model(x)
                loss = F.cross_entropy(logits[:, 1:].reshape(-1, 6), y[:, 1:].reshape(-1))
                if not torch.isfinite(loss):
                    reason = "nonfinite_training_loss"
                    break
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if not torch.isfinite(grad_norm):
                    reason = "nonfinite_gradient"
                    break
                optimizer.step()
                scheduler.step()
                updates = index + 1
                if updates == 1 or updates % cfg["log_every"] == 0 or updates == cfg["max_updates"]:
                    row = dict(update=updates, group_length=length, loss=float(loss.detach()),
                               elapsed_seconds=time.monotonic() - start,
                               lr=[group["lr"] for group in optimizer.param_groups])
                    log.write(json.dumps(row) + "\n")
                    log.flush()
                    print(json.dumps(row), flush=True)
    except Exception as exc:
        reason = f"exception:{type(exc).__name__}:{exc}"
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
    training_seconds = time.monotonic() - start
    complete = updates == cfg["max_updates"] and reason is None
    status = "TRAINING_COMPLETE" if complete else "NOT_COMPLETE"
    checkpoint = args.output / ("final.pt" if complete else "interrupted_state.pt")
    state_dict = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    torch.save(dict(model_state_dict=state_dict, provenance=provenance, training=cfg,
                    seed=args.seed, updates=updates, status=status, optimizer_resume_supported=False), checkpoint)
    dev = None
    dev_error = None
    try:
        model.eval()
        with torch.inference_mode():
            x, y = upstream.make_batch(table, identity, cfg["dev_batch_size"], 32,
                                       np.random.default_rng(cfg["dev_seed"]), args.device, bos_id=6)
            dev = sequence_metrics(model(x), y, bos=True)
    except Exception as exc:
        dev_error = f"{type(exc).__name__}:{exc}"
    result = dict(manifest, status=status, termination_reason=reason,
                  completed_updates=updates, scheduled_updates=cfg["max_updates"],
                  training_seconds=training_seconds, checkpoint=checkpoint.name,
                  checkpoint_sha256=file_sha256(checkpoint), dev32=dev, dev_error=dev_error)
    result["eligible_for_conditional_seeds"] = bool(
        complete and dev is not None and dev["all_prefix_survival"] >= .90
        and dev["final_quarter_accuracy"] >= .95
    )
    result_path = args.output / "result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    (args.output / "result.sha256").write_text(file_sha256(result_path) + "\n")
    print(json.dumps({"status": status, "completed_updates": updates,
                      "result": str(result_path), "dev32": dev}), flush=True)
    return 0 if complete and dev_error is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
