"""Frozen CPU evaluation of learned CKDA using byte-only online candidate caches."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

CASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CASE_ROOT))

import numpy as np
import torch

from codec.groups import frozen_sequences, make_group
from codec.learned import (ATOL, RTOL, coefficients, create_model, file_sha256,
                           load_upstream, physical_forward, read_logits, transition)
from codec.online import NonfiniteStateError, OnlineAdapter, OnlineState
from codec.packed import NativeFloatCodec, PackedCodec
from codec.survival import first_failure_times, summarize_first_failures

VERSION = "case010-learned-evaluation-v2-canonical-token-torch"
SHAPE = (12, 16, 16)
ARM_NAMES = (['NATIVE_FP32'] + [f'UNIFORM_{b}' for b in range(2, 17)] +
             ['STOCHASTIC_4', 'FULL_RESIDUAL_4_4', 'FULL_RESIDUAL_4_8',
              'FULL_RESIDUAL_4_FP32', 'LOWRANK_4_8_R1', 'LOWRANK_4_8_R2',
              'LOWRANK_4_8_R4', 'UNTRANSPORTED_4_4', 'MIXED_4_8', 'MIXED_6_8'])


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def verify_frozen_json(path):
    path = Path(path)
    checksum = path.with_suffix(".sha256")
    expected = checksum.read_text().split()[0]
    actual = file_sha256(path)
    if expected != actual:
        raise ValueError(f"Frozen file checksum mismatch: {path}")
    return json.loads(path.read_text()), actual


@torch.inference_mode()
def token_table(model):
    """Token-local projections for the seven input IDs, without gold or future data."""
    # One token per projection preserves the official sequential B=1,T=1 route.
    rows = [coefficients(model, torch.tensor([[token]])) for token in range(7)]
    arrays = {name: torch.cat([row[name][0] for row in rows], 0).cpu().numpy().copy()
              for name in rows[0]}
    for value in arrays.values():
        value.setflags(write=False)
    return arrays


def table_bytes(table):
    entries, chunks, offset = {}, [], 0
    for name in sorted(table):
        value = np.asarray(table[name], dtype="<f4", order="C")
        raw = value.tobytes()
        entries[name] = dict(shape=list(value.shape), dtype="<f4", offset=offset, bytes=len(raw))
        chunks.append(raw)
        offset += len(raw)
    config = canonical(dict(format="ckda-token-coefficients-v2-b1-t1", token_ids=list(range(7)), entries=entries))
    return config, b"".join(chunks)


def coefficient_metadata(model, table):
    if any(value.dtype != np.float32 or not np.isfinite(value).all() for value in table.values()):
        raise ValueError("Every token coefficient must be finite FP32")
    alpha, beta = table["alpha"], table["beta"]
    if np.any(np.abs(alpha) > 1) or np.any((beta < 0) | (beta > 2)):
        raise ValueError("Actual gates exceed CKDA's declared parameter ranges")
    result = dict(parameter_count=sum(p.numel() for p in model.parameters()),
                  native_cache_elements=int(np.prod(SHAPE)), native_cache_dtype="float32",
                  native_cache_value_bytes=int(np.prod(SHAPE))*4, heads=12,
                  head_key_dim=16, head_value_dim=16, state_shape_per_stream=list(SHAPE),
                  coefficient_shapes={k:list(v.shape) for k,v in table.items()},
                  coefficient_dtypes={k:str(v.dtype) for k,v in table.items()},
                  alpha_min=float(alpha.min()), alpha_max=float(alpha.max()),
                  alpha_negative_fraction=float((alpha < 0).mean()),
                  beta_min=float(beta.min()), beta_max=float(beta.max()))
    for name in ("q", "k"):
        norms = np.linalg.norm(table[name], axis=-1)
        result[f"{name}_norm_min"] = float(norms.min())
        result[f"{name}_norm_max"] = float(norms.max())
    if result["parameter_count"] != 56530:
        raise ValueError("Unexpected parameter count for the frozen architecture")
    return result


def gather(table, ids):
    return {name: value[ids] for name, value in table.items()}


@torch.inference_mode()
def torch_transition(state, coeff):
    """FP32 physical recurrence using the original adapter's PyTorch operations."""
    if state.dtype != np.float32:
        raise ValueError("Learned recurrence state must be float32")
    tensors = {name: torch.from_numpy(value) for name, value in coeff.items()}
    return transition(torch.from_numpy(state), tensors).numpy()


@torch.inference_mode()
def logits_from_numpy(model, state, coeff):
    # Match the official naive read's einsum reduction, then original modules.
    layer = model.layer
    q, gate, embedding = (torch.from_numpy(coeff[name]) for name in ("q", "g", "e"))
    out = torch.einsum("b h k, b h k v -> b h v", q * layer.head_k_dim**-0.5,
                       torch.from_numpy(state))
    out = layer.o_norm(out.unsqueeze(1), gate.unsqueeze(1)).squeeze(1)
    out = layer.o_proj(out.flatten(-2))
    return model.mlp(model.norm(embedding + out))


def input_identity(batch):
    return dict(split=batch.split, seed=batch.seed, group=batch.group_name,
                sequences=batch.tokens.shape[0], group_length=batch.tokens.shape[1],
                token_sha256=digest(batch.tokens.astype("<i8").tobytes()),
                gold_sha256=digest(batch.gold.astype("<i8").tobytes()),
                sequence_ids=list(batch.sequence_ids), generator="split-tagged per-sequence PCG64")


def full_tokens(tokens):
    return np.column_stack((np.full(len(tokens), 6, dtype=np.int64), tokens))


@torch.inference_mode()
def verify_route_parity(model, table, group_tokens):
    """Compare canonical B1,T1 official steps, including nonzero cache handoff.

    The original whole-sequence GEMM route is a different floating-point
    execution policy. Every table entry here is an exact official one-token
    projection. Physical recurrence and original readout can then be batched.
    """
    from fla.models.utils import Cache

    tokens = torch.from_numpy(full_tokens(group_tokens))
    initial = (torch.arange(len(tokens) * np.prod(SHAPE), dtype=torch.float32)
               .reshape((len(tokens),) + SHAPE).remainder(31) - 15) / 100
    report = []
    for nonzero in (False, True):
        start = initial if nonzero else torch.zeros_like(initial)
        native_rows, physical_rows, native_states, physical_states = [], [], [], []
        for row in range(len(tokens)):
            cache = Cache()
            cache.update(recurrent_state=start[row:row+1].clone(), conv_state=None, layer_idx=0, offset=0)
            physical_state = start[row:row+1].clone()
            native_steps, physical_steps = [], []
            for position in range(tokens.shape[1]):
                one_token = tokens[row:row+1, position:position+1]
                embedding = model.emb(one_token)
                native_out, _, cache = model.layer(embedding, past_key_values=cache, use_cache=True)
                native_steps.append(model.mlp(model.norm(embedding + native_out)))
                physical_out, physical_state = physical_forward(model, one_token, physical_state)
                physical_steps.append(physical_out)
            native_rows.append(torch.cat(native_steps, 1))
            physical_rows.append(torch.cat(physical_steps, 1))
            native_states.append(cache[0]["recurrent_state"])
            physical_states.append(physical_state)
        native, physical = torch.cat(native_rows), torch.cat(physical_rows)
        native_state, physical_state = torch.cat(native_states), torch.cat(physical_states)
        state, outputs = start.numpy().copy(), []
        for position in range(tokens.shape[1]):
            coeff = gather(table, tokens[:, position].numpy())
            state = torch_transition(state, coeff)
            outputs.append(logits_from_numpy(model, state, coeff))
        actual = torch.stack(outputs, 1)
        torch.testing.assert_close(actual, physical, atol=ATOL, rtol=RTOL)
        torch.testing.assert_close(actual, native, atol=ATOL, rtol=RTOL)
        torch.testing.assert_close(torch.from_numpy(state), physical_state, atol=ATOL, rtol=RTOL)
        torch.testing.assert_close(torch.from_numpy(state), native_state, atol=ATOL, rtol=RTOL)
        report.append(dict(nonzero_initial_state=nonzero,
                           maximum_logit_difference=float((actual-native).abs().max()),
                           maximum_state_difference=float((torch.from_numpy(state)-native_state).abs().max()),
                           native_final_state_bitwise_equal=bool(torch.equal(torch.from_numpy(state),native_state)),
                           logit_tolerance_mismatch_count=int(((actual-native).abs() > ATOL+RTOL*native.abs()).sum())))
    return dict(passed=True, atol=ATOL, rtol=RTOL, comparisons=report,
                group_sequences=len(group_tokens),group_horizon=group_tokens.shape[1],
                token_sha256=digest(np.asarray(group_tokens,dtype="<i8").tobytes()),
                official_scope="each sequence replayed through original Cache independently with B=1,T=1; physical_forward likewise B=1,T=1; candidate physical recurrence/readout vectorized across these sequences",
                scope="route parity only, both zero/nonzero initial states; no model/codec selection; original bulk GEMM route is not claimed bitwise equivalent")


def column_norm_squared(coeff):
    """Column norms of (I-beta k k^T)Diag(alpha), including full rank-one action."""
    key = coeff["k"]
    beta = coeff["beta"][..., None]
    key_norm2 = (key * key).sum(-1, keepdims=True, dtype=np.float32)
    return coeff["alpha"]**2 * (1 + (beta**2 * key_norm2 - 2 * beta) * key**2)


def calibrate(table, tokens):
    """CAL-only statistics from a causal base INT4 self-update trajectory."""
    codec = PackedCodec(SHAPE, bits=4)
    stored = codec.encode(np.zeros((len(tokens),) + SHAPE, dtype=np.float32))
    covariance = np.zeros((SHAPE[0], SHAPE[1], SHAPE[1]), dtype=np.float64)
    error_mean = np.zeros(SHAPE[:2], dtype=np.float64)
    energy = np.zeros(SHAPE[:2], dtype=np.float64)
    next_norm = np.zeros(SHAPE[:2], dtype=np.float64)
    paired_risk = np.zeros(SHAPE[:2], dtype=np.float64)
    token_ids = full_tokens(tokens)
    for position in range(token_ids.shape[1]):
        coeff = gather(table, token_ids[:, position])
        updated = torch_transition(codec.decode(stored), coeff)
        stored = codec.encode(updated)
        error = updated - codec.decode(stored)
        # Offline statistics have FP64 accumulation; every recurrence stays FP32.
        error64 = error.astype(np.float64)
        covariance += np.einsum("bhkv,bhlv->hkl", error64, error64)
        error_mean += error64.sum(axis=(0,3))
        if position < token_ids.shape[1] - 1:
            energy += (error64**2).sum(axis=(0, 3))
            next_coeff = gather(table, token_ids[:, position + 1])
            next_column_norm = column_norm_squared(next_coeff).astype(np.float64)
            next_norm += next_column_norm.sum(0)
            paired_risk += ((error64**2).mean(-1)*next_column_norm).sum(0)
    writes = token_ids.shape[1]
    covariance /= len(tokens) * writes * SHAPE[2]
    error_mean /= len(tokens) * writes * SHAPE[2]
    covariance -= np.einsum("hk,hl->hkl",error_mean,error_mean)
    energy /= len(tokens) * len(tokens[0]) * SHAPE[2]
    next_norm /= len(tokens) * len(tokens[0])
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(-values, axis=1, kind="stable")
    vectors = np.take_along_axis(vectors, order[:, None, :], axis=2)
    values = np.take_along_axis(values, order, axis=1)
    for head in range(SHAPE[0]):
        for column in range(SHAPE[1]):
            pivot = int(np.argmax(np.abs(vectors[head, :, column])))
            if vectors[head, pivot, column] < 0:
                vectors[head, :, column] *= -1
    scores = paired_risk / (len(tokens)*len(tokens[0]))
    rankings = np.argsort(-scores, axis=1, kind="stable")
    mixed4 = np.full(SHAPE[:2], 4, dtype=np.uint8)
    mixed6 = np.full(SHAPE[:2], 6, dtype=np.uint8)
    for head in range(SHAPE[0]):
        mixed4[head, rankings[head, :4]] = 8
        mixed6[head, rankings[head, :8]] = 8
    return dict(basis=vectors[:, :, :4].astype(np.float32), eigenvalues=values,
                covariance=covariance, error_mean=error_mean, error_energy=energy, next_column_norm2=next_norm,
                mixed_scores=scores, mixed4_bits=mixed4, mixed6_bits=mixed6)


def make_arms(calibration):
    arms = {"NATIVE_FP32": OnlineAdapter(NativeFloatCodec(SHAPE))}
    arms.update({f"UNIFORM_{b}": OnlineAdapter(PackedCodec(SHAPE, bits=b)) for b in range(2, 17)})
    arms["STOCHASTIC_4"] = OnlineAdapter(PackedCodec(SHAPE, bits=4, stochastic=True))
    for residual in (4, 8, "FP32"):
        residual_codec = NativeFloatCodec(SHAPE) if residual == "FP32" else PackedCodec(SHAPE, bits=residual)
        arms[f"FULL_RESIDUAL_4_{residual}"] = OnlineAdapter(PackedCodec(SHAPE, bits=4), residual_codec)
    for rank in (1, 2, 4):
        arms[f"LOWRANK_4_8_R{rank}"] = OnlineAdapter(
            PackedCodec(SHAPE, bits=4), PackedCodec((12, rank, 16), bits=8),
            calibration["basis"][:, :, :rank])
    arms["UNTRANSPORTED_4_4"] = OnlineAdapter(
        PackedCodec(SHAPE, bits=4), PackedCodec(SHAPE, bits=4), mode="untransported")
    arms["MIXED_4_8"] = OnlineAdapter(PackedCodec(SHAPE, bits=4, mixed_bits=calibration["mixed4_bits"]))
    arms["MIXED_6_8"] = OnlineAdapter(PackedCodec(SHAPE, bits=6, mixed_bits=calibration["mixed6_bits"]))
    if list(arms) != ARM_NAMES:
        raise AssertionError("Arm order differs from the frozen family")
    return arms


def stream_seeds(seed, count):
    return np.arange(count, dtype=np.uint64) + np.uint64(seed) * np.uint64(1000003)


@torch.inference_mode()
def sequence_call(model, table, tokens, adapter, seeds=None, initial=None, include_bos=True, diagnostics=True):
    """Complete causal call; gold is absent and only OnlineState crosses steps."""
    start = time.perf_counter()
    state = adapter.initial(np.zeros((len(tokens),) + SHAPE, dtype=np.float32), seeds=seeds) if initial is None else initial
    if initial is not None and include_bos:
        raise ValueError("A continued stream must not prepend another BOS")
    offset = int(include_bos)
    predictions = np.full((len(tokens), tokens.shape[1] + offset), -1, dtype=np.int16)
    totals = dict(writeback_mse=0., projection_leakage_mse=0., state_norm_mean=0.)
    completed = 0
    status = "COMPLETE"
    nonfinite_rows = 0
    absorbed = np.zeros(len(tokens), dtype=bool)
    first_invalid = [None]*len(tokens)

    def absorb(mask, position):
        for row in np.flatnonzero(mask & ~absorbed):
            first_invalid[int(row)] = int(position)
        absorbed[:] |= mask

    for position in range(tokens.shape[1] + offset):
        ids = np.full(len(tokens), 6, dtype=np.int64) if include_bos and position == 0 else tokens[:, position - offset]
        coeff = gather(table, ids)

        def update(decoded):
            updated = torch_transition(decoded, coeff)
            finite = np.isfinite(updated).all(axis=(1,2,3))
            absorb(~finite, position)
            if absorbed.any():
                updated = updated.copy()
                updated[absorbed] = 0
            return updated

        with np.errstate(over="ignore", invalid="ignore"):
            try:
                state, represented, diagnostic = adapter.step(state, update, diagnostics=diagnostics)
            except NonfiniteStateError as exc:
                # A residual addition can fail after the transition callback.
                # Only failed rows are reset; healthy incoming bytes/RNG survive.
                absorb(exc.bad_rows, position)
                bad_seeds = None if seeds is None else seeds[exc.bad_rows]
                safe = adapter.initial(np.zeros((int(exc.bad_rows.sum()),)+SHAPE,dtype=np.float32),seeds=bad_seeds)
                safe.payload[:, :8] = state.payload[exc.bad_rows, :8]
                incoming = state.payload.copy()
                incoming[exc.bad_rows] = safe.payload
                state, represented, diagnostic = adapter.step(OnlineState(incoming),update,diagnostics=diagnostics)
        logits = logits_from_numpy(model, represented, coeff)
        finite = torch.isfinite(logits).all(-1)
        nonfinite_rows += int((~finite).sum())
        prediction = logits.argmax(-1).numpy()
        prediction[absorbed | ~finite.numpy()] = -1
        predictions[:, position] = prediction
        if absorbed.any():
            status = "NONFINITE_ROWS_ABSORBED"
        elif nonfinite_rows:
            status = "NONFINITE_LOGITS_WITH_FINITE_STATE"
        if diagnostics:
            for key in totals:
                totals[key] += diagnostic[key]
        completed += 1
    elapsed = time.perf_counter() - start
    return predictions, state, dict(
        status=status, completed_writes=completed, elapsed_seconds=elapsed,
        nonfinite_output_sequence_steps=nonfinite_rows,
        absorbed_sequences=int(absorbed.sum()), first_invalid_write_position=first_invalid,
        invalid_write_position_zero_is_bos=include_bos,
        diagnostics={key: value / max(completed, 1) for key, value in totals.items()} if diagnostics else None,
        diagnostics_include_bos=True,
        diagnostic_invalid_row_policy="after state nonfinite detection, failed rows use zero placeholders and all later predictions remain failures; a nonfinite readout alone marks that prediction invalid while the finite state continues; BOS readout is unscored",
    )


def ledger_with_table(adapter, state, shared_table_bytes):
    ledger = adapter.ledger(state)
    ledger["shared_codec_bytes"] = ledger["shared_bytes"]
    ledger["shared_token_coefficient_bytes"] = shared_table_bytes
    ledger["shared_bytes"] += shared_table_bytes
    ledger["total_bytes"] = {str(n): ledger["shared_bytes"] + n * adapter.bytes_per_stream for n in (1, 16, 128)}
    ledger["scope"] = "exact serialized recurrent cache plus shared codec/basis/token table; common checkpoint separate; temporary workspaces excluded"
    return ledger


def summarize_arm(model, table, adapter, batch, horizons, family_size, model_seed, alpha, epsilons, shared_table_bytes,
                  correctness_output=None):
    predictions, state, execution = sequence_call(model, table, batch.tokens, adapter, stream_seeds(batch.seed, len(batch.tokens)))
    correct = predictions[:, 1:] == batch.gold
    taus = first_failure_times(correct)
    summary = summarize_first_failures(
        taus, batch.tokens.shape[1], horizons=horizons, sequence_ids=batch.sequence_ids,
        family_size=family_size, alpha=alpha, epsilons=epsilons, correct=correct, model_seed=model_seed)
    summary["execution"] = execution
    summary["bos"] = dict(accuracy=float((predictions[:, 0] == 0).mean()),
                          prediction_counts=np.bincount(predictions[:, 0] + 1, minlength=7).tolist(),
                          count_labels=[-1, 0, 1, 2, 3, 4, 5], gold_label=0, excluded_from_tau=True)
    summary["step_correct_counts"] = correct.sum(0).tolist()
    summary["step_accuracy_by_position"] = correct.mean(0).tolist()
    summary["token_counts_at_horizons"] = [dict(
        horizon=t, prefix_correct=int(correct[:, :t].sum()), prefix_total=int(len(correct)*t),
        final_quarter_correct=int(correct[:, t-max(t//4,1):t].sum()),
        final_quarter_total=int(len(correct)*max(t//4,1))) for t in horizons]
    summary["ledger"] = ledger_with_table(adapter, state, shared_table_bytes)
    summary["config_sha256"] = digest(adapter.config_bytes)
    summary["basis_sha256"] = digest(adapter.basis_bytes)
    summary["final_payload_sha256"] = digest(state.payload.tobytes())
    summary["read_boundary"] = "updated packed state plus decoded residual after write"
    if correctness_output is not None:
        path = Path(correctness_output)
        np.savez_compressed(path, packed=np.packbits(correct,axis=1,bitorder="little"),
                            shape=np.asarray(correct.shape,dtype=np.int64))
        summary["correctness_artifact"] = dict(file=path.name,sha256=file_sha256(path),
                                               shape=list(correct.shape),packing="np.packbits(axis=1)",
                                               bitorder="little",bos_included=False)
    return summary


def benchmark(model, table, arms, runtime):
    cfg = runtime["benchmark"]
    batch = frozen_sequences("S3", "DEV", cfg["sequences"], cfg["length"], cfg["seed"])
    result = {}
    # A common native call warms the readout and projection path before timing.
    sequence_call(model, table, batch.tokens[:, :2], arms["NATIVE_FP32"])
    for name, adapter in arms.items():
        rows = []
        for _ in range(cfg["repeats"]):
            _, _, execution = sequence_call(model, table, batch.tokens, adapter, stream_seeds(cfg["seed"], len(batch.tokens)), diagnostics=False)
            rows.append(execution)
        seconds = [row["elapsed_seconds"] for row in rows]
        result[name] = dict(median_seconds=float(np.median(seconds)), all_seconds=seconds,
                            milliseconds_per_group_token=1000*float(np.median(seconds))/(cfg["sequences"]*cfg["length"]),
                            all_complete=all(row["status"] == "COMPLETE" for row in rows))
    return dict(config=cfg, cpu_threads=runtime["cpu_threads"], arms=result,
                scope="full causal sequence call: allocation, initial packing, BOS, token-table gather, decode, transition, projection, encode, after-write read, original FP32 norm/MLP; excludes optional MSE/norm diagnostics, gold, summaries, file I/O, model load and shared-table construction",
                speed_claim="CPU reference timing only; no GPU kernel or model-wide throughput claim")


def validate_protocol(protocol, runtime):
    if not protocol.get("frozen_utc") or protocol.get("arms") != ARM_NAMES:
        raise ValueError("Expected the frozen Case010 protocol and exact arm list")
    for split, expected in {"CAL":(64,32,1001), "DEV":(128,128,2001), "TEST":(512,2048,3001)}.items():
        cfg = protocol["splits"][split]
        if (cfg["sequences"],cfg["length"],cfg["seed"]) != expected:
            raise ValueError(f"Unexpected frozen {split} split")
    if runtime["cpu_threads"] not in (1, 2):
        raise ValueError("Runtime CPU threads must be 1 or 2")
    if runtime["benchmark"] != dict(sequences=16, length=128, repeats=3, seed=2002):
        raise ValueError("Unexpected benchmark declaration")
    if runtime.get("schema") != "case010-evaluation-runtime-v2-canonical-token-torch":
        raise ValueError("Expected the frozen canonical one-token Torch execution route")
    if runtime["route_parity"] != dict(split="DEV", sequences=2, length=2048, seed=2004,
                                       selection_allowed=False, atol=ATOL, rtol=RTOL):
        raise ValueError("Unexpected long DEV parity declaration")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--runtime", type=Path, default=CASE_ROOT/"configs/evaluation_runtime.json")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--smoke", action="store_true", help="Only DEV two sequences x32; never CAL/TEST or study claims")
    args = parser.parse_args(argv)
    protocol, protocol_hash = verify_frozen_json(args.protocol)
    runtime, runtime_hash = verify_frozen_json(args.runtime)
    validate_protocol(protocol, runtime)
    torch.set_num_threads(runtime["cpu_threads"])
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if payload["status"] != "TRAINING_COMPLETE":
        raise ValueError("Primary evaluation requires the final completed scheduled checkpoint")
    if payload["provenance"]["protocol_sha256"] != protocol_hash:
        raise ValueError("Checkpoint was trained under a different protocol hash")
    model = create_model(load_upstream(args.upstream), backend="naive_recurrent", checkpoint=args.checkpoint).eval()
    table = token_table(model)
    group = make_group("S3")
    upstream = sys.modules["group_word_problems.train_wordproblem"]
    _, upstream_table, identity = upstream.perm_group("s3")
    np.testing.assert_array_equal(group.multiplication, upstream_table)
    if identity != group.identity:
        raise AssertionError("Independent and official identity labels differ")
    args.output.mkdir(parents=True, exist_ok=False)
    config, raw = table_bytes(table)
    (args.output/"token_coefficients.json").write_bytes(config)
    (args.output/"token_coefficients.bin").write_bytes(raw)
    route = runtime["route_parity"]
    parity_batch = frozen_sequences(group,"DEV",2,32,protocol["splits"]["DEV"]["seed"]) if args.smoke else frozen_sequences(
        group,"DEV",route["sequences"],route["length"],route["seed"])
    parity = verify_route_parity(model, table, parity_batch.tokens)
    parity["split"] = "DEV"
    parity["seed"] = parity_batch.seed
    write_json(args.output/"parity.json", parity)
    source_files = ["codec/learned.py", "codec/online.py", "codec/packed.py", "codec/groups.py", "codec/survival.py", "scripts/evaluate_learned.py"]
    frozen_source = args.output/"frozen-source"
    for relative in source_files:
        target = frozen_source/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((CASE_ROOT/relative).read_bytes())
    for name, source in (("protocol.json",args.protocol),("evaluation_runtime.json",args.runtime)):
        (args.output/name).write_bytes(source.read_bytes())
        (args.output/Path(name).with_suffix(".sha256")).write_text(file_sha256(source)+"\n")
    manifest = dict(schema=VERSION, phase="DEV_SMOKE" if args.smoke else "FROZEN_PRIMARY",
                    protocol_sha256=protocol_hash, runtime_sha256=runtime_hash,
                    checkpoint_sha256=file_sha256(args.checkpoint), model_seed=payload["seed"],
                    upstream_commit=payload["provenance"]["upstream_commit"],
                    required_upstream_source_sha256=payload["provenance"]["required_source_sha256"],
                    source_sha256={p:file_sha256(frozen_source/p) for p in source_files},
                    frozen_source_directory="frozen-source",
                    token_table_config_sha256=digest(config), token_table_data_sha256=digest(raw),
                    shared_token_table_bytes=len(config)+len(raw), model_checkpoint_bytes=args.checkpoint.stat().st_size,
                    torch_version=str(torch.__version__), numpy_version=np.__version__,
                    actual_coefficients=coefficient_metadata(model, table),
                    coefficient_execution_policy="each of seven token IDs projected separately at B=1,T=1; exact token-local table shared by every arm; no sequence-length-dependent projection; state and readout vectorized across streams",
                    checkpoint_training_status=payload["status"], checkpoint_updates=payload["updates"],
                    native_arm="physical-coordinate FP32 Torch recurrence with canonical B1,T1 coefficient table and exact official einsum read reduction; original FP32 readout modules; parity checked against official one-token gauge Cache path",
                    arithmetic="FP32 learned recurrence/weights/readout; FP64 offline CAL statistics/eigensolver and diagnostic reductions; codec quantizer uses FP64 temporary normalization; no FP64 learned recurrent-state arm",
                    test_accessed=False)
    write_json(args.output/"manifest.json", manifest)
    if args.smoke:
        calibration = calibrate(table, parity_batch.tokens)
        batches = {"DEV_SMOKE":parity_batch}
        manifest["calibration_source_split"] = "DEV_SMOKE_PIPELINE_ONLY"
    else:
        cfg = protocol["splits"]["CAL"]
        cal_batch = frozen_sequences(group, "CAL", cfg["sequences"], cfg["length"], cfg["seed"])
        calibration = calibrate(table, cal_batch.tokens)
        manifest["calibration_source_split"] = "CAL"
        write_json(args.output/"CAL_inputs.json", input_identity(cal_batch))
        # TEST is generated only after every DEV arm is complete below.
        batches = {"DEV":None, "TEST":None}
    np.savez(args.output/"calibration.npz", **calibration)
    manifest["calibration_sha256"] = file_sha256(args.output/"calibration.npz")
    arms = make_arms(calibration)
    config_dir = args.output/"codecs"
    config_dir.mkdir()
    for name, adapter in arms.items():
        (config_dir/f"{name}.json").write_bytes(adapter.config_bytes)
        if adapter.basis_bytes:
            (config_dir/f"{name}.basis.bin").write_bytes(adapter.basis_bytes)
    family_size = len(arms)*len(protocol["evaluation"]["horizons"])*protocol["evaluation"]["max_model_seeds"]
    results = {}
    for split in batches:
        if args.smoke:
            batch = batches[split]
        else:
            cfg = protocol["splits"][split]
            batch = frozen_sequences(group, split, cfg["sequences"], cfg["length"], cfg["seed"])
        if split == "TEST":
            manifest["test_accessed"] = True
        write_json(args.output/"manifest.json", manifest)
        write_json(args.output/f"{split}_inputs.json", input_identity(batch))
        horizons = [h for h in protocol["evaluation"]["horizons"] if h <= batch.tokens.shape[1]]
        folder = args.output/split
        folder.mkdir()
        results[split] = {}
        for name, adapter in arms.items():
            summary = summarize_arm(model, table, adapter, batch, horizons, family_size, payload["seed"],
                                    protocol["evaluation"]["alpha"],
                                    (protocol["evaluation"]["epsilon_primary"],protocol["evaluation"]["epsilon_secondary"]),
                                    len(config)+len(raw),correctness_output=folder/f"{name}.correctness.npz")
            write_json(folder/f"{name}.json", summary)
            results[split][name] = dict(file=f"{split}/{name}.json", sha256=file_sha256(folder/f"{name}.json"),
                                       execution=summary["execution"]["status"],
                                       final_survival=summary["horizons"][-1]["survival_probability"],
                                       rmst=summary["restricted_mean_failure_free_length"])
            print(json.dumps(dict(split=split, arm=name, **results[split][name])), flush=True)
    if not args.smoke:
        write_json(args.output/"timing.json", benchmark(model, table, arms, runtime))
    write_json(args.output/"index.json", dict(schema=VERSION, family_size=family_size, manifest=manifest, splits=results))
    (args.output/"index.sha256").write_text(file_sha256(args.output/"index.json")+"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
