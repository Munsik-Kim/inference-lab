"""Measure one post-hoc CAL fitting call per seed without changing candidates."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import benchmark_isolated as provenance


def compare_arrays(actual, reference):
    """Describe exact differences without modifying either array collection."""
    import numpy as np
    rows = {}
    for name in sorted(set(actual) | set(reference)):
        left, right = actual.get(name), reference.get(name)
        row = dict(actual_present=left is not None, reference_present=right is not None,
                   byte_exact=False, maximum_absolute_difference=None)
        for prefix, value in (("actual",left),("reference",right)):
            if value is not None:
                array = np.asarray(value)
                row[prefix] = dict(shape=list(array.shape),dtype=array.dtype.str,nbytes=int(array.nbytes),
                                   sha256=hashlib.sha256(array.tobytes(order="C")).hexdigest(),
                                   nonfinite_elements=int((~np.isfinite(array)).sum()))
        if left is not None and right is not None:
            left,right = np.asarray(left),np.asarray(right)
            row["byte_exact"] = (left.shape == right.shape and left.dtype == right.dtype
                                 and left.tobytes(order="C") == right.tobytes(order="C"))
            if left.shape == right.shape and np.isfinite(left).all() and np.isfinite(right).all():
                difference = np.abs(left.astype(np.float64)-right.astype(np.float64))
                row["maximum_absolute_difference"] = float(difference.max(initial=0.))
                row["different_value_count"] = int(np.count_nonzero(left != right))
        rows[name] = row
    return dict(array_keys_equal=set(actual) == set(reference),
                all_arrays_byte_exact=all(row["byte_exact"] for row in rows.values()),arrays=rows,
                returned_stats_array_nbytes=sum(int(np.asarray(value).nbytes) for value in actual.values()),
                storage_scope="returned stats arrays only; not peak temporary or allocator memory")


def measure_once(calibrator, table, tokens, reference, clock=time.perf_counter):
    start = clock()
    actual = calibrator(table,tokens)
    elapsed = clock()-start
    # Comparison, hashing and storage accounting are deliberately outside timing.
    comparison = compare_arrays(actual,reference)
    return dict(wall_seconds=elapsed,timed_calls=1,comparison=comparison,
                returned_stats_array_nbytes=comparison["returned_stats_array_nbytes"],
                stats_storage_is_peak_memory=False)


def validate_config(config, inspections):
    if (config.get("schema"),config.get("phase")) != ("case010-calibration-cost-v1","POST_HOC_COST_ONLY"):
        raise ValueError("Expected the frozen post-hoc cost-only declaration")
    if not config.get("frozen_utc") or (config["cpu_threads"],config["timed_calls_per_model_seed"]) != (2,1):
        raise ValueError("Expected one calibration call with two CPU threads")
    expected = dict(sequences=64,length=32,seed=1001)
    if config["split"] != "CAL" or {name:config[name] for name in expected} != expected:
        raise ValueError("Calibration cost must use the original CAL split")
    if config["model_seeds"] != [0,1,2] or sorted(row["manifest"]["model_seed"] for row in inspections) != [0,1,2]:
        raise ValueError("Exactly the three trained seeds are required")
    for key in ("no_model_codec_or_threshold_changes","no_test_generation","no_gpu","no_retry_for_favorable_cost"):
        if config[key] is not True:
            raise ValueError(f"Required cost-only restriction missing: {key}")
    for row in inspections:
        if row["manifest"]["protocol_sha256"] != config["original_protocol_sha256"] or row["protocol"]["splits"]["CAL"] != expected:
            raise ValueError("Primary protocol differs from the declared CAL fitting source")


def verify_isolated_complete(directory, inspections):
    directory = Path(directory)
    index,index_hash = provenance.frozen_json(directory/"index.json")
    if index.get("status") != "COMPLETE" or index.get("serial") is not True or set(index["seeds"]) != {"0","1","2"}:
        raise ValueError("All serial isolated inference timings must complete first")
    for row in inspections:
        seed = row["manifest"]["model_seed"]
        entry = index["seeds"][str(seed)]
        path = directory/entry["file"]
        if not path.resolve().is_relative_to(directory.resolve()) or provenance.sha256(path) != entry["sha256"]:
            raise ValueError("Isolated timing artifact identity mismatch")
        timing = json.loads(path.read_text())
        if timing["model_seed"] != seed or timing["checkpoint_sha256"] != row["manifest"]["checkpoint_sha256"]:
            raise ValueError("Isolated timing uses a different checkpoint")
    return index_hash


def run_child(plan_path, seed):
    plan = json.loads(Path(plan_path).read_text())
    if provenance.sha256(__file__) != plan["runner_sha256"] or provenance.sha256(provenance.__file__) != plan["provenance_helper_sha256"]:
        raise ValueError("Cost runner or provenance helper changed after the plan was written")
    config,config_hash = provenance.frozen_json(Path(plan["output"])/"calibration_cost.json")
    if config_hash != plan["config_sha256"]:
        raise ValueError("Calibration cost config changed")
    row = next(item for item in plan["evaluations"] if item["seed"] == seed)
    inspection = provenance.inspect_evaluation(row["directory"],row["checkpoint"])
    if inspection["primary_index_sha256"] != row["primary_index_sha256"]:
        raise ValueError("Primary identity changed after the cost plan was written")
    evaluation = provenance.load_frozen_evaluator(inspection)
    evaluation.torch.set_num_threads(2)
    model = evaluation.create_model(evaluation.load_upstream(Path(plan["upstream"])),
                                     checkpoint=Path(row["checkpoint"]),device="cpu").eval()
    table = evaluation.token_table(model)
    descriptor,data = evaluation.table_bytes(table)
    primary = Path(row["directory"])
    if descriptor != (primary/"token_coefficients.json").read_bytes() or data != (primary/"token_coefficients.bin").read_bytes():
        raise ValueError("Canonical coefficient table differs from primary bytes")
    batch = evaluation.frozen_sequences("S3","CAL",config["sequences"],config["length"],config["seed"])
    with evaluation.np.load(primary/"calibration.npz",allow_pickle=False) as stored:
        reference = {name:stored[name] for name in stored.files}
    started = datetime.now(timezone.utc).isoformat()
    result = measure_once(evaluation.calibrate,table,batch.tokens,reference)
    finished = datetime.now(timezone.utc).isoformat()
    if provenance.sha256(primary/"calibration.npz") != inspection["manifest"]["calibration_sha256"]:
        raise ValueError("Retained primary calibration changed during cost measurement")
    result.update(schema="case010-calibration-cost-result-v1",phase="POST_HOC_COST_ONLY",model_seed=seed,
                  checkpoint_sha256=inspection["manifest"]["checkpoint_sha256"],cpu_threads=2,
                  status="COMPLETE" if result["comparison"]["all_arrays_byte_exact"] else "COMPLETE_WITH_ARRAY_DIFFERENCES",
                  started_utc=started,finished_utc=finished,config_sha256=config_hash,
                  runner_sha256=plan["runner_sha256"],provenance_helper_sha256=plan["provenance_helper_sha256"],
                  original_protocol_sha256=inspection["manifest"]["protocol_sha256"],
                  runtime_sha256=inspection["manifest"]["runtime_sha256"],
                  source_sha256=inspection["manifest"]["source_sha256"],
                  primary_index_sha256=inspection["primary_index_sha256"],
                  isolated_timing_index_sha256=plan["isolated_timing_index_sha256"],
                  calibration_sha256=inspection["manifest"]["calibration_sha256"],
                  token_table_config_sha256=inspection["manifest"]["token_table_config_sha256"],
                  token_table_data_sha256=inspection["manifest"]["token_table_data_sha256"],
                  input_identity=evaluation.input_identity(batch),retained_calibration_unchanged=True,
                  candidate_refit_or_overwrite=False,test_generated=False,gpu_used=False,
                  timed_scope=config["timed_scope"],excluded_from_timer=config["excluded_from_timer"])
    output = Path(plan["output"])/f"seed{seed}.json"
    provenance.write_json(output,result)
    output.with_suffix(".sha256").write_text(provenance.sha256(output)+"\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream",type=Path)
    parser.add_argument("--evaluations",type=Path,nargs=3,metavar="PRIMARY_DIR")
    parser.add_argument("--checkpoints",type=Path,nargs=3,metavar="FINAL_PT")
    parser.add_argument("--isolated-timing",type=Path)
    parser.add_argument("--config",type=Path,default=ROOT/"configs/calibration_cost.json")
    parser.add_argument("--output",type=Path)
    parser.add_argument("--child-plan",type=Path,help=argparse.SUPPRESS)
    parser.add_argument("--child-seed",type=int,help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.child_plan:
        run_child(args.child_plan,args.child_seed)
        return 0
    if any(value is None for value in (args.upstream,args.evaluations,args.checkpoints,args.isolated_timing,args.output)):
        parser.error("--upstream, --evaluations, --checkpoints, --isolated-timing and --output are required")
    config,config_hash = provenance.frozen_json(args.config)
    inspections = [provenance.inspect_evaluation(directory,checkpoint) for directory,checkpoint in zip(args.evaluations,args.checkpoints)]
    validate_config(config,inspections)
    isolated_hash = verify_isolated_complete(args.isolated_timing,inspections)
    inspections.sort(key=lambda row:row["manifest"]["model_seed"])
    primary_parent = Path(os.path.commonpath([row["directory"] for row in inspections]))
    with (primary_parent/".case010-isolated-benchmark.lock").open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        args.output.mkdir(parents=True,exist_ok=False)
        plan = dict(schema="case010-calibration-cost-plan-v1",phase="POST_HOC_COST_ONLY",
                    upstream=str(args.upstream.resolve()),output=str(args.output.resolve()),
                    runner_sha256=provenance.sha256(__file__),provenance_helper_sha256=provenance.sha256(provenance.__file__),
                    config_sha256=config_hash,isolated_timing_index_sha256=isolated_hash,
                    evaluations=[dict(seed=row["manifest"]["model_seed"],directory=row["directory"],checkpoint=row["checkpoint"],
                                      primary_index_sha256=row["primary_index_sha256"]) for row in inspections],
                    serial=True,retry_policy="none")
        plan_path = args.output/"plan.json"
        provenance.write_json(plan_path,plan)
        (args.output/"measure_calibration_cost.py").write_bytes(Path(__file__).read_bytes())
        (args.output/"calibration_cost.json").write_bytes(args.config.read_bytes())
        (args.output/"calibration_cost.sha256").write_text(config_hash+"\n")
        env = dict(os.environ,CUDA_VISIBLE_DEVICES="",PYTHONDONTWRITEBYTECODE="1",
                   OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2")
        completed,receipts = [],{}
        for row in plan["evaluations"]:
            seed = row["seed"]
            command = [sys.executable,"-B",str(Path(__file__).resolve()),"--child-plan",str(plan_path.resolve()),"--child-seed",str(seed)]
            with (args.output/f"seed{seed}.private.log").open("w") as log:
                result = subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:
                provenance.write_json(args.output/"status.json",dict(status="FAILED_NO_RETRY",seed=seed,completed_seeds=completed))
                raise RuntimeError(f"Calibration cost seed {seed} failed; no retry attempted")
            completed.append(seed)
            receipt = json.loads((args.output/f"seed{seed}.json").read_text())
            receipts[str(seed)] = dict(file=f"seed{seed}.json",sha256=provenance.sha256(args.output/f"seed{seed}.json"),
                                       status=receipt["status"],all_arrays_byte_exact=receipt["comparison"]["all_arrays_byte_exact"])
            print(json.dumps(dict(seed=seed,**receipts[str(seed)])),flush=True)
        provenance.write_json(args.output/"index.json",dict(schema=plan["schema"],phase=plan["phase"],status="COMPLETE",serial=True,
                              config_sha256=config_hash,runner_sha256=plan["runner_sha256"],
                              seeds=receipts))
        (args.output/"index.sha256").write_text(provenance.sha256(args.output/"index.json")+"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
