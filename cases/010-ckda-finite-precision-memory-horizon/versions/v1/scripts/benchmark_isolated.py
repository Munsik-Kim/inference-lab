"""Run the frozen complete-call benchmark serially after all primary runs finish.

Each seed runs exactly once in a fresh child process. This wrapper does not fit
CAL, generate TEST, change codecs, or retry a failed timing measurement.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import types

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def frozen_json(path):
    path = Path(path)
    expected = path.with_suffix(".sha256").read_text().split()[0]
    if sha256(path) != expected:
        raise ValueError(f"Frozen JSON hash mismatch: {path}")
    return json.loads(path.read_text()), expected


def inspect_evaluation(directory, checkpoint, require_complete=True):
    """Verify checkpoint, frozen sources and fitted artifacts without inference."""
    directory, checkpoint = Path(directory).resolve(), Path(checkpoint).resolve()
    manifest = json.loads((directory/"manifest.json").read_text())
    if manifest["phase"] != "FROZEN_PRIMARY" or manifest["checkpoint_training_status"] != "TRAINING_COMPLETE":
        raise ValueError("Expected a completed-training primary evaluation")
    if sha256(checkpoint) != manifest["checkpoint_sha256"]:
        raise ValueError("Checkpoint hash differs from the evaluated checkpoint")
    protocol, protocol_hash = frozen_json(directory/"protocol.json")
    runtime, runtime_hash = frozen_json(directory/"evaluation_runtime.json")
    if (protocol_hash, runtime_hash) != (manifest["protocol_sha256"], manifest["runtime_sha256"]):
        raise ValueError("Evaluation protocol/runtime identity differs from its manifest")
    source_root = directory/manifest["frozen_source_directory"]
    if source_root.resolve().parent != directory:
        raise ValueError("Frozen source directory must be inside this evaluation")
    for relative, expected in manifest["source_sha256"].items():
        path = source_root/relative
        if not path.resolve().is_relative_to(source_root.resolve()) or sha256(path) != expected:
            raise ValueError(f"Frozen source identity mismatch: {relative}")
    required = {"codec/learned.py", "codec/online.py", "codec/packed.py", "codec/groups.py",
                "codec/survival.py", "scripts/evaluate_learned.py"}
    if set(manifest["source_sha256"]) != required:
        raise ValueError("Unexpected frozen evaluator source set")
    artifacts = {"token_coefficients.json":manifest["token_table_config_sha256"],
                 "token_coefficients.bin":manifest["token_table_data_sha256"],
                 "calibration.npz":manifest["calibration_sha256"]}
    for relative, expected in artifacts.items():
        if sha256(directory/relative) != expected:
            raise ValueError(f"Fitted artifact identity mismatch: {relative}")
    codecs = {}
    for name in protocol["arms"]:
        summary = json.loads((directory/"DEV"/f"{name}.json").read_text())
        config_path = directory/"codecs"/f"{name}.json"
        basis_path = directory/"codecs"/f"{name}.basis.bin"
        basis_hash = sha256(basis_path) if basis_path.exists() else hashlib.sha256(b"").hexdigest()
        if sha256(config_path) != summary["config_sha256"] or basis_hash != summary["basis_sha256"]:
            raise ValueError(f"Serialized codec differs from DEV: {name}")
        codecs[name] = dict(config_sha256=summary["config_sha256"], basis_sha256=basis_hash)
    index_hash = None
    if require_complete:
        index, index_hash = frozen_json(directory/"index.json")
        if index["manifest"] != manifest:
            raise ValueError("Completed index and manifest differ")
        for split in ("DEV", "TEST"):
            if set(index["splits"][split]) != set(protocol["arms"]):
                raise ValueError(f"Primary {split} is incomplete")
        if not manifest["test_accessed"] or not (directory/"timing.json").is_file():
            raise ValueError("All primary work, including embedded timing, must finish first")
    return dict(directory=str(directory), checkpoint=str(checkpoint), manifest=manifest,
                protocol=protocol, runtime=runtime, codecs=codecs,
                manifest_sha256=sha256(directory/"manifest.json"), primary_index_sha256=index_hash)


def load_frozen_evaluator(inspection):
    """Import only the verified snapshot, including its namespace package."""
    if any(name == "codec" or name.startswith("codec.") for name in sys.modules):
        raise RuntimeError("A fresh process is required; codec modules were already imported")
    source_root = Path(inspection["directory"])/inspection["manifest"]["frozen_source_directory"]
    # Primary snapshots intentionally contain only the six hashed source files.
    # An explicit package prevents a live checkout's __init__.py taking priority.
    package = types.ModuleType("codec")
    package.__path__ = [str(source_root/"codec")]
    package.__package__ = "codec"
    package.__spec__ = importlib.machinery.ModuleSpec("codec", loader=None, is_package=True)
    sys.modules["codec"] = package
    name = "case010_frozen_evaluation"
    spec = importlib.util.spec_from_file_location(name, source_root/"scripts/evaluate_learned.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    for relative in inspection["manifest"]["source_sha256"]:
        if relative.startswith("codec/"):
            imported = sys.modules[relative[:-3].replace("/", ".")]
            if Path(imported.__file__).resolve() != (source_root/relative).resolve():
                raise RuntimeError(f"Imported a source outside the verified snapshot: {relative}")
    return module


def validate_policy(policy, inspections):
    expected = dict(sequences=16, length=128, repeats=3, seed=2002)
    if policy["schema"] != "case010-cpu-execution-policy-v1" or not policy.get("frozen_utc"):
        raise ValueError("Expected the frozen execution policy")
    if policy["threads_per_process"] != 2 or not policy["no_new_gpu_forward"]:
        raise ValueError("Expected two-thread CPU-only timing")
    timing = policy["timing_policy"]
    if (timing["seeds"],timing["repeats_per_arm"],timing["DEV_generator_seed"],
            timing["sequences"],timing["length"],timing["no_retry_for_favorable_cost"]) != ([0,1,2],3,2002,16,128,True):
        raise ValueError("Timing policy differs from the frozen benchmark")
    if sorted(row["manifest"]["model_seed"] for row in inspections) != [0,1,2]:
        raise ValueError("Exactly the three distinct trained seeds are required")
    if len({row["directory"] for row in inspections}) != 3:
        raise ValueError("Primary seed directories must differ")
    for row in inspections:
        if row["runtime"]["benchmark"] != expected or row["runtime"]["cpu_threads"] != 2:
            raise ValueError("Runtime benchmark does not match the execution policy")


def run_child(plan_path, seed):
    plan = json.loads(Path(plan_path).read_text())
    if sha256(__file__) != plan["wrapper_sha256"]:
        raise ValueError("Isolated wrapper changed after the plan was written")
    _, policy_hash = frozen_json(Path(plan["output"])/"execution_policy.json")
    if policy_hash != plan["execution_policy_sha256"]:
        raise ValueError("Execution policy changed after the plan was written")
    row = next(item for item in plan["evaluations"] if item["seed"] == seed)
    inspection = inspect_evaluation(row["directory"], row["checkpoint"])
    if inspection["primary_index_sha256"] != row["primary_index_sha256"]:
        raise ValueError("Primary completion index changed after the plan was written")
    evaluation = load_frozen_evaluator(inspection)
    evaluation.torch.set_num_threads(2)
    model = evaluation.create_model(evaluation.load_upstream(Path(plan["upstream"])),
                                    checkpoint=Path(row["checkpoint"]), device="cpu").eval()
    table = evaluation.token_table(model)
    config, data = evaluation.table_bytes(table)
    primary = Path(row["directory"])
    if config != (primary/"token_coefficients.json").read_bytes() or data != (primary/"token_coefficients.bin").read_bytes():
        raise ValueError("Reconstructed canonical coefficients differ from primary bytes")
    arms = {}
    for name in inspection["protocol"]["arms"]:
        basis_path = primary/"codecs"/f"{name}.basis.bin"
        basis = basis_path.read_bytes() if basis_path.exists() else b""
        arms[name] = evaluation.OnlineAdapter.from_shared((primary/"codecs"/f"{name}.json").read_bytes(),basis)
    started = datetime.now(timezone.utc).isoformat()
    # The frozen function passes diagnostics=False and retains all three repeats.
    timing = evaluation.benchmark(model, table, arms, inspection["runtime"])
    finished = datetime.now(timezone.utc).isoformat()
    timing["model_seed"] = seed
    timing["checkpoint_sha256"] = inspection["manifest"]["checkpoint_sha256"]
    cfg = inspection["runtime"]["benchmark"]
    input_identity = evaluation.input_identity(evaluation.frozen_sequences(
        "S3", "DEV", cfg["sequences"], cfg["length"], cfg["seed"]))
    timing["isolated_provenance"] = dict(
        seed=seed, started_utc=started, finished_utc=finished, fresh_process=True,
        wrapper_sha256=sha256(__file__), execution_policy_sha256=plan["execution_policy_sha256"],
        primary_index_sha256=inspection["primary_index_sha256"],
        primary_manifest_sha256=inspection["manifest_sha256"],
        checkpoint_sha256=inspection["manifest"]["checkpoint_sha256"],
        source_sha256=inspection["manifest"]["source_sha256"],
        runtime_sha256=inspection["manifest"]["runtime_sha256"],
        calibration_sha256=inspection["manifest"]["calibration_sha256"],
        token_table_config_sha256=inspection["manifest"]["token_table_config_sha256"],
        token_table_data_sha256=inspection["manifest"]["token_table_data_sha256"],
        codec_sha256=inspection["codecs"], input_identity=input_identity,
        no_calibration_refit=True, test_generated=False,
        timed_function="unchanged frozen scripts.evaluate_learned.benchmark", retries=0)
    output = Path(plan["output"])/f"seed{seed}.json"
    write_json(output, timing)
    output.with_suffix(".sha256").write_text(sha256(output)+"\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream",type=Path)
    parser.add_argument("--evaluations",type=Path,nargs=3,metavar="PRIMARY_DIR")
    parser.add_argument("--checkpoints",type=Path,nargs=3,metavar="FINAL_PT")
    parser.add_argument("--policy",type=Path,default=ROOT/"configs/execution_policy.json")
    parser.add_argument("--output",type=Path)
    parser.add_argument("--child-plan",type=Path,help=argparse.SUPPRESS)
    parser.add_argument("--child-seed",type=int,help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.child_plan:
        run_child(args.child_plan,args.child_seed)
        return 0
    if any(value is None for value in (args.upstream,args.evaluations,args.checkpoints,args.output)):
        parser.error("--upstream, --evaluations, --checkpoints and --output are required")
    policy, policy_hash = frozen_json(args.policy)
    inspections = [inspect_evaluation(directory,checkpoint) for directory,checkpoint in zip(args.evaluations,args.checkpoints)]
    validate_policy(policy,inspections)
    inspections.sort(key=lambda row:row["manifest"]["model_seed"])
    # All helper invocations for these three primaries share a nonblocking lock.
    primary_parent = Path(os.path.commonpath([row["directory"] for row in inspections]))
    with (primary_parent/".case010-isolated-benchmark.lock").open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        args.output.mkdir(parents=True,exist_ok=False)
        plan = dict(schema="case010-isolated-timing-v1",status="STARTED",
                    upstream=str(args.upstream.resolve()),output=str(args.output.resolve()),
                    execution_policy_sha256=policy_hash,wrapper_sha256=sha256(__file__),
                    evaluations=[dict(seed=row["manifest"]["model_seed"],directory=row["directory"],
                                      checkpoint=row["checkpoint"],primary_index_sha256=row["primary_index_sha256"])
                                 for row in inspections],serial=True,retry_policy="none")
        plan_path = args.output/"plan.json"
        write_json(plan_path,plan)
        (args.output/"benchmark_isolated.py").write_bytes(Path(__file__).read_bytes())
        (args.output/"execution_policy.json").write_bytes(args.policy.read_bytes())
        (args.output/"execution_policy.sha256").write_text(policy_hash+"\n")
        env = dict(os.environ,CUDA_VISIBLE_DEVICES="",PYTHONDONTWRITEBYTECODE="1",
                   OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2")
        completed_seeds = []
        for row in plan["evaluations"]:
            seed = row["seed"]
            command = [sys.executable,"-B",str(Path(__file__).resolve()),"--child-plan",str(plan_path.resolve()),"--child-seed",str(seed)]
            with (args.output/f"seed{seed}.private.log").open("w") as log:
                result = subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:
                write_json(args.output/"status.json",dict(status="FAILED_NO_RETRY",seed=seed,
                           exit_code=result.returncode,completed_seeds=completed_seeds))
                raise RuntimeError(f"Isolated seed {seed} failed; no timing retry was attempted")
            completed_seeds.append(seed)
            print(json.dumps(dict(seed=seed,status="COMPLETE",file=f"seed{seed}.json")),flush=True)
        write_json(args.output/"index.json",dict(schema=plan["schema"],status="COMPLETE",serial=True,
                   execution_policy_sha256=policy_hash,wrapper_sha256=sha256(__file__),
                   seeds={str(seed):dict(file=f"seed{seed}.json",sha256=sha256(args.output/f"seed{seed}.json"))
                          for seed in completed_seeds}))
        (args.output/"index.sha256").write_text(sha256(args.output/"index.json")+"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
