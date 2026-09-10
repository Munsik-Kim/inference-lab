# SPDX-License-Identifier: Apache-2.0
# Adapted from the locally executed STEP 03 smoke; no runtime monkeypatching.
"""Offline full-model W8A8 smoke test using the production vLLM paths."""
import argparse
import hashlib
import json
import os
import runpy
import platform
import pathlib
import sys
import time
import traceback


def inspect_quantized_layers(model):
    """Read model state through LLM.apply_model; do not alter dispatch or weights."""
    import collections
    import sys
    import torch

    layers = []
    for name, module in model.named_modules():
        scheme = getattr(module, "scheme", None)
        if type(scheme).__name__ != "CompressedTensorsW8A8Int8":
            continue
        kernel = scheme.kernel
        layers.append({
            "name": name,
            "module_class": type(module).__name__,
            "scheme_class": type(scheme).__name__,
            "kernel_class": type(kernel).__name__,
            "weight_dtype": str(module.weight.dtype),
            "weight_device": str(module.weight.device),
            "weight_shape": list(module.weight.shape),
            "weight_stride": list(module.weight.stride()),
            "scale_dtype": str(module.weight_scale.dtype),
            "scale_shape": list(module.weight_scale.shape),
            "logical_widths": list(module.logical_widths),
            "is_channelwise": kernel.config.is_channelwise,
            "is_static_input_scheme": kernel.config.is_static_input_scheme,
            "input_symmetric": kernel.config.input_symmetric,
        })
    return {
        "worker_python": sys.executable,
        "gpu_name": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "model_class": type(model).__name__,
        "kernel_counts": dict(collections.Counter(x["kernel_class"] for x in layers)),
        "quantized_layer_count": len(layers),
        "fused_layer_count": sum(len(x["logical_widths"]) > 1 for x in layers),
        "layers": layers,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=pathlib.Path, required=True)
    parser.add_argument("--run-dir", type=pathlib.Path, required=True)
    parser.add_argument("--phase", choices=["before", "after"], required=True)
    parser.add_argument("--model-dir", type=pathlib.Path, required=True)
    parser.add_argument("--expected-prefix", required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    run = args.run_dir
    (run / "input.json").write_text(json.dumps(config, indent=2) + "\n")

    verified = runpy.run_path(str(pathlib.Path(__file__).with_name("prepare_model.py")))["verify_model"](args.model_dir.resolve())
    assert verified["model_id"] == config["model_id"] and verified["revision"] == config["revision"]
    (run / "model_verification.json").write_text(json.dumps(verified, indent=2) + "\n")
    import torch
    import vllm
    import triton
    from vllm import LLM, SamplingParams
    from vllm.model_executor.kernels.linear.scaled_mm.cutlass import CutlassInt8ScaledMMLinearKernel

    prefix = pathlib.Path(sys.prefix)
    expected_prefix = args.expected_prefix
    assert str(prefix) == expected_prefix
    assert pathlib.Path(torch.__file__).is_relative_to(prefix)
    assert pathlib.Path(vllm.__file__).is_relative_to(prefix)
    assert torch.cuda.is_available() and torch.cuda.get_device_capability(0) == (12, 0)
    cutlass_file = pathlib.Path(vllm.__file__).parent / "model_executor/kernels/linear/scaled_mm/cutlass.py"
    runtime = {
        "phase": args.phase, "python": sys.executable, "python_version": sys.version,
        "torch": torch.__version__, "torch_cuda_build": torch.version.cuda,
        "torch_import": torch.__file__, "vllm": vllm.__version__, "vllm_import": vllm.__file__,
        "gpu_name": torch.cuda.get_device_name(0), "compute_capability": [12, 0],
        "cutlass_support_sm120": CutlassInt8ScaledMMLinearKernel.is_supported(120),
        "cutlass_file_sha256": hashlib.sha256(cutlass_file.read_bytes()).hexdigest(),
        "triton": triton.__version__, "triton_import": triton.__file__,
    }
    assert runtime["cutlass_file_sha256"] == config["cutlass_sha256"][args.phase]
    assert runtime["cutlass_support_sm120"][0] == (args.phase == "before")
    assert pathlib.Path(triton.__file__).is_relative_to(prefix)
    assert platform.python_version() == config["runtime_versions"]["python"]
    for key in ["torch", "vllm", "triton", "torch_cuda_build"]:
        assert runtime[key] == config["runtime_versions"][key], (key, runtime[key])
    runtime["source_and_environment_verified"] = True
    (run / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    print("RUNTIME " + json.dumps(runtime), flush=True)
    model_path = args.model_dir.resolve()
    assert model_path.is_dir()
    model_config = json.loads((model_path / "config.json").read_text())
    assert model_config["quantization_config"]["quant_method"] == "compressed-tensors"
    assert not model_config.get("auto_map")
    print("CONFIG " + json.dumps(config), flush=True)
    started = time.monotonic()
    stage = "engine_initialization"
    try:
        engine = dict(config["engine"])
        engine["download_dir"] = os.environ["HF_HUB_CACHE"]
        llm = LLM(model=str(model_path), tokenizer=str(model_path), **engine)
        loaded = time.monotonic()
        print("MODEL_LOADED", flush=True)
        stage = "read_only_model_audit"
        audits = llm.apply_model(inspect_quantized_layers)
        (run / "model_audit.json").write_text(json.dumps(audits, indent=2) + "\n")
        assert len(audits) == 1
        audit = audits[0]
        assert audit["quantized_layer_count"] == config["expected_quantized_layers"]
        assert audit["fused_layer_count"] == config["expected_fused_layers"]
        expected_kernel = "CutlassInt8ScaledMMLinearKernel" if args.phase == "before" else "TritonInt8ScaledMMLinearKernel"
        assert audit["kernel_counts"] == {expected_kernel: config["expected_quantized_layers"]}
        for layer in audit["layers"]:
            assert layer["weight_dtype"] == "torch.int8" and layer["weight_device"] == "cuda:0"
            assert layer["is_channelwise"] and layer["input_symmetric"] and not layer["is_static_input_scheme"]
        print("MODEL_AUDIT " + json.dumps({k: v for k, v in audit.items() if k != "layers"}), flush=True)
        stage = "generation"
        request_start = time.monotonic()
        outputs = llm.chat(config["messages"], SamplingParams(**config["sampling"]), use_tqdm=False)
        done = time.monotonic()
        (run / "request_output_repr.txt").write_text(repr(outputs) + "\n")
        assert len(outputs) == 1 and outputs[0].finished
        output = outputs[0]
        completion = output.outputs[0]
        (run / "generated_text.txt").write_text(completion.text)
        assert completion.text.strip(), "Empty generation"
        assert 0 < len(completion.token_ids) <= config["sampling"]["max_tokens"], "Invalid generated token count"
        result = {
            "status": "PASS_W8A8_MODEL_SMOKE" if args.phase == "after" else "UNEXPECTED_BEFORE_SUCCESS",
            "model_id": config["model_id"], "revision": config["revision"], "phase": args.phase,
            "rendered_prompt": output.prompt, "prompt_token_ids": output.prompt_token_ids,
            "generated_text": completion.text, "generated_token_ids": list(completion.token_ids),
            "generated_tokens": len(completion.token_ids), "finish_reason": completion.finish_reason,
            "stop_reason": completion.stop_reason, "kernel_counts": audit["kernel_counts"],
            "quantized_layers": audit["quantized_layer_count"], "fused_layers": audit["fused_layer_count"],
            "load_wall_seconds": loaded - started, "request_wall_seconds": done - request_start,
            "timing_note": "One cold smoke request; includes JIT and is not a performance benchmark.",
        }
        (run / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print("RESULT " + json.dumps(result), flush=True)
        assert args.phase == "after", "Reassess original issue: the unmodified model unexpectedly generated"
    except Exception as exc:
        error = {"stage": stage, "exception_type": type(exc).__name__, "exception": str(exc), "traceback": traceback.format_exc()}
        (run / "exception.json").write_text(json.dumps(error, indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
