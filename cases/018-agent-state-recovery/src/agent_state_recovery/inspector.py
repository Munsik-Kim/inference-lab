"""Owned-worker observers. No top-level Torch/model imports or upstream edits."""

from contextlib import contextmanager


@contextmanager
def technical_uncompiled_context(module):
    """Official forward-context flag for TECH only; always restore binding."""
    original = module.set_forward_context
    @contextmanager
    def observed(*args, **kwargs):
        kwargs["skip_compiled"] = True
        with original(*args, **kwargs) as value:
            yield value
    module.set_forward_context = observed
    try:
        yield
    finally:
        module.set_forward_context = original


@contextmanager
def attention_impl_observer(model, is_tensor, finite):
    """Observe the real attention implementation, restoring every instance attr."""
    layers = {id(m): (name, m) for name, m in model.named_modules()
              if hasattr(m, "kv_cache_dtype") and hasattr(m, "impl")}
    rows = {name: {"calls": 0, "finite": True, "shapes": [], "metadata_present_calls": 0,
                   "query_dtypes": []} for name, _ in layers.values()}
    restores, seen = [], set()
    for _, layer in layers.values():
        impl = layer.impl
        if id(impl) in seen:
            continue
        seen.add(id(impl))
        had_own, previous = "forward" in impl.__dict__, impl.__dict__.get("forward")
        original = impl.forward
        restores.append((impl, had_own, previous))
        def observed(*args, _original=original, **kwargs):
            result = _original(*args, **kwargs)
            actual_layer = args[0] if args else kwargs["layer"]
            name = layers[id(actual_layer)][0]
            row = rows[name]
            row["calls"] += 1
            metadata = kwargs.get("attn_metadata", args[5] if len(args) > 5 else None)
            row["metadata_present_calls"] += int(metadata is not None)
            query = kwargs.get("query", args[1] if len(args) > 1 else None)
            dtype = str(getattr(query, "dtype", "synthetic"))
            if dtype not in row["query_dtypes"]: row["query_dtypes"].append(dtype)
            output = kwargs.get("output", args[6] if len(args) > 6 else None)
            tensors = [t for t in (result, output) if is_tensor(t)]
            if not tensors:
                raise AssertionError("attention output tensor unobservable")
            for tensor in tensors:
                row["finite"] &= finite(tensor)
                shape = list(getattr(tensor, "shape", []))
                if shape not in row["shapes"]: row["shapes"].append(shape)
            return result
        impl.forward = observed
    try:
        yield rows
    finally:
        for impl, had_own, previous in restores:
            if had_own: impl.forward = previous
            else: del impl.__dict__["forward"]


def __getattr__(name):
    if name != "RecoveryInspector":
        raise AttributeError(name)
    from nvfp4_technical.worker import TechnicalInspector

    class RecoveryInspector(TechnicalInspector):
        def recovery_inspect(self):
            r = self.technical_inspect()
            c = self.vllm_config
            r.update(model_dtype=str(c.model_config.dtype), model_path=str(c.model_config.model),
                     tokenizer_path=str(c.model_config.tokenizer), max_model_len=c.model_config.max_model_len,
                     structured_outputs_backend=c.structured_outputs_config.backend,
                     cudagraph_policy=str(c.compilation_config.cudagraph_mode),
                     scheduler={key: getattr(c.scheduler_config, key, None) for key in
                                ("max_num_seqs", "max_num_batched_tokens", "enable_chunked_prefill", "async_scheduling")})
            return r

        def recovery_observer_start(self):
            import torch
            worker = self.model_runner.structured_outputs_worker
            if worker is None:
                raise RuntimeError("actual V2 structured output worker unavailable")
            self._recovery_mask_original = worker.apply_grammar_bitmask
            self._recovery_mask_rows = []
            self._recovery_mask_calls = 0
            original = self._recovery_mask_original
            def observed(logits, batch, request_ids, bitmask):
                self._recovery_mask_calls += 1
                inspect = len(self._recovery_mask_rows) < 6
                before = int(torch.isneginf(logits).sum().item()) if inspect else None
                result = original(logits, batch, request_ids, bitmask)
                if inspect:
                    self._recovery_mask_rows.append({"logits_shape": list(logits.shape), "negative_inf_before": before,
                        "negative_inf_after": int(torch.isneginf(logits).sum().item()),
                        "remaining_finite": int(torch.isfinite(logits).sum().item()),
                        "NaN_after": bool(torch.isnan(logits).any().item()),
                        "structured_request_ids": list(request_ids), "grammar_bitmask_shape": list(bitmask.shape)})
                return result
            worker.apply_grammar_bitmask = observed
            return {"status": "OBSERVING", "scope": "read-only wrapper around actual V2 structured_outputs_worker.apply_grammar_bitmask; invokes original unchanged"}

        def recovery_observer_state(self):
            return {"mask_calls": getattr(self, "_recovery_mask_calls", 0),
                    "mask_rows": getattr(self, "_recovery_mask_rows", []),
                    "model_parameters_modified": False}

        def recovery_direct_finite(self):
            import torch
            import vllm.v1.worker.gpu_model_runner as module
            from vllm.config.compilation import CUDAGraphMode
            rows = []
            for label, size, decode in (("prefill", 8, False), ("decode", 1, True)):
                model = self.model_runner.get_model()
                with attention_impl_observer(model, lambda t: isinstance(t, torch.Tensor), lambda t: bool(torch.isfinite(t).all().item())) as observed:
                    with torch.inference_mode(), technical_uncompiled_context(module):
                        outputs = self.model_runner._dummy_run(size, cudagraph_runtime_mode=CUDAGraphMode.NONE,
                            force_attention=True, uniform_decode=decode, allow_microbatching=False,
                            remove_lora=False, is_profile=True)
                        finite = all(bool(torch.isfinite(t).all().item()) for t in outputs if isinstance(t, torch.Tensor))
                        torch.cuda.synchronize()
                observation = {"rows": observed, "all36_finite": len(observed) == 36 and all(r["calls"] > 0 and r["metadata_present_calls"] > 0 and r["finite"] for r in observed.values()),
                    "scope": "Read-only wrappers of actual impl.forward outputs with non-null attention metadata. Explicit unpadded TECH forward (is_profile=True), graph NONE/skip_compiled for observation only. All bindings restored; main serving GRAPH unchanged. Not generation API requests."}
                rows.append({"path": label, "num_tokens": size, "finite_returned_hidden_states": finite, **observation})
            return {"rows": rows, "direct_model_forward_probes": 2}

    RecoveryInspector.__module__ = __name__
    RecoveryInspector.__qualname__ = name
    globals()[name] = RecoveryInspector
    return RecoveryInspector
