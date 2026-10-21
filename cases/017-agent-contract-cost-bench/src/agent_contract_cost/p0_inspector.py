"""Case017-only observer through the official worker extension interface.

Case016 TechnicalInspector is reused unchanged through explicit runtime
PYTHONPATH entries. All torch imports remain inside server-only methods.
"""


def validate_support(receipt, kv):
    if receipt.get('cuda_capability') != [12, 0] or len(receipt.get('layers', [])) != 36:
        raise AssertionError('Actual SM120 and all36 attention layers required')
    if receipt['cache_config']['enable_prefix_caching']:
        raise AssertionError('Prefix cache must be disabled')
    expected = 'torch.bfloat16' if kv == 'K0' else 'torch.uint8' if kv == 'K1' else None
    if expected is None or not receipt.get('cache_tensors') or any(t['dtype'] != expected for t in receipt['cache_tensors']):
        raise AssertionError('Actual KV tensor storage does not match registered condition')
    if receipt.get('structured_outputs_backend') != 'xgrammar':
        raise AssertionError('Registered xgrammar backend not active')
    if receipt.get('model_dtype') != 'torch.bfloat16':
        raise AssertionError('Weights must remain BF16')
    if kv == 'K1':
        if len(receipt.get('nvfp4_views', [])) != 36:
            raise AssertionError('All36 packed-cache views required')
        if not all(l['is_kvcache_nvfp4'] for l in receipt['layers']):
            raise AssertionError('NVFP4 implementation path absent')
        if any(v['scale_dtype'] != ['torch.float8_e4m3fn'] * 2 for v in receipt['nvfp4_views']):
            raise AssertionError('Actual FP8 block scales absent')
        if not all(v['data_scale_storage_shared'] for v in receipt['nvfp4_views']):
            raise AssertionError('Packed data/scale view storage identity differs')
    return True


def validate_schema_cpu(schema, snapshot):
    """Actual installed tokenizer/compiler smoke, never a model load/forward.

    This helper is intentionally optional/lazy. Its CPU compile success does not
    establish that the server applied a mask; P0 raw responses and config do.
    """
    import xgrammar as xgr
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    import json
    from pathlib import Path
    vocab_size = json.loads((Path(snapshot) / 'config.json').read_text())['vocab_size']
    info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=vocab_size)
    compiler = xgr.GrammarCompiler(info, max_threads=2, cache_enabled=False)
    compiled = compiler.compile_json_schema(schema)
    return {'status': 'CPU_COMPILED', 'grammar': str(compiled.grammar),
            'tokenizer_length': len(tokenizer), 'scope': 'actual tokenizer/xgrammar compile; server mask NOT_RUN'}


def __getattr__(name):
    # Importing CPU support validators must not require Case016 or torch.
    if name != 'ContractInspector':
        raise AttributeError(name)
    from nvfp4_technical.worker import TechnicalInspector

    class ContractInspector(TechnicalInspector):
        def inspect_case017(self):
            r = self.technical_inspect()
            r['structured_outputs_backend'] = self.vllm_config.structured_outputs_config.backend
            r['model_dtype'] = str(self.vllm_config.model_config.dtype)
            r['max_model_len'] = self.vllm_config.model_config.max_model_len
            r['scheduler'] = {k: getattr(self.vllm_config.scheduler_config, k, None)
                              for k in ('max_num_seqs', 'max_num_batched_tokens', 'enable_chunked_prefill', 'async_scheduling')}
            r['cudagraph_policy'] = str(self.vllm_config.compilation_config.cudagraph_mode)
            r['observer_scope'] = 'read-only cache/config/parameter sample; parameter sample is not full weight hash'
            return r

    ContractInspector.__module__ = __name__
    ContractInspector.__qualname__ = name
    globals()[name] = ContractInspector
    return ContractInspector
