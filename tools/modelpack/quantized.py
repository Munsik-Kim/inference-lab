"""One pinned GPTQ W4A16 recipe. This module runs only in the build environment."""
from pathlib import Path
import time
import shutil
from .common import CASE, checksums, digest, inventory, new_external, read, require, sha, write
from .artifact import TOKENIZER_FILES


def inspect_quantized(root: Path, expected: dict) -> dict:
    import torch
    from safetensors import safe_open
    tensors={}
    for path in sorted(root.glob('*.safetensors')):
        with safe_open(path,framework='pt',device='cpu') as f:
            for key in f.keys():
                require(key not in tensors,'Duplicate shard tensor')
                t=f.get_tensor(key)
                require(bool(torch.isfinite(t).all()),'Nonfinite stored quantized tensor')
                tensors[key]={'shape':list(t.shape),'dtype':str(t.dtype),'bytes':t.numel()*t.element_size()}
                if key.endswith('weight_scale'):require(bool((t>0).all()),'Invalid zero/negative scale')
    require(tensors,'No safetensors')
    for name,shape in expected.items():
        require(name+'.weight' not in tensors,'Dense target fallback in artifact')
        packed=tensors.get(name+'.weight_packed');scales=tensors.get(name+'.weight_scale')
        require(packed is not None and packed['dtype']=='torch.int32' and packed['shape']==[shape[0],shape[1]//8], 'Wrong packed shape/dtype')
        require(scales is not None and scales['shape']==[shape[0],shape[1]//128], 'Wrong group scale shape')
    cfg=read(root/'config.json');q=cfg['quantization_config']
    require(q['quant_method']=='compressed-tensors','Unexpected serialization backend')
    require(len([x for x in tensors if x.endswith('weight_packed')])==len(expected),'Target count mismatch')
    return {'status':'PASS','target_modules':expected,'tensor_inventory':tensors,'quantization_config':q,
            'safetensors_bytes':sum(f.stat().st_size for f in root.glob('*.safetensors'))}


def build(snapshot: Path, output: Path, smoke: bool = False) -> dict:
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM,AutoTokenizer,Qwen3Config,Qwen3ForCausalLM
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import GPTQModifier
    from compressed_tensors.quantization import QuantizationScheme,QuantizationArgs
    from .runtime import gpu_guard
    guard=gpu_guard(10 if not smoke else 4)
    cfg=read(CASE/'configs/protocol_draft.json')['Q']
    require(snapshot.name==cfg['revision'],'Wrong source revision')
    out=new_external(output)
    tok=AutoTokenizer.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False)
    if smoke:
        c=Qwen3Config(vocab_size=151936,hidden_size=256,intermediate_size=512,num_hidden_layers=2,
                      num_attention_heads=8,num_key_value_heads=2,head_dim=32,tie_word_embeddings=True)
        model=Qwen3ForCausalLM(c).to(dtype=torch.bfloat16,device='cuda').eval()
    else:
        model=AutoModelForCausalLM.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False,
                   dtype=torch.bfloat16,attn_implementation='sdpa').eval().cuda()
    expected={name:list(module.weight.shape) for name,module in model.named_modules()
              if isinstance(module,torch.nn.Linear) and name!='lm_head'}
    rows=read(CASE/'inputs/Q'/('smoke.json' if smoke else 'calibration.json'))
    loader=DataLoader([{'input_ids':torch.tensor(r['token_ids']),
                       'attention_mask':torch.ones(len(r['token_ids']),dtype=torch.long)} for r in rows],batch_size=1,shuffle=False)
    scheme=QuantizationScheme(targets=['Linear'],weights=QuantizationArgs(num_bits=4,type='int',
                                      symmetric=True,strategy='group',group_size=128))
    recipe=GPTQModifier(config_groups={'group_0':scheme},ignore=['lm_head'],block_size=128,
                         dampening_frac=.01,actorder=None,offload_hessians=True)
    write(out/'intended_recipe.json',{'method':'GPTQ','num_bits':4,'group_size':128,'symmetric':True,
                 'actorder':None,'dampening_frac':.01,'block_size':128,'offload_hessians':True,
                 'pipeline':'sequential','target_modules':expected,'input_hash':digest(rows),'fixture':smoke})
    start=time.perf_counter()
    oneshot(model=model,tokenizer=tok,dataset=loader,recipe=recipe,pipeline='sequential',
            sequential_targets=['Qwen3DecoderLayer'],sequential_offload_device='cpu',
            num_calibration_samples=len(rows),shuffle_calibration_samples=False,
            max_seq_length=cfg['max_length'],pad_to_max_length=False,enable_compile=False,
            output_dir=str(out/'checkpoint'),log_dir=str(out/'private-logs'))
    elapsed=time.perf_counter()-start
    # Original tokenizer bytes remain the continuation boundary for the serving process.
    for name in TOKENIZER_FILES:
        if (snapshot/name).is_file():shutil.copyfile(snapshot/name,out/'checkpoint'/name)
    inspected=inspect_quantized(out/'checkpoint',expected)
    report={**inspected,'build':'BUILD_COMPLETE','fixture':smoke,'guard':guard,'conversion_seconds':elapsed,
            'calibration_documents':len(rows),'calibration_tokens':sum(r['length'] for r in rows),
            'max_allocated':torch.cuda.max_memory_allocated(),'max_reserved':torch.cuda.max_memory_reserved(),
            'files':inventory(out/'checkpoint')}
    write(out/'build_report.json',report)
    checksums(out/'checkpoint')
    return report
