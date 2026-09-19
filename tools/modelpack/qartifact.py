"""Finalize the standard W4 checkpoint without changing any stored tensor bytes."""
from pathlib import Path
import shutil
from .common import read,write,require,checksums,verify_checksums,inventory,sha,new_external
from .quantized import inspect_quantized


def inspect_standard(path: Path, expected: dict) -> dict:
    from safetensors import safe_open
    cfg=read(path/'config.json');n=cfg['num_hidden_layers'];hidden=cfg['hidden_size'];head=cfg['head_dim']
    result=inspect_quantized(path,expected);tensors=result['tensor_inventory'];names=set()
    for name,shape in expected.items():
        for suffix in ['weight_packed','weight_scale','weight_shape']:names.add(name+'.'+suffix)
    exclusions={'model.embed_tokens.weight':[cfg['vocab_size'],hidden],'model.norm.weight':[hidden]}
    if not cfg.get('tie_word_embeddings'):exclusions['lm_head.weight']=[cfg['vocab_size'],hidden]
    for i in range(n):
        for s in ['input_layernorm','post_attention_layernorm']:exclusions[f'model.layers.{i}.{s}.weight']=[hidden]
        for s in ['q_norm','k_norm']:exclusions[f'model.layers.{i}.self_attn.{s}.weight']=[head]
    names.update(exclusions)
    require(set(tensors)==names,'Complete Q checkpoint key set mismatch')
    for name,shape in exclusions.items():require(tensors[name]['shape']==shape and tensors[name]['dtype']=='torch.bfloat16','Wrong excluded native parameter')
    for f in path.glob('*.safetensors'):
        with safe_open(f,framework='pt',device='cpu') as stream:
            for name in stream.keys():
                if name.endswith('.weight_shape'):
                    require(stream.get_tensor(name).tolist()==expected[name[:-13]],'Packed original shape mismatch')
    index=path/'model.safetensors.index.json'
    if index.exists():
        data=read(index);require(set(data['weight_map'])==names,'Shard index key mismatch')
        require(set(data['weight_map'].values())=={p.name for p in path.glob('*.safetensors')},'Shard file mismatch')
    result['excluded_native_modules']=exclusions;return result


def finalize(build: Path,output: Path):
    report=read(build/'build_report.json');require(report['build']=='BUILD_COMPLETE','Incomplete conversion')
    old=verify_checksums(build/'checkpoint');new_external(output);shutil.copytree(build/'checkpoint',output,dirs_exist_ok=True)
    require(verify_checksums(output)==old,'Copied checkpoint differs')
    checked=inspect_standard(output,report['target_modules'])
    (output/'checksums.sha256').unlink()
    manifest={'schema':1,'artifact_type':'quantized_standard','build':'BUILD_COMPLETE','reload':'NOT_RUN',
      'source':{'model':'Qwen/Qwen3-4B-Instruct-2507','revision':'cdbee75f17c01a7cc42f958dc650907174af0554'},
      'recipe':read(build/'intended_recipe.json'),'tensor_inventory':checked['tensor_inventory'],
      'excluded_native_modules':checked['excluded_native_modules'],'runtime':'vLLM 0.29.0 / compressed-tensors 0.17.0',
      'original_build_report_sha256':sha(build/'build_report.json'),'payload_files':inventory(output)}
    write(output/'artifact.json',manifest);checksums(output)
    return manifest


def inspect(path: Path):
    verify_checksums(path);m=read(path/'artifact.json')
    require(m['schema']==1 and m['artifact_type']=='quantized_standard','Wrong Q artifact')
    require(m['payload_files']==inventory(path,('artifact.json','checksums.sha256')),'Q payload mismatch')
    report=inspect_standard(path,m['recipe']['target_modules'])
    require(report['tensor_inventory']==m['tensor_inventory'],'Q tensor metadata mismatch')
    return {'status':'PASS','type':'quantized_standard','manifest_sha256':sha(path/'artifact.json'),
            'safetensors_bytes':report['safetensors_bytes'],'target_count':len(m['recipe']['target_modules'])}
