"""Capture full layer-13 normalized/rotated Q/K, V and native BF16 O privately."""
import argparse,json,sys,time,gc
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import CASE,sha,write_json,utc


def main():
    p=argparse.ArgumentParser();p.add_argument('--split',choices=['dev','fresh'],required=True);p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--work-dir',type=Path,required=True);a=p.parse_args()
    from src.protocol import verify_phase,configure,gpu_guard
    spec,spec_sha=verify_phase('b' if a.split=='fresh' else 'a')
    configure();gpu_guard()
    import torch
    from transformers import AutoModel
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from src.anchor import Adapter
    torch.set_num_threads(4);torch.manual_seed(400113);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available():raise RuntimeError('GPU_NOT_AVAILABLE')
    if torch.cuda.mem_get_info()[0]<4*2**30:raise RuntimeError('GPU_BUSY')
    torch.cuda.set_per_process_memory_fraction(13*2**30/torch.cuda.get_device_properties(0).total_memory)
    manifest=json.loads((CASE/f'inputs/{a.split}_manifest.json').read_text());assert a.snapshot.name==manifest['revision']
    docs=[d for d in manifest['documents'] if d['split']==a.split]
    outdir=a.work_dir/'traces'/a.split;outdir.mkdir(parents=True,exist_ok=True)
    started=utc();load=time.perf_counter()
    model=AutoModel.from_pretrained(a.snapshot,dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True,trust_remote_code=False).eval().cuda()
    torch.cuda.synchronize();load_seconds=time.perf_counter()-load
    assert (model.config.num_attention_heads,model.config.num_key_value_heads,model.config.head_dim)==(16,8,128)
    original=ALL_ATTENTION_FUNCTIONS['sdpa'];state={};records=[];adapter=Adapter('default')
    def capture(module,query,key,value,attention_mask,scaling,dropout=0.,**kwargs):
        output,weights=original(module,query,key,value,attention_mask,scaling=scaling,dropout=dropout,**kwargs)
        if module.layer_idx!=13:return output,weights
        assert attention_mask is None and module.is_causal and module.sliding_window is None and dropout==0.
        n=query.shape[2];assert query.shape==(1,16,n,128) and key.shape==value.shape==(1,8,n,128)
        native=output.transpose(1,2).contiguous()
        replay=adapter(query,key,value,causal=True,scale=scaling)
        error=float((replay.float()-native.float()).norm()/native.float().norm())
        doc=state['doc'];status='VALID' if torch.equal(replay,native) else 'INVALID'
        meta=dict(document_id=doc['document_id'],split=a.split,language=doc['language'],length=n,layer=13,hq=16,hkv=8,dim=128,causal=True,scaling=scaling,dtype='bfloat16',layout='BHND',all_query_positions=True,query_count=n,input_sha256=doc['prefix_sha256'][str(n)],model_revision=manifest['revision'],spec_sha256=spec_sha,native_replay_relative_error=error,native_replay_bitwise_equal=bool(torch.equal(replay,native)),status=status)
        path=outdir/f"{doc['document_id']}-n{n}.pt";assert not path.exists(),'Refuse to overwrite trace'
        tensors={name:t.detach().cpu().contiguous() for name,t in [('q',query),('k',key),('v',value),('native',native)]}
        torch.save({**tensors,'meta':meta},path)
        records.append({**meta,'trace_file':path.name,'trace_sha256':sha(path),'bytes':path.stat().st_size})
        return output,weights
    ALL_ATTENTION_FUNCTIONS.register('sdpa',capture)
    try:
        with torch.inference_mode():
            for d in docs:
                state['doc']=d
                assert sha(CASE/d['text_file'])==d['text_sha256']
                for n in ([4096] if a.split=='dev' else [4096,512,2048]):
                    model(input_ids=torch.tensor([d['token_ids_4096'][:n]],device='cuda'),use_cache=False,return_dict=True)
                    torch.cuda.synchronize();print(json.dumps({'document_id':d['document_id'],'length':n,'status':records[-1]['status'],'native_error':records[-1]['native_replay_relative_error']}),flush=True)
    finally:ALL_ATTENTION_FUNCTIONS.register('sdpa',original)
    peak=torch.cuda.max_memory_allocated();del model;gc.collect();torch.cuda.empty_cache()
    write_json(CASE/f'provenance/{a.split}_capture.json',dict(started_at_utc=started,completed_at_utc=utc(),load_seconds=load_seconds,torch_peak_allocated_bytes=peak,model_revision=manifest['revision'],scope='Full Q/K/V and native BF16 outputs stored privately; no input fabrication or KV expansion; model process ends before operator timing',traces=records))
    assert len(records)==len(docs)*(1 if a.split=='dev' else 3)
if __name__=='__main__':main()
