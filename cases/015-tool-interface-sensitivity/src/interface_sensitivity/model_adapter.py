"""Explicit GPU boundary: verify all saved files, never fit or encode."""
import gc
import importlib
from pathlib import Path
import sys
import time
from .common import CASE, atomic, digest, read, require, sha, token_hash
from .runner import ObservationError


def forbid(*args, **kwargs):
    raise RuntimeError('FORBIDDEN: solve/encode/ENUM/enhanced-feedback/full sweep in Case015')


def verify_saved_artifacts(runtime, case=CASE):
    """CPU/hash only; complete coverage before any deserialize or model loading."""
    identities = read(Path(case)/'provenance/artifact_identity.json')
    checked = {}
    for method in ('RTN','D16'):
        directory = Path(runtime['artifact_root'])/method; identity = identities[method]
        require(sha(directory/'manifest.json') == identity['manifest_sha256'], 'artifact manifest identity mismatch')
        entries = identity['per_projection_identity']
        require(len(entries)==252 and len({e['layer'] for e in entries})==252, '252 distinct projection files required')
        require({p.stem for p in directory.glob('*.rpi4')} == {e['layer'] for e in entries}, 'missing/extra saved artifact')
        for entry in entries:
            p = directory/(entry['layer']+'.rpi4')
            require(p.stat().st_size==entry['bytes'] and sha(p)==entry['sha256'], 'saved artifact hash/bytes mismatch: '+entry['layer'])
        checked[method]={'projection_count':len(entries),'payload_bytes':sum(e['bytes'] for e in entries),'manifest_sha256':identity['manifest_sha256'],'all_files_sha256_checked':True}
    return checked


class Adapter:
    def __init__(self, case, runtime, method, output, worker_deadline):
        # No GPU imports occur on CLI --help, validate, oracle-check or audit.
        import torch
        import transformers
        from transformers import LogitsProcessor
        self.torch=torch; self.case=Path(case); self.method=method; self.deadline=worker_deadline
        source=self.case/'src/interface_sensitivity/upstream_case013'
        hashes=read(self.case/'provenance/reuse.json')['sources']
        for n,e in hashes.items():
            require(sha(source/n)==e['sha256'], 'pinned reference source changed')
        sys.path.insert(0,str(source))
        modules={n:importlib.import_module(n) for n in ('reference_int4','paper_int4','paper_model_runner')}
        for n,m in modules.items():
            require(Path(m.__file__).resolve()==source/(n+'.py'), 'wrong imported reference module')
        reference,quant,original=(modules[n] for n in ('reference_int4','paper_int4','paper_model_runner'))
        for mod,names in [(reference,['quantize_weight','quantize_element_int4','compare_methods','pack_nibbles']),
                          (quant,['solve','serialize_fast']), (original,['solve','serialize_fast'])]:
            for name in names:
                setattr(mod,name,forbid)
        reference.QuantizedWeight.serialize=forbid
        class ReadOnlyWorker(original.Worker):
            apply_artifacts=forbid
            run_all=forbid
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
        t=time.monotonic()
        self.worker=ReadOnlyWorker(Path(runtime['original_result_root']),'4b',method,output_override=output)
        w=self.worker
        torch.manual_seed(610132)
        self.tokenizer=w.tokenizer
        require(w.identity['id']=='Qwen/Qwen3-4B-Instruct-2507' and w.identity['revision']=='cdbee75f17c01a7cc42f958dc650907174af0554','model identity mismatch')
        names=[n for n,m in w.model.named_modules() if isinstance(m,torch.nn.Linear) and n.split('.')[-1] in quant.PROJECTIONS]
        require(len(names)==252,'projection coverage mismatch')
        loaded=[]
        if method!='BF16':
            metadata=read(self.case/'provenance/artifact_identity.json')[method]
            entries={e['layer']:e for e in metadata['per_projection_identity']}
            require(set(names)==set(entries),'saved/model module set mismatch')
            for name in names:
                if time.monotonic()>=self.deadline:
                    raise ObservationError('WORKER_BUDGET','worker deadline during saved artifact loading')
                data=(Path(runtime['artifact_root'])/method/(name+'.rpi4')).read_bytes()
                import hashlib
                require(hashlib.sha256(data).hexdigest()==entries[name]['sha256'],'artifact changed after preflight')
                q=reference.QuantizedWeight.deserialize(data)
                layer=w.model.get_submodule(name)
                require(list(q.shape)==list(layer.weight.shape)==entries[name]['shape'] and q.dtype==torch.float32,'shape or saved scale dtype mismatch')
                require(q.method==('rtn' if method=='RTN' else 'row_column'),'wrong saved quantizer method')
                quant.replace_projection(w.model,name,q,False)
                require(not w.model.get_submodule(name).activation4,'D4 disabled')
                loaded.append({'layer':name,'sha256':entries[name]['sha256'],'shape':entries[name]['shape']})
                del q,data
        for n,p in w.model.named_parameters():
            if n in w.original_small_hashes:
                import hashlib
                require(hashlib.sha256(p.detach().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()==w.original_small_hashes[n],'original norm changed')
        require((w.model.model.embed_tokens.weight.data_ptr()==w.model.lm_head.weight.data_ptr())==w.base_shared_embedding,'embedding/head tie changed')
        gc.collect();w.model.cuda();torch.cuda.synchronize()
        self.model=w.model; self.load_seconds=time.monotonic()-t
        self.parameter_identity=[(n,id(p),p._version) for n,p in w.model.named_parameters()]
        outer=self
        class FiniteBudget(LogitsProcessor):
            def __call__(self,ids,scores):
                if time.monotonic()>=outer.deadline:
                    raise ObservationError('WORKER_BUDGET','cumulative worker wall budget')
                if time.monotonic()>=outer.episode_deadline:
                    raise ObservationError('TIMEOUT','episode deadline')
                if not bool(torch.isfinite(scores).all()):
                    raise ObservationError('INFRA_ERROR','nonfinite unmasked logits')
                return scores
        self.processor=FiniteBudget()
        atomic(Path(output)/'artifact_readback.json',{'status':'PASS','method':method,'loaded':loaded,
               'new_solver_calls':0,'new_encode_calls':0,'artifact_writes':0,'guards_installed':True,
               'reference_imports':{n:'src/interface_sensitivity/upstream_case013/'+n+'.py' for n in modules}})
        atomic(Path(output)/'environment.json',{'python':sys.version,'torch':torch.__version__,'transformers':transformers.__version__,
            'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),'capability':list(torch.cuda.get_device_capability()),
            'TF32':torch.backends.cuda.matmul.allow_tf32,'BF16_reduced_reduction':torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            'FP16_reduced_reduction':torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,'attention':'sdpa','threads':torch.get_num_threads(),
            'model_generation_config':w.model.generation_config.to_dict(),'model_config':w.model.config.to_dict(),
            'execution_boundary':'saved INT4 -> dense BF16; FP32 input beta transport; no packed serving',
            'load_and_decode_seconds':self.load_seconds,'device_free_total_at_load':list(torch.cuda.mem_get_info())})

    def generate(self,messages,tools,deadline):
        torch=self.torch; self.episode_deadline=deadline
        try:
            ids=self.tokenizer.apply_chat_template(messages,tools=tools,tokenize=True,return_dict=False,add_generation_prompt=True,enable_thinking=False)
            if len(ids)+192>24576:
                raise ObservationError('CONTEXT_LIMIT','no automatic truncation')
            if time.monotonic()>=self.deadline:
                raise ObservationError('WORKER_BUDGET','cumulative worker deadline')
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.monotonic()
            with torch.inference_mode():
                x=torch.tensor(ids,device='cuda').unsqueeze(0)
                output=self.model.generate(x,attention_mask=torch.ones_like(x),do_sample=False,max_new_tokens=192,use_cache=True,
                    logits_to_keep=1,pad_token_id=self.tokenizer.eos_token_id,return_dict_in_generate=False,logits_processor=[self.processor])
                torch.cuda.synchronize();new=output[0,len(ids):].cpu().tolist()
            require(self.parameter_identity==[(n,id(p),p._version) for n,p in self.model.named_parameters()],'parameters changed between episodes')
            raw=self.tokenizer.decode(new,skip_special_tokens=False)
            return {'text':self.tokenizer.decode(new,skip_special_tokens=True),'raw_text':raw,'raw_utf8_sha256':__import__('hashlib').sha256(raw.encode()).hexdigest(),
                    'output_token_ids':new,'input_token_sha256':token_hash(ids),'input_tokens':len(ids),'output_tokens':len(new),
                    'messages_tools_hash':digest({'messages':messages,'tools':tools}), 'seconds':time.monotonic()-start,
                    'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
                    'device_free_total_after_request':list(torch.cuda.mem_get_info()),'enum_applied':False,'finite_unmasked_logits':True}
        except torch.cuda.OutOfMemoryError as e:
            raise ObservationError('OOM',str(e)) from e
        except ObservationError:
            raise
        except Exception as e:
            raise ObservationError('INFRA_ERROR',type(e).__name__+': '+str(e)) from e
