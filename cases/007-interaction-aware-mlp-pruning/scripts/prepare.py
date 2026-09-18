"""CPU-only snapshot audit, exact inputs and local protocol freeze; never downloads."""
import argparse,sys,json,inspect,platform,hashlib,importlib.metadata as md
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.core import *
from src.data import scenario,render

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--snapshot',type=Path,required=True);a=p.parse_args();case=Path(__file__).resolve().parents[1]
 rev='c1899de289a04d12100db370d81485cdf75e47ca'
 if a.snapshot.name!=rev:raise ValueError('Revision mismatch')
 previous=load(case.parent/'006-attention-decision-stability/provenance/model.json');actual={}
 for f in previous['files']:
  h=file_hash(a.snapshot/f['name'])
  if h!=f['verified_sha256']:raise ValueError('Model file hash mismatch: '+f['name'])
  actual[f['name']]=h
 from transformers import AutoTokenizer
 from transformers.models.qwen3 import modeling_qwen3 as source
 import torch
 tok=AutoTokenizer.from_pretrained(a.snapshot,local_files_only=True,trust_remote_code=False)
 inputs=[]
 for split,n in {**COUNTS,'SMOKE':6}.items():
  rows=[render(scenario(split,task,i),tok) for task in TASKS for i in range(n//3)]
  write_new(case/f'inputs/{split}.json',rows);inputs.extend(rows)
 for key in ['id','facts_hash','text_hash','token_hash']:
  if len({str(r[key]) for r in inputs})!=len(inputs):raise ValueError('Split/content overlap '+key)
 cfg=load(a.snapshot/'config.json')
 architecture={'layers':cfg['num_hidden_layers'],'hidden_size':cfg['hidden_size'],'intermediate_size':cfg['intermediate_size'],'layer':13,'gate_proj':[3072,1024],'up_proj':[3072,1024],'down_proj':[1024,3072],'bias':False,'activation':cfg['hidden_act'],'expression':'down_proj(silu(gate_proj(x)) * up_proj(x))','input_boundary':'actual input to zero-based decoder layer 13 MLP, after post_attention_layernorm','source_sha256':file_hash(inspect.getfile(source)),'mlp_class_source_sha256':hashlib.sha256(inspect.getsource(source.Qwen3MLP).encode()).hexdigest()}
 write_new(case/'provenance/architecture.json',architecture)
 env={'python':platform.python_version(),'torch':torch.__version__,'cuda_runtime':torch.version.cuda,'transformers':md.version('transformers'),'numpy':md.version('numpy'),'triton':md.version('triton'),'kernel':platform.release(),'wsl':'microsoft' in platform.release().lower(),'source_sha256':architecture['source_sha256'],'sageattention':'installed but NOT_USED in Case007','model_revision':rev,'model_hashes':actual,'tokenizer_template_sha256':hashlib.sha256(tok.chat_template.encode()).hexdigest()}
 write_new(case/'provenance/environment.json',env)
 spec={'version':1,'kind':'LOCAL_PROTOCOL_FREEZE_BEFORE_CALIBRATION','frozen_at_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),'model':'Qwen/Qwen3-0.6B','model_revision':rev,'architecture':architecture,'groups':groups(3072),'budgets':[4,8],'random_sets':{str(b):random_sets(b) for b in [4,8]},'splits':COUNTS,'smoke':6,'length':512,'positions':positions(512),'seed':SEED,'bootstrap_repeats':5000,'bootstrap_seed':SEED,'quantile':'numpy linear','reference_rms_floor':1e-6,'statistic':'mean of per-prompt relative sliced BF16 output error; PAIRWISE minus INDEPENDENT; 192 HELD_OUT prompts, equally represented tasks','decision':'positive if both budgets 95% CI upper<0; negative if both lower>0; otherwise no clear transfer; no deployment/quality gate','objective':'FP32 c_i from represented BF16 native inputs/weights; per-token inner products accumulated CPU FP64, equal prompt/token weighting','selection':'CALIBRATION only; exhaustive, signed Q; lexicographic tie; same selected sets for all splits; no tuning on DEV or HELD_OUT','local_random':'20 frozen random sets per budget evaluated at native inputs for all splits; no random model-level runs','model_candidates':'B + INDEPENDENT/PAIRWISE at both budgets; HELD_OUT only','validity':{'bf16_sliced_masked_relative_frobenius_max':.01,'fp32_decomposition_relative_max':1e-5,'full_outputs_finite_required':True,'common_mlp_input_hash_required':True,'near_zero':'undefined; block outcome'},'timing':{'rounds':3,'prompts_per_task':2,'warmup':5,'blocks':5,'calls':5,'order':'seeded shuffled method order per paired block','boundary':['full 512-token MLP','prompt forward including last-position LM head, no cache'],'primary':'synchronized wall-clock block mean; paired B/candidate ratio median','CI':'resample task-stratified prompts and shared process IDs then paired blocks, 5000 reps','device_events':'separate corresponding block; secondary'},'posthoc':['split Q stability after primary results fixed'],'resource':{'max_allocated_GiB':13,'minimum_free_GiB':8,'downloads':False,'new_builds':0},'inputs':{s:file_hash(case/f'inputs/{s}.json') for s in [*COUNTS,'SMOKE']},'code_hashes':{str(f.relative_to(case)):file_hash(f) for d in ['src','scripts','tests'] for f in sorted((case/d).glob('*.py'))}}
 write_new(case/'configs/protocol.json',spec);(case/'configs/protocol.sha256').write_text(file_hash(case/'configs/protocol.json')+'\n')
 print(json.dumps({'inputs':len(inputs),'architecture':architecture,'protocol_sha256':file_hash(case/'configs/protocol.json')}))
if __name__=='__main__':main()
