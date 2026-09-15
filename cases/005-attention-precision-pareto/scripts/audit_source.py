"""Read the pinned installed source, binaries and inherited DEV trace manifest."""
import argparse,json,sys,subprocess,inspect
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import CASE,sha,write_json,utc,fingerprint
from src.runtime import environment

def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--case004',type=Path,required=True);p.add_argument('--traces',type=Path,required=True);p.add_argument('--private-output',type=Path,required=True);a=p.parse_args()
 import torch,sageattention,sageattention.core as core
 from transformers.models.qwen3 import modeling_qwen3
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
 torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False;torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
 env=environment();env['reference_settings']={n:getattr(torch.backends.cuda.matmul,n) for n in ['allow_tf32','allow_fp16_reduced_precision_reduction','allow_bf16_reduced_precision_reduction']};env['float32_matmul_precision']=torch.get_float32_matmul_precision()
 old=json.loads((a.case004/'configs/experiment_spec.json').read_text())['environment']
 for key in ['sage_binary_sha256','sage_core_sha256','qwen_modeling_sha256','torch_runtime','cuda_build','gpu','sm','driver']:assert env[key]==old[key],key
 commit=subprocess.check_output(['git','-C',str(a.source),'rev-parse','HEAD'],text=True).strip();assert commit=='d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5'
 source_status=subprocess.check_output(['git','-C',str(a.source),'status','--porcelain'],text=True).strip();assert not source_status,source_status
 installed=Path(core.__file__);assert sha(installed)==sha(a.source/'sageattention/core.py')==env['sage_core_sha256']
 hashes={}
 for name in ['sageattention/core.py','sageattention/quant.py','sageattention/sm80_compile.py','sageattention/sm89_compile.py','csrc/mma.cuh','setup.py']:
  hashes[name]=sha(a.source/name)
 fn=inspect.getsource(modeling_qwen3.Qwen3Attention.forward)
 assert fn.index('self.q_norm')<fn.index('apply_rotary_pos_emb')<fn.index('attention_interface(')
 captures=json.loads((CASE/'provenance/dev_capture.json').read_text())['traces'];m=json.loads((CASE/'inputs/dev_manifest.json').read_text())
 for t in captures:
  d=next(d for d in m['documents'] if d['document_id']==t['document_id'])
  assert t['all_query_positions'] and t['query_count']==4096 and t['layer']==13 and t['model_revision']==m['revision']
  assert t['input_sha256']==d['prefix_sha256']['4096'];assert sha(a.traces/t['trace_file'])==t['trace_sha256']
  assert t['native_replay_bitwise_equal'] and t['status']=='VALID'
 private={'python':sys.executable,'torch':torch.__file__,'sage':str(installed),'sage_source':str(a.source),'trace_source':str(a.traces),'case004':str(a.case004),'worktree':str(CASE.parents[1])}
 write_json(a.private_output,private)
 out=dict(checked_at_utc=utc(),environment=env,environment_fingerprint=fingerprint(env),sage_commit=commit,source_files=hashes,source_clean=True,installed_core_matches_pinned_source=True,case004_binary_and_capture_source_match=True,verified_dev_full_traces=len(captures),layer_indexing='zero-based module.layer_idx=13',source_checks={'dispatch':'core.py 152-153: SM120 hardcodes per_warp, fp32+fp16; **kwargs not forwarded','fp8_bundle':'805-809: 2.25 for fp32+fp16; 448.0 otherwise','ignored_smooth_v':'797-803 FP8 fp32+fp32/fp32+fp16; 608-610 FP16 fp32/fp16+fp32','v4':'612-614: V to FP16 inside public call; sm80_compile qk_int8_sv_f16_accum_f32_attn','qk':'per_warp; query warp32, query tile128, key block64; K centering unchanged','accumulation_limit':'Option and source path names are not proof of effective accumulator bits on SM120'},source_url='https://github.com/thu-ml/SageAttention/blob/'+commit+'/sageattention/core.py',new_installations=0,new_builds=0)
 write_json(CASE/'provenance/source_audit.json',out);print(json.dumps({k:out[k] for k in ['environment_fingerprint','verified_dev_full_traces','installed_core_matches_pinned_source']}))
if __name__=='__main__':main()
