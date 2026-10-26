"""P1 official CPU example and independent NumPy arithmetic checks."""
import argparse, pathlib, subprocess, sys, os, time
import numpy as np
from common import *
def run(out):
 import torch
 out=pathlib.Path(out);out.mkdir(parents=True,exist_ok=True);fixtures=ROOT/'inputs';fixtures.mkdir(exist_ok=True)
 env=os.environ.copy();env['PYTHONPATH']=str(ROOT/'vendor/optic/src');env['PYTHONDONTWRITEBYTECODE']='1';env['TMPDIR']=str(out)
 t=time.perf_counter();r=subprocess.run([sys.executable,'-B',str(ROOT/'vendor/optic/examples/decode_optic.py')],capture_output=True,text=True,env=env)
 (out/'official_example.stdout.txt').write_text(r.stdout);(out/'official_example.stderr.txt').write_text(r.stderr)
 dump(out/'official_example.receipt.json',{'exit_code':r.returncode,'wall_seconds':time.perf_counter()-t,'command':[sys.executable,'-B','vendor/optic/examples/decode_optic.py']})
 if r.returncode:raise RuntimeError('official CPU example failed')
 loader,_=upstream();rows=[];contracts=[];manifest=[]
 shape=dict(V=257,L=3,H=16,G=2,D=4,Z=8)
 with torch.inference_mode():
  for seed in [0,1,2]:
   meta,a=fixture(shape,seed)
   for control in ['random','bias_only','zero']:
    aa={k:v.copy() for k,v in a.items()}
    if control!='random':aa['decoder_weight_fp16'].fill(0)
    if control=='zero':aa['decoder_bias_fp16'].fill(0)
    path=fixtures/f'p1_seed{seed}_{control}.npz';save_fixture(path,meta,aa);mm,bb=read_fixture(path)
    assert all(np.array_equal(v,bb[k]) and v.tobytes()==bb[k].tobytes() for k,v in aa.items())
    manifest.append({'path':str(path.relative_to(ROOT)),**file_info(path),'metadata':mm,'arrays':array_inventory(bb),'integer_code_exact':True})
    ids_all=np.arange(257,dtype=np.int64);v=reference(mm,bb,ids_all);l=reference(mm,bb,ids_all,loop=True)
    np.testing.assert_allclose(v,l,rtol=1e-10,atol=1e-12)
    rows.append({'seed':seed,'control':control,'test':'fp64_loop_vector','pass':True,'max_abs':float(np.max(np.abs(v-l)))})
    dec=loader(path,device='cpu');table=dec(torch.from_numpy(ids_all));fs=functions(torch,dec,table)
    scenarios={'all':ids_all,'boundary':np.array([0,256],np.int64),'shuffled':np.random.default_rng(seed+99).permutation(ids_all),'repeated':np.array([256,0,7,7,256,1],np.int64),'single':np.array([256],np.int64),'multidimensional':np.array([[256,0],[4,4]],np.int64),'empty':np.array([],np.int64)}
    raw_outputs={}
    for scenario,ids in scenarios.items():
     validate_ids(ids,257);ref=reference(mm,bb,ids);outs={}
     for name,f in fs.items():
      x=f(torch.from_numpy(ids));assert x.dtype==torch.bfloat16;assert list(x.shape)==[*ids.shape,48]
      outs[name]=x.float().numpy().reshape(*ids.shape,3,16);m=metrics(outs[name],ref)
      rows.append({'seed':seed,'control':control,'test':scenario,'method':name,**m})
      if scenario=='all':raw_outputs[name]=x.view(torch.uint16).numpy().copy()
     for a_,b_ in [('A','B'),('B','C'),('A','C')]:rows.append({'seed':seed,'control':control,'test':scenario,'method':a_+'/'+b_,**metrics(outs[a_],outs[b_])})
    np.savez_compressed(out/f'cpu_seed{seed}_{control}_bf16_words.npz',**raw_outputs)
    if control=='random':
     for desc,ids in [('negative',np.array([-1],np.int64)),('out_of_range',np.array([257],np.int64)),('int32',np.array([0],np.int32)),('float',np.array([0.],np.float32))]:
      try:validate_ids(ids,257);raise AssertionError('adapter accepted invalid ID')
      except (IndexError,TypeError) as e:contracts.append({'seed':seed,'kind':desc,'adapter_rejected':True,'exception':type(e).__name__})
      if desc in ['negative','out_of_range']:
       try:dec(torch.from_numpy(ids));raise AssertionError('official CPU accepted invalid ID')
       except (IndexError,RuntimeError) as e:contracts.append({'seed':seed,'kind':desc,'official_rejected':True,'exception':type(e).__name__})
     raw=dec(torch.tensor([0],dtype=torch.int32));contracts.append({'seed':seed,'kind':'official_int32','behavior':'silently casts to long','dtype':str(raw.dtype)})
     try:dec(torch.tensor([0],device='meta'));contracts.append({'seed':seed,'kind':'device_mismatch_meta','behavior':'accepted meta index; outside adapter contract'})
     except Exception as e:contracts.append({'seed':seed,'kind':'device_mismatch_meta','behavior':'rejected','exception':type(e).__name__})
 result={'status':'PASS' if all(x['pass'] for x in rows) else 'BLOCKED_NUMERICS','rows':rows,'contracts':contracts,'empty_contract':'official B and common A/B/C return empty BF16 [0,L*H]; NumPy [0,L,H]','official_example_exit':r.returncode,'torch_version':torch.__version__}
 dump(out/'cpu_correctness.json',result);dump(ROOT/'provenance/fixture_manifest.json',manifest)
 print(result['status'],len(rows),'arithmetic records');return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',default=str(ROOT/'results/20261026-1215/p1'));a=p.parse_args();r=run(a.out);sys.exit(0 if r['status']=='PASS' else 1)
