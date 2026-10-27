"""Input construction and two explicit freezes, no GPU imports."""
import argparse
from pathlib import Path
import numpy as np
from common import read,write,identity,array_hash,freeze
from fixtures import structured
from numerics import bfbits32

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--phase',choices=['plan','finalize'],required=True);a=p.parse_args();r=a.root;loc=read(r/'references/locations.json');case=Path(loc['original_case'])
    external=read(r/'references/external_inputs.json');ext={x['path']:{k:x[k] for k in ['bytes','sha256']} for x in external['artifacts']}
    ext.update(external['small_model_files']);ext.update(external['existing_runtime_sources'])
    for name in ['source/common.py','source/layout.py','source/runtime.py','PUBLIC_MANIFEST.json','inputs/manifest.json']:ext[str(case/name)]=identity(case/name)
    if a.phase=='plan':
        folder=r/'inputs/structured';folder.mkdir(exist_ok=False);rows=[]
        for name,kind,z,n,detail in structured():
            path=folder/(name+'.npy');bits=bfbits32(z);np.save(path,bits,allow_pickle=False)
            rows.append({'fixture':name,'kind':kind,'layers':[0,25,27],'path':str(path),'identity':identity(path),'array_sha256':array_hash(bits),'encoding':'BF16_BITS_UINT16','shape':[512,1024],'valid_rows':n,'norm64':float(np.linalg.norm(z.astype(np.float64))),'min':float(z.min()),'max':float(z.max()),'construction':detail})
        write(r/'inputs/structured_manifest.json',{'inputs':rows,'same_I_shares_exact_bytes':True})
        rng=np.random.default_rng(np.random.SeedSequence([61027026,0]));workers=[]
        for block in [0,1,2]:
            for layer in rng.permutation([0,25,27]).tolist():workers.append({'layer':layer,'block':block})
        write(r/'protocol_C.json',{'scope':'POSTHOC; projections fixed from historical probe and existing selected layers','projection_paths':[f'model.layers.{i}.self_attn.q_proj' for i in [0,25,27]],'M':512,'I':1024,'O':2048,'fixtures_per_projection':15,'valid_rows_per_projection_per_block':7680,'first_block_full_retention':0,'operator_workers':workers,'method_order':'Alternating D1/P by fixture and block, layer-specific seeded starting parity','method_seed':61027026,'structured_seed':61027024,'CPU_oracle_row_chunk':64,'scalar_audit_rows':[0,1,255,511],'scalar_audit_columns':[0,1,127,1023,2047],'oracle_scalar_gate':{'atol':1e-12,'rtol':1e-10},'two_oracles':['exact signed code * actual BF16 scale','actual D1 BF16 reconstructed weight promoted to FP64'],'weight_storage':'No transformed weight matrix retained','preflight_per_first_block':{'module_core_calls':4,'profile_P_calls':1,'uses_first_fixed_DEV_document':True},'main_core_calls_planned':270,'control_calls_planned':12,'profile_calls_planned':3,'full_model_input_extraction':0,'native_input_norm_calls_planned':24,'allocated_peak_cap_bytes':2**31,'output_max_mismatch_coordinates':256,'observed_before_freeze':'Original Case021 and P4B tables known; P4A first attempt timed out. No new projection oracles/output observed.'})
        paths=list((r/'source').glob('*.py'))+list((r/'tests').glob('*.py'))+list(folder.glob('*.npy'))+[r/'protocol.json',r/'protocol_C.json',r/'inputs/structured_manifest.json']+list((r/'references').glob('*.json'))
        write(r/'records/freeze_C_preparation.json',freeze(r,paths,ext))
        print('C preparation source/config/structured inputs frozen')
    else:
        prep=read(r/'records/gpu-inputs/data/input_preparation.json');rows=read(r/'inputs/structured_manifest.json')['inputs']
        if prep['full_model_forwards']!=0 or len(prep['rows'])!=24:raise ValueError('INPUT_PREPARATION_COVERAGE')
        rows=rows+[{**x,'layers':[x['layer']]} for x in prep['rows']]
        orders={}
        for layer in [0,25,27]:
            initial=int(np.random.default_rng(np.random.SeedSequence([61027026,layer])).integers(0,2))
            selected=[x for x in rows if layer in x['layers']]
            for block in [0,1,2]:
                for index,row in enumerate(selected):orders[f'{layer}:{block}:{row["fixture"]}']=['D1','P'] if (initial+block+index)%2==0 else ['P','D1']
        write(r/'inputs/projection_input_manifest.json',{'inputs':rows,'method_orders':orders,'original_DEV_seen_before_followup':True,'input_preparation_receipt':identity(r/'records/gpu-inputs/data/input_preparation.json'),'actual_Z_from':'Same stored D0_R0 block input -> installed native input norm -> beta once -> BF16','fresh_full_forward_count':0})
        prior=read(r/'records/freeze_C_preparation.json');paths=[r/x for x in prior['files']]+[r/'inputs/projection_input_manifest.json',r/'records/gpu-inputs/data/input_preparation.json']+list((r/'inputs/actual').glob('*.npy'))
        write(r/'records/freeze_C_main.json',freeze(r,paths,prior['external']))
        print('C main source/config/exact actual Z/method order frozen before projection output')

if __name__=='__main__':main()
