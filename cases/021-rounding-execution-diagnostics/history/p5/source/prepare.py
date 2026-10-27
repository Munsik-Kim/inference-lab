"""Bounded CPU selection before new GPU outputs; freeze both B and conditional C."""
import argparse,importlib.util,sys
from pathlib import Path
import numpy as np
from common import read,write,identity,csvwrite,array_hash
from arithmetic_v2 import decode,round_bits,candidates,exact_candidates,finite,onehot,nested

def load_artifact(r,layer):
    loc=read(r/'references/locations.json');case=Path(loc['original_case']);p4=Path(loc['p4_root']);e=read(p4/'references/external_inputs.json')
    artifact=next(x for x in e['artifacts'] if x['projection']==f'model.layers.{layer}.self_attn.q_proj');assert identity(artifact['path'])=={k:artifact[k] for k in ['bytes','sha256']}
    spec=importlib.util.spec_from_file_location('p5_rpi_schema',case/'source/layout.py');mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod);q=mod.load_rpi4(artifact['path']);assert q.shape==(2048,1024)
    scale=round_bits(q.alpha_inv.astype(np.float32)*q.gamma.astype(np.float32));s=decode(scale).astype(np.float64)
    previous=read(p4/f'records/gpu-q{layer}-b0/data/operator_receipt.json')['contract'];assert array_hash(scale.T.copy())==previous['actual_unpermuted_scale_array_sha256'];assert array_hash(q.codes)==previous['codes_array_sha256'];assert previous['bias']=='NONE'
    wd=round_bits(q.codes.astype(np.float64).reshape(2048,16,64)*s[...,None]).reshape(2048,1024);assert array_hash(wd)==previous['actual_D1_weight_bits_sha256']
    return q,s,artifact,previous
def stat(z):
    x=decode(z).astype(np.float64);nz=x[x!=0];e=((z&0x7f80)>>7).astype(int)
    return {'norm64':float(np.linalg.norm(x)),'min':float(x.min()),'max':float(x.max()),'nonzero_elements':int(nz.size),'nonzero_per_row_min':int(np.count_nonzero(x,axis=1).min()),'nonzero_per_row_max':int(np.count_nonzero(x,axis=1).max()),'nonzero_exponent_min':None if not nz.size else int(e[x!=0].min()-127),'nonzero_exponent_max':None if not nz.size else int(e[x!=0].max()-127)}
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root;proto=read(r/'P5_PROTOCOL.json');loc=read(r/'references/locations.json');p4=Path(loc['p4_root']);inputs=read(p4/'inputs/projection_input_manifest.json')['inputs'];dev=sorted({x['document_id'] for x in inputs if x['kind']=='ACTUAL_DEV'})[:2];assert len(dev)==2
    target=r/'private_inputs/prepared';target.mkdir(exist_ok=False);fixtures=[];pools=[];audit=[];contracts=[];exclusions=[]
    for layer in [0,25]:
        q,s,artifact,old=load_artifact(r,layer);contracts.append({'layer':layer,'artifact':artifact,'P4_contract':old});actual=[];seen=set();zsrc={}
        for di,doc in enumerate(dev):
            src=next(x for x in inputs if x['kind']=='ACTUAL_DEV' and x['document_id']==doc and layer in x['layers']);assert identity(src['path'])==src['identity'];z=np.load(src['path'],allow_pickle=False);zsrc[doc]=(src,z)
            rng=np.random.default_rng(np.random.SeedSequence([61027027,layer,di,1]));coords=rng.choice(512*1024,1024,replace=False)
            for flat in coords:
                token,axis=divmod(int(flat),1024);bits=int(z[token,axis]);key=(axis,bits)
                if not bool(finite(np.array(bits,np.uint16))) or bits&0x7fff==0 or key in seen:continue
                seen.add(key);actual.append({'origin':'ACTUAL_DEV','document_id':doc,'token':token,'axis':axis,'coefficient_bits':bits,'source_Z_sha256':src['identity']['sha256']})
        fixed=[];values=[1,-1,.125,-.125,1.125,-1.125,1.5,-1.5,1.75,-1.75]
        for i in range(2048):
            axis=i%1024;value=values[(axis+i//1024)%10];fixed.append({'origin':'FIXED','axis':axis,'coefficient_bits':int(round_bits(value)),'fixed_value':value})
        pool=actual+fixed;assert len(pool)<=4096;challenge=[]
        for index,row in enumerate(pool):
            axis=row['axis'];aa=float(decode(np.array(row['coefficient_bits'],np.uint16)));A,B=candidates(aa,q.codes[:,axis],s[:,axis//64]);valid=finite(A)&finite(B);different=valid&(A!=B)
            item={'layer':layer,'pool_index':index,**row,'A_B_different_outputs':int(different.sum()),'finite_outputs':int(valid.sum()),'first_discriminating_output':None if not different.any() else int(np.flatnonzero(different)[0])};pools.append(item)
            if not valid.all():exclusions.append({'layer':layer,'pool_index':index,'reason':'NONFINITE_CANDIDATE_OUTPUT'});continue
            if different.any() and len(challenge)<512:challenge.append({**item,'selected_output':int(np.flatnonzero(different)[0])})
        if len(actual)<512:raise ValueError('INSUFFICIENT_ACTUAL_COEFFICIENTS_NO_INPUT_EXPANSION')
        rowspec=[]
        for label,vals in [('POWER1',[1,-1]),('POWER_EIGHTH',[.125,-.125]),('FIXED_NONPOWER',[1.125,-1.125,1.5,-1.5,1.75,-1.75])]:
            rowspec.append((label,[{'origin':'FIXED','axis':(i*13+7)%1024,'coefficient_bits':int(round_bits(vals[i%len(vals)])),'fixed_value':vals[i%len(vals)]} for i in range(512)]))
        rowspec.append(('ACTUAL_ONE',actual[:512]))
        if challenge:rowspec.append(('CHALLENGE',challenge))
        for name,selection in rowspec:
            z=onehot(selection);f=target/f'B_l{layer}_{name}.npy';np.save(f,z,allow_pickle=False);fixtures.append({'phase':'B','layer':layer,'fixture':name,'path':str(f),'identity':identity(f),'array_sha256':array_hash(z),'shape':[512,1024],'valid_rows':len(selection),'zero_control_rows':512-len(selection),'rows':selection,**stat(z)})
        # Conditional C inputs are committed to before B outcomes exist.
        for di,doc in enumerate(dev):
            src,z=zsrc[doc]
            for K,masked,axes in nested(z,layer,di):
                f=target/f'C_l{layer}_{doc}_K{K}.npy'
                if K==1024:dest=Path(src['path'])
                else:np.save(f,masked,allow_pickle=False);dest=f
                fixtures.append({'phase':'C','layer':layer,'fixture':doc+f'_K{K}','document_id':doc,'document_index':di,'K':K,'path':str(dest),'identity':identity(dest),'array_sha256':array_hash(masked),'mask_axes_sha256':array_hash(axes),'original_Z_path':src['path'],'original_Z_identity':src['identity'],'shape':[512,1024],'valid_rows':512,'zero_control_rows':0,**stat(masked)})
        rng=np.random.default_rng(np.random.SeedSequence([61027028,layer]));selected=[(int(i),int(o)) for i,o in zip(rng.integers(0,len(pool),128),rng.integers(0,2048,128))]
        selected += [(x['pool_index'],x['selected_output']) for x in challenge[:64]]
        for i,o in selected:
            x=pool[i];aa=float(decode(np.array(x['coefficient_bits'],np.uint16)));axis=x['axis'];code=int(q.codes[o,axis]);scale=float(s[o,axis//64]);v=tuple(int(v) for v in candidates(aa,code,scale));exact=exact_candidates(aa,code,scale);assert v==exact,'EXACT_CANDIDATE_AUDIT';audit.append({'layer':layer,'pool_index':i,'output_axis':o,'A':v[0],'B':v[1],'exact_A':exact[0],'exact_B':exact[1]})
    orders={};workers={}
    for phase in ['B','C']:
        workers[phase]=[]
        for block in range(3):
            order=np.random.default_rng(np.random.SeedSequence([61027029,ord(phase),block])).permutation([0,25]).tolist()
            for layer in order:workers[phase].append({'phase':phase,'layer':layer,'block':block})
        for f in [x for x in fixtures if x['phase']==phase]:
            for block in range(3):
                parity=(block+len(orders))%2;orders[f"{phase}:{f['layer']}:{block}:{f['fixture']}"]=['D1','P'] if parity==0 else ['P','D1']
    write(r/'inputs_manifest/prepared.json',{'DEV_documents_sorted_first_two':dev,'fixtures':fixtures,'method_orders':orders,'worker_orders':workers,'conditional_C':'Only branch D after B audit; prepared does not mean GPU executed','contracts':contracts,'exclusions':exclusions})
    csvwrite(r/'outputs/candidate_pool.csv',pools);csvwrite(r/'outputs/candidate_exact_audit.csv',audit)
    write(r/'outputs/candidate_selection.json',{'documents':dev,'pool_rows':{str(l):sum(x['layer']==l for x in pools) for l in [0,25]},'actual_rows':{str(l):sum(x['layer']==l and x['origin']=='ACTUAL_DEV' for x in pools) for l in [0,25]},'challenge_valid_rows':{str(l):next((x['valid_rows'] for x in fixtures if x['phase']=='B' and x['layer']==l and x['fixture']=='CHALLENGE'),0) for l in [0,25]},'exact_scalar_coordinates':len(audit),'nonfinite_exclusions':exclusions,'new_GPU_outputs_seen':False,'A_B_hypotheses':'Arithmetic discriminators only; neither assumed to be the Marlin implementation'})
    print(read(r/'outputs/candidate_selection.json'))
if __name__=='__main__':main()
