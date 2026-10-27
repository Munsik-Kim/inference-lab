"""Same-input local rounding cancellation with a FP64 norm-RoPE reference."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from common import read,write,identity,csvwrite,array_hash
from numerics import bfdecode,bfbits32,norm_rope,norm_rope_scalar,weight_vector,decomposition

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    start=time.process_time();loc=read(a.root/'references/locations.json');case=Path(loc['original_case']);private=Path(loc['original_private']);original=read(private/'locations.json');model=Path(original['model'])
    config=read(model/'config.json');manifest=read(case/'inputs/manifest.json');positions=manifest['selected_positions']['512'];epsilon=config['rms_norm_eps'];weights={};rows=[];comparisons=[];sources={};bitchecks=0;maxscalar=0.;actual_subset={}
    for split in ['DEV','EVAL']:
        local_receipt=read(private/'raw'/('p2-local-dev' if split=='DEV' else 'p2-local-eval')/'local_metrics.json')
        for doc in manifest['document_order'][split]:
            ctx=private/'contexts'/split/doc
            pos=np.load(ctx/'position_ids.npy',allow_pickle=False)
            if not np.array_equal(pos,np.arange(512)[None]):raise ValueError('POSITION_IDS')
            cos=bfdecode(np.load(ctx/'cos.npy',allow_pickle=False))[:,positions,:].astype(np.float64);sin=bfdecode(np.load(ctx/'sin.npy',allow_pickle=False))[:,positions,:].astype(np.float64)
            for layer in [25,27]:
                input_id=identity(ctx/f'layer{layer}_input.npy')
                refs=[r for r in local_receipt['rows'] if r.get('document_id')==doc and r.get('layer')==layer and r.get('reference')=='D1' and r.get('candidate')=='P' and r.get('boundary')=='post_rope_q']
                if len(refs)!=2 or any(r['input_identity']!=input_id or not r['same_block_input'] for r in refs):raise ValueError('SAME_BLOCK_INPUT')
                packets={}
                for path in ['D1','P']:
                    for r in ['R0','R1']:
                        f=private/'local'/split/f'{doc}_layer{layer}_{path}_{r}_selected.npz';packets[path+'_'+r]=np.load(f,allow_pickle=False);sources[str(f)]=identity(f)
                for kind,heads in [('q',16),('k',8)]:
                    key=f'model.layers.{layer}.self_attn.{kind}_norm.weight'
                    if key not in weights:weights[key]=weight_vector(model/'model.safetensors',key,128)
                    weight,whash=weights[key];references={}
                    for path in ['D1','P']:
                        x0=packets[path+'_R0'][f'self_attn.{kind}_proj'];x1=packets[path+'_R1'][f'self_attn.{kind}_proj']
                        if x0.dtype!=np.float32 or x1.dtype!=x0.dtype or x0.shape!=(1,16,heads*128) or x0.tobytes()!=x1.tobytes():raise ValueError('PROJECTION_RAW_BIT_MISMATCH')
                        if not np.array_equal(x0,bfdecode(bfbits32(x0))):raise ValueError('PROJECTION_NOT_EXACT_BF16_PROMOTION')
                        bitchecks+=1;x=x0.astype(np.float64).reshape(1,16,heads,128).transpose(0,2,1,3)
                        u=norm_rope(x,weight,epsilon,cos,sin)
                        scalar=norm_rope_scalar(x[:,:1,:1,:],weight,epsilon,cos[:,:1,:],sin[:,:1,:])
                        if not np.allclose(u[:,:1,:1],scalar,rtol=1e-10,atol=1e-12):raise ValueError('PRIMITIVE_SCALAR')
                        maxscalar=max(maxscalar,float(np.max(np.abs(u[:,:1,:1]-scalar))));references[path]=u
                    by_r={}
                    for r in ['R0','R1']:
                        yd=packets['D1_'+r]['post_rope_'+kind].astype(np.float64);yp=packets['P_'+r]['post_rope_'+kind].astype(np.float64)
                        if yd.shape!=(1,heads,16,128) or yp.shape!=yd.shape:raise ValueError('ROPE_LAYOUT')
                        if not np.array_equal(yd,bfdecode(bfbits32(yd))) or not np.array_equal(yp,bfdecode(bfbits32(yp))):raise ValueError('POST_ROPE_NOT_EXACT_BF16_PROMOTION')
                        met=decomposition(references['D1'],references['P'],yd,yp);by_r[r]=met
                        rows.append({'analysis_split':'original_'+split+'_posthoc','document_id':doc,'layer':layer,'tensor':kind,'rounding':r,'positions':16,'heads':heads,'head_dim':128,'elements':yd.size,'projection_storage_dtype':'float32 exact promotion of BF16','output_storage_dtype':'float32 exact promotion of BF16','norm_weight_sha256':whash,'epsilon':epsilon,**met})
                        if split=='DEV' and doc==manifest['document_order']['DEV'][0] and layer==25 and kind=='q':
                            # Derived activation/reference values, not model norm weights.
                            for name,value in [('uD',references['D1']),('uP',references['P']),('yD',yd),('yP',yp)]:actual_subset[r+'_'+name]=value[:,:1,:1,:].copy()
                    z0,z1=by_r['R0'],by_r['R1'];pattern=z1['individual_error_sum']<z0['individual_error_sum'] and z1['pair_gap_SSE']>z0['pair_gap_SSE']
                    comparisons.append({'analysis_split':'original_'+split+'_posthoc','document_id':doc,'layer':layer,'tensor':kind,'positions':16,'heads':heads,'elements_per_path':heads*16*128,'d_energy':z0['d_energy'],'R0_individual_error_sum':z0['individual_error_sum'],'R1_individual_error_sum':z1['individual_error_sum'],'R0_pair_gap_SSE':z0['pair_gap_SSE'],'R1_pair_gap_SSE':z1['pair_gap_SSE'],'R0_cross':z0['signed_cross'],'R1_cross':z1['signed_cross'],'individual_error_change_R1_minus_R0':z1['individual_error_sum']-z0['individual_error_sum'],'pair_gap_change_R1_minus_R0':z1['pair_gap_SSE']-z0['pair_gap_SSE'],'noise_energy_change_R1_minus_R0':z1['n_energy']-z0['n_energy'],'cross_change_R1_minus_R0':z1['signed_cross']-z0['signed_cross'],'individual_improves_pair_gap_worsens':pattern,'R0_more_negative_cross':z0['signed_cross']<0 and z0['signed_cross']<z1['signed_cross']})
                for packet in packets.values():packet.close()
    docs=[]
    for split in ['DEV','EVAL']:
        for doc in manifest['document_order'][split]:
            subset=[r for r in comparisons if r['document_id']==doc]
            names=['R0_individual_error_sum','R1_individual_error_sum','R0_pair_gap_SSE','R1_pair_gap_SSE','R0_cross','R1_cross','individual_error_change_R1_minus_R0','pair_gap_change_R1_minus_R0','noise_energy_change_R1_minus_R0','cross_change_R1_minus_R0']
            docs.append({'analysis_split':'original_'+split+'_posthoc','document_id':doc,'tensor_pairs':len(subset),'elements_per_path':sum(r['elements_per_path'] for r in subset),'aggregation':'Sum over disjoint layer25/27 and Q/K coordinate spaces',**{k:sum(r[k] for r in subset) for k in names}})
    facts=[]
    for split in ['DEV','EVAL']:
        selected=[r for r in comparisons if r['analysis_split']=='original_'+split+'_posthoc']
        facts.append({'split':'original_'+split+'_posthoc','tensor_cells':len(selected),'individual_improves_pair_gap_worsens':sum(r['individual_improves_pair_gap_worsens'] for r in selected),'pattern_and_R0_more_negative_cross':sum(r['individual_improves_pair_gap_worsens'] and r['R0_more_negative_cross'] for r in selected),'opposite_individual_worsens_pair_gap_improves':sum(r['individual_error_change_R1_minus_R0']>0 and r['pair_gap_change_R1_minus_R0']<0 for r in selected),'pair_gap_R1_increases':sum(r['pair_gap_change_R1_minus_R0']>0 for r in selected),'pair_gap_R1_decreases':sum(r['pair_gap_change_R1_minus_R0']<0 for r in selected)})
    csvwrite(a.output/'rounding_cancellation_tensors.csv',rows);csvwrite(a.output/'rounding_cancellation_pairs.csv',comparisons);csvwrite(a.output/'rounding_cancellation_documents.csv',docs)
    np.savez(a.output/'derived_activation_audit_subset.npz',**actual_subset)
    write(a.output/'p4b_facts.json',{'scope':'POSTHOC same input, same selected coordinates only; not an attribution of full-model KL','groups':facts,'original_primary_verdict':'INCONCLUSIVE'})
    write(a.output/'p4b_coverage.json',{'status':'COMPLETE','documents':32,'layers':[25,27],'Q_heads':16,'K_heads':8,'head_dim':128,'positions':positions,'tensor_rounding_rows':len(rows),'tensor_pair_cells':len(comparisons),'document_rows':len(docs),'projection_bit_equality_checks':bitchecks,'local_packets':len(sources),'sources':sources,'full_attention_reconstructed':False,'model_full_fp64_reference':False})
    write(a.output/'primitive_reference_verification.json',{'scalar_vector_atol':1e-12,'scalar_vector_rtol':1e-10,'maximum_selected_actual_scalar_difference':maxscalar,'energy_identity_relative_gate':1e-10,'max_energy_identity_relative_residual':max(abs(r['identity_residual'])/r['identity_scale'] if r['identity_scale'] else 0 for r in rows),'norm_weights':{k:v[1] for k,v in weights.items()},'norm_weight_bytes_published':0,'cpu_process_seconds':time.process_time()-start,'torch_imported':False})
    print(json.dumps({'status':'COMPLETE','groups':facts,'tensor_rows':len(rows),'bit_checks':bitchecks}))

if __name__=='__main__':main()
