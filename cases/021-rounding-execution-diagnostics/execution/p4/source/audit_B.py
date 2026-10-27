"""Separate scalar norm/rotation and fsum energies from original local packets."""
import argparse
import csv
import math
from pathlib import Path
import time
import numpy as np
from common import read,write,identity
from numerics import bfdecode,weight_vector

def transform(values,weight,epsilon,cos,sin,heads):
    values=np.asarray(values,np.float64).reshape(1,16,heads,128)
    result=np.empty((1,heads,16,128),np.float64)
    for t in range(16):
        for h in range(heads):
            row=values[0,t,h]
            denominator=math.sqrt(math.fsum(float(x)*float(x) for x in row)/128+epsilon)
            normalized=[float(row[j])/denominator*float(weight[j]) for j in range(128)]
            for j in range(128):
                partner=-normalized[j+64] if j<64 else normalized[j-64]
                result[0,h,t,j]=normalized[j]*float(cos[0,t,j])+partner*float(sin[0,t,j])
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--summary',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);start=time.process_time()
    loc=read(a.root/'references/locations.json');case=Path(loc['original_case']);private=Path(loc['original_private']);model=Path(read(private/'locations.json')['model']);eps=read(model/'config.json')['rms_norm_eps'];positions=read(case/'inputs/manifest.json')['selected_positions']['512']
    recorded=list(csv.DictReader((a.summary/'rounding_cancellation_tensors.csv').open()));maximum=0.;checked=0;cache={}
    for row in recorded:
        split=row['analysis_split'].split('_')[1];doc=row['document_id'];layer=int(row['layer']);kind=row['tensor'];rmode=row['rounding'];heads=16 if kind=='q' else 8
        ctx=private/'contexts'/split/doc;cos=bfdecode(np.load(ctx/'cos.npy',allow_pickle=False))[:,positions];sin=bfdecode(np.load(ctx/'sin.npy',allow_pickle=False))[:,positions];us={};ys={}
        weight,whash=weight_vector(model/'model.safetensors',f'model.layers.{layer}.self_attn.{kind}_norm.weight',128)
        if whash!=row['norm_weight_sha256']:raise ValueError('WEIGHT_IDENTITY')
        for path in ['D1','P']:
            filename=private/'local'/split/f'{doc}_layer{layer}_{path}_{rmode}_selected.npz'
            with np.load(filename,allow_pickle=False) as packet:
                key=(doc,layer,kind,path)
                if key not in cache:cache[key]=transform(packet['self_attn.'+kind+'_proj'],weight,eps,cos,sin,heads)
                us[path]=cache[key];ys[path]=packet['post_rope_'+kind].astype(np.float64)
        d=(us['P']-us['D1']).ravel();ed=(ys['D1']-us['D1']).ravel();ep=(ys['P']-us['P']).ravel();noise=ep-ed;gap=(ys['P']-ys['D1']).ravel()
        terms={'d_energy':math.fsum(d*d),'n_energy':math.fsum(noise*noise),'signed_cross':2*math.fsum(d*noise),'pair_gap_SSE':math.fsum(gap*gap),'D1_SSE':math.fsum(ed*ed),'P_SSE':math.fsum(ep*ep)}
        for key,value in terms.items():
            other=float(row[key]);maximum=max(maximum,abs(value-other))
            if not math.isclose(value,other,rel_tol=1e-10,abs_tol=1e-12):raise ValueError('SCALAR_ENERGY_MISMATCH:'+key)
        residual=terms['pair_gap_SSE']-terms['d_energy']-terms['n_energy']-terms['signed_cross'];scale=max(abs(v) for v in [terms['pair_gap_SSE'],terms['d_energy'],terms['n_energy'],terms['signed_cross']])
        if (scale==0 and residual!=0) or (scale and abs(residual)>1e-10*scale):raise ValueError('SCALAR_IDENTITY')
        checked+=1
        # Keep no more than one document's local oracle arrays in memory.
        if len(cache)>16:cache.clear()
    write(a.output/'receipt.json',{'status':'VERIFIED','tensor_rounding_rows':checked,'maximum_absolute_energy_difference':maximum,'scalar_relative_tolerance':1e-10,'scalar_absolute_tolerance':1e-12,'energy_identity_relative_tolerance':1e-10,'shared_code':'File schema, BF16 decoding and norm-vector byte reader only; independent scalar norm/rotate and fsum energy formulas','CPU_process_seconds':time.process_time()-start,'Torch_imported':False})
    print(str({'status':'VERIFIED','rows':checked,'maximum_absolute_difference':maximum}))

if __name__=='__main__':main()
