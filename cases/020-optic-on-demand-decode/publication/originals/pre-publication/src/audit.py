"""Independent CPU reaggregation. Deliberately imports neither Torch nor benchmark code."""
import argparse,csv,hashlib,json,pathlib,sys,copy
import numpy as np
DEFAULT=pathlib.Path(__file__).resolve().parents[1]
def read(p):return json.loads(pathlib.Path(p).read_text())
def digest(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def check(ok,msg):
 if not ok:raise ValueError(msg)
def check_hash(p,h):check(digest(p)==h,f'hash mismatch: {p}')
def words_float(words):return (words.astype(np.uint32)<<16).view(np.float32)
def independent_reference(meta,a,ids):
 """Alternate vector formulation, one NumPy FP64 matrix product per layer."""
 G=meta['n_groups'];V=meta['ple_shape'][0];codes=a['token_codes_packed_uint8'].reshape(V,G);cb=a['codebooks_fp16'].astype('float64');flat=ids.reshape(-1)
 latent=np.concatenate([cb[g,codes[flat,g]] for g in range(G)],axis=1)
 w=a['decoder_weight_fp16'].astype('float64');b=a['decoder_bias_fp16'].astype('float64')
 return np.stack([latent@w[l].T+b[l] for l in range(len(w))],axis=1)
def errors(x,r):
 d=x.astype('float64')-r;nr=float(np.linalg.norm(r));ma=float(np.abs(d).max(initial=0));rel=float(np.linalg.norm(d)/max(nr,1e-12));finite=bool(np.isfinite(x).all())
 return {'relative_L2':rel,'max_abs':ma,'rmse':float(np.sqrt(np.mean(d*d))) if d.size else 0.,'finite':finite,'zero_reference':nr==0,'gate':finite and (ma<=1e-6 if nr==0 else rel<=1e-2)}
def unique_storage(rows):
 seen={};total=0
 for x in rows:
  sizes={'torch.uint8':1,'torch.int64':8,'torch.float16':2,'torch.bfloat16':2,'torch.float32':4,'torch.uint16':2}
  numel=int(np.prod(x['shape'],dtype=np.int64));check(numel==x['numel'],'numel mismatch');check(x['element_size']==sizes[x['dtype']],'dtype size mismatch');check(x['payload_bytes']==numel*sizes[x['dtype']],'payload mismatch')
  key=tuple(x['storage_key']);check(key[2]==x['storage_bytes'],'storage-key size mismatch');check(x['unique_storage']==(key not in seen),'alias accounting mismatch')
  if key not in seen:total+=x['storage_bytes'];seen[key]=True
 return total
def validate_raw(raw,cell,protocol):
 c=raw['cell']
 for k in ['shape','M','U','condition','method','block','cell_id','ids_bytes_sha256','sha256']:
  check(c[k]==cell[k],f'cell identity mismatch {k}')
 check(raw['upstream_sha']==protocol['upstream_sha'],'source mismatch')
 check(raw['output_dtype']=='torch.bfloat16','output dtype')
 s=protocol['shapes'][c['shape']];check(raw['output_shape']==[c['M'],s['L']*s['H']],'output shape')
 for key in ['host_us','event_bracket_per_call_us','floor_host_us','floor_empty_event_us']:
  a=np.array(raw['timing'][key]);check(len(a)==100 and np.isfinite(a).all() and (a>=0).all(),'invalid timing samples')
 check(raw['timing']['event_K']==protocol['event_calls'][str(c['M'])],'event K mismatch')
 check(raw['policy_after']['fp16_reduction'] is False and raw['policy_after']['fp16_accumulation'] is False,'arithmetic policy mismatch')
 check(raw['policy_after']['tf32'] is False,'TF32 policy mismatch')
 check(raw['correctness']['finite'],'nonfinite result')
 for err in [raw['correctness'],*raw['correctness']['layers']]:check(err['max_abs']<=1e-6 if err['zero_reference'] else err['relative_L2']<=1e-2,'numeric gate failed')
 inventory=raw['resident_inventory'];resident=unique_storage(inventory)
 expected=(s['V']*s['L']*s['H']*2 if c['method']=='A' else s['V']*s['G']+s['G']*256*s['D']*2+s['L']*s['H']*s['Z']*2+s['L']*s['H']*2+s['G']*8)
 check(resident==expected,'resident storage versus shape formula mismatch')
 check(unique_storage(raw['id_inventory'])==c['M']*8,'IDs payload')
 check(unique_storage(raw['output_inventory'])==c['M']*s['L']*s['H']*2,'output payload')
 if c['method']!='A':check(resident-raw['upstream_compact_resident_bytes']==s['G']*8,'nonpersistent buffer accounting')
 delta=max(raw['setup']['allocated_peak'],raw['cold_peak']['allocated'],raw['timed_memory']['allocated_peak'])-raw['baseline']['allocated'];check(delta<=protocol['max_allocated_delta_bytes'],'2GiB gate failed');check(delta==raw['allocated_delta_peak'],'allocated delta mismatch')
 return resident
def collect(root=DEFAULT):
 root=pathlib.Path(root);protocol=read(root/'configs/protocol.json');freeze=read(root/'provenance/freeze.json');run=root/'results'/protocol['run_id'];ledger=read(run/'worker_ledger.json');plans=read(root/'provenance/planned_cells.json');inputs=read(root/'provenance/input_manifest.json')
 if (root/'PUBLIC_MANIFEST.json').exists():
  public=read(root/'PUBLIC_MANIFEST.json');actual_files={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p.name!='PUBLIC_MANIFEST.json'}
  check(actual_files==set(public['files']),'public file set mismatch')
  for rel,info in public['files'].items():check_hash(root/rel,info['sha256']);check((root/rel).stat().st_size==info['bytes'],'public byte count mismatch')
 for rel,info in freeze['main_files'].items():check_hash(root/rel,info['sha256']);check((root/rel).stat().st_size==info['bytes'],'frozen bytes mismatch')
 check(len(plans)==90 and len({c['cell_id'] for c in plans})==90,'planned grid mismatch')
 for inp in inputs:
  check_hash(root/inp['path'],inp['sha256']);ids=np.load(root/inp['path'],allow_pickle=False);check(ids.dtype==np.int64,'ID dtype');check(len(ids)==inp['M'] and len(np.unique(ids))==inp['U'],'M/U mismatch');check(hashlib.sha256(ids.tobytes()).hexdigest()==inp['ids_bytes_sha256'],'ID bytes hash');check(ids.min()>=0 and ids.max()<protocol['shapes'][inp['shape']]['V'],'ID range')
  _,counts=np.unique(ids,return_counts=True);check((counts==(1 if inp['condition']=='unique' else 4)).all(),'duplication contract')
  check_hash(root/inp['artifact']['path'],inp['artifact']['sha256'])
 plan_by={c['cell_id']:c for c in plans};actual={};failed=[];latency=[];memory=[];correctness=[]
 for receipt in ledger['workers']:
  raw_path=run/receipt['raw']
  if raw_path.exists():check_hash(raw_path,receipt['raw_sha256'])
  if receipt['kind']!='cell':continue
  if receipt['exit_code']!=0 or not raw_path.exists():failed.append(receipt['label']);continue
  r=read(raw_path);c=plan_by[receipt['label']];resident=validate_raw(r,c,protocol);check(c['cell_id'] not in actual,'duplicate cell');actual[c['cell_id']]=r
  check_hash(root/'src/gpu_worker.py',receipt['worker_source_sha256']);check_hash(root/'configs/protocol.json',receipt['protocol_sha256'])
  with np.load(root/c['artifact']['path'],allow_pickle=False) as ar:
   metadata_unicode_payload=ar['metadata'].nbytes;metadata_utf8_bytes=len(ar['metadata'].item().encode());payload=sum(ar[k].nbytes for k in ar.files if k!='metadata')
  check(payload==c['artifact']['payload_bytes'],'artifact array payload mismatch')
  common={k:c[k] for k in ['shape','M','U','condition','method','block','cell_id']};h=np.array(r['timing']['host_us']);e=np.array(r['timing']['event_bracket_per_call_us']);q=lambda x:float(np.quantile(x,.95,method='linear'))
  latency.append({**common,'host_median_us':float(np.median(h)),'host_p95_us':q(h),'host_min_us':float(h.min()),'event_bracket_mean_per_call_us':float(e.mean()),'event_bracket_p95_per_call_us':q(e),'event_K':r['timing']['event_K'],'setup_seconds':r['setup']['wall_seconds'],'floor_host_median_us':float(np.median(r['timing']['floor_host_us'])),'floor_empty_event_median_us':float(np.median(r['timing']['floor_empty_event_us']))})
  memrow={**common,'artifact_npz_bytes':c['artifact']['bytes'],'array_payload_bytes':payload,'metadata_cpu_utf8_bytes':metadata_utf8_bytes,'metadata_cpu_numpy_unicode_bytes':metadata_unicode_payload,'gpu_resident_tensor_bytes':resident,'upstream_compact_resident_bytes':r['upstream_compact_resident_bytes'],'baseline_allocated_bytes':r['baseline']['allocated'],'baseline_reserved_bytes':r['baseline']['reserved'],'warm_resident_allocated_bytes':r['resident_after_warmup']['allocated'],'warm_reserved_bytes':r['resident_after_warmup']['reserved'],'steady_allocated_peak_bytes':r['timed_memory']['allocated_peak'],'reserved_start_bytes':r['timed_memory']['reserved_start'],'steady_reserved_peak_bytes':r['timed_memory']['reserved_peak'],'setup_allocated_peak_bytes':r['setup']['allocated_peak'],'setup_reserved_peak_bytes':r['setup']['reserved_peak'],'cold_allocated_peak_bytes':r['cold_peak']['allocated'],'cold_reserved_peak_bytes':r['cold_peak']['reserved'],'max_baseline_delta_bytes':r['allocated_delta_peak'],'id_payload_bytes':c['M']*8,'output_payload_bytes':c['M']*protocol['shapes'][c['shape']]['L']*protocol['shapes'][c['shape']]['H']*2}
  if c['method']=='C':
   unique_storage(r['unique_inventory'])
   for item in r['unique_inventory']:memrow[item['name']+'_diagnostic_payload_bytes']=item['payload_bytes']
  memory.append(memrow)
  correctness.append({**common,'scope':'GPU_worker','relative_L2':r['correctness']['relative_L2'],'max_abs':r['correctness']['max_abs'],'rmse':r['correctness']['rmse'],'gate':True})
 # Independently re-evaluate all P1 outputs against alternate FP64 expression, all layers.
 p1=run/'p1_with_raw'
 for fixture_info in read(root/'provenance/fixture_manifest.json'):
  path=root/fixture_info['path'];check_hash(path,fixture_info['sha256'])
  with np.load(path,allow_pickle=False) as z:meta=json.loads(z['metadata'].item());arrays={k:z[k] for k in z.files if k!='metadata'}
  for k,a in arrays.items():check(hashlib.sha256(a.tobytes()).hexdigest()==fixture_info['arrays'][k]['sha256'],'fixture array bytes')
  name=path.stem;seed=int(name.split('_')[1][4:]);control='_'.join(name.split('_')[2:]);ref=independent_reference(meta,arrays,np.arange(257,dtype=np.int64))
  with np.load(p1/f'cpu_seed{seed}_{control}_bf16_words.npz',allow_pickle=False) as z:
   for method in ['A','B','C']:
    x=words_float(z[method]).reshape(257,3,16)
    for layer in [-1,0,1,2]:
     err=errors(x if layer<0 else x[:,layer],ref if layer<0 else ref[:,layer]);check(err['gate'],'independent CPU reference failed');correctness.append({'scope':'P1_independent_raw','seed':seed,'control':control,'method':method,'layer':layer,**err})
 # Independent P2 formula from preserved B block0 requested BF16 outputs, no benchmark import.
 for inp in inputs:
  key=f"b0_{inp['shape']}_M{inp['M']}_{inp['condition']}_B"
  if key not in actual:continue
  raw=actual[key];output=run/raw['raw_bf16_output']['path'];check_hash(output,raw['raw_bf16_output']['sha256'])
  with np.load(root/inp['artifact']['path'],allow_pickle=False) as z:meta=json.loads(z['metadata'].item());arrays={k:z[k] for k in z.files if k!='metadata'}
  ref=independent_reference(meta,arrays,np.load(root/inp['path'],allow_pickle=False));s=protocol['shapes'][inp['shape']]
  with np.load(output,allow_pickle=False) as z:x=words_float(z['words']).reshape(inp['M'],s['L'],s['H'])
  for layer in [-1,*range(s['L'])]:
   err=errors(x if layer<0 else x[:,layer],ref if layer<0 else ref[:,layer]);check(err['gate'],'independent GPU raw reference failed');correctness.append({'scope':'P2_independent_raw','shape':inp['shape'],'M':inp['M'],'U':inp['U'],'condition':inp['condition'],'method':'B','block':0,'layer':layer,**err})
 pairs=read(run/'gpu_pair_correctness.json') if (run/'gpu_pair_correctness.json').exists() else []
 for r in pairs:
  for er in [r,*r['layers']]:check(er['finite'] and (er['max_abs']<=1e-6 if er['zero_reference'] else er['relative_L2']<=1e-2),'paired numeric gate')
  correctness.append({k:v for k,v in dict(r,scope='GPU_method_pair',gate=True).items() if k!='layers'})
 paired=[];lat_by={x['cell_id']:x for x in latency}
 for block in range(3):
  for inp in inputs:
   base=f"b{block}_{inp['shape']}_M{inp['M']}_{inp['condition']}"
   if not all(base+'_'+m in lat_by for m in ['A','B','C']):continue
   a,b,c=[lat_by[base+'_'+m] for m in ['A','B','C']]
   paired.append({'shape':inp['shape'],'M':inp['M'],'U':inp['U'],'condition':inp['condition'],'block':block,'B_over_A_host':b['host_median_us']/a['host_median_us'],'C_over_B_host':c['host_median_us']/b['host_median_us'],'B_minus_A_host_us':b['host_median_us']-a['host_median_us'],'C_minus_B_host_us':c['host_median_us']-b['host_median_us'],'B_over_A_event':b['event_bracket_mean_per_call_us']/a['event_bracket_mean_per_call_us'],'C_over_B_event':c['event_bracket_mean_per_call_us']/b['event_bracket_mean_per_call_us']})
 spent=sum(x['wall_seconds'] for x in ledger['workers']);check(abs(spent-ledger['spent_seconds'])<1e-8,'worker wall accounting');check(spent<=1800,'worker budget exceeded')
 missing=sorted(set(plan_by)-set(actual));labels=[]
 expected_pairs={(b,i['shape'],i['M'],i['U'],i['condition'],m) for b in range(3) for i in inputs for m in ['A/B','B/C','A/C']}
 observed_pairs={(r['block'],r['shape'],r['M'],r['U'],r['condition'],r['methods']) for r in pairs}
 check(len(observed_pairs)==len(pairs),'duplicate numerical pair')
 if len(actual)==90:check(observed_pairs==expected_pairs,'missing numerical pair')
 probe=read(run/'probe.json');check(probe['sm']==[12,0] and probe['shape']==[257,48] and probe['dtype']=='torch.bfloat16','probe geometry/dtype')
 for er in [probe['correctness'],*probe['correctness']['layers']]:check(er['finite'] and er['relative_L2']<=1e-2,'probe numeric gate')
 ops=[o['name'] for o in probe['profiler_ops']]
 check(any('index_select' in s for s in ops) and any('gemm' in s.lower() for s in ops) and any('bfloat16' in s for s in ops) and 'aten::add_' in ops,'profiler path missing')
 if len(actual)==90 and observed_pairs==expected_pairs:labels=['OPERATOR_VALIDATED','MEMORY_TRADEOFF_OBSERVED']
 gains=[]
 for inp in inputs:
  these=[p for p in paired if all(p[k]==inp[k] for k in ['shape','M','U','condition'])]
  if len(these)==3 and all(p['C_over_B_host']<1 for p in these):gains.append({'shape':inp['shape'],'M':inp['M'],'U':inp['U'],'condition':inp['condition'],'ratios':[p['C_over_B_host'] for p in these]})
 if gains:labels.append('DEDUP_CONDITIONAL_BENEFIT')
 elif len(actual)==90:
  labels.append('DEDUP_UNRESOLVED_BLOCK_VARIATION' if any(p['C_over_B_host']<1 for p in paired) else 'NO_DEDUP_BENEFIT_OBSERVED')
 decision={'status':'COMPLETE' if len(actual)==90 and not failed and not missing else 'COMPLETE_PARTIAL','planned_cells':30,'planned_cell_blocks':90,'actual_cells':len({(r['cell']['shape'],r['cell']['condition'],r['cell']['M'],r['cell']['method']) for r in actual.values()}),'success_cell_blocks':len(actual),'failed_cell_blocks':len(failed),'failed':failed,'missing_cell_blocks':len(missing),'missing':missing,'labels':labels,'dedup_gain_conditions':gains,'host_C_faster_paired_blocks':sum(p['C_over_B_host']<1 for p in paired),'paired_comparisons':len(paired),'judgment_note':'No condition had C<B in all three host blocks. This descriptive consistency rule was selected after seeing block variation; it is not a preregistered statistical threshold. Faster individual blocks remain visible.','gpu_workers_total':len(ledger['workers']),'gpu_worker_wall_seconds':spent,'gpu_worker_budget_remaining_seconds':1800-spent,'independent_method_pair_records':len(pairs),'cpu_only_reaggregation':True,'torch_imported': 'torch' in sys.modules,'quantile':'linear','public_raw_bf16_scope':'B block0 all ten inputs plus P1 A/B/C all257tokens, seeds0/1/2; requested outputs only'}
 return {'latency':latency,'memory':memory,'correctness':correctness,'paired':paired,'decision':decision,'actual':actual}
def save_tables(root,out,data):
 out=pathlib.Path(out);out.mkdir(parents=True,exist_ok=True)
 for key in ['correctness','memory','latency','paired']:
  rows=data[key];fields=list(dict.fromkeys(k for row in rows for k in row))
  with (out/f'{key}.csv').open('w',newline='') as f:
   w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
 (out/'decision.json').write_text(json.dumps(data['decision'],indent=2)+'\n')
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['verify','summarize','figures']);p.add_argument('--root',type=pathlib.Path,default=DEFAULT);p.add_argument('--out',type=pathlib.Path);a=p.parse_args();data=collect(a.root)
 if a.command=='summarize':save_tables(a.root,a.out or a.root/'results/summary',data)
 if a.command=='figures':
  from figures import render
  render(data,a.out or a.root/'figures')
 print(json.dumps(data['decision'],indent=2))
if __name__=='__main__':main()
