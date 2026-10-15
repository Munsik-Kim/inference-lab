"""Supplementary post-hoc arithmetic robustness; never alters raw A/B/C/D labels.
From e=R_run-R, |SSE_run-SSE| <= 2||R||||e||+||e||^2.
This analytic bound uses no fitted threshold or model selection.
"""
import argparse,json,math
from pathlib import Path

def main(run,out):
 run=Path(run);rows=[json.loads(s) for p in sorted((run/'cells').glob('*.jsonl')) for s in p.read_text().splitlines()];by={(r['block'],r['projection'],r['split'],r['document_id_hash'],r['alpha']):r for r in rows};result=[]
 for r in rows:
  if r['alpha']==0:continue
  b=by[(r['block'],r['projection'],r['split'],r['document_id_hash'],0)];bound=2*math.sqrt(r['SSE_after']*r['arithmetic_residual_norm2'])+r['arithmetic_residual_norm2']+2*math.sqrt(b['SSE_after']*b['arithmetic_residual_norm2'])+b['arithmetic_residual_norm2'];actual=r['BF16_output_residual_norm2']-b['BF16_output_residual_norm2']
  result.append({'block':r['block'],'projection':r['projection'],'split':r['split'],'alpha':r['alpha'],'document_id_hash':r['document_id_hash'],'algebra_Delta':r['Delta'],'actual_BF16_module_Delta':actual,'arithmetic_Delta_absolute_bound':bound,'algebra_sign_certified_against_arithmetic_bound':abs(r['Delta'])>bound,'actual_and_algebra_sign_agree':(actual>0)==(r['Delta']>0) if actual and r['Delta'] else None,'scope':'supplementary post-hoc diagnostic; does not redefine FP64 decomposition or change selected alpha'})
 Path(out).write_text(json.dumps({'definition':'2sqrt(SSE*arithmetic_residual_norm2)+residual_norm2 for each arm, summed across candidate and alpha0','rows':result},indent=2))
 print('arithmetic-bound certified',sum(r['algebra_sign_certified_against_arithmetic_bound'] for r in result),'/',len(result))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True);a=p.parse_args();main(a.run,a.output)
