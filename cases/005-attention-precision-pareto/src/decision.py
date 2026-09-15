"""Fixed local screens; no model quality or deployment decision."""
import numpy as np

def numerical(units,expected):
 vals=[u.get('relative_output_error') for u in units]
 invalid=sum(u.get('invalid_rows',0) for u in units)
 defined=len(units)==expected and all(x is not None and np.isfinite(x) for x in vals)
 status='FAIL' if invalid else 'INCONCLUSIVE' if not defined else 'PASS' if np.median(vals)<=.01 and np.quantile(vals,.95,method='linear')<=.03 else 'FAIL'
 return dict(status=status,median=float(np.median(vals)) if defined else None,p95=float(np.quantile(vals,.95,method='linear')) if defined else None,invalid_rows=invalid,missing_units=max(expected-len(units),0),undefined_units=sum(x is None or not np.isfinite(x) for x in vals),units=len(units))

def choose_dev(rows):
 candidates=[x for x in rows if x['config_id'] in ['V1','V2','V3','V4'] and x['execution_status']=='EXECUTABLE_VERIFIED' and x['numerical']['status']=='PASS' and x.get('full_output_nonfinite',0)==0 and x['speedup'] is not None and x['speedup']>=1.50]
 candidates.sort(key=lambda x:(-x['speedup'],x['numerical']['p95'],x['config_id']))
 return candidates[0]['config_id'] if candidates else None

def final_decision(finalist,fresh,valid,error,speed,ci):
 if not finalist:return 'STOP_DEV_SCREEN'
 if not fresh:return 'EXPLORATORY_ONLY_NO_FRESH_CONFIRM'
 if not valid:return 'BLOCKED_VALIDITY'
 if error['status']=='INCONCLUSIVE':return 'INCONCLUSIVE_FIDELITY'
 if error['status']!='PASS' or speed<1.50:return 'NO_QUALIFYING_CANDIDATE'
 if ci is None or ci[0]<=1:return 'INCONCLUSIVE_TIMING'
 return 'GO_LOCAL_CANDIDATE'

def dominates(a,b):
 keys=['wall_median_ms','median_error','p95_error']
 return all(a[k]<=b[k] for k in keys) and any(a[k]<b[k] for k in keys)

def empirical_pareto(rows):
 points=[dict(config_id=r['config_id'],wall_median_ms=r['wall_median_ms'],median_error=r['numerical']['median'],p95_error=r['numerical']['p95']) for r in rows if r['numerical']['median'] is not None and r['wall_median_ms'] is not None]
 return {p['config_id']:[q['config_id'] for q in points if dominates(q,p)] for p in points}
