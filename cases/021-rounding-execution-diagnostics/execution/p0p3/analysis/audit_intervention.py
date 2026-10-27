"""Second full-vocabulary calculation for stored boundary-replacement logits."""
import argparse
import csv
import json
from pathlib import Path
import sys
import time
import numpy as np
from audit import decode,digest,independent_kl

def main():
    p=argparse.ArgumentParser();p.add_argument('--private',type=Path,required=True);p.add_argument('--summary',type=Path,required=True);p.add_argument('--split',choices=['DEV','EVAL'],required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);start=time.process_time()
    raw=a.private/'raw'/('p2-intervention-'+a.split.lower());receipt=json.loads((raw/'intervention_receipt.json').read_text());expected=list(csv.DictReader((a.summary/'intervention_positions.csv').open()));maximum=0.;count=0
    for row in receipt['rows']:
        doc=row['document_id'];ref=decode(a.private/'main'/a.split/'D1_R0'/f'{doc}_n512_r0_logits.npy');base=decode(a.private/'main'/a.split/'P_R0'/f'{doc}_n512_r0_logits.npy');path=Path(row['logits']['path'])
        if digest(path)!=row['logits']['sha256'] or digest(row['same_current_P_input']['path'])!=row['same_current_P_input']['sha256']:raise ValueError('INTERVENTION_INPUT_OR_LOGIT_HASH')
        actual=decode(path);before=independent_kl(ref,base);after=independent_kl(ref,actual)
        rows=sorted([r for r in expected if r['document_id']==doc and int(r['layer'])==row['layer'] and r['boundary']==row['boundary']],key=lambda r:int(r['position']))
        if len(rows)!=16:raise ValueError('POSITION_COUNT')
        for key,values in [('baseline_KL',before),('replaced_KL',after),('KL_change',after-before)]:
            target=np.asarray([float(r[key]) for r in rows]);maximum=max(maximum,float(np.max(np.abs(values-target))))
            if not np.allclose(values,target,rtol=1e-9,atol=2e-11):raise ValueError('INTERVENTION_KL_MISMATCH')
        count+=16
    result={'verification':'RECALCULATED_FROM_LOGITS','split':a.split,'full_vocabulary_positions':count,'full_forward_records':len(receipt['rows']),'max_absolute_difference':maximum,'cpu_process_seconds':time.process_time()-start,'torch_imported':'torch' in sys.modules,'cuda_initialization':'NOT_CALLED','scope':'Stored16 full-vocabulary positions and saved current-P block-input identities'}
    (a.output/'receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
