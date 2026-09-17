"""Optional local audit of complete vocabulary vectors; vectors are not published."""
import argparse,json,math,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.storage import load,write_new


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    checked=0;largest=0.;repeat_count=0;repeat_differences=0
    for stage in ('dev-paired','standard-eval','boundary-eval','length-support'):
        root=a.run/stage
        for f in sorted((root/'cells').glob('*--B.json')):
            b=load(f)['payload'];item=b['item_id']
            with np.load(root/'private_vectors'/(item+'--B.npz'),allow_pickle=False) as archive:
                bz=archive['full_logits'].astype(np.float64)
            bp=np.exp(bz-np.max(bz));bp/=np.sum(bp)
            blse=float(np.max(bz)+np.log(np.exp(bz-np.max(bz)).sum()))
            if not math.isclose(blse,b['full_lse'],abs_tol=1e-12):raise ValueError('B normalizer mismatch')
            repeat_stage={'dev-paired':'dev-b','boundary-eval':'boundary-pool'}.get(stage)
            if repeat_stage:
                with np.load(a.run/repeat_stage/'private_vectors'/(item+'--B.npz'),allow_pickle=False) as archive:
                    repeat_count+=1;repeat_differences+=not np.array_equal(archive['full_logits'],bz)
            for arm in ('A_PUBLIC','V4'):
                r=load(root/'cells'/(item+'--'+arm+'.json'))['payload']
                z=np.load(root/'private_vectors'/(item+'--'+arm+'.npy'),allow_pickle=False).astype(np.float64)
                if z.shape!=bz.shape or not np.isfinite(z).all():raise ValueError('Invalid complete vector')
                lse=float(np.max(z)+np.log(np.exp(z-np.max(z)).sum()))
                actual=float(np.dot(bp,(bz-z)+(lse-blse)))
                delta=abs(actual-r['full_vocab_kl_B_to_candidate']);largest=max(largest,delta)
                if delta>1e-11 or not math.isclose(lse,r['full_lse'],abs_tol=1e-12):raise ValueError('Full-vocabulary KL/normalizer mismatch')
                if z[r['label_ids']].tolist()!=r['option_logits'] or int(np.argmax(z))!=r['full_argmax']:
                    raise ValueError('Retained option/argmax mismatch')
                checked+=1
    result={'status':'PASS','full_vocabulary_KL_pairs_checked':checked,'max_absolute_KL_recalculation_difference':largest,
            'repeated_B_full_vectors_checked':repeat_count,'repeated_B_full_vectors_differed':repeat_differences,
            'scope':'Local complete-vector arithmetic audit; vectors remain excluded from public package. Not new GPU replication.'}
    write_new(a.output,result);print(json.dumps(result))


if __name__=='__main__':main()
