"""Copy only the predeclared DEV4 x2 positions, preserving actual BF16 bits."""
import argparse
import csv
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from common import CASE,read,write,file_row

def main():
    p=argparse.ArgumentParser();p.add_argument('--private',type=Path,required=True);p.add_argument('--summary',type=Path,required=True);a=p.parse_args()
    manifest=read(CASE/'inputs/manifest.json');docs=manifest['document_order']['DEV'][:4];conditions=manifest['condition_order'];indices=[0,8]
    dest=CASE/'fixtures/public_logits';dest.mkdir(parents=True,exist_ok=False);files={}
    for doc in docs:
        for condition in conditions:
            source=a.private/'main/DEV'/condition/f'{doc}_n512_r0_logits.npy';raw=np.load(source,allow_pickle=False)
            name=doc+'_'+condition+'.npy';np.save(dest/name,raw[indices],allow_pickle=False)
            files[name]={**file_row(dest/name),'source_raw':file_row(source),'selection_indices_within16':indices,'dtype':str(raw.dtype),'encoding':'BF16_BITS_UINT16' if raw.dtype==np.uint16 else str(raw.dtype)}
    rows=list(csv.DictReader((a.summary/'position_metrics.csv').open()));comparisons=[]
    for doc in docs:
        for ref,cand in [('D0_R0','D1_R0'),('D1_R0','P_R0'),('D0_R0','P_R0'),('D0_R1','D1_R1'),('D1_R1','P_R1'),('D0_R1','P_R1')]:
            selected=[r for r in rows if r['document_id']==doc and int(r['length'])==512 and r['reference']==ref and r['candidate']==cand and int(r['position']) in [manifest['selected_positions']['512'][i] for i in indices]]
            selected.sort(key=lambda x:int(x['position']))
            if len(selected)!=2:raise ValueError('PUBLIC_SUBSET_DENOMINATOR')
            comparisons.append({'document_id':doc,'reference_file':doc+'_'+ref+'.npy','candidate_file':doc+'_'+cand+'.npy','positions':[int(r['position']) for r in selected],'expected_KL':[float(r['KL']) for r in selected]})
    write(dest/'manifest.json',{'scope':'DEV4 documents, length512, selected-position indices0/8 (positions30/271), six conditions; not all384 forwards','documents':docs,'conditions':conditions,'files':files,'comparisons':comparisons,'selection_rule':'first4 frozen DEV order; first and ninth of16 fixed selected positions; no performance selection'})
    print(sum(x['bytes'] for x in files.values()))

if __name__=='__main__':main()
