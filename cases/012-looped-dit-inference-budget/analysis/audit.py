"""Independent scalar/image audit: no runner aggregation functions imported."""
import argparse,json,hashlib,math
from pathlib import Path
from PIL import Image

def h(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);ap.add_argument('--output',type=Path);a=ap.parse_args();root=a.root
    parse=lambda p:json.loads(p.read_text(),parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
    summary=parse(root/'analysis/summary.json');rows=[]
    for p in sorted((root/'results/attempts').glob('*/attempt-*/record.json')):
        if p.parent.name.endswith('.partial'):continue
        r=parse(p);rows.append(r)
        if r['status']=='SUCCESS':
            image=p.parent/'image.png';assert h(image)==r['image_sha256'];im=Image.open(image);im.verify()
            with Image.open(image) as im:assert im.size==(512,512) and im.mode=='RGB'
            assert all(math.isfinite(r[k]) and r[k]>0 for k in ['complete_seconds','sampler_gpu_ms'])
    successes=[r for r in rows if r['status']=='SUCCESS'];assert len({r['job_id'] for r in successes})==len(successes)
    for phase,expected in [('main',192),('smoke',24)]:
        rr=[r for r in successes if r['job']['phase']==phase]
        assert len(rr)==expected,(phase,len(rr),expected)
        paired={}
        for r in rr:
            j=r['job'];paired.setdefault((j['prompt_id'],j['seed_label']),[]).append(r)
        assert len(paired)==expected//3
        for triplet in paired.values():
            assert len(triplet)==3 and len({r['initial_noise_sha256'] for r in triplet})==1
            assert len({r['input_ids_sha256'] for r in triplet})==1 and len({r['attention_mask_sha256'] for r in triplet})==1
    for setting,t in summary['timing'].items():
        rr=[r for r in successes if r['job']['phase']=='main' and r['job']['setting']['id']==setting]
        vals=sorted(r['complete_seconds'] for r in rr);assert len(vals)==64
        assert abs((vals[31]+vals[32])/2-t['main_complete_median_seconds'])<1e-12
    repeats=[r for r in successes if r['job']['phase']=='repeat']
    assert len(repeats)==18
    mainmap={(r['job']['prompt_id'],r['job']['seed_label'],r['job']['setting']['id']):r for r in successes if r['job']['phase']=='main'}
    repeat_image_matches=0
    for r in repeats:
        j=r['job'];m=mainmap[(j['prompt_id'],j['seed_label'],j['setting']['id'])]
        assert r['initial_noise_sha256']==m['initial_noise_sha256'] and r['input_ids_sha256']==m['input_ids_sha256']
        repeat_image_matches+=int(r['image_sha256']==m['image_sha256'])
    for p in (root/'configs').glob('*.json'):parse(p)
    if summary['status']=='ANNOTATION_PENDING':assert summary['quality']['primary'] is None
    assert summary['budget']['full_generation_starts']<=320 and summary['budget']['generation_process_seconds']<=14400
    parity=parse(root/'results/parity.json');assert all(r['pre_pil_bitwise'] and r['official_generate_png_equal'] and r['finite'] for r in parity['rows'])
    receipt={'status':'PASS','scope':'independent saved-record/image/median/noise audit, no model forward or quality judgement','images_verified':len(successes),'main_images':192,'smoke_images':24,'annotation_status':summary['status'],'repeat_timing_rows':len(repeats),'repeat_png_same_as_main':repeat_image_matches,'source_summary_sha256':h(root/'analysis/summary.json')}
    if a.output:a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
