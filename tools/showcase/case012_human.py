"""Audit a completed, visually grouped user assessment from recorded scalars only."""
import argparse
import math
import statistics
from collections import Counter
from pathlib import Path
from common import ROOT, read, require, sha, json_text
from case012 import C12

PUBLIC = 'presentation/evaluations/case012-human-v1'
ANNOTATION_SHA = 'c3d403898d1b339b318a79ecc450c3fd7a7acbf927b11450ae92c116307cbf21'
COHORT_SHA = 'd204dd88a27b317e1e94b5f77e840cede78bb4945e59b817d8e4f96ebb2a3e30'
GROUP_SHA = '5b054f388145d7600647fee711915188e9b49d85788c6426ed0dd104fa145d5a'
SETTINGS = ('A_time', 'B_time', 'C_time')
VALUES = {'satisfied', 'not_satisfied', 'uncertain'}
SOURCE_PATHS = (C12+'/analysis/record_index.json', C12+'/analysis/annotation_items.json',
                C12+'/configs/rubric.json', C12+'/publication/assessment_v1/annotations.json',
                'presentation/content/case012-similarity.json')


def calculate(root, annotations):
    """Independent Python arithmetic: no frozen quality() or bootstrap helper."""
    items = {i['image_id']: i for i in read(root/C12/'analysis/annotation_items.json')['items'] if i['split']=='MAIN'}
    records = {r['job_id']: r for r in read(root/C12/'analysis/record_index.json')['rows'] if r['job']['phase']=='main'}
    ai = {r['image_id']: r for r in read(root/C12/'publication/assessment_v1/annotations.json')['rows']}
    group_path = root/'presentation/content/case012-similarity.json'
    require(sha(group_path)==GROUP_SHA, 'Human assessment visual-group source changed')
    groups = {g['group_id']:g for g in read(group_path)['groups']}
    require(annotations['schema']=='case012-annotations-v2' and annotations['scope']=='MAIN_ONLY'
            and annotations['status']=='COMPLETE' and not annotations['draft_rows']
            and not annotations['incomplete_image_ids'], 'Completed MAIN human export required')
    require(annotations['evaluator']=={'type':'human','id':'anonymous-local-draft'}, 'Anonymous human assessment identity')
    require(annotations['image_set_sha256']==COHORT_SHA
            and annotations['rubric_sha256']==sha(root/C12/'configs/rubric.json'), 'Human input/rubric identity')
    require(annotations['review']['grouping_sha256']==GROUP_SHA
            and annotations['review']['mode']=='similarity'
            and annotations['review']['separated_image_ids']==[], 'Human grouping identity')
    rows = {r['image_id']:r for r in annotations['rows']}
    require(len(rows)==len(annotations['rows'])==192 and set(rows)==set(items)==set(records)==set(ai),
            'Human MAIN missing/duplicate IDs')
    counts=Counter(); direct_ids=set(); reused_ids=set(); results={}; paired_index={}
    for image_id,row in rows.items():
        item=items[image_id]; record=records[image_id]; keys={c['id'] for c in item['constraints']}
        require(row['image_sha256']==item['image_sha256']==record['image_sha256'], 'Human PNG hash mismatch')
        require(set(row['values'])==set(row['origins'])==keys and set(row['values'].values())<=VALUES,
                'Human constraint coverage/value mismatch')
        for cid,value in row['values'].items():
            origin=row['origins'][cid]
            if origin=={'kind':'direct'}:
                counts['direct_constraints']+=1; direct_ids.add(image_id)
            else:
                require(set(origin)=={'kind','source_image_id','group_id'} and origin['kind']=='similarity_reuse',
                        'Human reuse origin schema')
                group=groups.get(origin['group_id']); source=rows.get(origin['source_image_id'])
                require(group is not None and source is not None and source['image_id']!=image_id
                        and source['image_id'] in group['members'] and image_id in group['members']
                        and group['prompt_id']==record['job']['prompt_id'], 'Human reuse membership')
                require(source['values'].get(cid)==value and source['origins'].get(cid)=={'kind':'direct'},
                        'Human reuse source/value mismatch')
                counts['reused_constraints']+=1; reused_ids.add(image_id)
            counts['uncertain_constraints']+=value=='uncertain'
            counts['human_AI_constraint_disagreements']+=value!=ai[image_id]['values'][cid]
        values=list(row['values'].values()); passed=all(v=='satisfied' for v in values)
        results[image_id]={'passed':passed, 'optimistic_pass':all(v!='not_satisfied' for v in values),
                           'fraction':values.count('satisfied')/len(values), 'values':row['values'],
                           'direct_constraints':sum(o=={'kind':'direct'} for o in row['origins'].values()),
                           'reused_constraints':sum(o['kind']=='similarity_reuse' for o in row['origins'].values())}
        counts['human_AI_image_pass_disagreements']+=passed!=all(v=='satisfied' for v in ai[image_id]['values'].values())
        key=(record['job']['prompt_id'],record['job']['seed_label'],record['job']['setting']['id'])
        require(key not in paired_index, 'Duplicate human prompt-seed-setting pair')
        paired_index[key]=image_id
    def aggregate(ids):
        rr=[results[i] for i in ids]; n=len(rr)
        return {'images':n, 'passed_images':sum(r['passed'] for r in rr),
                'all_constraint_pass_rate':sum(r['passed'] for r in rr)/n,
                'mean_constraint_fraction':math.fsum(r['fraction'] for r in rr)/n,
                'all_uncertain_as_pass_rate':sum(r['optimistic_pass'] for r in rr)/n}
    settings={s:aggregate([i for i,r in records.items() if r['job']['setting']['id']==s]) for s in SETTINGS}
    require(all(v['images']==64 for v in settings.values()), 'Human per-setting denominator')
    categories={c:{s:aggregate([i for i,r in records.items() if r['job']['setting']['id']==s and items[i]['category']==c])
                   for s in SETTINGS} for c in sorted({i['category'] for i in items.values()})}
    require(all(v['images']==16 for cat in categories.values() for v in cat.values()), 'Human category denominator')
    pairs=Counter({'both_pass':0,'A_only':0,'C_only':0,'neither':0}); gains=losses=0
    base={(p,seed) for p,seed,s in paired_index}
    require(len(base)==64 and len({p for p,seed in base})==16, 'Human pair/prompt denominator')
    for p,seed in sorted(base):
        a=results[paired_index[p,seed,'A_time']]; c=results[paired_index[p,seed,'C_time']]
        pairs['both_pass' if a['passed'] and c['passed'] else 'A_only' if a['passed'] else 'C_only' if c['passed'] else 'neither']+=1
        for cid,av in a['values'].items():
            cv=c['values'][cid]; gains+=av!='satisfied' and cv=='satisfied'; losses+=av=='satisfied' and cv!='satisfied'
    primary={'direction':'C_time minus A_time, higher better',
             'estimate':settings['C_time']['all_constraint_pass_rate']-settings['A_time']['all_constraint_pass_rate'],
             'interval':None,'level':None,'prompt_clusters':16,
             'method':'descriptive assigned-label comparison; no confidence interval for visually reused judgments'}
    require(primary['estimate']==(pairs['C_only']-pairs['A_only'])/64, 'Human paired difference mismatch')
    return {'schema':'case012-human-publication-v1','status':'GROUPED_REVIEW_DESCRIPTIVE',
            'evidence_kind':'USER_GROUPED_HUMAN_REVIEW','evaluator':annotations['evaluator'],
            'coverage':{'complete':True,'annotated':192,'expected':192,'missing':[]},
            'label_provenance':{'direct_constraints':counts['direct_constraints'],
                'reused_constraints':counts['reused_constraints'],'direct_images':len(direct_ids),
                'reused_images':len(reused_ids),'review_groups':len(groups)},
            'primary':primary,'settings':settings,'categories':categories,'paired':dict(pairs),
            'constraint_transition':{'C_gains':gains,'C_losses':losses},
            'uncertain_constraint_count':counts['uncertain_constraints'],
            'AI_comparison':{'image_pass_disagreements':counts['human_AI_image_pass_disagreements'],
                'constraint_disagreements':counts['human_AI_constraint_disagreements'],
                'images':192,'constraints':counts['direct_constraints']+counts['reused_constraints'],
                'meaning':'different assigned judgments, not ground-truth accuracy or independent inter-rater reliability'},
            'timing':{s:{'median_complete_seconds':statistics.median(r['complete_seconds'] for r in records.values() if r['job']['setting']['id']==s)} for s in SETTINGS},
            'per_image':results,'annotation_file_sha256':ANNOTATION_SHA,
            'source_hashes':{p:sha(root/p) for p in SOURCE_PATHS},
            'new_generations':0,'published_AI_scores_modified':False}


def load_human(root):
    folder=root/PUBLIC
    require(sha(folder/'annotations.json')==ANNOTATION_SHA, 'Published human annotation identity changed')
    result=calculate(root,read(folder/'annotations.json'))
    require(read(folder/'summary.json')==result, 'Human summary does not match independent scalar audit')
    manifest=read(folder/'manifest.json')
    actual={p.name for p in folder.iterdir() if p.is_file()}
    require(set(manifest['files'])==actual-{'manifest.json'}, 'Human publication inventory mismatch')
    for name,meta in manifest['files'].items():
        path=folder/name
        require(Path(name).name==name and not path.is_symlink() and path.stat().st_size==meta['bytes']
                and sha(path)==meta['sha256'], 'Human publication member mismatch: '+name)
    require(manifest['source_annotation_sha256']==ANNOTATION_SHA and manifest['new_generations']==0,
            'Human publication scope mismatch')
    return result


def report_markdown(d, lang):
    ko=lang=='ko';t=lambda en,korean:korean if ko else en
    lines=[t('# Case 012 — Completed human review','# Case 012 — 완료한 사람 평가'),'',
        t('The user completed the image checklist. One hundred images were directly checked; answers from those images were explicitly reused for 92 visually similar images of the same request. All 192 images have assigned answers.',
          '사용자가 이미지 체크를 완료했습니다. 직접 체크한 100장의 답을 같은 요청의 비슷한 사진 92장에도 적용했고, 전체 192장에 답이 있습니다.'),'',
        t('## Observed results','## 확인한 결과'),'',
        t('| Setting | All constraints met | Rate | Median complete request |','| 설정 | 모든 조건 충족 | 충족률 | 전체 요청 시간 중앙값 |'),
        '|---|---:|---:|---:|']
    for s,label in zip(SETTINGS,('L1 / S89','L2 / S66','L4 / S50')):
        q=d['settings'][s];lines.append(f'| {label} | {q["passed_images"]}/{q["images"]} | {100*q["all_constraint_pass_rate"]:.2f}% | {d["timing"][s]["median_complete_seconds"]:.3f} s |')
    pair=d['paired'];p=d['primary']
    lines += ['',t(f'L4 gained all-constraint passes in {pair["C_only"]} inputs and lost them in {pair["A_only"]}, for a net difference of one image ({100*p["estimate"]:.4f} percentage points). Both settings passed {pair["both_pass"]} inputs; neither passed {pair["neither"]}.',
        f'L4는 같은 입력에서 {pair["C_only"]}개를 얻고 {pair["A_only"]}개를 잃었습니다. 순차이는 1장({100*p["estimate"]:.4f}%p)입니다. 두 설정 모두 맞힌 입력은 {pair["both_pass"]}개, 모두 못 맞힌 입력은 {pair["neither"]}개입니다.'),'',
        t('L2 had the shortest recorded median time. L4 had one more pass than L1 and two more than L2 in this grouped review. The measured times differ; this is not an exact equal-time comparison.',
          '기록된 요청 시간 중앙값은 L2가 가장 짧았습니다. 이번 묶음 평가에서 L4는 L1보다 1장, L2보다 2장 더 조건을 맞혔습니다. 측정 시간이 서로 달라, 정확히 같은 시간의 비교는 아닙니다.'),'',
        t('## Which requests were difficult?','## 어떤 요청이 어려웠나요?'),'',
        t('| Requested condition | L1 | L2 | L4 |','| 요청 종류 | L1 | L2 | L4 |'), '|---|---:|---:|---:|']
    names={'count':t('Count','개수'),'color_binding':t('Object-color binding','대상별 색'),
           'left_right':t('Left/right','좌우'),'compound':t('Combined constraints','복합 조건')}
    for cat in ('count','color_binding','left_right','compound'):
        cc=d['categories'][cat];lines.append('| '+names[cat]+' | '+' | '.join(f'{cc[s]["passed_images"]}/{cc[s]["images"]}' for s in SETTINGS)+' |')
    lines += ['',t('The combined requests were hardest: all constraints were met in roughly half of their images. This checklist scores count, requested colors and spatial relations. It does not separately score realism or the natural shape of identifiable objects.',
          '복합 요청이 가장 어려웠고, 모든 조건을 맞힌 이미지는 약 절반이었습니다. 이 체크리스트는 개수·요청한 색·공간 관계를 평가합니다. 알아볼 수 있는 물체의 형태가 자연스러운지, 사진이 사실적인지는 별도 점수에 포함하지 않습니다.'),'',
        t('## How these answers were counted','## 답을 어떻게 집계했나요?'),'',
        t(f'Of 480 constraint answers, {d["label_provenance"]["direct_constraints"]} were direct and {d["label_provenance"]["reused_constraints"]} were reused. Their image/group/source identities are preserved in annotations.json. Visual grouping is a workload convenience, not a claim that different PNGs are identical.',
          f'전체 480개 항목 답 중 직접 체크는 {d["label_provenance"]["direct_constraints"]}개, 묶음 적용은 {d["label_provenance"]["reused_constraints"]}개입니다. annotations.json에 이미지·묶음·답을 가져온 원본의 ID를 보존했습니다. 묶음은 작업량을 줄이는 기능이며 서로 다른 PNG가 동일하다는 뜻은 아닙니다.'),
        t('Uncertain answers count as unmet: three constraint answers were uncertain. If uncertain answers were instead accepted, passes would be 52/64, 53/64 and 54/64. This is a sensitivity calculation, not a replacement assessment.',
          '판단 불확실인 항목 3개는 미충족으로 계산했습니다. 불확실도 충족으로 가정하는 보조 계산에서는 52/64·53/64·54/64가 됩니다. 이는 민감도 계산이며 기존 체크를 바꾸지 않습니다.'),
        t('The 192 assigned labels are not 192 independent human judgments. No confidence interval or significance claim is attached to reused judgments. Sixteen prompts, each with four noise seeds, were evaluated under three settings. The reviewer had access to the previously published site; independent full blinding is not claimed.',
          '답이 있는 192장은 192번의 독립적인 사람 판독과 같지 않습니다. 묶음 답안에는 신뢰구간이나 통계적 우위 주장을 붙이지 않았습니다. 문장 16개마다 초기 잡음 4개를 사용해 세 설정을 비교했습니다. 평가자는 기존 공개 사이트를 볼 수 있었으므로 독립된 완전 블라인드 평가라고 부르지 않습니다.'),'',
        t('## Comparison with the preserved AI assessment','## 보존된 AI 평가와 비교'),'',
        t(f'The original AI passes remain 51/64, 53/64 and 53/64. Human and AI pass judgments differ on {d["AI_comparison"]["image_pass_disagreements"]}/192 images and {d["AI_comparison"]["constraint_disagreements"]}/480 constraint answers. These are disagreements, not a ground-truth test of either assessor.',
          f'기존 AI 평가는 51/64·53/64·53/64로 보존했습니다. 사람과 AI의 이미지 통과 여부가 다른 것은 {d["AI_comparison"]["image_pass_disagreements"]}/192장, 항목 답이 다른 것은 {d["AI_comparison"]["constraint_disagreements"]}/480개입니다. 이는 판단 차이이며 어느 평가자의 정답률을 검증한 수치가 아닙니다.'),'',
        t('## Check the records without models','## 모델 없이 직접 검산'),'',
        '```bash','python -B tools/showcase/case012_human.py --output ../case012-human-audit.json','```','',
        t('Run from a repository clone with the existing CPU requirements installed. The output must be a new file outside the repository. The auditor reads recorded answers, image IDs, timing and origins; no generation or model download is performed.',
          '기존 CPU 요구 패키지를 설치한 저장소 clone에서 실행합니다. 출력은 저장소 밖의 새 파일이어야 합니다. 기록된 답·이미지 ID·시간·출처만 읽으며 이미지 생성이나 모델 다운로드는 수행하지 않습니다.'),'',
        '[Annotations](annotations.json) · [Summary](summary.json) · [Manifest](manifest.json)',
        '[Original study](../../../'+C12+'/README'+('.ko' if ko else '')+'.md)',
        '',t('DIOVA added the assessment export, provenance checks and scalar comparison tools. OpenAI Codex assisted implementation and analysis. The ratings themselves are user-provided; no additional AI ratings were substituted.',
          'DIOVA는 체크 결과 저장·출처 검사·스칼라 비교 도구를 구현했습니다. OpenAI Codex가 구현과 분석을 지원했습니다. 답안은 사용자가 제공한 것이며 추가 AI 판단으로 대체하지 않았습니다.'),'']
    return '\n'.join(lines)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',type=Path,default=ROOT)
    p.add_argument('--output',type=Path,required=True,help='New external JSON receipt')
    args=p.parse_args();root=args.repo.resolve();out=args.output.resolve()
    require(not out.exists() and not out.is_relative_to(root), 'Use a new output file outside repository')
    result=load_human(root);out.parent.mkdir(parents=True,exist_ok=True)
    receipt={k:v for k,v in result.items() if k not in ('per_image','source_hashes')}
    receipt['audit_status']='PASS';out.write_text(json_text(receipt)+'\n')
    print(json_text({'status':'PASS','images':192,'settings':result['settings'],'primary':result['primary']}))


if __name__=='__main__':main()
