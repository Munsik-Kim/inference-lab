"""Paired API regression, template grouping, explicit missing observations."""
import random
from collections import Counter
from .common import METHODS,APIS,read,require,digest


def summarize(records,plan,seed=6101623,replicates=10000):
    expected={r['episode_id']:r for r in plan};require(len(expected)==96,'exact registered96cell plan required');seen=set();prompts={}
    for r in records:
        require(r['episode_id'] not in seen and r['episode_id'] in expected,'duplicate/unplanned episode');seen.add(r['episode_id']);p=expected[r['episode_id']]
        for k in ('method','api_version','scenario_id','canonical_task_hash','surface_prompt_hash'):require(r[k]==p[k],'identity mismatch: '+k)
        require(r['evidence_kind']=='measurement','synthetic trace must not enter measured result analysis')
        require(r['VERIFIED_WORKFLOW_SUCCESS'] is None or type(r['VERIFIED_WORKFLOW_SUCCESS']) is bool,'invalid workflow endpoint')
        require(not any(r[k] for k in ('enum_enabled','enhanced_feedback','aliases_enabled')),'baseline mitigation enabled')
        k=(r['scenario_id'],r['api_version'])
        if r['input_token_hash']:
            require(r['input_token_hash']==p['input_token_sha256'],'registered first input changed')
            prompts.setdefault(k,r['input_token_hash']);require(prompts[k]==r['input_token_hash'],'method pairing prompt mismatch')
    ids=sorted({r['scenario_id'] for r in plan});lookup={(r['scenario_id'],r['method'],r['api_version']):r for r in records}
    table=[];paired=[];templates={}
    fields=['STATE_ONLY_SUCCESS','FINAL_ANSWER_CORRECT','REQUIRED_ACTIONS_DONE','OBSERVATION_ORDER_VALID','UNKNOWN_TOOL','ARGUMENT_VALUE_WRONG','UNSUPPORTED_SUCCESS_CLAIM']
    for m in METHODS:
        for api in APIS:
            rr=[r for r in records if r['method']==m and r['api_version']==api];observed=[r for r in rr if r['VERIFIED_WORKFLOW_SUCCESS'] is not None]
            table.append({'method':m,'api_version':api,'verified_success':sum(r['VERIFIED_WORKFLOW_SUCCESS'] for r in observed),'observed':len(observed),'planned':16,
                **{k:sum(r[k] for r in observed) for k in fields},'observation_violation_episodes':sum(bool(r['observation_violations']) for r in observed),
                'model_calls':sum(r['model_calls'] for r in rr),'tool_attempts':sum(r['attempted_tool_calls'] for r in rr),'generated_tokens':sum(r['generated_tokens'] for r in rr),'episode_wall_seconds':sum(r['wall_seconds'] for r in rr)})
        counts=Counter();complete=[]
        for sid in ids:
            a=lookup.get((sid,m,APIS[0]));b=lookup.get((sid,m,APIS[1]))
            if a is None or b is None or a['VERIFIED_WORKFLOW_SUCCESS'] is None or b['VERIFIED_WORKFLOW_SUCCESS'] is None:counts['unobserved']+=1;continue
            x=a['VERIFIED_WORKFLOW_SUCCESS'];y=b['VERIFIED_WORKFLOW_SUCCESS'];counts['both_success' if x and y else 'loss' if x else 'gain' if y else 'both_fail']+=1;complete.append(sid)
        paired.append({'method':m,**{k:counts[k] for k in ['both_success','loss','gain','both_fail','unobserved']},'complete_pairs':len(complete),'all_API_success':counts['both_success']})
    for t in sorted({r['template_id'] for r in plan}):
        templates[t]=[{'method':m,'api_version':a,'success':sum(r['VERIFIED_WORKFLOW_SUCCESS'] is True for r in records if r['method']==m and r['api_version']==a and r['template_id']==t),'planned':4} for m in METHODS for a in APIS]
    complete=[sid for sid in ids if all((sid,m,a) in lookup and lookup[(sid,m,a)]['VERIFIED_WORKFLOW_SUCCESS'] is not None for m in METHODS for a in APIS)]
    auxiliary=None
    if len(complete)==16:
        per={sid:(int(lookup[sid,'D16',APIS[1]]['VERIFIED_WORKFLOW_SUCCESS'])-int(lookup[sid,'D16',APIS[0]]['VERIFIED_WORKFLOW_SUCCESS']))-(int(lookup[sid,'BF16',APIS[1]]['VERIFIED_WORKFLOW_SUCCESS'])-int(lookup[sid,'BF16',APIS[0]]['VERIFIED_WORKFLOW_SUCCESS'])) for sid in complete}
        groups=[[per[sid] for sid in complete if lookup[sid,'BF16',APIS[0]]['template_id']==t] for t in sorted(templates)]
        rng=random.Random(seed);draws=[]
        for _ in range(replicates):
            sample=[v for _ in groups for v in groups[rng.randrange(len(groups))]];draws.append(sum(sample)/len(sample))
        draws.sort()
        def q(x):
            pos=(len(draws)-1)*x;i=int(pos);w=pos-i;return draws[i]*(1-w)+draws[min(i+1,len(draws)-1)]*w
        auxiliary={'D16_API_change_minus_BF16_API_change':sum(per.values())/16,'exploratory_template_bootstrap_CI95':[q(.025),q(.975)],'template_groups':4,'replicates':replicates,'seed':seed,'scope':'secondary exploratory, fixed one workflow and API rename; zero-width interval is not equivalence'}
    return {'status':'COMPLETED_A1' if len(records)==96 and len(complete)==16 else 'PARTIAL','fresh_gpu_run':True,'planned':96,'measured_records':len(records),
        'missing':[r for k,r in expected.items() if k not in seen],'unobserved':[r['episode_id'] for r in records if r['VERIFIED_WORKFLOW_SUCCESS'] is None],
        'workflow_count':1,'scenario_count':16,'template_count':4,'score_version':'case015-a1-verified-workflow-v1','success_table':table,'paired_regression':paired,'template_results':templates,'auxiliary':auxiliary}
