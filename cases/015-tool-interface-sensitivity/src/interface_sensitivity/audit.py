"""CPU replay of saved model text, actual fixture transitions and paired scalars."""
import copy
import hashlib
import math
from pathlib import Path
import random
import tempfile
from .common import inputs, digest, read, rows, require, INTERFACES
from .runner import run_episode
from .transform import make_interface
from .analysis import summarize


def independent_verdict(task,final,state,events):
    answer=isinstance(final,dict) and all(final.get(k)==v for k,v in task['gold'].items())
    executed=[e for e in events if e['execution_status']=='SUCCESS']
    valid=any(e['semantic_tool_id']=='fixture.validate' and e['tool_result'].get('valid') is True for e in executed)
    legacy=answer and (task['family']!='fixture_patch' or (state==task['fixture']['expected_config'] and valid))
    obligations={'document':['document.read'],'updates':['updates.read'],'combine':['record.read','math.multiply'],
                 'fixture_patch':['fixture.patch','fixture.validate']}[task['family']]
    cursor=-1;passed=True
    for required in obligations:
        found=None
        for i,e in enumerate(events):
            if i<=cursor or e['execution_status']!='SUCCESS' or e['semantic_tool_id']!=required:
                continue
            args=e['raw_arguments']
            if required=='math.multiply':
                r=task['fixture']['record']
                if sorted(args.values())!=sorted([r['quantity'],r['unit_price']]):continue
            if required=='fixture.patch' and args!=task['fixture']['expected_config']:continue
            if required=='fixture.validate' and not (e['tool_result'].get('valid') is True and e['state_after']==state):continue
            found=i;break
        if found is None:passed=False;break
        cursor=found
    expected=task['fixture']['expected_config'] if task['family']=='fixture_patch' else task['fixture']['config']
    return bool(answer and passed and state==expected),bool(legacy)


def audit(case,records_path):
    case=Path(case);records=rows(records_path);tasks,messages,tools=inputs(case);tasks={t['id']:t for t in tasks}
    protocol=read(case/'specs/protocol.json');plan=read(case/'results/p1_r1/planned_cells.json')
    checked=events_count=responses_count=0
    with tempfile.TemporaryDirectory(prefix='case015-raw-audit-') as tmp:
        for row in records:
            task=tasks[row['task_id']];spec=make_interface(task,messages[task['id']],tools,row['interface_id'])
            if row['observation_status']!='OBSERVED':
                # Missing model actions cannot be recreated from absent text.
                continue
            queue=iter(row['responses'])
            def generate(history,declared,deadline):
                r=copy.deepcopy(next(queue))
                require(digest({'messages':history,'tools':declared})==r['messages_tools_hash'],'raw history/declaration mismatch')
                require(hashlib.sha256(r['raw_text'].encode()).hexdigest()==r['raw_utf8_sha256'],'raw response bytes mismatch')
                require(len(r['output_token_ids'])==r['output_tokens'],'output token count mismatch')
                return r
            identity={k:row[k] for k in ('method','model_id','model_revision','artifact_hash','generation_config_hash','execution_source_sha256')}
            replay=run_episode(task,spec,generate,Path(tmp)/row['episode_id'],protocol['generation'],identity,row['episode_id'])
            for key in ('tool_events','functional_success','legacy_success','final_answer','final_fixture','model_calls','tool_attempts','generated_tokens','first_error_type','first_error_turn','terminal_reason'):
                require(replay[key]==row[key],'replay differs: '+row['episode_id']+': '+key)
            function,legacy=independent_verdict(task,replay['final_answer'],replay['final_fixture'],replay['tool_events'])
            if row['terminal_reason'] in ('BUDGET_EXHAUSTED','TIMEOUT','CONTEXT_LIMIT'):
                function=legacy=False
            require((function,legacy)==(row['functional_success'],row['legacy_success']),'independent action/state oracle mismatch')
            checked+=1;events_count+=len(replay['tool_events']);responses_count+=len(replay['responses'])
    summary=summarize(records,plan,protocol['bootstrap_replicates'],protocol['bootstrap_seed'])
    # Independently construct the primary contrast without analysis.interaction.
    lookup={(r['task_id'],r['method'],r['interface_id']):r['functional_success'] for r in records}
    values=[]
    for t in summary['complete_tasks']:
        values.append(sum((int(lookup[t,'D16',i])-int(lookup[t,'BF16',i]))/3 for i in INTERFACES[1:])
                      -int(lookup[t,'D16',INTERFACES[0]])+int(lookup[t,'BF16',INTERFACES[0]]))
    if values:
        require(abs(sum(values)/len(values)-summary['contrasts']['D16-minus-BF16']['estimate'])<1e-14,'independent primary arithmetic mismatch')
        rng=random.Random(protocol['bootstrap_seed']);samples=[]
        for _ in range(protocol['bootstrap_replicates']):
            samples.append(sum(values[rng.randrange(len(values))] for _ in values)/len(values))
        samples.sort()
        def q(frac):
            pos=(len(samples)-1)*frac;i=int(pos);w=pos-i
            return samples[i]*(1-w)+samples[min(i+1,len(samples)-1)]*w
        independent_ci=[q(.025),q(.975)]
        require(all(abs(a-b)<1e-14 for a,b in zip(independent_ci,summary['contrasts']['D16-minus-BF16']['CI95'])),'independent bootstrap mismatch')
    else:independent_ci=None
    return {'status':'PASS','observed_episodes_replayed':checked,'actual_fixture_actions_reexecuted':events_count,
        'saved_model_responses_checked':responses_count,'planned_cells':len(plan),'missing_cells':len(summary['missing']),
        'independent_primary_CI95':independent_ci,'summary_recomputed_from_raw':True,
        'scope':'CPU saved-response/action/state/scalar/paired bootstrap audit; not GPU inference/logit or model-weight replication'}
