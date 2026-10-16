"""Task-paired interface interactions; stdlib only, missingness is explicit."""
import itertools
import math
import random
from collections import Counter
from .common import METHODS, INTERFACES, require


def quantile(values, q):
    s = sorted(values); x = (len(s)-1)*q; lo = int(x); hi = min(lo+1, len(s)-1)
    return s[lo]*(hi-x) + s[hi]*(x-lo) if hi != lo else s[lo]


def interaction(task_rows, candidate='D16', baseline='BF16'):
    def d(method):
        return math.fsum(task_rows[(method, i)] for i in INTERFACES[1:])/3-task_rows[(method, INTERFACES[0])]
    return d(candidate)-d(baseline)


def validate_records(records, plan):
    expected = {x['episode_id']: x for x in plan}
    require(len(expected) == len(plan), 'duplicate planned cell')
    seen = set(); canonical = {}; prompt = {}
    for r in records:
        require(r['episode_id'] not in seen, 'duplicate completed episode')
        seen.add(r['episode_id']); require(r['episode_id'] in expected, 'unplanned episode')
        p = expected[r['episode_id']]
        for key in ('method', 'interface_id', 'task_id', 'canonical_task_hash', 'rendered_prompt_hash'):
            require(r[key] == p[key], 'planned identity mismatch: '+key)
        require(type(r['functional_success']) is bool or r['functional_success'] is None, 'invalid endpoint')
        require(r['method'] in METHODS and r['interface_id'] in INTERFACES, 'unknown condition')
        require(not any(r[k] for k in ('enum_enabled', 'aliases_enabled', 'enhanced_feedback_enabled')), 'baseline mitigation enabled')
        k = r['task_id']; canonical.setdefault(k, r['canonical_task_hash'])
        require(canonical[k] == r['canonical_task_hash'], 'canonical task mismatch')
        k = (r['task_id'], r['interface_id'])
        if r['input_token_hash'] is not None:
            prompt.setdefault(k, r['input_token_hash'])
            require(prompt[k] == r['input_token_hash'], 'within-interface model input mismatch')
    return [p for k, p in expected.items() if k not in seen]


def summarize(records, plan, bootstrap=10000, seed=6101603):
    missing = validate_records(records, plan)
    tasks = sorted({p['task_id'] for p in plan}); lookup = {(r['task_id'],r['method'],r['interface_id']):r for r in records}
    table = []
    for m in METHODS:
        for i in INTERFACES:
            rr = [lookup.get((t,m,i)) for t in tasks]
            observed = [r for r in rr if r is not None and r['functional_success'] is not None]
            table.append({'method':m,'interface_id':i,'success':sum(r['functional_success'] for r in observed),
                          'observed_tasks':len(observed),'planned_tasks':len(tasks),'legacy_success':sum(r['legacy_success'] for r in observed),
                          'task_results':{t:lookup[(t,m,i)]['functional_success'] if (t,m,i) in lookup else None for t in tasks}})
    complete = [t for t in tasks if all((t,m,i) in lookup and lookup[(t,m,i)]['functional_success'] is not None for m in METHODS for i in INTERFACES)]
    per_task = {}; paired = []; changes = []
    for t in tasks:
        if t in complete:
            per_task[t] = {(m,i):int(lookup[(t,m,i)]['functional_success']) for m in METHODS for i in INTERFACES}
    contrasts = {}
    rng = random.Random(seed)
    indexes = [[rng.randrange(len(complete)) for _ in complete] for _ in range(bootstrap)] if complete else []
    for a,b in [('D16','BF16'),('RTN','BF16'),('D16','RTN')]:
        values = [interaction(per_task[t],a,b) for t in complete]
        samples = [math.fsum(values[j] for j in idx)/len(idx) for idx in indexes]
        contrasts[a+'-minus-'+b] = {'estimate':math.fsum(values)/len(values) if values else None,
                                  'CI95':[quantile(samples,.025),quantile(samples,.975)] if samples else None,
                                  'per_task':dict(zip(complete,values)), 'complete_task_count':len(complete)}
    for m in METHODS:
        for i in INTERFACES[1:]:
            pairs = [(lookup.get((t,m,INTERFACES[0])),lookup.get((t,m,i))) for t in tasks]
            count = Counter()
            for a,b in pairs:
                if a is None or b is None or a['functional_success'] is None or b['functional_success'] is None:
                    count['unobserved'] += 1; continue
                x,y = a['functional_success'],b['functional_success']
                count['both_success' if x and y else 'baseline_only' if x else 'changed_only' if y else 'both_fail'] += 1
            paired.append({'method':m,'baseline':INTERFACES[0],'changed':i,**{k:count[k] for k in ['both_success','baseline_only','changed_only','both_fail','unobserved']}})
        complete_m = [t for t in tasks if all((t,m,i) in lookup and lookup[(t,m,i)]['functional_success'] is not None for i in INTERFACES)]
        abs_changes = [abs(int(lookup[(t,m,i)]['functional_success'])-int(lookup[(t,m,INTERFACES[0])]['functional_success'])) for t in complete_m for i in INTERFACES[1:]]
        changes.append({'method':m,'S':math.fsum(abs_changes)/len(abs_changes) if abs_changes else None,
                        'changed_pairs':sum(abs_changes),'observed_pairs':len(abs_changes),'complete_tasks':len(complete_m),
                        'all_interface_success':sum(all(lookup[(t,m,i)]['functional_success'] for i in INTERFACES) for t in complete_m)})
    family_map = {p['task_id']:p['family_id'] for p in plan}; families = {}
    for f in sorted(set(family_map.values())):
        ts = [t for t in complete if family_map[t] == f]
        other = [t for t in complete if family_map[t] != f]
        families[f] = {'tasks':ts,'primary_D16_minus_BF16':math.fsum(interaction(per_task[t]) for t in ts)/len(ts) if ts else None,
                       'leave_this_family_out':math.fsum(interaction(per_task[t]) for t in other)/len(other) if other else None}
    # A linear contrast permits exact extrema: each shared I0 coefficient appears
    # once with its full weight; it is never imputed independently three times.
    bounds = []
    for t in tasks:
        lo=hi=0.
        for m,sign in [('D16',1),('BF16',-1)]:
            for i in INTERFACES:
                coefficient = sign * (-1 if i==INTERFACES[0] else 1/3)
                r = lookup.get((t,m,i)); value = r.get('functional_success') if r else None
                if value is None:
                    lo += min(0,coefficient); hi += max(0,coefficient)
                else:
                    lo += coefficient*value; hi += coefficient*value
        bounds.append((lo,hi))
    metrics = []
    for m in METHODS:
        for i in INTERFACES:
            rr=[r for r in records if r['method']==m and r['interface_id']==i]
            ev=[e for r in rr for e in r['tool_events']]; name_ev=[e for e in ev if isinstance(e['raw_name'],str)]
            semantic_ev=[e for e in ev if e['semantic_tool_id'] is not None]
            arg_ev=[e for e in semantic_ev if e['schema_result']!='NOT_EVALUABLE']
            rec=[r for r in rr if any(e['subsequent_action_observed'] for e in r['recovery'])]
            metrics.append({'method':m,'interface_id':i,'name_valid':sum(e['semantic_tool_id'] is not None for e in name_ev),'name_evaluable':len(name_ev),
                            'unnameable_call_objects':len(ev)-len(name_ev),'semantic_correct':sum(e['semantic_choice_correct'] for e in semantic_ev),'semantic_evaluable':len(semantic_ev),
                            'schema_valid':sum(e['schema_result']=='VALID' for e in arg_ev),'schema_evaluable':len(arg_ev),
                            'argument_value_correct':sum(e['argument_value_correct'] is True for e in semantic_ev),'argument_value_evaluable':sum(e['argument_value_correct'] is not None for e in semantic_ev),
                            'recovery_episodes_evaluable':len(rec),'episodes_with_later_valid_action':sum(any(e['registered_valid_later_action'] for e in r['recovery']) for r in rec),
                            'first_error_counts':dict(Counter(r['first_error_type'] for r in rr if r['first_error_type'])),
                            'event_error_counts':dict(Counter(label for e in ev for label in e['error_labels'])),
                            'no_call_episodes':sum(r['tool_attempts']==0 for r in rr),'no_call_validity':'NOT_APPLICABLE'})
    return {'status':'COMPLETED_P1' if len(records)==len(plan) and not missing and len(complete)==len(tasks) else 'PARTIAL',
            'planned_episodes':len(plan),'completed_records':len(records),'complete_tasks':complete,'missing':missing,
            'unobserved_episodes':[r['episode_id'] for r in records if r['functional_success'] is None],
            'success_table':table,'contrasts':contrasts,'paired_interface_counts':paired,'sensitivity':changes,'families':families,
            'primary_worst_best_all_planned_tasks':[math.fsum(x[0] for x in bounds)/len(tasks),math.fsum(x[1] for x in bounds)/len(tasks)],
            'call_metrics':metrics,'bootstrap':{'replicates':bootstrap,'seed':seed,'unit':'whole task 12-cell vector; fixed observed DEV tasks','interval':'exploratory task-paired percentile95; not external population guarantee'},
            'actual_model_calls':sum(r['model_calls'] for r in records),'actual_generated_tokens':sum(r['generated_tokens'] for r in records),
            'actual_tool_attempts':sum(r['tool_attempts'] for r in records)}
