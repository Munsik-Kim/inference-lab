"""Synthetic, owned fixtures. Oracle data never enters messages or tool schema."""
import copy
import hashlib
import random
from .common import digest, dumps

FAMILIES=('A','B','C','D')
DESCRIPTIONS={
 'A':'Choose the lowest-price eligible quote received by the cutoff; ties use the smaller source ID.',
 'B':'Choose the highest source version received by the cutoff, regardless of arrival order.',
 'C':'Ignore arrivals after the cutoff. Among remaining events choose the latest event_time; ties use the higher version.',
 'D':'Choose the highest version received by the cutoff. A cancellation at that version takes priority over a correction.'}

def oracle(task):
    visible=task['visible'];events=[r for r in visible['sources'] if r['arrival_time']<=visible['cutoff']]
    fam=task['family']
    if fam=='A':selected=min(events,key=lambda r:(r['unit_price_minor'],r['source_id']))
    elif fam=='B':selected=max(events,key=lambda r:r['version'])
    elif fam=='C':selected=max(events,key=lambda r:(r['event_time'],r['version']))
    else:selected=max(events,key=lambda r:(r['version'],r['kind']=='cancel'))
    quantity=0 if selected['kind']=='cancel' else visible['quantity']
    value=(quantity*selected['unit_price_minor']*(100-visible['discount_percent'])+50)//100
    return {'source_id':selected['source_id'],'quantity':quantity,'unit_price_minor':selected['unit_price_minor'],
            'discount_percent':visible['discount_percent'],'total_minor':value,'source_version':selected['version'],
            'status':'cancelled' if selected['kind']=='cancel' else 'active','record_id':visible['record_id']}

def make_tasks(split,n_each,seed):
    rng=random.Random(seed);tasks=[]
    for family in FAMILIES:
        for i in range(n_each):
            tid=f'{split}-{family}{i:02d}'
            ids=['S-'+hashlib.sha256(f'{seed}|{tid}|{j}'.encode()).hexdigest()[:10] for j in range(3)]
            prices=rng.sample(range(100,20001),3);versions=[1,3,2]
            events=[{'source_id':ids[j],'version':versions[j],'event_time':f'2026-10-{15+j:02d}T08:00:00Z',
                     'arrival_time':f'2026-10-{16+(2-j):02d}T12:00:00Z','unit_price_minor':prices[j],
                     'kind':'cancel' if family=='D' and j==1 else 'quote' if family=='A' else 'correction'} for j in range(3)]
            # Delayed newer event is unavailable at the registered cutoff for C.
            if family=='C':events[2]['arrival_time']='2026-10-21T12:00:00Z'
            rng.shuffle(events)
            v={'record_id':'R-'+hashlib.sha256((tid+'target').encode()).hexdigest()[:10],
               'quantity':rng.randint(2,40),'discount_percent':rng.choice([0,5,10,15]),
               'cutoff':'2026-10-19T00:00:00Z','rule':DESCRIPTIONS[family],'sources':events}
            state={v['record_id']:{'total_minor':None,'source_version':0,'status':'pending','protected_tag':'target'},
                   'CONTROL-'+tid:{'total_minor':12345,'source_version':7,'status':'active','protected_tag':'unchanged'}}
            t={'task_id':tid,'family':family,'split':split,'visible':v,'initial_state':state,
               'evidence_kind':'synthetic_fixture','document_unit':'one generator instance within one of four families',
               'obligation_order':['source','calculation','write','validation','final'],
               'essential_facts':['selection_rule_and_cutoff','quantity_and_discount','source_candidates'],
               'dependency_edges':[['source','calculation'],['calculation','write'],['write','validation'],['validation','final']]}
            t['oracle']=oracle(t);t['canonical_task_hash']=digest({k:v for k,v in t.items() if k!='oracle'});tasks.append(t)
    return tasks

def tools(menu=8):
    def tool(name,description,props,required=None):
        return {'type':'function','function':{'name':name,'description':description,'parameters':{
            'type':'object','properties':props,'required':list(props) if required is None else required,'additionalProperties':False}}}
    s={'type':'string'};integer={'type':'integer'}
    base=[
      tool('read_source','Read exactly one identified source. It does not select a latest source or return other sources.',{'source_id':s}),
      tool('calculate_amount','Return floor((quantity * unit_price_minor * (100-discount_percent) + 50)/100). All values are integer minor units.',{'quantity':integer,'unit_price_minor':integer,'discount_percent':integer}),
      tool('update_state','Store the supplied values in one record without correcting them.',{'record_id':s,'total_minor':integer,'source_version':integer,'status':{'type':'string','enum':['active','cancelled']}}),
      tool('validate_state','Check public field types and nonnegative values in the current record. This does not check task correctness.',{'record_id':s}),
      tool('check_timestamp','Return whether a string uses the documented UTC timestamp format.',{'timestamp':s}),
      tool('format_reference','Echo a reference without looking up data.',{'reference':s}),
      tool('read_currency_policy','Return that money is represented in integer minor units. No task facts are returned.',{}),
      tool('service_status','Return availability of this isolated synthetic service.',{})]
    for i in range(menu-8):base.append(tool(f'catalogue_info_{i+1:02d}','Return unrelated static catalogue metadata. Does not read or modify workflow records.',{}))
    if menu not in (8,32):raise ValueError('registered menus are 8 and 32')
    return base

SYSTEM='You are completing a synthetic record reconciliation task. Use the supplied tools; do not generate executable code. No real account is connected.'
def message_parts(task):
    v=task['visible']
    early='TASK POLICY\n'+dumps({'rule':v['rule'],'cutoff':v['cutoff']})+'\n'
    middle='REQUIRED RECORD FACTS\n'+dumps({'record_id':v['record_id'],'quantity':v['quantity'],'discount_percent':v['discount_percent']})+'\n'
    late='SOURCE INDEX\n'+dumps(v['sources'])+'\n'
    request=('WORK REQUEST\nReconcile record '+v['record_id']+'. Apply the policy and cutoff above to select a source. '
      'Read that exact source using read_source, then after receiving its result calculate the amount with calculate_amount. '
      'Use the original quantity unless the selected source is a cancellation; then quantity is zero. '
      'Use the stated discount. After seeing the calculation result, store total_minor, the selected source_version and status '
      '(active or cancelled) using update_state. Validate the updated record, then after receiving that validation return only JSON '
      'with exactly record_id:string, total_minor:integer, source_version:integer, status:string, validated:boolean. '
      'Other records and protected fields must remain unchanged. update_state and validate_state may be called in the same response '
      'in that order. A public validation is not a correctness oracle.\n')
    return early,middle,late,request

def short_messages(task):
    return [{'role':'system','content':SYSTEM},{'role':'user','content':''.join(message_parts(task))}]
