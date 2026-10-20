"""A closed callable registry, no dynamic code/path/network dispatch."""
import copy
import json
import pathlib
import re
from .fixture_generator import tools
from .common import digest, save

class Fixture:
    def __init__(self,task,root,menu=8):
        self.task=copy.deepcopy(task);self.root=pathlib.Path(root)
        if self.root.exists() and (self.root.is_symlink() or any(self.root.iterdir())):raise ValueError('fresh non-symlink fixture root required')
        self.root.mkdir(parents=True,exist_ok=True);self.path=self.root/'state.json';self.state=copy.deepcopy(task['initial_state']);self.version=0;self.events=[];self.integrity_violation=False
        self.schemas={x['function']['name']:x['function']['parameters'] for x in tools(menu)};save(self.path,self.state)
    def snapshot(self):return copy.deepcopy(self.state)
    def integrity_ok(self):
        target=self.task['visible']['record_id'];initial=self.task['initial_state']
        return set(self.state)==set(initial) and all(self.state[k]==initial[k] for k in initial if k!=target) and self.state[target].get('protected_tag')==initial[target]['protected_tag']
    def execute(self,name,args,turn,index,available):
        before=self.snapshot();version=self.version;status='SUCCESS';result=None;error=None
        schema=self.schemas.get(name)
        try:
            if schema is None:raise ValueError('UNKNOWN_TOOL')
            props=schema['properties']
            if type(args)is not dict or set(args)!=set(props):raise ValueError('BAD_ARGUMENT_KEYS')
            for k,p in props.items():
                if p['type']=='integer' and type(args[k])is not int:raise ValueError('BAD_ARGUMENT_TYPE')
                if p['type']=='string' and type(args[k])is not str:raise ValueError('BAD_ARGUMENT_TYPE')
                if 'enum'in p and args[k]not in p['enum']:raise ValueError('BAD_ARGUMENT_VALUE')
            if name=='read_source':
                rows=[r for r in self.task['visible']['sources'] if r['source_id']==args['source_id']]
                if not rows:raise ValueError('UNKNOWN_SOURCE')
                result=copy.deepcopy(rows[0])
            elif name=='calculate_amount':
                result={'total_minor':(args['quantity']*args['unit_price_minor']*(100-args['discount_percent'])+50)//100}
            elif name=='update_state':
                if args['record_id'] not in self.state:raise ValueError('UNKNOWN_RECORD')
                self.state[args['record_id']].update({k:v for k,v in args.items() if k!='record_id'});self.version+=1
                result={'updated':True,**args}
            elif name=='validate_state':
                row=self.state.get(args['record_id'])
                valid=bool(row and type(row.get('total_minor'))is int and row['total_minor']>=0 and type(row.get('source_version'))is int and row['source_version']>=0 and row.get('status')in('active','cancelled'))
                result={'record_id':args['record_id'],'valid':valid,'state_version':self.version}
            elif name=='check_timestamp':result={'valid':bool(re.fullmatch(r'\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ',args['timestamp']))}
            elif name=='format_reference':result={'reference':args['reference']}
            elif name=='read_currency_policy':result={'money_representation':'integer minor units'}
            else:result={'available':True,'metadata':'synthetic static catalogue'}
        except ValueError as e:status='REJECTED';error=str(e);result={'error':error}
        self.integrity_violation|=not self.integrity_ok();save(self.path,self.state)
        e={'event_id':f'e{len(self.events)}','name':name,'arguments':args,'generated_in_turn':turn,'call_index':index,
           'generated_after_event_ids':list(available),'status':status,'result':result,'error':error,
           'state_before':before,'state_after':self.snapshot(),'before_digest':digest(before),'after_digest':digest(self.state),
           'version_before':version,'version_after':self.version}
        self.events.append(e);return e
