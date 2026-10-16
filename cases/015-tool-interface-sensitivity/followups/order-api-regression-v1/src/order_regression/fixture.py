"""Real isolated JSON files, fixed ID-to-filename mapping, no executable input."""
import copy
from pathlib import Path
from .common import atomic, read, require, digest, same
from .registry import DEFINITIONS


class Fixture:
    def __init__(self, scenario, registry, root):
        self.scenario=copy.deepcopy(scenario);self.registry=registry
        self.root=Path(root).absolute()
        require(not self.root.exists(),'fixture root must be new')
        require(all(not p.is_symlink() for p in [self.root,*self.root.parents]),'symlink fixture path')
        self.root.mkdir(parents=True)
        self.pristine=copy.deepcopy(scenario['orders'])
        self.paths={oid:self.root/('order-'+str(i).zfill(2)+'.json') for i,oid in enumerate(sorted(self.pristine))}
        for oid,p in self.paths.items():atomic(p,self.pristine[oid],True)
        self.version=0;self.last_digest=digest(self.pristine);self.integrity_violation=False;self.events=[]

    def snapshot(self):
        require(set(p.name for p in self.root.iterdir())==set(p.name for p in self.paths.values()),'unexpected fixture file')
        state={}
        for oid,p in self.paths.items():
            require(p.is_file() and not p.is_symlink(),'unsafe fixture file')
            state[oid]=read(p)
        current=digest(state)
        if current!=self.last_digest:self.version+=1;self.last_digest=current
        self.integrity_violation |= not integrity_ok(self.scenario,state)
        return state

    def execute(self, raw_name, args, turn, index, available_ids):
        before=self.snapshot();vb=self.version;semantic=self.registry.resolve(raw_name)
        status='SUCCESS';schema='VALID';labels=[];result=None
        try:
            if semantic is None:
                labels.append('UNKNOWN_TOOL');schema='NOT_EVALUABLE';raise ValueError()
            if not isinstance(args,dict):schema='NOT_OBJECT';raise ValueError()
            properties=next(p for s,d,p in DEFINITIONS if s==semantic)
            if set(args)!=set(properties):schema='KEYS';raise ValueError()
            for key,definition in properties.items():
                if type(args[key]) is not (str if definition['type']=='string' else int):schema='TYPE';raise ValueError()
            if semantic!='amount.multiply' and args['order_id'] not in self.paths:
                schema='UNKNOWN_ORDER';raise ValueError()
            if semantic=='order.read':
                oid=args['order_id'];r=before[oid]
                result={'order_id':oid,'quantity':r['quantity'],'unit_price_minor':r['unit_price_minor'],'current_total_minor':r['total_minor']}
            elif semantic=='amount.multiply':result={'product':args['a']*args['b']}
            elif semantic=='order.write':
                oid=args['order_id'];row=copy.deepcopy(before[oid]);row['total_minor']=args['total_minor'];atomic(self.paths[oid],row)
                result={'updated':True,'order_id':oid,'total_minor':args['total_minor']}
            else:
                oid=args['order_id'];r=before[oid];original=self.pristine[oid]
                immutable={k:v for k,v in r.items() if k!='total_minor'}
                valid=same(immutable,{k:v for k,v in original.items() if k!='total_minor'}) and type(r.get('total_minor')) is int and r['total_minor']==original['quantity']*original['unit_price_minor']
                result={'valid':bool(valid),'order_id':oid}
        except (ValueError,TypeError):
            status='ERROR';result={'error':'unknown tool or invalid arguments'}
            if semantic is not None:labels.append('BAD_ARGUMENTS')
        after=self.snapshot()
        event={'event_id':f'g{turn}-c{index}','generated_in_turn':turn,'call_index':index,
            'generated_after_event_ids':list(available_ids),'raw_name':raw_name,'semantic_tool_id':semantic,
            'raw_arguments':copy.deepcopy(args),'declared_names_hash':self.registry.names_hash,'schema_result':schema,
            'execution_status':status,'tool_result':result,'diagnostic_labels':labels,
            'state_before_execution':before,'state_after_execution':after,
            'state_before_execution_digest':digest(before),'state_after_execution_digest':digest(after),
            'state_version_before':vb,'state_version_after':self.version,'integrity_violation_so_far':self.integrity_violation}
        self.events.append(event);return event


def integrity_ok(scenario,state):
    pristine=scenario['orders'];target=scenario['target_order_id']
    if set(state)!=set(pristine):return False
    for oid,original in pristine.items():
        if oid!=target:
            if not same(state[oid],original):return False
        elif not same({k:v for k,v in state[oid].items() if k!='total_minor'},{k:v for k,v in original.items() if k!='total_minor'}):return False
    return True
