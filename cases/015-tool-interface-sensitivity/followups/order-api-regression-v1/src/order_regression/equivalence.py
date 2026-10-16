"""CPU gold traces: actual execution with result receipt at generation boundaries."""
import tempfile
from pathlib import Path
from .common import ROOT,APIS,read,require,digest,same
from .fixture import Fixture
from .registry import Registry
from .scoring import assess,expected_total


def gold_trace(scenario,api,root,combined_write_validate=True):
    registry=Registry(api);fixture=Fixture(scenario,registry,root);target=scenario['target_order_id'];order=scenario['orders'][target]
    available=[];responses=[]
    stages=[[('order.read',{'order_id':target})],[('amount.multiply',{'a':order['quantity'],'b':order['unit_price_minor']})],
        [('order.write',{'order_id':target,'total_minor':expected_total(scenario)})], [('order.validate',{'order_id':target})]]
    if combined_write_validate:stages=[stages[0],stages[1],stages[2]+stages[3]]
    for turn,calls in enumerate(stages):
        frozen=tuple(available);responses.append({'available_result_event_ids':list(frozen),'generated_in_turn':turn})
        made=[]
        for i,(semantic,args) in enumerate(calls):
            e=fixture.execute(registry.expose(semantic),args,turn,i,frozen);made.append(e['event_id'])
        available.extend(made)
    responses.append({'available_result_event_ids':list(available),'generated_in_turn':len(stages)})
    final={'order_id':target,'total_minor':expected_total(scenario),'validated':True};state=fixture.snapshot()
    verdict=assess(scenario,fixture.events,responses,final,state,fixture.version,fixture.integrity_violation)
    return fixture,responses,verdict


def oracle_check(root=ROOT):
    scenarios=read(Path(root)/'data/scenarios.json');records=[]
    with tempfile.TemporaryDirectory(prefix='case015-a1-oracle-') as tmp:
        for scenario in scenarios:
            base=None
            for api in APIS:
                fixture,responses,verdict=gold_trace(scenario,api,Path(tmp)/scenario['scenario_id']/api)
                normalized=[{k:e[k] for k in ['semantic_tool_id','raw_arguments','tool_result','state_before_execution','state_after_execution','state_version_after','generated_after_event_ids']} for e in fixture.events]
                require(verdict['VERIFIED_WORKFLOW_SUCCESS'],'gold workflow did not pass')
                if base is None:base=normalized
                require(same(base,normalized),'API state/result/observation equivalence mismatch')
                records.append({'scenario_id':scenario['scenario_id'],'api_version':api,'evidence_kind':'synthetic_contract_test','normalized_trace':normalized,'trace_sha256':digest(normalized),'verdict':verdict})
    return {'status':'PASS','synthetic_contract_episodes':len(records),'actual_tool_actions':sum(len(r['normalized_trace']) for r in records),'model_forwards':0,'records':records}
