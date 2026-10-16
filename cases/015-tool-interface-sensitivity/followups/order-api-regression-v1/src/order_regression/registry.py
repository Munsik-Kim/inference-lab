"""Four-tool bijection. Both API versions share descriptions and parameters."""
import copy
from .common import require, digest, APIS

MAPS = {
    'API_V1': {'order.read':'read_order','amount.multiply':'multiply_amount','order.write':'write_order_total','order.validate':'validate_order'},
    'API_V2': {'order.read':'readOrder','amount.multiply':'multiplyAmount','order.write':'writeOrderTotal','order.validate':'validateOrder'},
}
DEFINITIONS = [
    ('order.read','Read the requested order quantity, unit price in integer minor units, and current total.',{'order_id':{'type':'string'}}),
    ('amount.multiply','Multiply two integers and return their integer product.',{'a':{'type':'integer'},'b':{'type':'integer'}}),
    ('order.write','Store the provided integer total in minor units for the requested order, preserving all other fields.',{'order_id':{'type':'string'},'total_minor':{'type':'integer'}}),
    ('order.validate','Check the current stored order and return whether it is valid.',{'order_id':{'type':'string'}}),
]


class Registry:
    def __init__(self, api, mapping=None):
        require(api in APIS,'unknown API version')
        self.api=api;self.mapping=dict(MAPS[api] if mapping is None else mapping)
        require(set(self.mapping)==set(MAPS['API_V1']),'missing/extra semantic identity')
        require(all(type(n) is str and n.isascii() and n.replace('_','').isalnum() for n in self.mapping.values()),'invalid exposed name')
        require(len(set(self.mapping.values()))==4,'mapping collision')
        self.inverse={n:s for s,n in self.mapping.items()}
        self.names_hash=digest(list(self.mapping.values()))

    def resolve(self, name):
        return self.inverse.get(name) if type(name) is str else None

    def tools(self):
        return [{'type':'function','function':{'name':self.mapping[s],'description':d,
            'parameters':{'type':'object','properties':copy.deepcopy(p),'required':list(p),'additionalProperties':False}}} for s,d,p in DEFINITIONS]

    def expose(self, semantic):
        return self.mapping[semantic]
