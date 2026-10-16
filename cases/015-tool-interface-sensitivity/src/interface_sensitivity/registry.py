"""Exact bijection between exposed names and private semantic identities."""
from .common import require, digest

SEMANTICS = {
    'read_document': 'document.read', 'read_record': 'record.read',
    'query_table': 'updates.read', 'calculate': 'math.multiply',
    'apply_fixture_patch': 'fixture.patch', 'validate_fixture_test': 'fixture.validate',
}
ORIGINAL_NAMES = {v: k for k, v in SEMANTICS.items()}


class Registry:
    def __init__(self, mapping):
        require(set(mapping) == set(ORIGINAL_NAMES), 'missing/extra semantic tools')
        require(all(isinstance(x, str) and x and x.isascii() and x.replace('_', '').isalnum() for x in mapping.values()), 'invalid exposed name')
        require(len(set(mapping.values())) == len(mapping), 'exposed-name collision')
        self.mapping = dict(mapping)
        self.inverse = {name: semantic for semantic, name in mapping.items()}
        self.names_hash = digest(sorted(self.inverse))

    def resolve(self, name):
        return self.inverse.get(name) if isinstance(name, str) else None

    def expose(self, semantic):
        return self.mapping[semantic]

    def original(self, name):
        semantic = self.resolve(name)
        return ORIGINAL_NAMES.get(semantic)
