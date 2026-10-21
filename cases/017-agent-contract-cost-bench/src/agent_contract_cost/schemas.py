"""Registered backend scope, separate from the shared strict host contract.

The installed xgrammar 0.2.8 compiles tool-specific oneOf schemas, but CPU
token-mask evidence shows it accepts duplicate array items despite uniqueItems.
We therefore omit that unsupported keyword from the *server* grammar rather
than pretend it is enforced. Every policy keeps the same strict host validator.
No task identifiers, task state or hidden answer enters this transformation.
"""
import copy

from .contracts import grammar_schema

G_SCHEMA_SCOPE = "TOOL_BRANCH_SCHEMA_HOST_ONLY_UNIQUE_ITEMS"


def server_schema(definitions):
    schema = copy.deepcopy(grammar_schema(definitions))

    def visit(node):
        if isinstance(node, dict):
            node.pop("uniqueItems", None)
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(schema)
    return schema
