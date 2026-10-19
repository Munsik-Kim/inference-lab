"""Optional installed-library CPU checks; not measured recovery responses."""
import json
from pathlib import Path
import sys
import unittest
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'src'))
from json_recovery_p2.grammar import accepts_text, finite_then_mask, schemas
from json_recovery_p2.upstream import Registry


class GrammarContracts(unittest.TestCase):
    def test_all_tools_and_free_wrong_values_without_gold(self):
        for api in ['API_V1', 'API_V2']:
            registry = Registry(api); tools = registry.tools()
            values = [{'order_id': 'WRONG-ID'}, {'a': -12345, 'b': 999}, {'order_id': 'ANY-ID', 'total_minor': -99}, {'order_id': ''}]
            for name, arguments in zip(registry.mapping.values(), values):
                text = '<tool_call>' + json.dumps({'name': name, 'arguments': arguments}) + '</tool_call>'
                self.assertTrue(accepts_text(tools, text), text)
            for flag in [False, True]:
                self.assertTrue(accepts_text(tools, json.dumps({'order_id': 'free', 'total_minor': -1, 'validated': flag})))

    def test_name_argument_keys_types_and_unknown_name_rejected(self):
        tools = Registry('API_V1').tools()
        for name, args in [('write_order_total', {'a': 2, 'b': 3}), ('write_order_total', {'order_id': 'free', 'total_minor': '123'}),
                           ('write_order_total', {'order_id': 'free', 'total_minor': True}),
                           ('write_order_total', {'order_id': 'free', 'total_minor': 1, 'extra': 0}), ('unknown_tool', {'order_id': 'free'})]:
            with self.subTest(name=name, args=args):
                self.assertFalse(accepts_text(tools, '<tool_call>' + json.dumps({'name': name, 'arguments': args}) + '</tool_call>'))

    def test_multicall_write_validate_key_order_and_complete_end(self):
        tools = Registry('API_V2').tools()
        first = '<tool_call>' + json.dumps({'arguments': {'total_minor': 0, 'order_id': 'wrong'}, 'name': 'writeOrderTotal'})
        second = '<tool_call>' + json.dumps({'name': 'validateOrder', 'arguments': {'order_id': 'wrong'}}) + '</tool_call>'
        self.assertTrue(accepts_text(tools, first)); self.assertTrue(accepts_text(tools, first + '</tool_call>\n' + second))
        self.assertFalse(accepts_text(tools, first[:-2]))

    def test_only_registered_names_enumerated_and_final_flag_free(self):
        schema = schemas(Registry('API_V1').tools())
        for branch in schema['tool_call']['anyOf']:
            for arg in branch['properties']['arguments']['properties'].values():
                self.assertNotIn('const', arg); self.assertNotIn('enum', arg)
        self.assertEqual(schema['final_answer']['properties']['validated'], {'type': 'boolean'})

    def test_finite_unmasked_before_intentional_negative_infinity_mask(self):
        result = finite_then_mask([1.0, 2.0, 3.0], [0, 2]); self.assertEqual(result, [1.0, -float('inf'), 3.0])
        for bad in [[1.0, float('nan')], [1.0, float('inf')], result]:
            with self.assertRaises(ValueError): finite_then_mask(bad, [0])


if __name__ == '__main__': unittest.main()
