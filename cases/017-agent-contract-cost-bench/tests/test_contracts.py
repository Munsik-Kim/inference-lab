"""SYNTHETIC_CONTRACT_TEST: parser/schema checks are not model performance."""
import json
import unittest

from agent_contract_cost.contracts import (ContractError, check_schema, grammar_schema,
                                            parse_action, registry_mapping, validate_schema)


def registry():
    return {
        "read_record": {"parameters": {"type": "object", "properties": {
            "record_id": {"type": "string"}}, "required": ["record_id"], "additionalProperties": False}},
        "write_record": {"parameters": {"type": "object", "properties": {
            "record_id": {"type": "string"}, "amount": {"type": "integer", "minimum": 0}},
            "required": ["record_id", "amount"], "additionalProperties": False}},
        "final": {"parameters": {"type": "object", "properties": {
            "task_id": {"type": "string"}, "completed": {"type": "boolean"},
            "result": {"type": "object", "properties": {"amount": {"type": "integer"}},
                       "required": ["amount"], "additionalProperties": False}},
            "required": ["task_id", "completed", "result"], "additionalProperties": False}},
    }


def raw(name="read_record", **arguments):
    return json.dumps({"tool": name, "arguments": arguments})


class StrictContractTests(unittest.TestCase):
    def test_only_single_complete_object_with_json_whitespace(self):
        valid = raw(record_id="anything")
        self.assertTrue(parse_action(" \r\n\t" + valid + " \n", registry()).accepted)
        for bad in ("Here: " + valid, valid + "\n" + valid, valid + " done", "```json\n" + valid + "\n```", valid[:-1]):
            with self.subTest(bad=bad):
                result = parse_action(bad, registry())
                self.assertFalse(result.accepted)
                self.assertFalse(result.parsed_object)

    def test_duplicate_keys_nested_and_top_are_rejected_without_extraction(self):
        for bad in ('{"tool":"read_record","tool":"final","arguments":{}}',
                    '{"tool":"read_record","arguments":{"record_id":"a","record_id":"b"}}'):
            result = parse_action(bad, registry())
            self.assertEqual(result.error.code, "DUPLICATE_KEY")
            self.assertFalse(result.parsed_object)

    def test_valid_dict_rejections_consume_parsed_attempt_but_arrays_do_not(self):
        for text in ('{}', '{"tool":"read_record"}', '{"tool":0,"arguments":{}}',
                     '{"tool":"READ_RECORD","arguments":{}}'):
            result = parse_action(text, registry())
            self.assertFalse(result.accepted)
            self.assertTrue(result.parsed_object)
        for text in ('[]', 'null', '1', 'true', '"text"'):
            self.assertFalse(parse_action(text, registry()).parsed_object)

    def test_no_argument_coercion_bool_int_or_extra_key(self):
        for amount in ("101", True, 101.0, -1):
            result = parse_action(raw("write_record", record_id="R", amount=amount), registry())
            self.assertFalse(result.accepted)
        for completed in (0, 1, "true"):
            self.assertFalse(parse_action(raw("final", task_id="T", completed=completed, result={"amount": 1}), registry()).accepted)
        self.assertFalse(parse_action(raw(record_id="R", extra="ignored?"), registry()).accepted)
        self.assertFalse(parse_action('{"tool":"write_record","arguments":{"record_id":"R","amount":NaN}}', registry()).accepted)

    def test_grammar_allows_wrong_id_and_wrong_but_typed_amount(self):
        schema = grammar_schema(registry())
        for amount in (99, 100, 101, 10**1000):
            action = {"tool": "write_record", "arguments": {"record_id": "a completely wrong ID", "amount": amount}}
            self.assertIsNone(validate_schema(action, schema))
            self.assertTrue(parse_action(json.dumps(action), registry()).accepted)
        for completed in (False, True):
            action = {"tool": "final", "arguments": {"task_id": "wrong", "completed": completed, "result": {"amount": -77}}}
            self.assertIsNone(validate_schema(action, schema))

    def test_public_schema_only_and_unsupported_keyword_fail_closed(self):
        mapping = registry()
        definitions = [{"name": key, "description": "public tool", **value} for key, value in mapping.items()]
        self.assertEqual(grammar_schema(definitions), grammar_schema(mapping))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            registry_mapping(definitions + definitions[:1])
        with self.assertRaisesRegex(ValueError, "unsupported"):
            check_schema({"type": "string", "pattern": "secret-record-id"})
        self.assertNotIn("oracle", json.dumps(grammar_schema(mapping)))
        self.assertNotIn("const", json.dumps(grammar_schema(mapping)))

    def test_error_feedback_has_only_public_fields_not_supplied_values(self):
        bad = raw("write_record", record_id="hidden-input", amount="guessed-secret")
        public = parse_action(bad, registry()).error.public()
        self.assertEqual(public, {"code": "TYPE_ERROR", "path": "$.arguments.amount", "expected": "integer"})
        self.assertNotIn("guessed-secret", json.dumps(public))

    def test_array_nested_types_enum_range_and_uniqueness(self):
        schema = {"type": "array", "minItems": 1, "maxItems": 2, "uniqueItems": True,
                  "items": {"type": "string", "enum": ["a", "b"]}}
        self.assertIsNone(validate_schema(["b", "a"], schema))
        for value in ([], ["a", "a"], ["a", "b", "a"], [True], ["c"]):
            self.assertIsInstance(validate_schema(value, schema), ContractError)


if __name__ == "__main__":
    unittest.main()
