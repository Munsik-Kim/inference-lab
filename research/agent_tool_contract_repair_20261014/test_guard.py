import unittest
from guard import State, active_state, advance, feedback

NAMES = ('read_record', 'apply_fixture_patch', 'validate_fixture_test')


class NameContractTests(unittest.TestCase):
    def test_wrong_name_is_rejected_at_first_character(self):
        state = active_state('<tool_call>\n{"name": "', NAMES)
        self.assertIsNone(advance(state, 'Apply', NAMES))
        self.assertIsNotNone(advance(state, 'apply', NAMES))

    def test_token_can_cross_key_colon_quote_and_name(self):
        state = active_state('<tool_call>{"na', NAMES)
        self.assertIsNone(advance(state, 'me": "Wrong', NAMES))
        self.assertIsNotNone(advance(state, 'me": "read_record",', NAMES))

    def test_name_requires_complete_enum_and_no_escape(self):
        state = active_state('<tool_call>{"name":"apply', NAMES)
        self.assertIsNone(advance(state, '"', NAMES))
        self.assertIsNone(advance(state, '\\u005f', NAMES))
        self.assertIsNotNone(advance(state, '_fixture_patch", "arguments":{}', NAMES))

    def test_name_in_arguments_is_not_tool_identifier(self):
        state = active_state('<tool_call>{"arguments":{"name":"', NAMES)
        self.assertIsNotNone(advance(state, 'Unknown", "n": 10}, "name": "read_record"}', NAMES))

    def test_final_answer_and_completed_call_are_unrestricted(self):
        self.assertIsNone(active_state('{"answer":"ApplyFixturePatch"}', NAMES))
        self.assertIsNone(active_state('<tool_call>{"name":"read_record"}</tool_call>', NAMES))

    def test_no_numeric_or_semantic_argument_correction(self):
        state = active_state('<tool_call>{"name":"apply_fixture_patch","arguments":', NAMES)
        self.assertIsNotNone(advance(state, '{"enabled":false,"limit":999}}', NAMES))

    def test_feedback_has_registry_but_no_gold(self):
        schema = [{'function': {'name': n, 'parameters': {'required': []}}} for n in NAMES]
        result = feedback('ApplyFixturePatch', {'limit': 999}, schema, ValueError('unknown'))
        self.assertEqual(result['error'], 'UNKNOWN_TOOL_NAME')
        self.assertEqual(result['allowed_tool_names'], sorted(NAMES))
        self.assertNotIn('arguments', result)
        self.assertNotIn('gold', result)
        result = feedback('read_record', {'wrong': 1}, schema, ValueError('bad keys'))
        self.assertEqual(result['error'], 'INVALID_TOOL_ARGUMENTS')


if __name__ == '__main__':
    unittest.main()
