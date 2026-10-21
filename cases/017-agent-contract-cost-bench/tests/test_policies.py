"""SYNTHETIC_CONTRACT_TEST: scripted policies/budgets never invoke a model."""
import copy
import json
import unittest

from agent_contract_cost.contracts import ContractError, parse_action
from agent_contract_cost.policies import (BudgetLimits, BudgetState, NEUTRAL_FEEDBACK,
                                         PolicySession, retry_messages)
from test_contracts import raw, registry


def consume(session, text):
    session.begin_request(100)
    session.budget.finish_response(10)
    return session.consume_response(text)


class PolicyBoundaryTests(unittest.TestCase):
    def test_timeout_attempt_and_partial_tokens_are_not_completed_response(self):
        budget = BudgetState()
        budget.begin_request(100)
        budget.fail_request(observed_output_tokens=3, abort_requested=True)
        self.assertEqual(budget.model_request_attempts, 1)
        self.assertEqual(budget.model_responses_received, 0)
        self.assertEqual(budget.generated_output_tokens, 3)
        self.assertEqual(budget.server_abort_requests, 1)

    def test_sixth_final_is_allowed_and_seventh_never_requested(self):
        session = PolicySession("C0", registry())
        for _ in range(5):
            decision = consume(session, raw(record_id="R"))
            self.assertTrue(decision.execute)
            session.budget.record_execution()
            self.assertEqual(session.after_execution(), "UNKNOWN")
        final = raw("final", task_id="T", completed=True, result={"amount": 1})
        self.assertTrue(consume(session, final).execute)
        self.assertEqual(session.after_execution(final=True, verified=True), "VERIFIED_SUCCESS")
        self.assertEqual(session.budget.parsed_tool_attempts, 6)
        self.assertEqual(session.budget.final_attempts, 1)
        self.assertEqual(session.budget.callable_executions, 5)
        with self.assertRaises(ValueError):
            session.begin_request(100)

    def test_sixth_nonfinal_explicit_policy_blocks_seventh_even_final(self):
        session = PolicySession("C0", registry())
        for _ in range(6):
            self.assertTrue(consume(session, raw(record_id="R")).execute)
            session.after_execution()
        self.assertEqual(session.terminal_reason, "TOOL_CALL_LIMIT")
        self.assertTrue(session.budget.final_opportunity_blocked_by_registered_attempt_budget)
        self.assertEqual(session.budget.model_request_attempts, 6)
        with self.assertRaises(ValueError):
            consume(session, raw("final", task_id="T", completed=True, result={"amount": 1}))

    def test_syntax_vs_registry_schema_attempt_accounting(self):
        for text, attempts in (("not JSON", 0), ("[]", 0), ('{"tool":"missing","arguments":{}}', 1),
                               (raw("write_record", record_id="R", amount=True), 1)):
            session = PolicySession("C0", registry())
            decision = consume(session, text)
            self.assertFalse(decision.execute)
            self.assertEqual(session.budget.parsed_tool_attempts, attempts)
            self.assertEqual(session.budget.model_request_attempts, 1)
            self.assertEqual(session.budget.contract_rejections, 1)
            self.assertEqual(session.budget.callable_executions, 0)

    def test_neutral_and_error_retry_once_and_count_actual_request(self):
        for policy in ("N", "E"):
            session = PolicySession(policy, registry())
            first = consume(session, "bad")
            self.assertTrue(first.retry)
            self.assertEqual(session.budget.retry_triggered, 1)
            self.assertEqual(session.budget.retry_consumed, 0)
            second = consume(session, "bad again")
            self.assertFalse(second.retry)
            self.assertEqual(second.terminal_reason, "CONTRACT_FAILURE")
            self.assertEqual(session.budget.retry_consumed, 1)
            self.assertEqual(session.budget.model_request_attempts, 2)

    def test_retry_context_or_deadline_blocked_is_not_consumed(self):
        for tokens, elapsed in ((24576, 0), (100, 900)):
            session = PolicySession("N", registry())
            self.assertTrue(consume(session, "bad").retry)
            with self.assertRaises(ValueError):
                session.begin_request(tokens, elapsed)
            self.assertEqual(session.budget.model_request_attempts, 1)
            self.assertEqual(session.budget.retry_consumed, 0)

    def test_retry_history_only_adds_failed_assistant_and_public_user_no_fake_event(self):
        initial = [{"role": "system", "content": "public contract"},
                   {"role": "user", "content": "public task"}]
        before = copy.deepcopy(initial)
        error = ContractError("TYPE_ERROR", "$.arguments.amount", "integer")
        neutral = retry_messages(initial, 'malformed "raw', "N", error)
        feedback = retry_messages(initial, 'malformed "raw', "E", error)
        self.assertEqual(initial, before)
        self.assertEqual(neutral[:-1], feedback[:-1])
        self.assertEqual(neutral[-1], {"role": "user", "content": NEUTRAL_FEEDBACK})
        self.assertEqual(json.loads(feedback[-1]["content"]), error.public())
        self.assertFalse(any(message["role"] == "tool" for message in neutral))
        self.assertEqual(neutral[-2]["content"], 'malformed "raw')

    def test_business_execution_error_does_not_activate_policy_retry(self):
        for policy in ("C0", "G", "N", "E"):
            session = PolicySession(policy, registry())
            decision = consume(session, raw(record_id="nonexistent"))
            self.assertTrue(decision.execute)
            self.assertFalse(decision.retry)
            # Host's UNKNOWN_RECORD tool result is an observation, not a
            # contract rejection; this layer never repairs the supplied ID.
            session.budget.record_execution()
            self.assertEqual(session.after_execution(), "UNKNOWN")
            self.assertEqual(session.budget.retry_triggered, 0)
            self.assertEqual(session.grammar_active, policy == "G")

    def test_model_token_context_deadline_boundaries_are_distinct(self):
        cases = (({"model_request_attempts": 8}, 100, 0, "MODEL_CALL_LIMIT"),
                 ({"generated_output_tokens": 4096}, 100, 0, "TOKEN_LIMIT"),
                 ({}, 24576, 0, "CONTEXT_LIMIT"), ({}, 100, 900, "EPISODE_DEADLINE"))
        for counters, prompt, elapsed, reason in cases:
            budget = BudgetState(**counters)
            self.assertEqual(budget.request_bounds(prompt, elapsed)["terminal_reason"], reason)
            with self.assertRaisesRegex(ValueError, reason):
                budget.begin_request(prompt, elapsed)
        budget = BudgetState(generated_output_tokens=4080)
        self.assertEqual(budget.begin_request(24570)["max_tokens"], 6)
        with self.assertRaisesRegex(ValueError, "SERVER_OUTPUT_BUDGET"):
            budget.finish_response(7)
        self.assertEqual(budget.generated_output_tokens, 4087)

    def test_correct_final_syntax_without_actual_work_is_not_verified(self):
        session = PolicySession("G", registry())
        decision = consume(session, raw("final", task_id="T", completed=True, result={"amount": 1}))
        self.assertTrue(decision.execute)
        self.assertEqual(session.after_execution(final=True, verified=False, state_correct=False), "INCORRECT_STATE")
        self.assertEqual(session.budget.side_effect_writes, 0)
        self.assertEqual(session.budget.final_attempts, 1)


if __name__ == "__main__":
    unittest.main()
