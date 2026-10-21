"""Four registered policies and attempt accounting; no model/fixture imports.

Final is an action and consumes one of six parsed attempts. Therefore attempt
six may be final; after six non-final parsed objects the episode stops without
a seventh generation. This explicit Case017 policy differs from the historical
Case016 policy, whose final did not consume a tool attempt.
"""
from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict, dataclass

from .contracts import ContractError, ParseResult, parse_action


NEUTRAL_FEEDBACK = "직전 호출은 실행되지 않았습니다. 동일 작업의 호출을 다시 작성하세요."
POLICIES = ("C0", "G", "N", "E")
TERMINAL_REASONS = ("UNKNOWN", "VERIFIED_SUCCESS", "CONTRACT_FAILURE", "INCORRECT_STATE",
                    "INCORRECT_FINAL", "MODEL_CALL_LIMIT", "TOOL_CALL_LIMIT", "TOKEN_LIMIT",
                    "CONTEXT_LIMIT", "REQUEST_TIMEOUT", "EPISODE_DEADLINE", "OOM",
                    "SERVER_ERROR", "INFRA_ERROR")


@dataclass(frozen=True)
class BudgetLimits:
    max_model_request_attempts: int = 8
    max_parsed_tool_attempts: int = 6
    max_generated_tokens: int = 4096
    max_new_tokens: int = 512
    context_tokens: int = 24576
    request_seconds: float = 180.0
    episode_seconds: float = 900.0


@dataclass
class BudgetState:
    limits: BudgetLimits = BudgetLimits()
    model_request_attempts: int = 0
    model_responses_received: int = 0
    server_abort_requests: int = 0
    generated_raw_objects: int = 0
    parsed_tool_attempts: int = 0
    contract_rejections: int = 0
    callable_executions: int = 0
    side_effect_writes: int = 0
    final_attempts: int = 0
    retry_triggered: int = 0
    retry_consumed: int = 0
    generated_output_tokens: int = 0
    final_opportunity_blocked_by_registered_attempt_budget: bool = False
    _inflight: bool = False
    _active_cap: int = 0

    def request_bounds(self, prompt_tokens: int, elapsed_seconds: float = 0.0) -> dict:
        if type(prompt_tokens) is not int or prompt_tokens < 0:
            raise ValueError("invalid prompt token count")
        if not math.isfinite(elapsed_seconds) or elapsed_seconds < 0:
            raise ValueError("invalid elapsed time")
        remaining = {"request_cap": self.limits.max_new_tokens,
                     "episode_tokens": self.limits.max_generated_tokens - self.generated_output_tokens,
                     "context_tokens": self.limits.context_tokens - prompt_tokens}
        cap = min(remaining.values())
        reason = None
        if elapsed_seconds >= self.limits.episode_seconds:
            reason = "EPISODE_DEADLINE"
        elif self.parsed_tool_attempts >= self.limits.max_parsed_tool_attempts:
            reason = "TOOL_CALL_LIMIT"
        elif self.model_request_attempts >= self.limits.max_model_request_attempts:
            reason = "MODEL_CALL_LIMIT"
        elif remaining["context_tokens"] <= 0:
            reason = "CONTEXT_LIMIT"
        elif remaining["episode_tokens"] <= 0:
            reason = "TOKEN_LIMIT"
        return {"allowed": reason is None, "terminal_reason": reason,
                "max_tokens": max(0, cap), "bounds": remaining,
                "limiting_factors": [key for key, value in remaining.items() if value == cap],
                "timeout_seconds": max(0.0, min(self.limits.request_seconds,
                    self.limits.episode_seconds - elapsed_seconds))}

    def begin_request(self, prompt_tokens: int, elapsed_seconds: float = 0.0) -> dict:
        if self._inflight:
            raise ValueError("previous request has not completed")
        plan = self.request_bounds(prompt_tokens, elapsed_seconds)
        if not plan["allowed"]:
            raise ValueError(plan["terminal_reason"])
        # Debit before awaiting HTTP, so a failed/timeout request is not free.
        self.model_request_attempts += 1
        self._active_cap = plan["max_tokens"]
        self._inflight = True
        return plan

    def finish_response(self, output_tokens: int) -> None:
        if not self._inflight:
            raise ValueError("no inflight request")
        if type(output_tokens) is not int or output_tokens < 0:
            raise ValueError("invalid output token count")
        self.model_responses_received += 1
        self.generated_output_tokens += output_tokens
        self._inflight = False
        if output_tokens > self._active_cap:
            raise ValueError("SERVER_OUTPUT_BUDGET_VIOLATION")

    def fail_request(self, *, observed_output_tokens=0, abort_requested=False) -> None:
        if not self._inflight:
            raise ValueError("no inflight request")
        if type(observed_output_tokens) is not int or observed_output_tokens < 0:
            raise ValueError("invalid observed partial token count")
        self.generated_output_tokens += observed_output_tokens
        self._inflight = False
        self.server_abort_requests += int(abort_requested)

    def record_parse(self, result: ParseResult) -> None:
        if result.parsed_object:
            if self.parsed_tool_attempts >= self.limits.max_parsed_tool_attempts:
                raise ValueError("TOOL_CALL_LIMIT")
            self.generated_raw_objects += 1
            self.parsed_tool_attempts += 1
        if not result.accepted:
            self.contract_rejections += 1
        elif result.action["tool"] == "final":
            self.final_attempts += 1

    def record_execution(self, *, side_effect=False) -> None:
        self.callable_executions += 1
        self.side_effect_writes += int(side_effect)

    def ledger(self) -> dict:
        result = asdict(self)
        result.pop("_inflight")
        result.pop("_active_cap")
        return result


@dataclass(frozen=True)
class ResponseDecision:
    parsed: ParseResult
    execute: bool
    retry: bool
    feedback: str | None
    terminal_reason: str | None


def error_feedback(error: ContractError) -> str:
    return json.dumps(error.public(), ensure_ascii=False, separators=(",", ":"))


def retry_messages(messages, failed_raw: str, policy: str, error: ContractError):
    if policy not in ("N", "E") or type(failed_raw) is not str:
        raise ValueError("retry requires N/E and unchanged raw string")
    feedback = NEUTRAL_FEEDBACK if policy == "N" else error_feedback(error)
    result = copy.deepcopy(messages)
    result.extend([{"role": "assistant", "content": failed_raw},
                   {"role": "user", "content": feedback}])
    return result


class PolicySession:
    def __init__(self, policy: str, registry, *, budget: BudgetState | None = None):
        if policy not in POLICIES:
            raise ValueError("unknown policy")
        self.policy = policy
        self.registry = copy.deepcopy(registry)
        self.budget = budget or BudgetState()
        self.terminal_reason = "UNKNOWN"
        self.first_failure = None
        self._retry_pending = False

    @property
    def grammar_active(self):
        return self.policy == "G"

    def begin_request(self, prompt_tokens, elapsed_seconds=0.0):
        if self.terminal_reason != "UNKNOWN":
            raise ValueError("session already terminated")
        plan = self.budget.begin_request(prompt_tokens, elapsed_seconds)
        if self._retry_pending:
            self.budget.retry_consumed += 1
            self._retry_pending = False
        return plan

    def consume_response(self, raw: str) -> ResponseDecision:
        """Call after finish_response. Host dispatches only execute=True actions.

        This method creates no observations, changes no fixture, and never turns
        a business execution error into a retry. Caller appends only genuine
        callable results to its observation list.
        """
        if self.terminal_reason != "UNKNOWN":
            raise ValueError("session already terminated")
        parsed = parse_action(raw, self.registry)
        self.budget.record_parse(parsed)
        if parsed.accepted:
            return ResponseDecision(parsed, True, False, None, None)
        if self.first_failure is None:
            self.first_failure = parsed.error.public()
        if self.policy in ("N", "E") and self.budget.retry_triggered == 0:
            self.budget.retry_triggered = 1
            room = self.budget.request_bounds(0)
            if room["allowed"]:
                self._retry_pending = True
                feedback = NEUTRAL_FEEDBACK if self.policy == "N" else error_feedback(parsed.error)
                return ResponseDecision(parsed, False, True, feedback, None)
            self.terminal_reason = room["terminal_reason"]
        else:
            self.terminal_reason = "CONTRACT_FAILURE"
        return ResponseDecision(parsed, False, False, None, self.terminal_reason)

    def after_execution(self, *, final=False, verified=False, state_correct=False):
        if final:
            self.terminal_reason = ("VERIFIED_SUCCESS" if verified else
                                    "INCORRECT_FINAL" if state_correct else "INCORRECT_STATE")
        elif self.budget.parsed_tool_attempts >= self.budget.limits.max_parsed_tool_attempts:
            self.budget.final_opportunity_blocked_by_registered_attempt_budget = True
            self.terminal_reason = "TOOL_CALL_LIMIT"
        return self.terminal_reason
