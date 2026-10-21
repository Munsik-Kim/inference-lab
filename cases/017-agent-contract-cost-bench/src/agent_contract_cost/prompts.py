"""Allowlisted model-visible input and observation serialization, CPU only."""
import copy
import json

from .fixtures import canonical
from .tools import tool_definitions


SYSTEM_INSTRUCTIONS = """You are completing one synthetic local record-management task through a fixed action interface. Your job is to carry out the requested changes, inspect the returned validation, and report the actual resulting values. The fixture is isolated; none of these actions connects to a real customer, payment system, warehouse or external account.

Each response must contain exactly one complete JSON object. Its only top-level keys are tool and arguments. tool is one of the exact registered action names below. arguments is one JSON object with precisely the properties required for that action. Generate JSON text directly: do not surround it with Markdown fences, explanations, a list, a tool-call marker, or a second JSON object. Do not write analysis before or after the object. A JSON integer must be a numeric integer rather than quoted digits, a decimal number or a boolean. A JSON boolean must be true or false. Preserve the spelling and case of all property names and registered action names.

The action definitions are the complete public interface for this task. They specify the accepted keys, value types, static ranges and behavior. Use only that interface. In particular, final is itself a registered action, with task_id, completed and the family-specific result object defined below. A plain success sentence does not finish the task. Do not invent an additional action, alternative spelling or alias. An error result does not mean that the requested change happened.

The next user message states the task ID and the requested business operation. It identifies the record or records to work on and the desired changes. Details available only through a read action must be obtained through that read action. After you emit an action, its real result is returned in a user message containing event_id, tool, status and result. Treat that returned result as an observation. Your next response can use the information it returned. Actions are executed only after the entire response has been generated; you cannot inspect the result of a call while writing the same response. Emit one action and wait for its result before creating another action that depends on it.

Never replace a returned value with a guess. For a purchase settlement, inspect the returned line quantities, unit prices and discount before calculating; use the calculation action's returned integer amount when writing. Integer minor units mean the smallest accounting unit used by this synthetic fixture; no exchange rate, tax or extra discount should be invented. For a warehouse movement, inspect the current source and destination stock and their revisions before submitting the atomic transfer. A transfer should use exactly the requested quantity. For document metadata, inspect the current revision before issuing a revision-checked update. The task supplies the desired metadata; the read result supplies the current revision.

Preserve every protected field and every unrelated record throughout the task. Write only values permitted by the requested operation. Do not alter a different record to make a validation message appear successful. Atomic warehouse transfers either update both endpoints or neither endpoint. A rejected revision check means that the update was not applied. Read the result and act according to the public contract; do not claim that an error changed the file.

Validation checks the public structure and invariants described in its action definition. It is not a substitute for choosing the requested record, quantity, amount or metadata correctly. Inspect validation after the intended update. Your final response must describe the actual requested resulting values, use the task ID from the user message, and be based on a successful validation of the current state that you have already received. An older validation is stale if a subsequent action changed the state. Do not report completed true just because a calculation succeeded or because you intend to validate later.

There are at most eight model responses and at most six parsed action attempts for the whole task, including final. A rejected registered-name or argument contract can still consume a parsed action attempt. final consumes one action attempt and performs no file write. The intended workflows fit within the limits: use the required reads, the requested calculation or mutation, validation and final. Additional reads are permitted, but they use the same finite action budget. The host may end the task when that budget is exhausted. Do not depend on a seventh action or an unrequested continuation.

These instructions are identical for every comparison condition. No private expected answer, hidden file, oracle, model identity or cache configuration is available through the interface. Use the task request, the public definitions and the results that have actually been returned to you.

Registered action definitions:
"""


def initial_messages(task):
    """Only task ID, public task instruction and family definitions enter prompts."""
    if type(task.get("task_id")) is not str or type(task.get("prompt")) is not str or task.get("family") not in ("A", "B", "C"):
        raise ValueError("task public prompt fields are invalid")
    definitions = tool_definitions(task["family"])
    user = task["prompt"]
    if task["task_id"] not in user:
        user = "Task " + task["task_id"] + ". " + user
    # Readable public definitions, fixed before any model observation. Formatting
    # is shared across all policies/KV conditions; no task facts are padded.
    declarations = json.dumps(definitions, ensure_ascii=False, sort_keys=True, indent=2)
    return [{"role": "system", "content": SYSTEM_INSTRUCTIONS + declarations}, {"role": "user", "content": user}]


def observation_view(event):
    """No fixture snapshots, oracle, semantic mapping or other audit fields."""
    return {key: copy.deepcopy(event[key]) for key in ("event_id", "tool", "status", "result")}


def append_result(messages, raw, event):
    result = copy.deepcopy(messages)
    result.append({"role": "assistant", "content": raw})
    result.append({"role": "user", "content": canonical(observation_view(event))})
    return result
