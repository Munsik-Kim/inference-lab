"""Endpoint checks from executed state and pre-generation observations, CPU only.

Historic Case015/016 observation distinctions are reused conceptually; this scorer
does not import either runner or trust a stored success flag.
"""
import itertools

from .fixtures import canonical, digest, integrity_ok, oracle, strict_equal


SCORER_VERSION = "case017-endpoint-v1"
OBLIGATIONS = {"A": ("read", "calculation", "write", "validation"), "B": ("source_read", "destination_read", "transfer", "validation"), "C": ("read", "update", "validation")}


def _correct_roles(task, event, gold):
    if event.get("status") != "SUCCESS":
        return []
    name, args, result = event["tool"], event["arguments"], event["result"]
    goal = task["goal"]
    family = task["family"]
    roles = []
    if family == "A":
        row = task["initial_state"]["orders"][goal["order_id"]]
        expected = gold["final_arguments"]["result"]["total_minor"]
        if name == "read_order" and args["order_id"] == goal["order_id"] and strict_equal(result.get("lines"), row["lines"]) and strict_equal(result.get("discount_bps"), row["discount_bps"]):
            roles.append("read")
        if name == "calculate_total" and sorted(canonical(item) for item in args["lines"]) == sorted(canonical(item) for item in row["lines"]) and strict_equal(args["discount_bps"], row["discount_bps"]) and strict_equal(result.get("total_minor"), expected):
            roles.append("calculation")
        if name == "write_order_total" and args["order_id"] == goal["order_id"] and strict_equal(args["total_minor"], expected):
            roles.append("write")
        if name == "validate_order" and args["order_id"] == goal["order_id"] and result.get("valid") is True:
            roles.append("validation")
    elif family == "B":
        initial = task["initial_state"]["warehouses"]
        if name == "read_stock" and args["sku"] == goal["sku"]:
            for role, key in (("source_read", "source_id"), ("destination_read", "destination_id")):
                if args["warehouse_id"] == goal[key] and all(strict_equal(result.get(field), initial[goal[key]][field]) for field in ("quantity", "revision", "sku")):
                    roles.append(role)
        if name == "transfer_stock" and all(strict_equal(args[key], goal[key]) for key in ("source_id", "destination_id", "sku", "quantity")) and args["source_revision"] == initial[goal["source_id"]]["revision"] and args["destination_revision"] == initial[goal["destination_id"]]["revision"] and result.get("transferred") is True:
            roles.append("transfer")
        if name == "validate_transfer" and all(strict_equal(args[key], goal[key]) for key in ("source_id", "destination_id", "sku")) and result.get("valid") is True:
            roles.append("validation")
    else:
        initial = task["initial_state"]["documents"][goal["document_id"]]
        if name == "read_document" and args["document_id"] == goal["document_id"] and strict_equal(result.get("revision"), initial["revision"]):
            roles.append("read")
        if name == "update_document" and all(strict_equal(args[key], goal[key]) for key in ("document_id", "status", "tags", "owner_id")) and strict_equal(args["expected_revision"], initial["revision"]) and result.get("updated") is True:
            roles.append("update")
        if name == "validate_document" and args["document_id"] == goal["document_id"] and result.get("valid") is True:
            roles.append("validation")
    return roles


def assess(task, events, final_action, state, state_version, final_available_result_event_ids, terminal=None):
    """A correct file or final JSON alone cannot satisfy workflow completion.

    ``final_available_result_event_ids`` belongs to the generation that produced
    final, not an execution-time list updated after final was already generated.
    A recovered earlier non-mutating error does not automatically fail a workflow.
    """
    gold = oracle(task)
    state_ok = strict_equal(state, gold["state"])
    final_ok = type(final_action) is dict and set(final_action) == {"tool", "arguments"} and final_action["tool"] == "final" and strict_equal(final_action["arguments"], gold["final_arguments"])
    integrity = integrity_ok(task, state) and all(integrity_ok(task, event["state_before"]) and integrity_ok(task, event["state_after"]) and not event.get("integrity_violation_so_far", False) for event in events)
    obligations = OBLIGATIONS[task["family"]]
    groups = {role: [] for role in obligations}
    for index, event in enumerate(events):
        for role in _correct_roles(task, event, gold):
            groups[role].append((index, event))
    available_final = set(final_available_result_event_ids)
    executed_chain = None
    observed_chain = None
    for combination in itertools.product(*(groups[role] for role in obligations)):
        positions = [item[0] for item in combination]
        if task["family"] == "B":
            if not max(positions[:2]) < positions[2] < positions[3] or positions[0] == positions[1]:
                continue
        elif positions != sorted(set(positions)):
            continue
        chain = {role: event for role, (_, event) in zip(obligations, combination)}
        if executed_chain is None:
            executed_chain = [event["event_id"] for _, event in combination]
        if task["family"] == "A":
            dependencies = chain["read"]["event_id"] in chain["calculation"]["generated_after_event_ids"] and chain["calculation"]["event_id"] in chain["write"]["generated_after_event_ids"]
        elif task["family"] == "B":
            dependencies = all(chain[role]["event_id"] in chain["transfer"]["generated_after_event_ids"] for role in ("source_read", "destination_read"))
        else:
            dependencies = chain["read"]["event_id"] in chain["update"]["generated_after_event_ids"]
        validation = chain["validation"]
        latest = validation["after_digest"] == digest(state) and validation["version_after"] == state_version
        if dependencies and latest and validation["event_id"] in available_final:
            observed_chain = [event["event_id"] for _, event in combination]
            break
    unmet = [role for role in obligations if not groups[role]]
    if not unmet and not observed_chain:
        unmet.append("observable_dependency_chain")
    if not state_ok:
        unmet.append("correct_state")
    if not integrity:
        unmet.append("protected_state")
    if not final_ok:
        unmet.append("final")
    terminal_ok = terminal in (None, "FINAL", "VERIFIED_SUCCESS")
    if not terminal_ok:
        unmet.append("terminal_budget_or_infrastructure")
    verified = bool(state_ok and final_ok and integrity and observed_chain and terminal_ok)
    claimed = type(final_action) is dict and type(final_action.get("arguments")) is dict and final_action["arguments"].get("completed") is True
    labels = []
    if any(event.get("status") == "CONTRACT_ERROR" for event in events):
        labels.append("CONTRACT_REJECTION")
    if not integrity:
        labels.append("UNAUTHORIZED_STATE_CHANGE")
    if executed_chain and not observed_chain:
        labels.append("OBSERVATION_ORDER_OR_STALE_VALIDATION")
    if claimed and not verified:
        labels.append("UNSUPPORTED_SUCCESS_CLAIM")
    return {"VERIFIED_WORKFLOW_SUCCESS": verified, "STATE_CORRECT": bool(state_ok and integrity), "STATE_ONLY_SUCCESS": bool(state_ok and integrity), "FINAL_ANSWER_CORRECT": bool(final_ok), "REQUIRED_ACTIONS_DONE": bool(executed_chain), "OBSERVATION_ORDER_VALID": bool(observed_chain), "integrity_valid_throughout": bool(integrity), "CLAIMED_DONE_WITH_WRONG_STATE": bool(claimed and not (state_ok and integrity)), "UNSUPPORTED_SUCCESS_CLAIM": bool(claimed and not verified), "executed_chain": executed_chain, "verified_chain": observed_chain, "unmet_obligations": unmet, "first_unmet_obligation": unmet[0] if unmet else None, "diagnostic_labels": labels, "scorer_version": SCORER_VERSION}
