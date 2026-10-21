"""Owned synthetic business records and independent, CPU-only expected states.

The generator does not open model outputs. HOLDOUT is deliberately not generated
on import: the caller must freeze DEV policy/scorer before calling that split.
"""
import copy
import hashlib
import json
import random


GENERATOR_VERSION = "case017-fixtures-v1"
SPLITS = {"DEV": (3, 3, 2), "HOLDOUT": (8, 8, 8), "COST": (2, 2, 2)}
SEEDS = {"DEV": 6102001, "HOLDOUT": 6102002, "COST": 6102003}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def strict_equal(left, right):
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return set(left) == set(right) and all(strict_equal(left[k], right[k]) for k in left)
    if type(left) is list:
        return len(left) == len(right) and all(strict_equal(a, b) for a, b in zip(left, right))
    return left == right


def total_minor(lines, discount_bps):
    """Round the discounted total to nearest minor unit; exact halves round up."""
    gross = sum(row["quantity"] * row["unit_price_minor"] for row in lines)
    return (gross * (10000 - discount_bps) + 5000) // 10000


def oracle(task):
    """Return an independent desired state and strict final result, never tool feedback."""
    state = copy.deepcopy(task["initial_state"])
    goal = task["goal"]
    if task["family"] == "A":
        record = state["orders"][goal["order_id"]]
        record["total_minor"] = total_minor(record["lines"], record["discount_bps"])
        result = {"order_id": goal["order_id"], "total_minor": record["total_minor"], "validated": True}
    elif task["family"] == "B":
        source = state["warehouses"][goal["source_id"]]
        destination = state["warehouses"][goal["destination_id"]]
        source["quantity"] -= goal["quantity"]
        destination["quantity"] += goal["quantity"]
        source["revision"] += 1
        destination["revision"] += 1
        result = {k: goal[k] for k in ("source_id", "destination_id", "sku")}
        result.update(source_quantity=source["quantity"], destination_quantity=destination["quantity"], validated=True)
    elif task["family"] == "C":
        document = state["documents"][goal["document_id"]]
        for key in ("status", "tags", "owner_id"):
            document[key] = copy.deepcopy(goal[key])
        document["revision"] += 1
        result = {"document_id": goal["document_id"], **{k: copy.deepcopy(document[k]) for k in ("revision", "status", "tags", "owner_id")}, "validated": True}
    else:
        raise ValueError("unknown task family")
    return {"state": state, "final_arguments": {"task_id": task["task_id"], "completed": True, "result": result}}


def integrity_ok(task, state):
    """Target mutable fields only; control records and all other fields are protected."""
    initial = task["initial_state"]
    if type(state) is not dict or set(state) != set(initial):
        return False
    family = task["family"]
    table = {"A": "orders", "B": "warehouses", "C": "documents"}[family]
    if not strict_equal({k: v for k, v in state.items() if k != table}, {k: v for k, v in initial.items() if k != table}):
        return False
    if type(state.get(table)) is not dict or set(state[table]) != set(initial[table]):
        return False
    goal = task["goal"]
    targets = {goal["order_id"]} if family == "A" else ({goal["source_id"], goal["destination_id"]} if family == "B" else {goal["document_id"]})
    mutable = {"A": {"total_minor"}, "B": {"quantity", "revision"}, "C": {"status", "tags", "owner_id", "revision"}}[family]
    for record_id, original in initial[table].items():
        current = state[table].get(record_id)
        if type(current) is not dict or set(current) != set(original):
            return False
        if record_id not in targets:
            if not strict_equal(current, original):
                return False
        elif not strict_equal({k: v for k, v in current.items() if k not in mutable}, {k: v for k, v in original.items() if k not in mutable}):
            return False
    return True


def generate_tasks(split="DEV", seed=None):
    if split not in SPLITS:
        raise ValueError("split must be DEV, HOLDOUT or COST")
    rng = random.Random(SEEDS[split] if seed is None else seed)
    tasks = []
    for family, count in zip(("A", "B", "C"), SPLITS[split]):
        for index in range(count):
            task_id = f"{split}-{family}{index:02d}"
            # IDs use independent random bits rather than quantities or answers.
            ids = [f"{prefix}-{rng.getrandbits(48):012x}" for prefix in ("R", "X", "Z")]
            target, control, extra = ids
            if family == "A":
                line_count = [1, 3, 5, 6][index % 4] if split == "HOLDOUT" else 2 + index % 3
                lines = [{"quantity": rng.randint(2, 40), "unit_price_minor": rng.randint(100, 20000)} for _ in range(line_count)]
                discount = ([0, 375, 825, 2250][index % 4] if split == "HOLDOUT" else rng.choice([0, 125, 375, 825, 1500, 2250]))
                initial_total = (17 + index if index % 2 else None) if split == "HOLDOUT" else None
                state = {"orders": {target: {"lines": lines, "discount_bps": discount, "total_minor": initial_total, "protected_customer": f"client-{rng.getrandbits(24):06x}"}, control: {"lines": [{"quantity": 2, "unit_price_minor": 701}], "discount_bps": 0, "total_minor": 1402, "protected_customer": "control-client"}}}
                goal = {"order_id": target}
                if split == "HOLDOUT":
                    wording = [f"Close the payable calculation for purchase {target}. Inspect its complete line list and discount, pass those to the calculation tool, persist the returned payable total, inspect validation, and issue final.", f"Reconcile the current total on purchase {target}. The existing amount may be missing or stale. Fetch the underlying line and discount fields, calculate from those fields, replace the total, validate the result, and complete via final.", f"Prepare purchase {target} for the ledger. Obtain all component amounts through read_order; calculate_total must determine the discounted integer amount. Save the calculation result, check its record validation, and submit the final record values.", f"Audit and settle purchase {target}: read the source line items and discount; obtain a tool-computed payable amount; write it back; verify public record invariants; then return final after reading validation."][index % 4]
                else:
                    wording = [f"Read purchase record {target}. Use the returned quantities, unit prices and discount in calculate_total. Store its returned total, validate the record, then send final.", f"Complete the settlement of {target}: retrieve its items, calculate the discounted amount with the calculation tool, save the amount, check validation, and finish with final.", f"The total on purchase record {target} is unfinished. Inspect that record, obtain the integer payable amount from calculate_total, save and validate it, and return final."][index % 3]
                prompt = wording + " Money is stored in integer minor units. A discount basis point is 1/10000; round the discounted total to nearest integer, with exact halves up. Wait for each read/calculation result before creating an action that needs it. Preserve all other records and fields."
                template_id = f"{split}-purchase-{index % (4 if split == 'HOLDOUT' else 3)}"
            elif family == "B":
                sku = f"SKU-{rng.getrandbits(24):06x}"
                quantity = rng.randint(2, 18)
                state = {"warehouses": {target: {"sku": sku, "quantity": quantity + rng.randint(5, 80), "revision": rng.randint(1, 12), "protected_label": "origin"}, control: {"sku": sku, "quantity": rng.randint(0, 60), "revision": rng.randint(1, 12), "protected_label": "destination"}, extra: {"sku": sku, "quantity": 93, "revision": 2, "protected_label": "unrelated"}}}
                if split == "HOLDOUT":
                    state["warehouses"][control]["quantity"] = 0 if index % 2 == 0 else rng.randint(40, 100)
                    for unused in range(index % 3):
                        unrelated = f"Z-{rng.getrandbits(48):012x}"
                        state["warehouses"][unrelated] = {"sku": sku, "quantity": rng.randint(1, 100), "revision": 1, "protected_label": "additional-unrelated"}
                goal = {"source_id": target, "destination_id": control, "sku": sku, "quantity": quantity}
                if split == "HOLDOUT":
                    wording = [f"Fulfil a stock relocation ticket: send exactly {quantity} units of {sku} from warehouse {target} to {control}. Discover both endpoint revisions and quantities, perform the atomic transfer, inspect validation of the resulting pair, and finish.", f"The destination warehouse {control} needs {quantity} units of {sku} from {target}. Inspect destination and source stock, transfer only the requested amount using current revisions, validate conservation, then issue final.", f"Record one inventory movement: SKU={sku}, units={quantity}, origin={target}, receiving warehouse={control}. Retrieve both current records before the atomic movement; review validation and report resulting quantities through final.", f"Handle transfer request {target} -> {control} for {quantity} units of {sku}. Obtain the source and destination snapshots, submit the revision-checked transfer, check the pair's invariant validation, and complete with final."][index % 4]
                else:
                    wording = [f"Move {quantity} units of {sku} from {target} to {control}. Read both stock records first, transfer atomically using their returned revisions, validate the pair, then send final.", f"Perform a warehouse transfer: {quantity} of {sku}, {target} -> {control}. Inspect both quantities and revisions, apply one atomic transfer, check validation, and finish with final.", f"Rebalance the two warehouse records by transferring exactly {quantity} units of {sku} out of {target} into {control}. Read both records, transfer, validate, then submit final."][index % 3]
                prompt = wording + " Preserve the combined quantity, protected fields, and every unrelated warehouse record. Wait for both read results before creating the transfer action."
                template_id = f"{split}-transfer-{index % (4 if split == 'HOLDOUT' else 3)}"
            else:
                owner = f"owner-{rng.getrandbits(24):06x}"
                tags = sorted(rng.sample(["finance", "operations", "reviewed", "internal", "release", "reference"], 2 + index % 2))
                status = ["review", "published", "archived"][index % 3]
                state = {"documents": {target: {"revision": rng.randint(2, 30), "status": "draft", "tags": ["pending"], "owner_id": "owner-old", "protected_title": f"Synthetic document {rng.getrandbits(20):05x}"}, control: {"revision": 8, "status": "published", "tags": ["reference"], "owner_id": "owner-control", "protected_title": "Control document"}}}
                if split == "HOLDOUT":
                    state["documents"][target]["status"] = ["published", "archived", "draft", "review"][index % 4]
                    state["documents"][target]["tags"] = [] if index % 2 == 0 else ["legacy", "pending", "retain"]
                    if index % 4 == 0:
                        tags = []
                    if index % 4 == 1:
                        owner = "owner-old"
                goal = {"document_id": target, "status": status, "tags": tags, "owner_id": owner}
                instruction = f"document={target}, status={status}, tags={json.dumps(tags)}, owner_id={owner}"
                if split == "HOLDOUT":
                    wording = [f"Process a document publication change request: {instruction}. Discover the current revision, apply the requested fields using that revision, inspect validation of the updated record, and submit final.", f"Synchronize document ownership and labels to these exact settings: {instruction}. First retrieve the existing record revision. Commit the metadata update, read validation, and complete with final.", f"An editorial ticket specifies {instruction}. Inspect the current document; replace the listed metadata under its current revision; validate the revised record; then send final with the resulting revision and metadata.", f"Apply this document metadata patch as a revision-checked update: {instruction}. Fetch the present revision before writing, confirm validation after the write, and report completion through final."][index % 4]
                else:
                    wording = [f"Update document metadata: {instruction}. Read its current revision first, update with that revision, validate the document, then send final.", f"Complete this document change: {instruction}. Inspect the current record, apply the new metadata with its returned revision, check validation, and finish with final."][index % 2]
                prompt = wording + " An update increments the revision by exactly one. Preserve the title and every other document. Wait for the current revision before creating the update action."
                template_id = f"{split}-metadata-{index % (4 if split == 'HOLDOUT' else 2)}"
            task = {"task_id": task_id, "family": family, "split": split, "template_id": template_id, "generator_version": GENERATOR_VERSION, "seed": SEEDS[split] if seed is None else seed, "prompt": f"Task {task_id}. " + prompt, "initial_state": state, "goal": goal}
            task["initial_state_sha256"] = digest(state)
            task["oracle_sha256"] = digest(oracle(task))
            task["canonical_task_sha256"] = digest(task)
            tasks.append(task)
    return tasks


def gold_actions(task):
    goal = task["goal"]
    initial = task["initial_state"]
    gold = oracle(task)
    def action(name, **arguments):
        return {"tool": name, "arguments": arguments}
    if task["family"] == "A":
        row = initial["orders"][goal["order_id"]]
        actions = [action("read_order", order_id=goal["order_id"]), action("calculate_total", lines=copy.deepcopy(row["lines"]), discount_bps=row["discount_bps"]), action("write_order_total", order_id=goal["order_id"], total_minor=gold["final_arguments"]["result"]["total_minor"]), action("validate_order", order_id=goal["order_id"])]
    elif task["family"] == "B":
        actions = [action("read_stock", warehouse_id=goal[key], sku=goal["sku"]) for key in ("source_id", "destination_id")]
        actions += [action("transfer_stock", **goal, source_revision=initial["warehouses"][goal["source_id"]]["revision"], destination_revision=initial["warehouses"][goal["destination_id"]]["revision"]), action("validate_transfer", **{k: goal[k] for k in ("source_id", "destination_id", "sku")})]
    else:
        actions = [action("read_document", document_id=goal["document_id"]), action("update_document", **goal, expected_revision=initial["documents"][goal["document_id"]]["revision"]), action("validate_document", document_id=goal["document_id"])]
    return actions + [{"tool": "final", "arguments": copy.deepcopy(gold["final_arguments"])}]
