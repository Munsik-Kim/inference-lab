"""Fixed local JSON fixture callable registry; no network, shell or model imports."""
import copy
import json
import os
from pathlib import Path

from .fixtures import canonical, digest, integrity_ok, strict_equal, total_minor


def obj(properties):
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


STRING = {"type": "string"}
INTEGER = {"type": "integer", "minimum": 0}
BOOLEAN = {"type": "boolean"}
STATUS = {"type": "string", "enum": ["draft", "review", "published", "archived"]}
TAGS = {"type": "array", "items": STRING, "minItems": 0, "maxItems": 8, "uniqueItems": True}
LINES = {"type": "array", "minItems": 1, "maxItems": 16, "items": obj({"quantity": INTEGER, "unit_price_minor": INTEGER})}


def tool_definitions(family):
    def tool(name, description, properties):
        return {"name": name, "description": description, "parameters": obj(properties)}
    if family == "A":
        result = obj({"order_id": STRING, "total_minor": INTEGER, "validated": BOOLEAN})
        definitions = [tool("read_order", "Read all line quantities and unit prices in integer minor units, discount basis points, and current total for a purchase record.", {"order_id": STRING}), tool("calculate_total", "Calculate sum(quantity*unit_price_minor), apply discount_bps/10000, round to nearest minor unit with exact halves up, and return total_minor.", {"lines": LINES, "discount_bps": {"type": "integer", "minimum": 0, "maximum": 10000}}), tool("write_order_total", "Store the supplied integer total on the specified existing purchase record; preserve all other fields.", {"order_id": STRING, "total_minor": INTEGER}), tool("validate_order", "Check public structure, integer total, and protected-field invariants. This does not compare the amount with the requested calculation.", {"order_id": STRING})]
    elif family == "B":
        result = obj({"source_id": STRING, "destination_id": STRING, "sku": STRING, "source_quantity": INTEGER, "destination_quantity": INTEGER, "validated": BOOLEAN})
        pair = {"source_id": STRING, "destination_id": STRING, "sku": STRING}
        definitions = [tool("read_stock", "Read current integer stock quantity and revision for one warehouse and SKU.", {"warehouse_id": STRING, "sku": STRING}), tool("transfer_stock", "Atomically subtract the supplied quantity from source and add it to destination; require sufficient stock and matching current revisions; increment both revisions once.", {**pair, "quantity": {"type": "integer", "minimum": 1}, "source_revision": INTEGER, "destination_revision": INTEGER}), tool("validate_transfer", "Check nonnegative quantities, protected fields, and conservation of total stock across the warehouse records; does not verify the requested transfer quantity.", pair)]
    elif family == "C":
        result = obj({"document_id": STRING, "revision": INTEGER, "status": STATUS, "tags": TAGS, "owner_id": STRING, "validated": BOOLEAN})
        definitions = [tool("read_document", "Read the current revision and metadata of an existing document.", {"document_id": STRING}), tool("update_document", "Set the supplied status, tags and owner on an existing document if expected_revision matches; increment revision exactly once and preserve all other fields.", {"document_id": STRING, "expected_revision": INTEGER, "status": STATUS, "tags": TAGS, "owner_id": STRING}), tool("validate_document", "Check public metadata structure, revision type and protected fields; does not compare metadata with a hidden requested answer.", {"document_id": STRING})]
    else:
        raise ValueError("unknown task family")
    definitions.append(tool("final", "Finish the task with its task ID, completed flag, exact resulting values, and whether you observed validation. This action does not change any file.", {"task_id": STRING, "completed": BOOLEAN, "result": result}))
    return copy.deepcopy(definitions)


def schema_error(value, schema, path="$.arguments"):
    """Small explicit validator for exactly the supported public schema vocabulary."""
    kind = schema.get("type")
    python_type = {"object": dict, "array": list, "string": str, "integer": int, "boolean": bool}.get(kind)
    if python_type is None or type(value) is not python_type:
        return {"code": "TYPE_ERROR", "path": path, "expected": kind}
    if "enum" in schema and not any(strict_equal(value, item) for item in schema["enum"]):
        return {"code": "ENUM_ERROR", "path": path, "expected": "declared public enum"}
    if kind == "object":
        properties = schema["properties"]
        missing = [key for key in schema["required"] if key not in value]
        extra = [key for key in value if key not in properties]
        if missing or extra:
            return {"code": "KEY_ERROR", "path": path + "." + (missing + extra)[0], "expected": "declared required keys only"}
        for key, child in properties.items():
            error = schema_error(value[key], child, path + "." + key)
            if error:
                return error
    elif kind == "array":
        if not schema.get("minItems", 0) <= len(value) <= schema.get("maxItems", float("inf")):
            return {"code": "ARRAY_LENGTH_ERROR", "path": path, "expected": "declared item count"}
        if schema.get("uniqueItems") and len({canonical(item) for item in value}) != len(value):
            return {"code": "DUPLICATE_ITEM", "path": path, "expected": "unique items"}
        for index, item in enumerate(value):
            error = schema_error(item, schema["items"], path + f"[{index}]")
            if error:
                return error
    elif kind == "integer" and not schema.get("minimum", -float("inf")) <= value <= schema.get("maximum", float("inf")):
        return {"code": "RANGE_ERROR", "path": path, "expected": "declared static integer range"}
    return None


class Fixture:
    def __init__(self, task, root):
        self.task = copy.deepcopy(task)
        self.root = Path(root).absolute()
        if self.root.exists() or any(path.is_symlink() for path in (self.root, *self.root.parents)):
            raise ValueError("fixture root must be new and have no symlink ancestors")
        self.root.mkdir(parents=True)
        self.path = self.root / "state.json"
        self.events = []
        self.version = 0
        self.integrity_violation = False
        self._last_digest = digest(task["initial_state"])
        self._write(task["initial_state"])

    def _write(self, state):
        if self.root.is_symlink() or (self.path.exists() and self.path.is_symlink()):
            raise ValueError("unsafe fixture file")
        partial = self.root / "state.partial"
        if partial.exists() or partial.is_symlink():
            raise ValueError("unexpected partial fixture file")
        with partial.open("x", encoding="utf-8") as stream:
            stream.write(canonical(state) + "\n")
        os.replace(partial, self.path)

    def snapshot(self):
        if self.root.is_symlink() or self.path.is_symlink() or not self.path.is_file() or {p.name for p in self.root.iterdir()} != {"state.json"}:
            raise ValueError("unsafe or unexpected fixture file")
        with self.path.open(encoding="utf-8") as stream:
            state = json.load(stream)
        current_digest = digest(state)
        if current_digest != self._last_digest:
            self.version += 1
            self._last_digest = current_digest
        self.integrity_violation |= not integrity_ok(self.task, state)
        return state

    @property
    def state(self):
        return self.snapshot()

    def execute(self, action, turn, available_result_event_ids):
        """Execute declared arguments only; event observations are fixed before execution."""
        available = list(available_result_event_ids)
        known = {event["event_id"]: event for event in self.events}
        if len(available) != len(set(available)) or any(event_id not in known or known[event_id]["generated_in_turn"] >= turn for event_id in available):
            raise ValueError("unknown, duplicate or same-generation observation event")
        before = self.snapshot()
        version_before = self.version
        name = action.get("tool") if type(action) is dict else None
        arguments = action.get("arguments") if type(action) is dict else None
        definitions = {item["name"]: item for item in tool_definitions(self.task["family"])}
        contract_error = None
        if type(action) is not dict or set(action) != {"tool", "arguments"}:
            contract_error = {"code": "ENVELOPE_ERROR", "path": "$", "expected": "tool and arguments only"}
        elif type(name) is not str or name not in definitions:
            contract_error = {"code": "UNKNOWN_TOOL", "path": "$.tool", "expected": "registered name"}
        else:
            contract_error = schema_error(arguments, definitions[name]["parameters"])
        callable_executed = contract_error is None and name != "final"
        status = "CONTRACT_ERROR" if contract_error else "SUCCESS"
        if contract_error:
            result = {"error": contract_error}
        else:
            try:
                result = self._dispatch(name, arguments, copy.deepcopy(before))
            except BusinessError as error:
                status = "ERROR"
                result = {"error": error.code}
        after = self.snapshot()
        event = {"event_id": f"event-{len(self.events):04d}", "tool": name, "arguments": copy.deepcopy(arguments), "status": status, "result": copy.deepcopy(result), "contract_error": contract_error, "callable_executed": callable_executed, "side_effect_write": not strict_equal(before, after), "generated_in_turn": turn, "generated_after_event_ids": available, "state_before": before, "state_after": after, "before_digest": digest(before), "after_digest": digest(after), "version_before": version_before, "version_after": self.version, "integrity_violation_so_far": self.integrity_violation}
        self.events.append(event)
        return copy.deepcopy(event)

    def _dispatch(self, name, args, state):
        if name == "final":
            return {"accepted": True}
        if name == "calculate_total":
            return {"total_minor": total_minor(args["lines"], args["discount_bps"])}
        if name in ("read_order", "write_order_total", "validate_order"):
            record_id = args["order_id"]
            if record_id not in state["orders"]:
                raise BusinessError("UNKNOWN_RECORD")
            row = state["orders"][record_id]
            if name == "read_order":
                return {"order_id": record_id, **copy.deepcopy(row)}
            if name == "write_order_total":
                row["total_minor"] = args["total_minor"]
                self._write(state)
                return {"updated": True, "order_id": record_id, "total_minor": args["total_minor"]}
            valid = type(row.get("total_minor")) is int and row["total_minor"] >= 0 and integrity_ok(self.task, state)
            return {"valid": bool(valid), "order_id": record_id}
        if name in ("read_stock", "transfer_stock", "validate_transfer"):
            table = state["warehouses"]
            ids = [args["warehouse_id"]] if name == "read_stock" else [args["source_id"], args["destination_id"]]
            if any(record_id not in table or table[record_id]["sku"] != args["sku"] for record_id in ids):
                raise BusinessError("UNKNOWN_STOCK_RECORD")
            if name == "read_stock":
                return {"warehouse_id": ids[0], **copy.deepcopy(table[ids[0]])}
            source, destination = (table[record_id] for record_id in ids)
            if name == "transfer_stock":
                if ids[0] == ids[1]:
                    raise BusinessError("SAME_WAREHOUSE")
                if source["revision"] != args["source_revision"] or destination["revision"] != args["destination_revision"]:
                    raise BusinessError("REVISION_CONFLICT")
                if source["quantity"] < args["quantity"]:
                    raise BusinessError("INSUFFICIENT_STOCK")
                source["quantity"] -= args["quantity"]
                destination["quantity"] += args["quantity"]
                source["revision"] += 1
                destination["revision"] += 1
                self._write(state)
                return {"transferred": True, "source_id": ids[0], "destination_id": ids[1], "quantity": args["quantity"], "source_quantity": source["quantity"], "destination_quantity": destination["quantity"], "source_revision": source["revision"], "destination_revision": destination["revision"]}
            original_total = sum(row["quantity"] for row in self.task["initial_state"]["warehouses"].values())
            valid = all(type(row["quantity"]) is int and row["quantity"] >= 0 and type(row["revision"]) is int for row in table.values()) and sum(row["quantity"] for row in table.values()) == original_total and integrity_ok(self.task, state)
            return {"valid": bool(valid), "source_id": ids[0], "destination_id": ids[1], "sku": args["sku"]}
        table = state["documents"]
        record_id = args["document_id"]
        if record_id not in table:
            raise BusinessError("UNKNOWN_DOCUMENT")
        row = table[record_id]
        if name == "read_document":
            return {"document_id": record_id, **copy.deepcopy(row)}
        if name == "update_document":
            if args["expected_revision"] != row["revision"]:
                raise BusinessError("REVISION_CONFLICT")
            for key in ("status", "tags", "owner_id"):
                row[key] = copy.deepcopy(args[key])
            row["revision"] += 1
            self._write(state)
            return {"updated": True, "document_id": record_id, **{k: copy.deepcopy(row[k]) for k in ("revision", "status", "tags", "owner_id")}}
        valid = type(row["revision"]) is int and row["revision"] >= 0 and row["status"] in STATUS["enum"] and type(row["owner_id"]) is str and schema_error(row["tags"], TAGS) is None and integrity_ok(self.task, state)
        return {"valid": bool(valid), "document_id": record_id}


class BusinessError(Exception):
    def __init__(self, code):
        self.code = code
