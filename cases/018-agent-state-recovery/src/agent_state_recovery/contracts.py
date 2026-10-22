"""New strict JSON envelope parser and a deliberately small public schema subset.

Conceptual reference: Case017 contracts.py. No import of another case or SDK.
Audit shares this parser/public schema checker, not execution or endpoint logic.
"""
import json


class ContractError(ValueError):
    def __init__(self, code, path="$", expected="single complete JSON object"):
        self.code, self.path, self.expected = code, path, expected
        super().__init__(code)

    def public(self):
        return {"code": self.code, "path": self.path, "expected": self.expected}


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ContractError("DUPLICATE_KEY")
        result[key] = value
    return result


def _constant(_):
    raise ContractError("NONFINITE_JSON")


def check(value, schema, path="$"):
    kind = schema["type"]
    types = {"object": dict, "array": list, "integer": int, "string": str, "boolean": bool}
    if type(value) is not types[kind]:
        raise ContractError("TYPE_ERROR", path, kind)
    if "enum" in schema and value not in schema["enum"]:
        raise ContractError("ENUM_ERROR", path, "public registered name")
    if kind == "object":
        properties = schema["properties"]
        if set(value) != set(schema["required"]) or set(properties) != set(schema["required"]):
            raise ContractError("KEY_ERROR", path, "exact public required fields")
        for key, child in properties.items():
            check(value[key], child, path + "." + key)
    if kind == "array":
        if not schema.get("minItems", 0) <= len(value) <= schema.get("maxItems", 100000):
            raise ContractError("ARRAY_LENGTH", path, "public size bounds")
        for index, item in enumerate(value):
            check(item, schema["items"], f"{path}[{index}]")
    if kind == "integer" and not schema.get("minimum", -10**30) <= value <= schema.get("maximum", 10**30):
        raise ContractError("RANGE_ERROR", path, "public integer bounds")
    if kind == "string" and not schema.get("minLength", 0) <= len(value) <= schema.get("maxLength", 100000):
        raise ContractError("STRING_LENGTH", path, "public string bounds")


def parse(raw, definitions):
    try:
        value = json.loads(raw, object_pairs_hook=_object, parse_constant=_constant)
    except ContractError:
        raise
    except (ValueError, TypeError, RecursionError) as error:
        raise ContractError("PARSE_ERROR") from error
    if type(value) is not dict:
        raise ContractError("ENVELOPE_TYPE")
    if set(value) != {"tool", "arguments"}:
        raise ContractError("ENVELOPE_KEYS")
    if type(value["tool"]) is not str or type(value["arguments"]) is not dict:
        raise ContractError("ENVELOPE_TYPE")
    registry = {entry["name"]: entry["parameters"] for entry in definitions}
    if value["tool"] not in registry:
        raise ContractError("UNKNOWN_TOOL", "$.tool", "registered public name")
    check(value["arguments"], registry[value["tool"]], "$.arguments")
    return value


def is_parsed_object(raw):
    try:
        return type(json.loads(raw, object_pairs_hook=_object, parse_constant=_constant)) is dict
    except (ValueError, TypeError, RecursionError):
        return False
