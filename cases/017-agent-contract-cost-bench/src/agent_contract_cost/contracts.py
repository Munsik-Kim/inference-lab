"""Strict, stdlib-only action parsing and public schema validation.

This is a new contract, not Case015's first-valid-dict extraction. A parsed
attempt is a syntactically valid JSON *object*, even if its envelope is invalid.
Arrays/scalars and syntax errors are contract rejections, not parsed attempts.
No values are repaired, coerced, extracted from prose, or dispatched here.
"""
from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class ContractError:
    code: str
    path: str
    expected: str

    def public(self) -> dict[str, str]:
        """Error feedback contains neither observed values nor hidden answers."""
        return {"code": self.code, "path": self.path, "expected": self.expected}


@dataclass(frozen=True)
class ParseResult:
    accepted: bool
    action: dict[str, Any] | None
    error: ContractError | None
    parsed_object: bool

    def as_dict(self) -> dict[str, Any]:
        return {"accepted": self.accepted, "action": copy.deepcopy(self.action),
                "error": self.error.public() if self.error else None,
                "parsed_object": self.parsed_object}


class _DuplicateKey(ValueError):
    pass


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKey("duplicate JSON key")
        result[key] = value
    return result


def _invalid_constant(_value):
    raise ValueError("non-finite JSON literal")


def registry_mapping(registry) -> dict[str, dict]:
    """Accept public definition lists or name -> definition/schema mappings."""
    if isinstance(registry, list):
        names = [entry["name"] for entry in registry]
        if len(set(names)) != len(names):
            raise ValueError("duplicate registered tool name")
        registry = {entry["name"]: entry for entry in registry}
    if not isinstance(registry, Mapping) or not registry:
        raise ValueError("registry must be a nonempty public definition mapping")
    result = {}
    for name, definition in registry.items():
        if not isinstance(name, str) or not name:
            raise ValueError("invalid registered tool name")
        if not isinstance(definition, Mapping):
            raise ValueError("tool definition is not an object")
        if "name" in definition and definition["name"] != name:
            raise ValueError("registered definition name mismatch")
        schema = definition.get("parameters", definition)
        check_schema(schema)
        if schema.get("type") != "object":
            raise ValueError("tool arguments require object schema")
        result[name] = copy.deepcopy(dict(schema))
    if "final" not in result:
        raise ValueError("final must be a registered action")
    return result


_KEYWORDS = {"type", "properties", "required", "additionalProperties", "items",
             "enum", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
             "minItems", "maxItems", "uniqueItems", "minLength", "maxLength", "oneOf", "anyOf",
             "description", "title", "$schema"}
_TYPES = {"object", "array", "string", "integer", "number", "boolean", "null"}


def check_schema(schema: dict) -> None:
    """Fail closed for unsupported keywords instead of silently ignoring them."""
    if not isinstance(schema, Mapping) or set(schema) - _KEYWORDS:
        raise ValueError("unsupported public schema keyword or schema type")
    kind = schema.get("type")
    if kind is not None and kind not in _TYPES:
        raise ValueError("unsupported JSON schema type")
    for branch_key in ("oneOf", "anyOf"):
        if branch_key in schema:
            if not isinstance(schema[branch_key], list) or not schema[branch_key]:
                raise ValueError("schema branches must be a nonempty list")
            for branch in schema[branch_key]:
                check_schema(branch)
    properties = schema.get("properties", {})
    if not isinstance(properties, Mapping):
        raise ValueError("properties must be an object")
    for name, child in properties.items():
        if not isinstance(name, str):
            raise ValueError("property name must be string")
        check_schema(child)
    if "items" in schema:
        check_schema(schema["items"])
    additional = schema.get("additionalProperties", True)
    if isinstance(additional, Mapping):
        check_schema(additional)
    elif type(additional) is not bool:
        raise ValueError("additionalProperties must be boolean or schema")
    required = schema.get("required", [])
    if not isinstance(required, list) or any(not isinstance(key, str) for key in required):
        raise ValueError("required must be a list of property names")
    if len(set(required)) != len(required) or any(key not in properties for key in required):
        raise ValueError("invalid required properties")
    if "enum" in schema and (not isinstance(schema["enum"], list) or not schema["enum"]):
        raise ValueError("enum must be a nonempty list")
    for key in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"):
        if key in schema and (type(schema[key]) not in (int, float) or not math.isfinite(schema[key])):
            raise ValueError("invalid numeric schema boundary")
    for key in ("minItems", "maxItems", "minLength", "maxLength"):
        if key in schema and (type(schema[key]) is not int or schema[key] < 0):
            raise ValueError("invalid nonnegative schema boundary")
    if "uniqueItems" in schema and type(schema["uniqueItems"]) is not bool:
        raise ValueError("uniqueItems must be boolean")


def _type_matches(value, kind):
    return {"object": lambda: type(value) is dict,
            "array": lambda: type(value) is list,
            "string": lambda: type(value) is str,
            "integer": lambda: type(value) is int,
            "number": lambda: type(value) is int or (type(value) is float and math.isfinite(value)),
            "boolean": lambda: type(value) is bool,
            "null": lambda: value is None}[kind]()


def _json_equal(a, b):
    # In particular, True is not equal to 1 under this contract.
    if type(a) is not type(b):
        return False
    if type(a) is dict:
        return set(a) == set(b) and all(_json_equal(a[key], b[key]) for key in a)
    if type(a) is list:
        return len(a) == len(b) and all(_json_equal(x, y) for x, y in zip(a, b))
    return a == b


def _property_path(path, key):
    # No model-supplied key is interpolated in feedback. Known schema keys only.
    return path + "." + key


def validate_schema(value, schema, path="$", *, check=True) -> ContractError | None:
    if check:
        check_schema(schema)
    kind = schema.get("type")
    if kind and not _type_matches(value, kind):
        return ContractError("TYPE_ERROR", path, kind)
    for branch_key in ("oneOf", "anyOf"):
        if branch_key in schema:
            successes = sum(validate_schema(value, branch, path, check=False) is None
                            for branch in schema[branch_key])
            if (branch_key == "oneOf" and successes != 1) or (branch_key == "anyOf" and successes == 0):
                return ContractError("SCHEMA_BRANCH_ERROR", path, branch_key)
    if "enum" in schema and not any(_json_equal(value, option) for option in schema["enum"]):
        return ContractError("ENUM_ERROR", path, "registered enum value")
    if type(value) is dict:
        props = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in value:
                return ContractError("MISSING_KEY", _property_path(path, key), props[key].get("type", "required field"))
        if schema.get("additionalProperties", True) is False and set(value) - set(props):
            return ContractError("EXTRA_KEY", path, "only registered keys")
        for key, child in props.items():
            if key in value:
                error = validate_schema(value[key], child, _property_path(path, key), check=False)
                if error:
                    return error
        extra_schema = schema.get("additionalProperties")
        if isinstance(extra_schema, Mapping):
            for key in set(value) - set(props):
                error = validate_schema(value[key], extra_schema, path + ".*", check=False)
                if error:
                    return error
    if type(value) is list:
        for key, op in (("minItems", lambda a, b: a < b), ("maxItems", lambda a, b: a > b)):
            if key in schema and op(len(value), schema[key]):
                return ContractError("LENGTH_ERROR", path, key)
        if schema.get("uniqueItems") and any(_json_equal(item, prior) for i, item in enumerate(value) for prior in value[:i]):
            return ContractError("DUPLICATE_ITEM", path, "unique items")
        if "items" in schema:
            for index, child in enumerate(value):
                error = validate_schema(child, schema["items"], f"{path}[{index}]", check=False)
                if error:
                    return error
    if type(value) is str:
        for key, op in (("minLength", lambda a, b: a < b), ("maxLength", lambda a, b: a > b)):
            if key in schema and op(len(value), schema[key]):
                return ContractError("LENGTH_ERROR", path, key)
    if type(value) in (int, float):
        if type(value) is float and not math.isfinite(value):
            return ContractError("TYPE_ERROR", path, "finite number")
        for key, violation in (("minimum", lambda a, b: a < b), ("maximum", lambda a, b: a > b),
                               ("exclusiveMinimum", lambda a, b: a <= b), ("exclusiveMaximum", lambda a, b: a >= b)):
            if key in schema and violation(value, schema[key]):
                return ContractError("RANGE_ERROR", path, "public numeric range")
    return None


def parse_action(raw: str, registry) -> ParseResult:
    schemas = registry_mapping(registry)
    if type(raw) is not str:
        return ParseResult(False, None, ContractError("PARSE_ERROR", "$", "single JSON object"), False)
    try:
        obj = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
    except _DuplicateKey:
        return ParseResult(False, None, ContractError("DUPLICATE_KEY", "$", "unique JSON keys"), False)
    except (ValueError, RecursionError):
        return ParseResult(False, None, ContractError("PARSE_ERROR", "$", "single JSON object"), False)
    if type(obj) is not dict:
        return ParseResult(False, None, ContractError("ENVELOPE_TYPE", "$", "object"), False)
    if set(obj) != {"tool", "arguments"}:
        return ParseResult(False, None, ContractError("ENVELOPE_KEYS", "$", "tool and arguments only"), True)
    if type(obj["tool"]) is not str:
        return ParseResult(False, None, ContractError("TYPE_ERROR", "$.tool", "string"), True)
    if obj["tool"] not in schemas:
        return ParseResult(False, None, ContractError("UNKNOWN_TOOL", "$.tool", "registered tool name"), True)
    error = validate_schema(obj["arguments"], schemas[obj["tool"]], "$.arguments", check=False)
    if error:
        return ParseResult(False, None, error, True)
    return ParseResult(True, copy.deepcopy(obj), None, True)


def grammar_schema(registry) -> dict:
    """Only public definitions enter decoding; task/oracle data is not accepted."""
    schemas = registry_mapping(registry)
    return {"oneOf": [{"type": "object", "properties": {
        "tool": {"type": "string", "enum": [name]}, "arguments": schema},
        "required": ["tool", "arguments"], "additionalProperties": False}
        for name, schema in schemas.items()]}
