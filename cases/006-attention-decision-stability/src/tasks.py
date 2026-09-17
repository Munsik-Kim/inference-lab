"""Fact-first synthetic tasks and a bounded AST interpreter. No eval/exec."""
from __future__ import annotations

import ast
import hashlib
import random
from typing import Any

from .storage import digest

TASKS = ("RETRIEVAL", "COMPARISON", "CODE")
LABELS = ("A", "B", "C", "D")
SPLITS = ("smoke", "dev", "standard", "boundary_pool")
COUNTS = {"smoke": 2, "dev": 32, "standard": 64, "boundary_pool": 64}


class OracleError(ValueError):
    pass


def interpret(program: str) -> int:
    """Integer-only Python subset, <=128 AST nodes, 512 operations, loops <=8."""
    tree = ast.parse(program)
    if len(list(ast.walk(tree))) > 128:
        raise OracleError("Program too large")
    allowed = (ast.Module, ast.Assign, ast.Name, ast.Load, ast.Store, ast.Constant,
               ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.Mod, ast.For, ast.Call)
    if any(not isinstance(node, allowed) for node in ast.walk(tree)):
        raise OracleError("AST node outside whitelist, including unreachable code")
    env: dict[str, int] = {}
    remaining = 512

    def tick() -> None:
        nonlocal remaining
        remaining -= 1
        if remaining < 0:
            raise OracleError("Operation budget exceeded")

    def name(node: ast.AST) -> str:
        if not isinstance(node, ast.Name) or node.id.startswith("_") or len(node.id) > 24:
            raise OracleError("Only simple named integers are allowed")
        return node.id

    def expr(node: ast.AST) -> int:
        tick()
        if isinstance(node, ast.Constant) and type(node.value) is int:
            value = node.value
        elif isinstance(node, ast.Name):
            key = name(node)
            if key not in env:
                raise OracleError("Undefined integer")
            value = env[key]
        elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Mod)):
            a, b = expr(node.left), expr(node.right)
            if isinstance(node.op, ast.Add):
                value = a + b
            elif isinstance(node.op, ast.Sub):
                value = a - b
            elif isinstance(node.op, ast.Mult):
                value = a * b
            else:
                if b == 0:
                    raise OracleError("Modulo by zero")
                value = a % b
        else:
            raise OracleError(f"Unsupported expression: {type(node).__name__}")
        if abs(value) > 1_000_000:
            raise OracleError("Integer bound exceeded")
        return value

    def statement(node: ast.AST) -> None:
        tick()
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            env[name(node.targets[0])] = expr(node.value)
        elif isinstance(node, ast.For) and not node.orelse:
            call = node.iter
            if (not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name)
                    or call.func.id != "range" or call.keywords or len(call.args) != 1
                    or not isinstance(call.args[0], ast.Constant)
                    or type(call.args[0].value) is not int or not 0 <= call.args[0].value <= 8):
                raise OracleError("Only range(literal integer 0..8) loops are allowed")
            key = name(node.target)
            for i in range(call.args[0].value):
                env[key] = i
                for child in node.body:
                    statement(child)
        else:
            raise OracleError(f"Unsupported statement: {type(node).__name__}")

    for node in tree.body:
        statement(node)
    if "answer" not in env:
        raise OracleError("No answer variable")
    return env["answer"]


def scenario(split: str, task: str, index: int, seed: int = 606160) -> dict[str, Any]:
    if split not in SPLITS or task not in TASKS or not 0 <= index < COUNTS[split]:
        raise ValueError("Outside fixed scenario budget")
    sid = f"c006-{split}-{task.lower()}-{index:03d}"
    numeric_seed = int(hashlib.sha256(f"{seed}/{sid}".encode()).hexdigest()[:16], 16)
    rng = random.Random(numeric_seed)
    if task == "CODE":
        initial, step, count, modulus = rng.randrange(11, 90), rng.randrange(2, 13), 2 + index % 6, rng.randrange(17, 37)
        # Disjoint initial-value ranges prevent identical programs across splits.
        # Decided by CPU overlap checks before any model output or design freeze.
        initial = 11 + 64 * SPLITS.index(split) + index
        program = f"total = {initial}\nfor i in range({count}):\n    total = total + {step} * i\nanswer = total % {modulus}"
        answer = (initial + step * count * (count - 1) // 2) % modulus
        if interpret(program) != answer:
            raise OracleError("AST and independent closed-form oracles disagree")
        facts = {"initial": initial, "step": step, "count": count, "modulus": modulus}
        core = "Program (integer arithmetic, Python range excludes its end):\n" + program
        question = "What integer value does answer hold after this program?"
        options = [str(answer), str(answer + 1), str(answer + 3), str(answer + 7)]
        support = ["C0", "C1", "C2", "C3"]
        answer = str(answer)
    else:
        count = 6 + index % 3
        values = rng.sample(range(100, 990), count)
        tag = hashlib.sha256(sid.encode()).hexdigest()[:6]
        facts = [{"id": f"R{i:02d}", "entity": f"unit-{tag}-{i:02d}", "value": values[i]} for i in range(count)]
        rng.shuffle(facts)
        first, second = rng.sample(facts, 2)
        core = "Records (each entity occurs exactly once):\n" + "\n".join(
            f"{r['id']}: {r['entity']} = {r['value']} units." for r in facts)
        if task == "RETRIEVAL":
            question = f"How many units are assigned to {first['entity']}?"
            answer = str(first["value"])
            options = [answer] + [str(r["value"]) for r in facts if r != first][:3]
            support = [first["id"]]
        else:
            question = f"Between {first['entity']} and {second['entity']}, which entity has the larger value?"
            winner = max((first, second), key=lambda x: x["value"])
            answer = winner["entity"]
            options = [answer] + [r["entity"] for r in facts if r != winner][:3]
            support = sorted([first["id"], second["id"]])
    rng.shuffle(options)
    # Balanced gold label by construction, not by baseline output.
    gold_index = index % 4
    old = options.index(answer)
    options[old], options[gold_index] = options[gold_index], options[old]
    if len(set(options)) != 4 or options.count(answer) != 1:
        raise OracleError("Gold/foil ambiguity")
    return {"base_id": sid, "task": task, "split": split, "language": "English instructions; Python subset for CODE",
            "seed": numeric_seed, "facts": facts, "base_facts_hash": digest({"task": task, "facts": facts}),
            "core": core, "question": question, "options": options,
            "gold": LABELS[gold_index], "gold_value": answer, "support_ids": support,
            "difficulty": {"records_or_iterations": len(facts) if isinstance(facts, list) else facts["count"]}}


def permute(item: dict, shift: int) -> dict:
    options = item["options"][shift:] + item["options"][:shift]
    return {**item, "options": options, "gold": LABELS[options.index(item["gold_value"])], "variant": f"cyclic-{shift}"}


def user_content(item: dict, filler: str = "", secondary: bool = False) -> str:
    # Explicit allowlist: no facts/gold/support metadata is formatted here.
    options = "\n".join(f"{label}. {value}" for label, value in zip(LABELS, item["options"]))
    instruction = ("Return only JSON with keys choice (A/B/C/D) and evidence (array of supporting record IDs). "
                   "For code use C0,C1,C2,C3 for the four source lines."
                   if secondary else "Answer with exactly one letter: A, B, C, or D. Do not explain.")
    return (f"{instruction}\n{item['core']}\n<irrelevant_notes>\n{filler}\n</irrelevant_notes>\n"
            f"Question: {item['question']}\nOptions:\n{options}\n")


def build_prompt(item: dict, tokenizer: Any, target: int, secondary: bool = False) -> dict:
    """Adjust only deterministic irrelevant filler; never truncate task evidence."""
    filler_unit = " Archive notes record ordinary paper storage and routine shelf cleaning."
    def render(repeats: int, extra: int = 0) -> tuple[str, list[int]]:
        filler = filler_unit * repeats + (" ." * extra)
        messages = [{"role": "user", "content": user_content(item, filler, secondary)}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tokenizer.encode(text, add_special_tokens=False)
        return text, ids
    _, minimum = render(0)
    if len(minimum) > target:
        raise ValueError("Essential question/evidence exceeds target; no truncation")
    lo, hi = 0, target
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if len(render(mid)[1]) <= target:
            lo = mid
        else:
            hi = mid - 1
    text, ids = render(lo)
    for extra in range(target - len(ids) + 3):
        candidate, tokens = render(lo, extra)
        if len(tokens) == target:
            text, ids = candidate, tokens
            break
    else:
        raise ValueError("Exact filler-only length unavailable; policy requires stopping")
    label_ids = []
    for label in LABELS:
        joined = tokenizer.encode(text + label, add_special_tokens=False)
        if joined[:-1] != ids or len(joined) != len(ids) + 1:
            raise ValueError("BLOCKED_ANSWER_INTERFACE: continuation changes prompt tokenization")
        label_ids.append(joined[-1])
    if len(set(label_ids)) != 4:
        raise ValueError("Distinct label tokens required")
    return {"item_id": f"{item['base_id']}-L{target}" + ("-secondary" if secondary else ""),
            "base_id": item["base_id"], "task": item["task"], "split": item["split"], "length": len(ids),
            "prompt": text, "token_ids": ids, "token_hash": digest(ids), "label_token_ids": label_ids,
            "prompt_hash": hashlib.sha256(text.encode()).hexdigest(), "secondary": secondary,
            "future_answer_tokens_present": False, "filler_policy": "Only irrelevant notes adjusted; no truncation"}
