"""Bounded integer AST oracle reused from Case006; see reuse ledger."""
import ast

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

