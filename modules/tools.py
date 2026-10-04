import ast
import operator

_ALLOWED = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

def calculate_expression(expression):
    node = ast.parse(expression, mode="eval").body
    return _evaluate(node)

def _evaluate(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value

    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED:
        left = _evaluate(node.left)
        right = _evaluate(node.right)
        return _ALLOWED[type(node.op)](left, right)

    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED:
        return _ALLOWED[type(node.op)](_evaluate(node.operand))

    raise ValueError("Only basic arithmetic expressions are allowed.")
