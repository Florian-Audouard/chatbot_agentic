from __future__ import annotations

import ast
import operator

from langchain_core.tools import tool


_BIN_OPS: dict[type[ast.operator], object] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS: dict[type[ast.unaryop], object] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _eval_arith(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_arith(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left = _eval_arith(node.left)
        right = _eval_arith(node.right)
        return float(_BIN_OPS[type(node.op)](left, right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return float(_UNARY_OPS[type(node.op)](_eval_arith(node.operand)))

    raise ValueError("Unsupported expression")


def safe_arithmetic(expression: str) -> str:
    """Safely evaluate a basic arithmetic expression.

    Supported: numbers, parentheses, + - * / % **, unary +/-.
    """

    try:
        tree = ast.parse(expression, mode="eval")
        value = _eval_arith(tree)
    except Exception as exc:
        return f"Error: {exc}"

    # Render as int if it is an integer value (avoid `4.0`).
    if value.is_integer():
        return str(int(value))
    return str(value)


@tool
def simple_math(expression: str) -> str:
    """Evaluate a basic arithmetic expression like '2+2*3'."""

    return safe_arithmetic(expression)
