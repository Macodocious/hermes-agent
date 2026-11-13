"""The approval notify path must send through the adapter that is registered
for the platform now, not the one the turn captured when the message arrived.

A platform reconnect builds a new adapter instance and swaps it into
``self.adapters`` while nulling the old instance's client, so a prompt sent
through the captured object fails with "Not connected" and the approval gate
then blocks with nothing in chat to click.
"""

import ast
import pathlib

RUN_PY = pathlib.Path(__file__).resolve().parents[2] / "gateway" / "run.py"


def _function_node(tree: ast.AST, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in gateway/run.py")


def test_approval_notify_uses_live_adapter_not_the_captured_one():
    tree = ast.parse(RUN_PY.read_text(encoding="utf-8"))
    node = _function_node(tree, "_approval_notify_sync")
    receivers = {
        n.value.id
        for n in ast.walk(node)
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
    }
    assert "_status_adapter" not in receivers, (
        "approval notify sends through the turn-captured adapter, which a "
        "reconnect leaves clientless"
    )
    assert "_notify_adapter" in receivers


def test_live_adapter_prefers_the_registry_and_falls_back_to_the_captured_one():
    tree = ast.parse(RUN_PY.read_text(encoding="utf-8"))
    node = _function_node(tree, "_live_adapter")
    returned = [n for n in ast.walk(node) if isinstance(n, ast.Return)]
    assert len(returned) == 1, "_live_adapter must have exactly one return"
    expression = ast.unparse(returned[0].value)
    assert "self._adapter_for_source(source)" in expression, expression
    assert _fallback_is_captured(returned[0].value), expression


def _fallback_is_captured(value: ast.expr) -> bool:
    if not isinstance(value, ast.BoolOp) or not isinstance(value.op, ast.Or):
        return False
    return any(
        isinstance(operand, ast.Name) and operand.id == "_status_adapter"
        for operand in value.values
    )
