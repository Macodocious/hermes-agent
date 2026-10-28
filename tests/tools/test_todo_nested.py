"""Tests for nested subtasks in the todo tool (parent field).

Covers: validation of the parent field, dangling/cyclic parent
sanitization, merge-mode parent updates, tree-aware post-compression
injection, and the flat-only reorder guard.

Fork model (differs from upstream, adapted here):
- Item ids are renumbered to their list position on a replace-mode write, so
  fixtures use numeric ids and parent refs are remapped with them.
- Injection stays flat and keeps completed/cancelled rows visible (finished
  work must persist across compression), so the subtree-rendering cases are
  skipped rather than deleted.
- The write path does not reorder items, so the flat reorder guard does not
  apply.
"""

import json

import pytest

from tools.todo_tool import TodoStore, todo_tool


def _item(i, status="pending", parent=None, content=None):
    d = {"id": i, "content": content or f"task {i}", "status": status}
    if parent is not None:
        d["parent"] = parent
    return d


class TestParentValidation:
    def test_parent_preserved(self):
        store = TodoStore()
        items = store.write([_item("1"), _item("2", parent="1")])
        assert items[1]["parent"] == "1"

    def test_self_parent_dropped(self):
        store = TodoStore()
        items = store.write([_item("1", parent="1")])
        assert "parent" not in items[0]

    def test_dangling_parent_dropped(self):
        store = TodoStore()
        items = store.write([_item("1", parent="ghost")])
        assert "parent" not in items[0]

    def test_cycle_broken(self):
        store = TodoStore()
        items = store.write([_item("1", parent="2"), _item("2", parent="1")])
        # At least one link removed; no item can walk a loop
        by_id = {i["id"]: i for i in items}
        for item in items:
            seen = set()
            node = item
            while node.get("parent"):
                assert node["parent"] not in seen
                seen.add(node["id"])
                node = by_id[node["parent"]]

    def test_empty_parent_omitted(self):
        store = TodoStore()
        items = store.write([_item("1", parent="")])
        assert "parent" not in items[0]


class TestMergeParent:
    def test_merge_sets_parent(self):
        store = TodoStore()
        store.write([_item("1"), _item("2")])
        items = store.write([{"id": "2", "parent": "1"}], merge=True)
        by_id = {i["id"]: i for i in items}
        assert by_id["2"]["parent"] == "1"

    def test_merge_clears_parent_with_empty_string(self):
        store = TodoStore()
        store.write([_item("1"), _item("2", parent="1")])
        items = store.write([{"id": "2", "parent": ""}], merge=True)
        by_id = {i["id"]: i for i in items}
        assert "parent" not in by_id["2"]

    def test_merge_new_child_appended(self):
        store = TodoStore()
        store.write([_item("1")])
        items = store.write([_item("2", parent="1")], merge=True)
        assert items[-1] == {"id": "2", "content": "task 2", "status": "pending", "parent": "1"}


class TestNestedInjection:
    @pytest.mark.skip(
        reason="fork renders a flat list with completed rows visible; subtree indentation is not implemented"
    )
    def test_children_indented(self):
        store = TodoStore()
        store.write([
            _item("wp1", status="in_progress"),
            _item("t1", parent="wp1"),
            _item("t2", parent="wp1"),
        ])
        text = store.format_for_injection()
        assert text is not None
        lines = text.split("\n")
        assert lines[1].startswith("- [>] wp1.")
        assert lines[2].startswith("  - [ ] t1.")
        assert lines[3].startswith("  - [ ] t2.")

    @pytest.mark.skip(
        reason="fork renders a flat list with completed rows visible; subtree indentation is not implemented"
    )
    def test_completed_parent_kept_when_child_active(self):
        store = TodoStore()
        store.write([
            _item("wp1", status="completed"),
            _item("t1", status="pending", parent="wp1"),
        ])
        text = store.format_for_injection()
        assert text is not None
        assert "wp1" in text  # parent context survives
        assert "[x]" in text
        assert "  - [ ] t1." in text

    @pytest.mark.skip(reason="fork keeps finished rows visible by design; it does not omit whole subtrees")
    def test_finished_subtree_omitted(self):
        store = TodoStore()
        store.write([
            _item("wp1", status="completed"),
            _item("t1", status="completed", parent="wp1"),
            _item("wp2", status="pending"),
        ])
        text = store.format_for_injection()
        assert text is not None
        assert "wp1" not in text
        assert "t1" not in text
        assert "wp2" in text

    @pytest.mark.skip(reason="fork keeps finished rows visible by design; injection is never None for a non-empty list")
    def test_all_finished_returns_none(self):
        store = TodoStore()
        store.write([
            _item("a", status="completed"),
            _item("a1", status="cancelled", parent="a"),
        ])
        assert store.format_for_injection() is None

    @pytest.mark.skip(reason="fork renumbers ids, so the rendered line carries the position, not the authored id")
    def test_flat_injection_unchanged(self):
        store = TodoStore()
        store.write([_item("a", status="in_progress"), _item("b")])
        text = store.format_for_injection()
        assert text is not None
        assert text.split("\n")[1] == "- [>] a. task a (in_progress)"


class TestOrderGuard:
    @pytest.mark.skip(reason="fork write path keeps authored order and does not reorder nested or flat lists")
    def test_nested_list_keeps_authored_order(self):
        store = TodoStore()
        items = store.write([
            _item("a", status="pending"),
            _item("b", status="in_progress"),
            _item("b1", parent="b"),
        ])
        assert [i["id"] for i in items] == ["a", "b", "b1"]

    @pytest.mark.skip(reason="fork write path keeps authored order and does not reorder nested or flat lists")
    def test_flat_list_still_reordered(self):
        store = TodoStore()
        items = store.write([
            _item("a", status="pending"),
            _item("b", status="in_progress"),
        ])
        assert [i["id"] for i in items] == ["b", "a"]


class TestToolRoundTrip:
    def test_parent_survives_json(self):
        store = TodoStore()
        out = todo_tool(todos=[_item("1"), _item("2", parent="1")], store=store)
        data = json.loads(out)
        assert data["todos"][1]["parent"] == "1"
        assert data["summary"]["total"] == 2

    def test_hydration_replay_preserves_parent(self):
        # Simulate _hydrate_todo_store: write the previous tool result's
        # todos array back into a fresh store in replace mode.
        store = TodoStore()
        out = json.loads(todo_tool(todos=[_item("1"), _item("2", parent="1")], store=store))
        fresh = TodoStore()
        replayed = fresh.write(out["todos"], merge=False)
        assert replayed[1]["parent"] == "1"
