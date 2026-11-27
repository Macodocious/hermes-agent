"""One row per task: a duplicate write reconciles into the row already held.

The reported defect: a write naming a task the list already held landed as a
second row, because the merge path appended whenever the id it was handed named
no row. The list then carried one task twice, and the only way to clear the copy
was to cancel it with a reason — a user-visible line per duplicate. Identity is
the plan and specification pair when both rows carry one, the content otherwise,
and a terminal row takes no part in the match.
"""

from tools.todo_tool import TodoStore

PLAN = "/root/.hermes/cache/plans/p/plan.md"
SPEC = "/root/.hermes/cache/plans/p/task.md"


def _seeded() -> TodoStore:
    store = TodoStore()
    store.write(
        [{"id": "seed-0", "content": "Implement `A`", "status": "pending", "plan": PLAN, "spec": SPEC}],
        merge=True,
    )
    return store


def test_identical_task_under_an_unknown_id_adds_no_row():
    store = _seeded()
    store.write(
        [{"id": "99", "content": "reworded", "status": "pending", "plan": PLAN, "spec": SPEC}],
        merge=True,
    )
    rows = store.read()
    assert len(rows) == 1
    assert rows[0]["content"] == "Implement `A`"


def test_missing_references_are_filled_and_the_row_is_kept():
    store = TodoStore()
    store.write([{"id": "1", "content": "Implement `A`", "status": "paused"}], merge=True)
    store.write(
        [{"id": "7", "content": "Implement `A`", "status": "pending", "plan": PLAN, "spec": SPEC}],
        merge=True,
    )
    rows = store.read()
    assert len(rows) == 1
    assert rows[0]["status"] == "paused"
    assert rows[0]["plan"] == PLAN
    assert rows[0]["spec"] == SPEC


def test_a_finished_task_can_be_created_again():
    store = TodoStore()
    store.write([{"id": "1", "content": "Implement `A`", "status": "completed"}], merge=True)
    store.write([{"id": "2", "content": "Implement `A`", "status": "pending"}], merge=True)
    assert len(store.read()) == 2


def test_one_merge_payload_carrying_the_same_task_twice_adds_one_row():
    store = TodoStore()
    store.write(
        [
            {"id": "1", "content": "Implement `A`", "status": "pending"},
            {"id": "2", "content": "Implement `A`", "status": "pending"},
        ],
        merge=True,
    )
    assert len(store.read()) == 1


def test_a_whole_list_write_keeps_one_row_per_task():
    store = TodoStore()
    store.write(
        [
            {"id": "1", "content": "Implement `A`", "status": "pending"},
            {"id": "2", "content": "Implement `A`", "status": "pending"},
        ]
    )
    assert len(store.read()) == 1
