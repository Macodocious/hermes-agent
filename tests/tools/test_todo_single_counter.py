"""One external task counter: the plan's position, never the store id.

The reported defect: the lifecycle lines numbered a task "2 of 11" from the
plan while the tracker numbered the same row "task 4" from its store id, two
meta rows having been added to the list ahead of the plan. Two counters, one
task. Every surface below therefore renders the plan-scoped label and nothing
else: the block, the refusal text, the tool result, and the pacing line.
"""

import json

from agent.task_manager import TASK_PACING_TOOL_REMINDER, tool_result_reminder
from tools.todo_tool import TodoStore, todo_tool

PLAN = "plan.md"

# The reported list: two agent-authored rows ahead of the plan's eleven, so the
# plan's task 2 carries store id 4.
INCIDENT = [
    {"id": "1", "content": "research", "status": "completed"},
    {"id": "2", "content": "write the plan", "status": "completed"},
] + [
    {"id": str(index + 3), "content": f"plan task {index}", "status": "pending", "plan": PLAN}
    for index in range(1, 12)
]


def _incident_store() -> TodoStore:
    store = TodoStore()
    store.write([dict(item) for item in INCIDENT])
    return store


class _Agent:
    def __init__(self, store):
        self._todo_store = store
        self._lifecycle_reminder_issued = False


def test_block_numbers_the_plan_not_the_tracker():
    store = _incident_store()

    block = store.format_for_turn() or ""

    # Plan task 2 is store row 4. The block names it 2/11 and never 4.
    assert "- [2/11] [pending] plan task 2" in block
    assert "- [1/11] [pending] plan task 1" in block
    assert "[4]" not in block
    # A row outside the plan is not one of the tasks being counted.
    assert "- [completed] research" not in block  # completed rows render [x]
    assert "- [x] research" in block


def test_a_row_outside_the_plan_carries_no_label():
    store = _incident_store()
    store.write([{"id": "1", "content": "research", "status": "in_progress"}], merge=True)

    block = store.format_for_turn() or ""

    assert "- [in_progress] research" in block
    assert "- [1/11] [pending] plan task 1" in block


def test_a_list_with_no_plan_numbers_its_own_rows():
    store = TodoStore()
    store.write([
        {"id": "1", "content": "a", "status": "pending"},
        {"id": "2", "content": "b", "status": "pending"},
    ])

    block = store.format_for_turn() or ""

    assert "- [1/2] [pending] a" in block
    assert "- [2/2] [pending] b" in block


def test_a_lifecycle_action_takes_the_label_from_the_block():
    store = _incident_store()

    # Plan task 1 is store row 3; the block called it 1/11.
    assert store.transition("begin", "1/11")["ok"] is True
    assert store.read()[2]["status"] == "in_progress"


def test_a_merge_write_can_rename_by_label():
    store = _incident_store()

    store.write([{"label": "2/11", "content": "renamed"}], merge=True)

    assert store.read()[3]["content"] == "renamed"


def test_a_refusal_names_rows_by_label():
    store = _incident_store()
    store.transition("begin", "1/11")

    refusal = store.transition("begin", "2/11")["error"]

    assert "1/11" in refusal
    assert "task 3" not in refusal
    assert "task 4" not in refusal


def test_the_tool_result_counts_the_plan():
    store = _incident_store()

    payload = json.loads(todo_tool(store=store))

    assert payload["summary"]["total"] == 11
    assert payload["summary"]["rows"] == 13
    assert [row["task"] for row in payload["todos"]][3] == "2/11"


def test_the_pacing_line_names_the_plan_position():
    store = _incident_store()
    store.transition("begin", "1/11")
    store.mark_work()

    line = tool_result_reminder(_Agent(store), "read_file")

    assert line.startswith('[Task lifecycle] Task 1/11 "plan task 1" is current')
    assert TASK_PACING_TOOL_REMINDER.format(
        label="1/11", content="plan task 1", work_count=1
    ) in line
