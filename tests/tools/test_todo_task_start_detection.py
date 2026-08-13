"""Unit tests for TodoStore.detect_task_start (task-started notification).

A task "starts" when a pending todo write transitions an item to
``in_progress`` that was not ``in_progress`` before the call — either an
existing item moving status, or a brand-new item created directly as
``in_progress``. Re-asserting the same item does not fire.
"""

from tools.todo_tool import TodoStore


class TestDetectTaskStart:
    """Transition detection semantics on the store's pre-call state."""

    def test_none_todos_returns_none(self):
        store = TodoStore()
        store.write([
            {"id": "1", "content": "task one", "status": "pending"},
        ])
        assert store.detect_task_start(None) is None

    def test_empty_list_returns_none(self):
        store = TodoStore()
        assert store.detect_task_start([]) is None

    def test_new_item_created_in_progress(self):
        store = TodoStore()
        store.write([
            {"id": "1", "content": "task one", "status": "pending"},
        ])
        started = store.detect_task_start([
            {"id": "1", "content": "task one", "status": "pending"},
            {"id": "2", "content": "task two", "status": "in_progress"},
        ])
        assert started is not None
        assert started["id"] == "2"
        assert started["content"] == "task two"
        assert started["status"] == "in_progress"

    def test_existing_item_moves_to_in_progress(self):
        store = TodoStore()
        store.write([
            {"id": "1", "content": "task one", "status": "pending"},
            {"id": "2", "content": "task two", "status": "pending"},
        ])
        started = store.detect_task_start([
            {"id": "1", "content": "task one", "status": "in_progress"},
            {"id": "2", "content": "task two", "status": "pending"},
        ])
        assert started is not None
        assert started["id"] == "1"

    def test_reasserting_current_task_returns_none(self):
        store = TodoStore()
        store.write([
            {"id": "1", "content": "task one", "status": "in_progress"},
        ])
        started = store.detect_task_start([
            {"id": "1", "content": "task one", "status": "in_progress"},
        ])
        assert started is None

    def test_no_transition_returns_none(self):
        store = TodoStore()
        store.write([
            {"id": "1", "content": "task one", "status": "pending"},
        ])
        started = store.detect_task_start([
            {"id": "1", "content": "task one", "status": "pending"},
            {"id": "2", "content": "task two", "status": "completed"},
        ])
        assert started is None

    def test_first_started_item_wins_by_input_order(self):
        store = TodoStore()
        started = store.detect_task_start([
            {"id": "3", "content": "third", "status": "in_progress"},
            {"id": "2", "content": "second", "status": "in_progress"},
        ])
        assert started is not None
        assert started["id"] == "3"

    def test_missing_content_falls_back_to_previous_item(self):
        store = TodoStore()
        store.write([
            {"id": "1", "content": "task one", "status": "pending"},
        ])
        started = store.detect_task_start([
            {"id": "1", "status": "in_progress"},
        ])
        assert started is not None
        assert started["content"] == "task one"

    def test_missing_content_and_id_skipped(self):
        store = TodoStore()
        started = store.detect_task_start([
            {"status": "in_progress"},
        ])
        assert started is None

    def test_string_payload_parsed(self):
        import json
        store = TodoStore()
        store.write([
            {"id": "1", "content": "task one", "status": "pending"},
        ])
        started = store.detect_task_start(json.dumps([
            {"id": "1", "content": "task one", "status": "in_progress"},
        ]))
        assert started is not None
        assert started["id"] == "1"

    def test_unparseable_string_returns_none(self):
        store = TodoStore()
        assert store.detect_task_start("not json") is None

    def test_non_list_returns_none(self):
        store = TodoStore()
        assert store.detect_task_start({"todos": []}) is None

    def test_content_capped_at_max_chars(self):
        from tools.todo_tool import MAX_TODO_CONTENT_CHARS
        store = TodoStore()
        started = store.detect_task_start([
            {"id": "1", "content": "x" * (MAX_TODO_CONTENT_CHARS + 500), "status": "in_progress"},
        ])
        assert started is not None
        assert len(started["content"]) <= MAX_TODO_CONTENT_CHARS
