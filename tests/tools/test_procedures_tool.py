"""Tests for the procedures tools: list_procedures, view_procedure, write_procedure.

The tools live in the hermes-expansion plugin; these tests load that module by
path so the tool surface is exercised as the agent sees it.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

RULEBOOK_PATH = Path.home() / ".hermes" / "plugins" / "hermes-expansion" / "rulebook.py"


def _load_rulebook(monkeypatch, procedures_dir: Path):
    if not RULEBOOK_PATH.is_file():
        pytest.skip(f"plugin module not present: {RULEBOOK_PATH}")
    package = types.ModuleType("procedures_plugin")
    package.__path__ = [str(RULEBOOK_PATH.parent)]
    sys.modules["procedures_plugin"] = package
    spec = importlib.util.spec_from_file_location("procedures_plugin.rulebook", RULEBOOK_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["procedures_plugin.rulebook"] = module
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "PROCEDURES_DIR", procedures_dir)
    monkeypatch.setattr(module, "_require_gate", lambda: None)
    return module


def _write_procedure_file(procedures_dir: Path, filename: str, name: str, description: str, body: str = "Body."):
    procedures_dir.mkdir(parents=True, exist_ok=True)
    (procedures_dir / filename).write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\n{body}\n",
        encoding="utf-8",
    )


def test_list_reads_the_directory(tmp_path, monkeypatch):
    procedures_dir = tmp_path / "procedures"
    _write_procedure_file(procedures_dir, "one.md", "one", "First procedure.")
    module = _load_rulebook(monkeypatch, procedures_dir)

    payload = json.loads(module._list_procedures_handler({}))

    assert [entry["name"] for entry in payload["procedures"]] == ["one"]
    assert payload["procedures"][0]["description"] == "First procedure."


def test_list_of_missing_directory_is_empty(tmp_path, monkeypatch):
    module = _load_rulebook(monkeypatch, tmp_path / "procedures")

    payload = json.loads(module._list_procedures_handler({}))

    assert payload["procedures"] == []


def test_view_returns_body_without_frontmatter(tmp_path, monkeypatch):
    procedures_dir = tmp_path / "procedures"
    _write_procedure_file(procedures_dir, "one.md", "one", "First.", body="Step 1. Step 2.")
    module = _load_rulebook(monkeypatch, procedures_dir)

    payload = json.loads(module._view_procedure_handler({"name": "one"}))

    assert payload["body"].strip() == "Step 1. Step 2."
    assert "description:" not in payload["body"]


def test_view_unknown_name_reports_an_error(tmp_path, monkeypatch):
    module = _load_rulebook(monkeypatch, tmp_path / "procedures")

    payload = json.loads(module._view_procedure_handler({"name": "missing"}))

    assert "error" in payload


def test_write_creates_the_file_and_the_directory(tmp_path, monkeypatch):
    procedures_dir = tmp_path / "procedures"
    module = _load_rulebook(monkeypatch, procedures_dir)

    payload = json.loads(module._write_procedure_handler({
        "reason": "test",
        "path": "new-one.md",
        "content": "---\nname: new-one\ndescription: New.\n---\n\nBody.\n",
    }))

    assert payload["status"] == "written"
    written = procedures_dir / "new-one.md"
    assert written.is_file()
    assert "name: new-one" in written.read_text(encoding="utf-8")


def test_write_outside_the_procedures_folder_is_refused(tmp_path, monkeypatch):
    procedures_dir = tmp_path / "procedures"
    module = _load_rulebook(monkeypatch, procedures_dir)

    payload = json.loads(module._write_procedure_handler({
        "reason": "test",
        "path": "../rules/02-design-philosophy.md",
        "content": "nope",
    }))

    assert "error" in payload
    assert not (tmp_path / "rules" / "02-design-philosophy.md").exists()
