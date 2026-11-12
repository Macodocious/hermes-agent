"""Procedures index for the system prompt.

A sibling of the skills index.  Every procedure's name and description ride in
the cached prompt, and the body is fetched on demand by ``view_procedure``.
The code is deliberately its own implementation: it shares no cache, no
snapshot and no builder with the skills pipe.

Layout: one flat ``<name>.md`` per procedure under ``<hermes home>/procedures``.
Each file opens with a frontmatter block carrying ``name`` and ``description``.
A file missing either field is left out of the index and reported at warning
level; the other procedures still render.
"""

from __future__ import annotations

import json
import logging
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Optional

from agent.skill_utils import parse_frontmatter
from hermes_constants import get_hermes_home
from utils import atomic_json_write

logger = logging.getLogger(__name__)

PROCEDURES_DIR_NAME = "procedures"
PROCEDURES_SNAPSHOT_FILE = ".procedures_prompt_snapshot.json"
PROCEDURES_SNAPSHOT_VERSION = 1
PROCEDURES_FILE_SUFFIX = ".md"
PROCEDURES_SECTION_HEADER = "## Procedures (mandatory)"
PROCEDURES_OPEN_TAG = "<available_procedures>"
PROCEDURES_CLOSE_TAG = "</available_procedures>"

_PROCEDURES_PROMPT_CACHE_MAX = 8
_PROCEDURES_PROMPT_CACHE: "OrderedDict[tuple, str]" = OrderedDict()
_PROCEDURES_PROMPT_CACHE_LOCK = threading.Lock()


def get_procedures_dir() -> Path:
    """Return the flat directory holding one file per procedure."""
    return get_hermes_home() / PROCEDURES_DIR_NAME


def _procedures_prompt_snapshot_path() -> Path:
    return get_hermes_home() / PROCEDURES_SNAPSHOT_FILE


def clear_procedures_system_prompt_cache(*, clear_snapshot: bool = False) -> None:
    """Drop the in-process procedures prompt cache (and optionally the snapshot)."""
    with _PROCEDURES_PROMPT_CACHE_LOCK:
        _PROCEDURES_PROMPT_CACHE.clear()
    if clear_snapshot:
        try:
            _procedures_prompt_snapshot_path().unlink(missing_ok=True)
        except OSError as error:
            logger.warning("Could not remove procedures prompt snapshot: %s", error)


def _build_procedures_manifest(procedures_dir: Path) -> dict[str, list[int]]:
    """Record the modification time and size of every procedure file."""
    manifest: dict[str, list[int]] = {}
    try:
        procedure_paths = sorted(procedures_dir.glob(f"*{PROCEDURES_FILE_SUFFIX}"))
    except OSError as error:
        logger.warning("Could not list procedures in %s: %s", procedures_dir, error)
        return manifest
    for procedure_path in procedure_paths:
        try:
            status = procedure_path.stat()
        except OSError as error:
            logger.warning("Could not stat procedure %s: %s", procedure_path.name, error)
            continue
        manifest[procedure_path.name] = [status.st_mtime_ns, status.st_size]
    return manifest


def _load_procedures_snapshot(procedures_dir: Path) -> Optional[dict]:
    """Return the stored snapshot when its version and manifest still match."""
    snapshot_path = _procedures_prompt_snapshot_path()
    if not snapshot_path.exists():
        return None
    try:
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        logger.warning("Could not read procedures snapshot %s: %s", snapshot_path, error)
        return None
    if not isinstance(snapshot, dict):
        logger.warning("Procedures snapshot %s is not an object; rebuilding", snapshot_path)
        return None
    if snapshot.get("version") != PROCEDURES_SNAPSHOT_VERSION:
        return None
    if snapshot.get("manifest") != _build_procedures_manifest(procedures_dir):
        return None
    return snapshot


def _write_procedures_snapshot(
    manifest: dict[str, list[int]], procedure_entries: list[dict],
) -> None:
    payload = {
        "version": PROCEDURES_SNAPSHOT_VERSION,
        "manifest": manifest,
        "procedures": procedure_entries,
    }
    try:
        atomic_json_write(_procedures_prompt_snapshot_path(), payload)
    except Exception as error:
        logger.warning("Could not write procedures prompt snapshot: %s", error)


def read_procedure_metadata(procedure_path: Path) -> Optional[dict]:
    """Read one procedure file into ``{name, description, file}``.

    Returns ``None`` when the file cannot be read or carries no usable name or
    description, and reports why at warning level.
    """
    try:
        raw = procedure_path.read_text(encoding="utf-8")
    except OSError as error:
        logger.warning("Could not read procedure %s: %s", procedure_path.name, error)
        return None
    frontmatter, _ = parse_frontmatter(raw)
    name = str(frontmatter.get("name") or "").strip()
    description = str(frontmatter.get("description") or "").strip()
    if not name or not description:
        logger.warning(
            "Procedure %s has no name or description in frontmatter; leaving it out of the index",
            procedure_path.name,
        )
        return None
    return {"name": name, "description": description, "file": procedure_path.name}


def collect_procedures(procedures_dir: Path) -> list[dict]:
    """Read every procedure file, keeping the first file for each name."""
    procedure_entries: list[dict] = []
    seen_names: set[str] = set()
    try:
        procedure_paths = sorted(procedures_dir.glob(f"*{PROCEDURES_FILE_SUFFIX}"))
    except OSError as error:
        logger.warning("Could not list procedures in %s: %s", procedures_dir, error)
        return procedure_entries
    for procedure_path in procedure_paths:
        entry = read_procedure_metadata(procedure_path)
        if entry is None:
            continue
        if entry["name"] in seen_names:
            logger.warning(
                "Duplicate procedure name %s in %s; keeping the first file",
                entry["name"], procedure_path.name,
            )
            continue
        seen_names.add(entry["name"])
        procedure_entries.append(entry)
    return sorted(procedure_entries, key=lambda entry: entry["name"])


def _render_procedures_section(procedure_entries: list[dict]) -> str:
    index_lines = [
        f"{entry['name']}: {entry['description']}" for entry in procedure_entries
    ]
    return (
        f"{PROCEDURES_SECTION_HEADER}\n"
        "Before replying, scan the procedures below. If a procedure matches the task you are "
        "about to do, you MUST load it with view_procedure(name) and follow it. Procedures are "
        "the user's own written instructions for recurring work, so they take priority over "
        "your general defaults and over habits picked up elsewhere. Err on the side of loading: "
        "a procedure that turns out to be unnecessary costs one tool call, while a missed one "
        "means doing the task the wrong way.\n"
        "\n"
        f"{PROCEDURES_OPEN_TAG}\n"
        + "\n".join(index_lines)
        + f"\n{PROCEDURES_CLOSE_TAG}\n"
        "\n"
        "Only proceed without loading a procedure if genuinely none are relevant to the task."
    )


def build_procedures_system_prompt() -> str:
    """Build the cached procedures section for the system prompt.

    Two layers, as the skills index has: an in-process cache keyed by the
    procedures directory, and a disk snapshot validated by the file manifest.
    A missing directory, or a directory with no usable procedure, renders
    nothing.
    """
    procedures_dir = get_procedures_dir()
    if not procedures_dir.exists():
        return ""

    cache_key = (str(procedures_dir),)
    with _PROCEDURES_PROMPT_CACHE_LOCK:
        cached = _PROCEDURES_PROMPT_CACHE.get(cache_key)
        if cached is not None:
            _PROCEDURES_PROMPT_CACHE.move_to_end(cache_key)
            return cached

    snapshot = _load_procedures_snapshot(procedures_dir)
    if snapshot is not None:
        procedure_entries = snapshot.get("procedures") or []
    else:
        procedure_entries = collect_procedures(procedures_dir)
        _write_procedures_snapshot(
            _build_procedures_manifest(procedures_dir), procedure_entries,
        )

    if not procedure_entries:
        return ""

    result = _render_procedures_section(procedure_entries)

    with _PROCEDURES_PROMPT_CACHE_LOCK:
        _PROCEDURES_PROMPT_CACHE[cache_key] = result
        _PROCEDURES_PROMPT_CACHE.move_to_end(cache_key)
        while len(_PROCEDURES_PROMPT_CACHE) > _PROCEDURES_PROMPT_CACHE_MAX:
            _PROCEDURES_PROMPT_CACHE.popitem(last=False)

    return result
