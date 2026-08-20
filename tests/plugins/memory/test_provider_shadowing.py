"""Regression tests for memory-provider loading under a polluted sys.path.

A plugin directory that shares a top-level name with a site-packages
package (e.g. ``mnemosyne``) can shadow the real package when a plugin
inserts the plugins dir at ``sys.path[0]``.  The provider's own imports
(``from mnemosyne.batch_tool import ...``) then resolve to the shadow
and fail, silently disabling memory for the whole session.

The loader pre-registers shadowable packages in ``sys.modules`` (import
machinery consults ``sys.modules`` before any path lookup), so provider
imports resolve to site-packages regardless of path pollution.
"""

import importlib
import importlib.util
import sys
import types
from pathlib import Path

import pytest

from plugins.memory import (
    _SHADOWABLE_TOP_LEVEL_NAMES,
    _register_shadowable_packages,
    _scrubbed_sys_path,
)


@pytest.fixture
def fake_plugins_dir(tmp_path: Path) -> Path:
    """A fake user plugins dir containing a shadowing ``mnemosyne`` package."""
    shadow = tmp_path / "mnemosyne"
    shadow.mkdir()
    (shadow / "__init__.py").write_text(
        "raise ImportError('shadow package must never be imported')\n"
    )
    return tmp_path


def _make_fake_site_package(tmp_path: Path, name: str) -> Path:
    """Create a real-looking site-packages package and import it."""
    pkg = tmp_path / name
    pkg.mkdir()
    (pkg / "__init__.py").write_text(
        f"VALUE = 'real-{name}'\n"
    )
    (pkg / "batch_tool.py").write_text(
        "BATCH = 'real-batch-tool'\n"
    )
    spec = importlib.util.spec_from_file_location(
        name, str(pkg / "__init__.py"),
        submodule_search_locations=[str(pkg)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return pkg


def test_scrubbed_sys_path_removes_plugins_dir(monkeypatch, tmp_path: Path):
    """The context manager hides the plugins dir from sys.path and restores it."""
    plugins_dir = tmp_path / "plugins"
    plugins_dir.mkdir()
    monkeypatch.setattr("plugins.memory._get_user_plugins_dir", lambda: plugins_dir)

    sys.path.insert(0, str(plugins_dir))
    sys.path.insert(0, str(plugins_dir / "github-issues"))
    try:
        with _scrubbed_sys_path():
            assert str(plugins_dir) not in sys.path
            assert str(plugins_dir / "github-issues") not in sys.path
        assert str(plugins_dir) in sys.path
        assert str(plugins_dir / "github-issues") in sys.path
    finally:
        sys.path.remove(str(plugins_dir))
        sys.path.remove(str(plugins_dir / "github-issues"))


def test_register_shadowable_packages_skips_plugins_dir_resolution(
    monkeypatch, tmp_path: Path, fake_plugins_dir: Path
):
    """A package that only resolves inside the plugins dir is never registered."""
    monkeypatch.setattr("plugins.memory._get_user_plugins_dir", lambda: fake_plugins_dir)
    monkeypatch.setattr(
        "plugins.memory._SHADOWABLE_TOP_LEVEL_NAMES", ("shadowed",)
    )

    # Only the shadow exists — find_spec under the scrubbed path finds nothing.
    sys.path.insert(0, str(fake_plugins_dir))
    try:
        _register_shadowable_packages()
        assert "shadowed" not in sys.modules
    finally:
        sys.path.remove(str(fake_plugins_dir))


def test_register_shadowable_packages_registers_real_package(
    monkeypatch, tmp_path: Path, fake_plugins_dir: Path
):
    """A real site-packages package is pre-registered even with a shadow present."""
    real_pkg = _make_fake_site_package(tmp_path, "shadowed")
    monkeypatch.setattr("plugins.memory._get_user_plugins_dir", lambda: fake_plugins_dir)
    monkeypatch.setattr(
        "plugins.memory._SHADOWABLE_TOP_LEVEL_NAMES", ("shadowed",)
    )

    sys.path.insert(0, str(fake_plugins_dir))
    try:
        _register_shadowable_packages()
        module = sys.modules.get("shadowed")
        assert module is not None
        assert module.__file__ == str(real_pkg / "__init__.py")
        assert module.VALUE == "real-shadowed"
    finally:
        sys.path.remove(str(fake_plugins_dir))
        sys.modules.pop("shadowed", None)


def test_provider_import_resolves_to_real_package_under_pollution(
    monkeypatch, tmp_path: Path, fake_plugins_dir: Path
):
    """A provider importing the shadowed name gets the real package.

    This is the end-to-end failure mode: the plugins dir is at
    sys.path[0] (as the github plugins used to do), yet the provider's
    ``from shadowed.batch_tool import ...`` must resolve to the real
    site-packages package because the loader pre-registered it.
    """
    real_pkg = _make_fake_site_package(tmp_path, "shadowed")
    monkeypatch.setattr("plugins.memory._get_user_plugins_dir", lambda: fake_plugins_dir)
    monkeypatch.setattr(
        "plugins.memory._SHADOWABLE_TOP_LEVEL_NAMES", ("shadowed",)
    )

    sys.path.insert(0, str(fake_plugins_dir))
    try:
        _register_shadowable_packages()
        from shadowed.batch_tool import BATCH  # noqa: F401

        assert BATCH == "real-batch-tool"
        assert sys.modules["shadowed"].__file__ == str(real_pkg / "__init__.py")
    finally:
        sys.path.remove(str(fake_plugins_dir))
        sys.modules.pop("shadowed", None)
        sys.modules.pop("shadowed.batch_tool", None)
