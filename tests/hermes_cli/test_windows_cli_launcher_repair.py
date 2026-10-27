"""``hermes update`` must not stage Windows launchers itself.

The repair moved to ``_install_repair.ensure_windows_bin_launchers`` (run at
process start) and ``migrate_windows_bin_path`` (run in this command's tail),
so the ACP-shim helper must return early on Windows instead of writing a
second copy — otherwise a POSIX shell shim lands next to the managed exe
launchers. Layout and form are covered by
``test_ensure_windows_bin_launchers``.
"""

from __future__ import annotations

from types import SimpleNamespace

from hermes_cli import update_cmd


def test_update_acp_shim_is_a_noop_on_windows(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    root = home / "hermes-agent"
    scripts = root / "venv" / "Scripts"
    scripts.mkdir(parents=True)
    (scripts / "hermes.exe").write_bytes(b"MZ-hermes")
    (scripts / "hermes-acp.exe").write_bytes(b"MZ-hermes-acp")
    monkeypatch.setenv("HERMES_HOME", str(home))
    fake_main = SimpleNamespace(
        sys=SimpleNamespace(platform="win32"),
        PROJECT_ROOT=root,
    )
    monkeypatch.setattr(update_cmd, "_m", lambda: fake_main)

    update_cmd._ensure_acp_launcher()

    assert not (home / "bin").exists()
    assert not (root / "bin").exists()
