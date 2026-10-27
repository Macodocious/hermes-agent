"""The ``hermes update`` ACP shim hands Windows launchers to the repair.

Launcher layout itself (canonical managed dir, legacy PATH-gated dir, exe vs
``.cmd`` form) is covered by ``test_ensure_windows_bin_launchers``. What is
still this file's business is the wiring: on Windows, ``update_cmd``'s
``_ensure_acp_launcher`` must delegate to that repair instead of writing the
POSIX shell shim.
"""

from __future__ import annotations

from types import SimpleNamespace

from hermes_cli import _install_repair as install_repair
from hermes_cli import update_cmd


def _make_managed(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    root = home / "hermes-agent"
    scripts = root / "venv" / "Scripts"
    scripts.mkdir(parents=True)
    (scripts / "hermes.exe").write_bytes(b"MZ-hermes")
    (scripts / "hermes-acp.exe").write_bytes(b"MZ-hermes-acp")
    (root / "venv" / "pyvenv.cfg").write_text(
        "home = X\nversion_info = 3.11.15\n", encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    fake_main = SimpleNamespace(
        sys=SimpleNamespace(platform="win32"),
        PROJECT_ROOT=root,
    )
    monkeypatch.setattr(update_cmd, "_m", lambda: fake_main)
    monkeypatch.setattr(install_repair, "_is_windows", lambda: True)
    return home, root


def test_update_restores_windows_launchers_into_managed_bin(tmp_path, monkeypatch):
    home, root = _make_managed(tmp_path, monkeypatch)

    update_cmd._ensure_acp_launcher()

    assert (home / "bin" / "hermes.exe").read_bytes() == b"MZ-hermes"
    assert (home / "bin" / "hermes-acp.exe").read_bytes() == b"MZ-hermes-acp"
