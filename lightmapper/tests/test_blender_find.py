import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.skipif(os.name != 'nt', reason='Blender discovery uses Windows paths and registry')


@pytest.fixture
def modules():
    from lightmapper import blender_find, cli, settings
    return blender_find, cli, settings


@pytest.fixture
def fake_cmd(tmp_path):
    def make(version):
        path = tmp_path / f'Blender {version}' / 'blender.cmd'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f'@echo Blender {version}\n')
        return path
    return make


@pytest.fixture
def fake_cmd_text(tmp_path):
    def make(text):
        path = tmp_path / 'garbage.cmd'
        path.write_text(f'@echo {text}\n')
        return path
    return make


def test_version_parsed(fake_cmd, modules):
    assert modules[0].blender_version(fake_cmd('5.2.2')) == (5, 2)


def test_version_of_missing_file_is_none(tmp_path, modules):
    assert modules[0].blender_version(tmp_path / 'nope.exe') is None


def test_version_of_garbage_is_none(fake_cmd_text, modules):
    assert modules[0].blender_version(fake_cmd_text('hello')) is None


def test_version_with_undecodable_output_is_parsed(tmp_path, monkeypatch, modules):
    finder = modules[0]
    path = tmp_path / 'odd-output.cmd'
    script = tmp_path / 'emit.py'
    script.write_text("import sys\nsys.stdout.buffer.write(b'\\xff\\nBlender 5.2.2\\n')\n")
    path.write_text(f'@echo off\r\n@"{sys.executable}" "{script}"\r\n')
    run = finder.subprocess.run
    def utf8_run(*args, **kwargs):
        errors = kwargs.pop('errors', 'strict')
        kwargs.pop('encoding', None)
        kwargs.pop('text', None)
        result = run(*args, text=False, **kwargs)
        assert b'\xff' in result.stdout
        result.stdout = result.stdout.decode('utf-8', errors=errors)
        return result
    monkeypatch.setattr(finder.subprocess, 'run', utf8_run)
    assert finder.blender_version(path) == (5, 2)


def test_registry_continues_after_hive_without_uninstall_key(monkeypatch, tmp_path, modules):
    finder = modules[0]
    install = tmp_path / 'Blender 5.2'
    install.mkdir()
    path = install / 'blender.exe'
    path.touch()

    class Key:
        def __init__(self, kind):
            self.kind = kind
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False

    winreg = SimpleNamespace(HKEY_LOCAL_MACHINE=1, HKEY_CURRENT_USER=2)
    def open_key(key, subkey):
        if key == winreg.HKEY_LOCAL_MACHINE and subkey == finder._UNINSTALL:
            raise FileNotFoundError(subkey)
        if isinstance(key, Key) and key.kind == 'hive' and subkey == 'Blender 5.2':
            return Key('install')
        if key == winreg.HKEY_CURRENT_USER and subkey == finder._UNINSTALL:
            return Key('hive')
        if key == winreg.HKEY_LOCAL_MACHINE and 'WOW6432Node' in subkey:
            return Key('empty')
        raise FileNotFoundError(str(subkey))
    def query_info(key):
        return (1 if key.kind == 'hive' else 0, 0, 0)
    def enum_key(key, index):
        return 'Blender 5.2'
    def query_value(key, name):
        return {'DisplayName': 'Blender 5.2', 'InstallLocation': str(install)}[name], 1
    winreg.OpenKey = open_key
    winreg.QueryInfoKey = query_info
    winreg.EnumKey = enum_key
    winreg.QueryValueEx = query_value
    monkeypatch.setitem(sys.modules, 'winreg', winreg)
    assert path in finder.registry_installs()


def test_newest_install_wins(monkeypatch, tmp_path, fake_cmd, modules):
    finder = modules[0]
    monkeypatch.delenv('LIGHTMAPPER_BLENDER', raising=False)
    old, newest = fake_cmd('5.2.2'), fake_cmd('6.0.0')
    monkeypatch.setattr(finder, 'registry_installs', lambda: [old])
    monkeypatch.setattr(finder, 'program_files_installs', lambda: [newest])
    assert finder.find_blender().path == newest


def test_old_explicit_blender_falls_through_to_newest(monkeypatch, fake_cmd, modules):
    finder = modules[0]
    old, current = fake_cmd('4.1.1'), fake_cmd('5.2.2')
    monkeypatch.setenv('LIGHTMAPPER_BLENDER', str(old))
    monkeypatch.setattr(finder, 'registry_installs', lambda: [current])
    monkeypatch.setattr(finder, 'program_files_installs', lambda: [])
    assert finder.find_blender().path == current


def test_explicit_blender_wins_over_newer_discovered(monkeypatch, fake_cmd, modules):
    finder = modules[0]
    explicit, newer = fake_cmd('5.2.2'), fake_cmd('6.0.0')
    monkeypatch.setenv('LIGHTMAPPER_BLENDER', str(explicit))
    monkeypatch.setattr(finder, 'program_files_installs', lambda: [newer])
    assert finder.find_blender().path == explicit


def test_remembered_used_when_env_unset(monkeypatch, fake_cmd, modules):
    finder = modules[0]
    remembered = fake_cmd('5.2.2')
    monkeypatch.delenv('LIGHTMAPPER_BLENDER', raising=False)
    assert finder.find_blender(str(remembered)).path == remembered


def test_missing_remembered_path_is_skipped(monkeypatch, fake_cmd, modules):
    finder = modules[0]
    found = fake_cmd('5.2.2')
    monkeypatch.delenv('LIGHTMAPPER_BLENDER', raising=False)
    monkeypatch.setattr(finder, 'registry_installs', lambda: [found])
    monkeypatch.setattr(finder, 'program_files_installs', lambda: [])
    assert finder.find_blender(r'C:\gone\blender.exe').path == found


def test_only_old_blender_says_install_newer(monkeypatch, fake_cmd, modules):
    finder = modules[0]
    old = fake_cmd('4.1.1')
    monkeypatch.delenv('LIGHTMAPPER_BLENDER', raising=False)
    monkeypatch.setattr(finder, 'registry_installs', lambda: [old])
    monkeypatch.setattr(finder, 'program_files_installs', lambda: [])
    with pytest.raises(finder.BlenderNotFound, match=r'Found Blender 4\.1, but OpenTool needs Blender 5\.2 or newer'):
        finder.find_blender()


def test_nothing_found_says_not_found(monkeypatch, modules):
    finder = modules[0]
    monkeypatch.delenv('LIGHTMAPPER_BLENDER', raising=False)
    monkeypatch.setattr(finder, 'registry_installs', lambda: [])
    monkeypatch.setattr(finder, 'program_files_installs', lambda: [])
    with pytest.raises(finder.BlenderNotFound, match='Blender 5.2 or newer was not found'):
        finder.find_blender()


def test_same_install_probed_once(monkeypatch, fake_cmd, modules):
    finder = modules[0]
    path = fake_cmd('5.2.2')
    calls = []
    original = finder.blender_version
    def probe(candidate, *args, **kwargs):
        calls.append(candidate)
        return original(candidate, *args, **kwargs)
    monkeypatch.delenv('LIGHTMAPPER_BLENDER', raising=False)
    monkeypatch.setattr(finder, 'registry_installs', lambda: [path])
    monkeypatch.setattr(finder, 'program_files_installs', lambda: [path])
    monkeypatch.setattr(finder, 'blender_version', probe)
    finder.find_blender()
    assert len(calls) == 1


def test_version_probe_has_no_window(monkeypatch, fake_cmd, modules):
    finder = modules[0]
    calls = []
    monkeypatch.setattr(finder.subprocess, 'run', lambda *a, **kw: calls.append(kw) or type('R', (), {'stdout': 'Blender 5.2.2', 'returncode': 0})())
    finder.blender_version(fake_cmd('5.2.2'))
    assert calls[0]['creationflags'] == finder.NO_WINDOW


def test_registry_lists_this_machines_blender(modules):
    if not Path(r'C:\Program Files\Blender Foundation\Blender 5.2').exists():
        pytest.skip('Blender 5.2 is not installed')
    assert any(p.parent.name == 'Blender 5.2' for p in modules[0].registry_installs())


def test_cli_falls_back_to_discovery_when_default_is_missing(monkeypatch, tmp_path, modules):
    finder, cli, settings_module = modules
    from lightmapper.settings import Settings
    monkeypatch.delenv('LIGHTMAPPER_BLENDER', raising=False)
    monkeypatch.setattr(settings_module, 'DEFAULT_BLENDER', str(tmp_path / 'gone' / 'blender.exe'))
    monkeypatch.setattr(cli, 'find_blender', lambda remembered=None: finder.Blender(Path('C:/B6/blender.exe'), (6, 0)))
    s = Settings(blender=settings_module.DEFAULT_BLENDER)
    assert cli._blender_fallback(s).blender == str(Path('C:/B6/blender.exe'))


def test_cli_keeps_an_explicit_blender(monkeypatch, tmp_path, modules):
    finder, cli, _ = modules
    from lightmapper.settings import Settings
    monkeypatch.delenv('LIGHTMAPPER_BLENDER', raising=False)
    monkeypatch.setattr(cli, 'find_blender', lambda remembered=None: pytest.fail('must not search'))
    s = Settings(blender=str(tmp_path / 'mine.exe'))
    assert cli._blender_fallback(s) is s


def test_cli_keeps_the_env_blender(monkeypatch, modules):
    _, cli, _ = modules
    from lightmapper.settings import Settings
    monkeypatch.setenv('LIGHTMAPPER_BLENDER', r'C:\env.exe')
    monkeypatch.setattr(cli, 'find_blender', lambda remembered=None: pytest.fail('must not search'))
    s = Settings()
    assert cli._blender_fallback(s) is s
