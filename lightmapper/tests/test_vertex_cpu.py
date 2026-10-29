import json
import os
import subprocess
import sys

from lightmapper import entry, frozen


def run_probe(out):
    done = subprocess.run(
        frozen.child_command(frozen.CPU_VERTEX_FLAG, '--probe', str(out)),
        capture_output=True, timeout=300)
    assert done.returncode == 0, done.stderr[-2000:]
    return json.loads(out.read_text())


def test_cpu_vertex_flag_dispatches_the_child(tmp_path, monkeypatch):
    monkeypatch.delattr(sys, 'frozen', raising=False)
    monkeypatch.setenv('NUMBA_CACHE_DIR', str(tmp_path / 'cache'))
    probe = run_probe(tmp_path / 'probe.json')
    assert probe['threading_layer'] in ('workqueue', 'omp', 'tbb')
    assert probe['threads'] >= 1
    assert probe['cache_dir'] == str(tmp_path / 'cache')
    assert probe['seconds'] > 0


def test_second_probe_loads_from_the_cache(tmp_path, monkeypatch):
    monkeypatch.delattr(sys, 'frozen', raising=False)
    monkeypatch.setenv('NUMBA_CACHE_DIR', str(tmp_path / 'cache'))
    first = run_probe(tmp_path / 'first.json')
    second = run_probe(tmp_path / 'second.json')
    assert first['cache_misses'] >= 1
    assert second['cache_hits'] >= 1
    assert second['cache_misses'] == 0


def test_cache_dir_defaults_under_local_app_data(tmp_path, monkeypatch):
    monkeypatch.setenv('LOCALAPPDATA', str(tmp_path))
    assert frozen.numba_cache_dir() == tmp_path / 'OpenTool' / 'numba-cache'


def test_cache_dir_falls_back_to_home(tmp_path, monkeypatch):
    monkeypatch.setenv('LOCALAPPDATA', '')
    monkeypatch.setattr(frozen.Path, 'home', lambda: tmp_path)
    assert frozen.numba_cache_dir() == tmp_path / 'OpenTool' / 'numba-cache'


def test_a_cache_dir_the_user_set_is_kept(tmp_path, monkeypatch):
    from lightmapper import vertex_cpu
    cache = str(tmp_path / 'user-cache')
    monkeypatch.setenv('NUMBA_CACHE_DIR', cache)
    seen = {}

    def child_main(argv):
        seen.update(cache=os.environ['NUMBA_CACHE_DIR'], argv=argv)
        return 17

    monkeypatch.setattr(vertex_cpu, 'child_main', child_main)
    assert entry.cli_entry([frozen.CPU_VERTEX_FLAG, '--probe', 'out.json']) == 17
    assert seen == {'cache': cache, 'argv': ['--probe', 'out.json']}


def test_entry_sets_cache_before_importing_vertex_cpu(tmp_path, monkeypatch):
    monkeypatch.delenv('NUMBA_CACHE_DIR', raising=False)
    monkeypatch.setenv('LOCALAPPDATA', str(tmp_path))
    code = (
        'from lightmapper import entry; '
        'entry.cli_entry(["--cpu-vertex-pass", "--probe", ' + repr(str(tmp_path / 'probe.json')) + '])')
    done = subprocess.run([sys.executable, '-c', code], capture_output=True, timeout=300)
    assert done.returncode == 0, done.stderr[-2000:]
    probe = json.loads((tmp_path / 'probe.json').read_text())
    assert probe['cache_dir'] == str(tmp_path / 'OpenTool' / 'numba-cache')
    assert list((tmp_path / 'OpenTool' / 'numba-cache').rglob('*.nbc'))


def test_spec_collects_vertex_cpu_as_source():
    text = (frozen.bundle_dir() / 'packaging' / 'opentool.spec').read_text()
    assert "module_collection_mode={'lightmapper.vertex_cpu': 'py'}" in text
