import io
import os
import struct
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
import pytest

from lightmapper import denoise_worker, entry, gpu, vertex_gpu

REPO = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('operation', ['setup', 'write', 'commit', 'execute', 'wait', 'read'])
def test_native_error_never_sends_a_stale_page(monkeypatch, capsys, operation):
    from lightmapper.tests import fake_oidn
    monkeypatch.setattr(fake_oidn, 'failure', operation)
    monkeypatch.setitem(sys.modules, 'pyoidn', fake_oidn)
    src, dst = io.BytesIO(), io.BytesIO()
    for value in (.25, .75):
        payload = io.BytesIO()
        np.save(payload, np.full((4, 4, 3), value, np.float32))
        data = payload.getvalue()
        src.write(struct.pack('<Q', len(data)) + data)
    src.seek(0)
    assert denoise_worker.serve('CUDA', src, dst) != 0
    assert f'native {operation} failure' in capsys.readouterr().err
    dst.seek(0)
    size = struct.unpack('<Q', dst.read(8))[0]
    ready = np.load(io.BytesIO(dst.read(size)))
    assert ready.tolist() == ([0] if operation == 'setup' else [1])
    if operation != 'setup':
        size = struct.unpack('<Q', dst.read(8))[0]
        np.testing.assert_array_equal(np.load(io.BytesIO(dst.read(size))), np.full((4, 4, 3), .25, np.float32))
    assert dst.read() == b''


def test_source_worker_runs_lightmap_py(monkeypatch):
    monkeypatch.delattr(sys, 'frozen', raising=False)
    command = denoise_worker.worker_command()
    assert command[0] == sys.executable and command[1].endswith('lightmap.py') and command[2] == '--denoise-worker'


def test_frozen_worker_is_the_console_exe(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, 'frozen', True, raising=False)
    monkeypatch.setattr(sys, 'executable', str(tmp_path / 'OpenTool.exe'))
    assert denoise_worker.worker_command() == [str(tmp_path / 'OpenTool-cli.exe'), '--denoise-worker']


def test_frozen_runtime_comes_from_the_bundle(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, 'frozen', True, raising=False)
    monkeypatch.setattr(sys, '_MEIPASS', str(tmp_path), raising=False)
    assert denoise_worker.runtime_dll_dir() == tmp_path


@pytest.mark.skipif(os.name != 'nt', reason='Windows runtime')
def test_source_runtime_comes_from_system32(monkeypatch):
    monkeypatch.delattr(sys, 'frozen', raising=False)
    assert denoise_worker.runtime_dll_dir() == Path(os.environ['SystemRoot']) / 'System32'


def test_worker_round_trip_through_the_real_command():
    # real OIDN only ever runs in the worker process: creating an OIDN device inside pytest is
    # exactly what the worker exists to avoid (pages.py:23-26)
    page = np.full((40, 40, 3), .5, np.float32)
    src = io.BytesIO()
    np.save(src, page)
    done = subprocess.run(denoise_worker.worker_command(), input=src.getvalue(), capture_output=True, timeout=120)
    assert done.returncode == 0, done.stderr.decode(errors='replace')
    out = np.load(io.BytesIO(done.stdout))
    assert out.shape == page.shape
    assert abs(out.mean() - .5) < .01
    assert np.abs(out - .5).max() < .05


def test_main_pipes_npy_through_pyoidn(monkeypatch):
    fake = types.SimpleNamespace(denoise=lambda img, filter_type: img * 2 if filter_type == 'RTLightmap' else None)
    monkeypatch.setitem(sys.modules, 'pyoidn', fake)
    monkeypatch.setattr(denoise_worker, 'runtime_dll_dir', lambda: Path('nowhere'))
    page = np.full((4, 4, 3), .5, np.float32)
    src, dst = io.BytesIO(), io.BytesIO()
    np.save(src, page)
    src.seek(0)
    assert denoise_worker.main(src, dst) == 0
    dst.seek(0)
    np.testing.assert_array_equal(np.load(dst), page * 2)


def test_entry_routes_the_worker_flag(monkeypatch):
    monkeypatch.setattr(denoise_worker, 'main', lambda: 7)
    assert entry.cli_entry(['--denoise-worker']) == 7


def test_entry_routes_commands_to_cli(monkeypatch):
    from lightmapper import cli
    monkeypatch.setattr(cli, 'main', lambda argv: len(argv))
    assert entry.cli_entry(['validate', 'x']) == 2


def test_entry_does_not_load_numpy():
    code = 'import sys; import lightmapper.entry; print("numpy" in sys.modules)'
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, cwd=REPO).stdout
    assert out.strip() == 'False'


def test_frozen_worker_command_carries_the_kind(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, 'frozen', True, raising=False)
    monkeypatch.setattr(sys, 'executable', str(tmp_path / 'OpenTool.exe'))
    assert denoise_worker.worker_command('CUDA') == [str(tmp_path / 'OpenTool-cli.exe'), '--denoise-worker', 'CUDA']


def test_entry_routes_the_worker_kind_probe_and_vertex_flags(monkeypatch):
    seen = []
    monkeypatch.setattr(denoise_worker, 'main', lambda kind=None: seen.append(kind) or 4)
    monkeypatch.setattr(gpu, 'probe_main', lambda: 5, raising=False)
    monkeypatch.setattr(vertex_gpu, 'child_main', lambda argv: len(argv), raising=False)
    assert entry.cli_entry(['--denoise-worker', 'CUDA']) == 4 and seen == ['CUDA']
    assert entry.cli_entry(['--gpu-probe']) == 5
    assert entry.cli_entry(['--vertex-pass', 'a', 'b']) == 2
