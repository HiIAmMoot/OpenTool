import os
import subprocess
import sys

import pytest

from lightmapper import blender_job, frozen, pages, resources, vertex_gpu
from lightmapper.blender_job import write_job, run_blender
from lightmapper.settings import Settings
from test_blender_job import fake, make_scene, make_suns, AMBIENT


def test_shader_is_a_required_file_where_the_pass_looks_for_it():
    assert 'lightmapper/shaders/vertex_pass.slang' in resources.REQUIRED and resources.missing() == []
    assert vertex_gpu.SHADERS == frozen.bundle_dir() / 'lightmapper' / 'shaders'


def test_run_blender_starts_blender_without_a_window(tmp_path, monkeypatch, fake):
    seen = {}
    real = blender_job.subprocess.run

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(blender_job.subprocess, 'run', spy)
    s = Settings(blender=fake)
    work = tmp_path / 'work'
    write_job(work, make_scene(), [(16, 8)], make_suns(), AMBIENT, s)
    run_blender(work, s)
    assert seen['creationflags'] == frozen.NO_WINDOW


def test_denoise_starts_without_a_window(monkeypatch):
    from test_pages import fake_run
    import io
    import numpy as np
    data = io.BytesIO()
    np.save(data, np.zeros((34, 34, 3), np.float32))
    run = fake_run(monkeypatch, stdout=lambda kw: data.getvalue())
    pages._denoise(np.ones((2, 2, 3), np.float32), 3)
    assert run.kwargs['creationflags'] == frozen.NO_WINDOW


@pytest.mark.skipif(os.name != 'nt', reason='Windows flag')
def test_no_window_is_the_windows_flag():
    assert frozen.NO_WINDOW == subprocess.CREATE_NO_WINDOW


def test_bundle_dir_follows_meipass_when_frozen(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, 'frozen', True, raising=False)
    monkeypatch.setattr(sys, '_MEIPASS', str(tmp_path), raising=False)
    assert frozen.is_frozen() and frozen.bundle_dir() == tmp_path


def test_bundle_dir_is_the_repo_in_source_mode():
    assert (frozen.bundle_dir() / 'lightmapper' / 'pipeline.py').is_file()


def test_cli_executable_sits_beside_the_running_exe(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, 'executable', str(tmp_path / 'OpenTool.exe'))
    assert frozen.cli_executable() == tmp_path / 'OpenTool-cli.exe'


def test_child_command_in_source_mode_runs_lightmap_py(monkeypatch):
    monkeypatch.delattr(sys, 'frozen', raising=False)
    assert frozen.child_command('--gpu-probe') == [sys.executable, str(frozen.bundle_dir() / 'lightmap.py'), '--gpu-probe']


def test_child_command_when_frozen_is_the_console_exe(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, 'frozen', True, raising=False)
    monkeypatch.setattr(sys, 'executable', str(tmp_path / 'OpenTool.exe'))
    assert frozen.child_command('--vertex-pass', 'w', '{}') == [str(tmp_path / 'OpenTool-cli.exe'), '--vertex-pass', 'w', '{}']


def test_child_command_when_frozen_runs_the_cpu_vertex_flag(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, 'frozen', True, raising=False)
    monkeypatch.setattr(sys, 'executable', str(tmp_path / 'OpenTool.exe'))
    assert frozen.child_command(frozen.CPU_VERTEX_FLAG, 'w') == [str(tmp_path / 'OpenTool-cli.exe'), '--cpu-vertex-pass', 'w']


def test_every_layout_is_a_required_resource():
    from lightmapper.tagio.layout import _GROUP_FILES
    assert {f'lightmapper/tagio/layouts/{name}.xml' for name in _GROUP_FILES.values()} <= set(resources.REQUIRED)
