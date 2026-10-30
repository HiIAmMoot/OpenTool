import pytest

from lightmapper.gpu import Adapter
from opentool.devices import device_hint, gpu_text, result_line, stage_rows, stage_rows_used

RTX = Adapter('NVIDIA GeForce RTX 4060 Laptop GPU', 0x10DE, (1,), 'vulkan', True)
RADEON = Adapter('AMD Radeon(TM) Graphics', 0x1002, (2,), 'vulkan', True)


def test_gpu_text():
    assert gpu_text(None, None) == 'Looking for GPUs…'
    assert gpu_text([RTX, RADEON], None) == 'GPU: NVIDIA GeForce RTX 4060 Laptop GPU, AMD Radeon(TM) Graphics'
    assert gpu_text([], 'slangpy: missing') == 'No GPU with ray tracing found (slangpy: missing)'
    assert gpu_text([], None) == 'No GPU with ray tracing found'


def test_gpu_text_skips_adapters_without_ray_queries():
    plain = Adapter('Basic', 0x1234, (3,), 'vulkan', False)
    assert gpu_text([plain, RTX], None) == 'GPU: NVIDIA GeForce RTX 4060 Laptop GPU'
    assert gpu_text([plain], None) == 'No GPU with ray tracing found'


@pytest.mark.parametrize('setting, adapters, hint', [
    ('auto', [RTX, RADEON], 'Bakes on NVIDIA GeForce RTX 4060 Laptop GPU.'),
    ('amd', [RTX, RADEON], 'Bakes on AMD Radeon(TM) Graphics.'),
    ('nvidia', [RADEON], 'No NVIDIA GPU with ray tracing was found, so the bake would stop. Pick Auto or CPU only.'),
    ('amd', [RTX], 'No AMD GPU with ray tracing was found, so the bake would stop. Pick Auto or CPU only.'),
    ('auto', [], "The vertex pass runs on the CPU; pages still try Cycles' GPU backends."),
    ('auto', [Adapter('Older NVIDIA', 0x10DE, (3,), 'vulkan', False)],
     "The vertex pass runs on the CPU; pages still try Cycles' GPU backends."),
    ('cpu', [RTX], 'Bakes on the CPU, with fewer rays for Draft, Normal and High.'),
    ('auto', None, '')])
def test_device_hint(setting, adapters, hint):
    assert device_hint(setting, adapters, None) == hint


def test_result_line_on_the_gpu():
    assert result_line({'page_bake': 'OPTIX', 'vertex_pass': 'NVIDIA GeForce RTX 4060 Laptop GPU (vulkan)', 'denoise': 'CUDA'},
                       {'samples': 1024, 'texel_samples': 1026, 'vertex_rays': 4096}, []) == (
        'Pages on the GPU (OPTIX), vertex pass on NVIDIA GeForce RTX 4060 Laptop GPU (vulkan), denoiser on CUDA. '
        '1024 rays per texel, 4096 per vertex.')


def test_result_line_on_the_cpu_with_warnings():
    assert result_line({'page_bake': 'CPU', 'vertex_pass': 'cpu', 'denoise': 'CPU'},
                       {'samples': 256, 'texel_samples': 252, 'vertex_rays': 1024}, ['a', 'b']) == (
        'Pages on the CPU, vertex pass on the CPU, denoiser on the CPU. 256 rays per texel, 1024 per vertex. '
        '2 warnings, see the log.')


def test_result_line_denoiser_off_no_rays_one_warning():
    assert result_line({'page_bake': 'CPU', 'vertex_pass': 'cpu', 'denoise': 'off'}, None, ['a']) == (
        'Pages on the CPU, vertex pass on the CPU, denoiser off. 1 warning, see the log.')


OLD_NVIDIA = Adapter('NVIDIA GeForce GTX 860M', 0x10DE, (4,), 'vulkan', False)


def test_stage_rows_while_looking():
    assert stage_rows('auto', None, None) == [
        ('Page bake', None, 'Looking for GPUs…'), ('Vertex pass', None, 'Looking for GPUs…'),
        ('Denoise', None, 'Looking for GPUs…')]


@pytest.mark.parametrize('adapter, kind', [(RTX, 'CUDA'), (RADEON, 'HIP')])
def test_stage_rows_everything_on_the_gpu(adapter, kind):
    # (the GPU line above the rows already names the card)
    assert stage_rows('auto', [adapter], None, {kind: True}) == [
        ('Page bake', 'GPU', ''), ('Vertex pass', 'GPU', ''), ('Denoise', 'GPU', '')]


@pytest.mark.parametrize('adapter, kind', [(RTX, 'CUDA'), (RADEON, 'HIP')])
def test_stage_rows_refused_gpu_denoiser_predicts_cpu(adapter, kind):
    rows = stage_rows('auto', [adapter], None, {kind: False})
    assert rows == [('Page bake', 'GPU', ''), ('Vertex pass', 'GPU', ''),
                    ('Denoise', 'CPU', 'the GPU denoiser could not start')]


def test_stage_rows_waits_for_denoiser_probe():
    assert stage_rows('auto', [RTX], None)[2] == ('Denoise', None, 'Checking the GPU denoiser…')


def test_stage_rows_other_vendor_denoises_on_cpu():
    arc = Adapter('Intel Arc', 0x8086, (3,), 'vulkan', True)
    assert [state for _, state, _ in stage_rows('auto', [arc], None)] == ['GPU', 'GPU', 'CPU']


def test_stage_rows_used_hip():
    assert stage_rows_used({'page_bake': 'HIP', 'vertex_pass': 'Radeon (vulkan)', 'denoise': 'HIP'}) == [
        ('Page bake', 'GPU', 'HIP'), ('Vertex pass', 'GPU', ''), ('Denoise', 'GPU', 'HIP')]


@pytest.mark.parametrize('failure', ['timeout', 'crash', 'malformed', 'refused', 'launch'])
def test_denoiser_probe_failures_are_unavailable(monkeypatch, failure):
    import subprocess
    from opentool import devices

    def run(command, **kwargs):
        assert command[-2:] == ['--denoise-worker', 'CUDA'] or command[-2:] == ['--denoise-worker', 'HIP']
        assert kwargs['input'] == b'' and 0 < kwargs['timeout'] <= 5
        if failure == 'timeout':
            raise subprocess.TimeoutExpired(command, kwargs['timeout'])
        if failure == 'launch':
            raise OSError('cannot start')
        return subprocess.CompletedProcess(command, 1 if failure == 'crash' else 0,
                                           b'bad frame' if failure == 'malformed' else ready_frame(failure != 'refused'), b'')

    monkeypatch.setattr(subprocess, 'run', run)
    assert devices.probe_denoiser() == {'CUDA': False, 'HIP': False}


def ready_frame(ready):
    import io
    import struct
    import numpy as np
    out = io.BytesIO()
    np.save(out, np.array([int(ready)], dtype=np.uint8))
    return struct.pack('<Q', len(out.getvalue())) + out.getvalue()


@pytest.mark.parametrize('supported', ['CUDA', 'HIP'])
def test_denoiser_probe_reports_each_backend(monkeypatch, supported):
    import subprocess
    from opentool import devices
    monkeypatch.setattr(subprocess, 'run', lambda command, **kwargs:
                        subprocess.CompletedProcess(command, 0, ready_frame(command[-1] == supported), b''))
    assert devices.probe_denoiser() == {'CUDA': supported == 'CUDA', 'HIP': supported == 'HIP'}


def test_denoiser_probe_uses_frozen_console_child(monkeypatch, tmp_path):
    import subprocess
    import sys
    from lightmapper import frozen
    from opentool import devices
    monkeypatch.setattr(sys, 'frozen', True, raising=False)
    monkeypatch.setattr(sys, 'executable', str(tmp_path / 'OpenTool.exe'))
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        assert kwargs['creationflags'] == frozen.NO_WINDOW
        return subprocess.CompletedProcess(command, 0, ready_frame(True), b'')

    monkeypatch.setattr(subprocess, 'run', run)
    assert devices.probe_denoiser() == {'CUDA': True, 'HIP': True}
    assert commands == [[str(tmp_path / 'OpenTool-cli.exe'), '--denoise-worker', kind]
                        for kind in ('CUDA', 'HIP')]


def test_real_denoiser_probe_child_returns_ready_frames(capsys, monkeypatch):
    import importlib.util
    import subprocess
    import time
    from opentool import devices
    if importlib.util.find_spec('pyoidn') is None:
        pytest.skip('OIDN is not installed')
    began = time.perf_counter()
    real_run = subprocess.run
    answers = {}

    def run(command, **kwargs):
        done = real_run(command, **kwargs)
        assert done.returncode in (0, 2), done.stderr.decode(errors='replace')
        assert done.stdout in (ready_frame(True), ready_frame(False))
        answers[command[-1]] = done.stdout == ready_frame(True)
        return done

    monkeypatch.setattr(subprocess, 'run', run)
    assert devices.probe_denoiser() == answers
    assert set(answers) == {'CUDA', 'HIP'}
    with capsys.disabled():
        print(f' OIDN probe child time: {time.perf_counter() - began:.3f} s')


def test_stage_rows_without_ray_queries_keep_only_the_pages_on_the_gpu():
    rows = stage_rows('auto', [OLD_NVIDIA], None)
    assert [(name, state) for name, state, _ in rows] == [('Page bake', 'GPU'), ('Vertex pass', 'CPU'), ('Denoise', 'CPU')]
    assert 'Cycles' in rows[0][2] and 'ray queries' in rows[1][2]


def test_stage_rows_no_gpu_at_all():
    assert [state for _, state, _ in stage_rows('auto', [], 'slangpy: missing')] == ['GPU', 'CPU', 'CPU']


def test_stage_rows_cpu_only():
    assert [state for _, state, _ in stage_rows('cpu', [RTX], None)] == ['CPU', 'CPU', 'CPU']


def test_stage_rows_when_the_bake_would_stop():
    rows = stage_rows('nvidia', [RADEON], None)
    assert [state for _, state, _ in rows] == [None, None, None]
    assert rows[0][2] == 'No NVIDIA GPU with ray tracing'


def test_stage_rows_used_reads_the_summary():
    assert stage_rows_used({'page_bake': 'OPTIX', 'vertex_pass': 'NVIDIA GeForce RTX 4060 Laptop GPU (vulkan)',
                            'denoise': 'CUDA'}) == [
        ('Page bake', 'GPU', 'OPTIX'), ('Vertex pass', 'GPU', ''),
        ('Denoise', 'GPU', 'CUDA')]
    assert stage_rows_used({'page_bake': 'CPU', 'vertex_pass': 'cpu', 'denoise': 'off'}) == [
        ('Page bake', 'CPU', ''), ('Vertex pass', 'CPU', ''), ('Denoise', 'Off', '')]
