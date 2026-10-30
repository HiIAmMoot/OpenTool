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


def test_stage_rows_everything_on_the_gpu():
    # (the GPU line above the rows already names the card)
    assert stage_rows('auto', [RTX], None) == [('Page bake', 'GPU', ''), ('Vertex pass', 'GPU', ''), ('Denoise', 'GPU', '')]


def test_stage_rows_amd_denoises_on_the_gpu_too():
    assert [state for _, state, _ in stage_rows('amd', [RADEON], None)] == ['GPU', 'GPU', 'GPU']


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
