import pytest

from lightmapper.gpu import Adapter
from opentool.devices import device_hint, gpu_text, result_line

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
