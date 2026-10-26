import dataclasses
import sys
from types import SimpleNamespace

import pytest

from lightmapper import gpu
from lightmapper.gpu import AMD, NVIDIA, Adapter, DeviceError, choose, probe_choice


NV = Adapter('NVIDIA GeForce RTX 4060 Laptop GPU', NVIDIA, (1,), 'vulkan', True)
RADEON = Adapter('AMD Radeon(TM) 680M', AMD, (2,), 'vulkan', True)


def test_auto_prefers_nvidia_then_amd():
    assert choose('auto', [RADEON, NV]).adapter == NV
    c = choose('auto', [RADEON])
    assert (c.vendor, c.cycles, c.oidn, c.required) == ('amd', ('HIP',), 'HIP', False)
    c = choose('nvidia', [NV])
    assert (c.cycles, c.oidn, c.required) == (('OPTIX', 'CUDA'), 'CUDA', True)


def test_auto_without_ray_queries_keeps_the_cycles_gpu_chain_with_a_warning():
    c = choose('auto', [dataclasses.replace(NV, ray_query=False)])
    assert (c.adapter, c.cycles, c.oidn, c.required) == (None, ('OPTIX', 'CUDA', 'HIP'), 'CPU', False)
    assert c.warnings == ('no GPU with ray queries (NVIDIA GeForce RTX 4060 Laptop GPU); vertex pass on the CPU',)


@pytest.mark.parametrize('adapters', [[NV], [dataclasses.replace(RADEON, ray_query=False)], []])
def test_named_vendor_missing_raises(adapters):
    with pytest.raises(DeviceError, match='amd'):
        choose('amd', adapters)


def test_cpu_and_other_vendor_choices():
    c = choose('cpu', [NV])
    assert (c.vendor, c.adapter, c.cycles, c.oidn, c.required, c.warnings) == (None, None, (), 'CPU', False, ())
    arc = Adapter('Intel Arc', 0x8086, (3,), 'vulkan', True)
    c = choose('auto', [arc])
    assert (c.vendor, c.adapter, c.cycles, c.oidn) == ('other', arc, ('OPTIX', 'CUDA', 'HIP'), 'CPU')
    assert c.warnings


@pytest.mark.parametrize('setting,vendor,cycles,oidn,required', [
    ('auto', None, ('OPTIX', 'CUDA', 'HIP'), 'CPU', False),
    ('NVIDIA', 'nvidia', ('OPTIX', 'CUDA'), 'CUDA', True),
    ('HIP', 'amd', ('HIP',), 'HIP', True),
    ('CPU', None, (), 'CPU', False)])
def test_unprobed_choices(setting, vendor, cycles, oidn, required):
    c = choose(setting, None)
    assert (c.vendor, c.adapter, c.cycles, c.oidn, c.required, c.warnings) == (
        vendor, None, cycles, oidn, required, ())


def test_summary_is_json_ready():
    assert choose('AUTO', [NV]).summary() == {
        'setting': 'auto', 'vendor': 'nvidia', 'adapter': NV.name,
        'backend': 'vulkan', 'cycles': ['OPTIX', 'CUDA'], 'oidn': 'CUDA'}
    assert choose('cpu', []).summary() == {
        'setting': 'cpu', 'vendor': None, 'adapter': None, 'backend': None, 'cycles': [], 'oidn': 'CPU'}
    with pytest.raises(dataclasses.FrozenInstanceError):
        NV.backend = 'd3d12'
    with pytest.raises(dataclasses.FrozenInstanceError):
        choose('auto', [NV]).vendor = 'amd'


def test_probe_choice_cpu_creates_no_device(monkeypatch):
    monkeypatch.setitem(sys.modules, 'slangpy', None)
    monkeypatch.setattr(gpu, 'list_adapters', lambda: pytest.fail('cpu probed the GPUs'))
    # the autouse fixture replaces the module attribute, not this original entry point
    assert probe_choice('cpu').cycles == ()


def test_probe_choice_uses_the_probed_adapters_and_failure_reason(monkeypatch):
    monkeypatch.setattr(gpu, 'list_adapters', lambda: ([RADEON], None))
    assert probe_choice('HIP').adapter == RADEON
    monkeypatch.setattr(gpu, 'list_adapters', lambda: ([], 'probe refused'))
    assert probe_choice('auto').warnings == (
        'no GPU with ray queries (probe refused); vertex pass on the CPU',)


def test_one_adapter_per_gpu_on_the_vendors_first_backend(monkeypatch):
    monkeypatch.setattr(gpu, 'BACKEND_ORDER', {NVIDIA: ('vulkan', 'd3d12'), AMD: ('d3d12', 'vulkan')})
    found = [dataclasses.replace(NV, backend='d3d12'), NV, RADEON,
             dataclasses.replace(RADEON, backend='d3d12'),
             dataclasses.replace(NV, luid=(9,), backend='d3d12', ray_query=False)]
    assert [(a.luid, a.backend, a.ray_query) for a in gpu.dedupe_adapters(found)] == [
        ((1,), 'vulkan', True), ((2,), 'd3d12', True), ((9,), 'd3d12', False)]


def test_ray_queries_beat_backend_order_and_unknown_backends_keep_first_seen(monkeypatch):
    monkeypatch.setattr(gpu, 'BACKEND_ORDER', {NVIDIA: ('vulkan', 'd3d12')})
    found = [dataclasses.replace(NV, ray_query=False), dataclasses.replace(NV, backend='d3d12')]
    assert gpu.dedupe_adapters(found) == [found[1]]
    found = [dataclasses.replace(NV, backend='unlisted', ray_query=False),
             dataclasses.replace(NV, backend='another', ray_query=False)]
    assert gpu.dedupe_adapters(found) == [found[0]]


def test_many_adapters_deduplicate_without_losing_physical_gpus():
    found = [dataclasses.replace(NV, luid=(i,)) for i in range(20000)]
    assert gpu.dedupe_adapters(found + found) == found


def test_missing_slangpy_makes_auto_the_cpu_with_a_warning(monkeypatch):
    monkeypatch.setitem(sys.modules, 'slangpy', None)
    adapters, reason = gpu.list_adapters()
    assert adapters == [] and 'slangpy' in reason
    assert 'slangpy' in choose('auto', adapters, reason).warnings[0]
    with pytest.raises(DeviceError, match='slangpy'):
        choose('nvidia', adapters, reason)
    with pytest.raises(DeviceError, match='slangpy'):
        gpu.open_device(NV)


def test_probe_only_enumerates_safe_backends_and_closes_devices(monkeypatch):
    closed = []
    calls = []
    def enumerate_adapters(kind):
        assert kind == 'vulkan'
        return [NV, RADEON, dataclasses.replace(NV, luid=(9,))]
    def device(**kwargs):
        calls.append(kwargs)
        if kwargs['adapter_luid'] == [9]:
            raise RuntimeError('adapter refused')
        return SimpleNamespace(has_feature=lambda feature: feature == 'ray_query',
                               close=lambda: closed.append(kwargs['adapter_luid']))
    device.enumerate_adapters = enumerate_adapters
    spy = SimpleNamespace(Device=device, DeviceType=SimpleNamespace(vulkan='vulkan'),
                          Feature=SimpleNamespace(ray_query='ray_query'))
    monkeypatch.setitem(sys.modules, 'slangpy', spy)
    found, reason = gpu.list_adapters()
    assert found == [NV, RADEON] and 'adapter refused' in reason
    assert closed == [[1], [2]]
    assert all(c['enable_hot_reload'] is False and c['enable_rhi_validation'] is False for c in calls)
    assert choose('auto', found, reason).adapter == NV


@pytest.mark.gpu_nvidia
def test_probe_finds_the_nvidia_gpu():
    assert any(a.vendor_id == NVIDIA and a.ray_query for a in gpu.list_adapters()[0])


@pytest.mark.gpu_amd
def test_probe_finds_the_amd_gpu():
    assert any(a.vendor_id == AMD and a.ray_query for a in gpu.list_adapters()[0])
