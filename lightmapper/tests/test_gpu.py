import dataclasses
import json
import sys
from types import SimpleNamespace

import pytest

from lightmapper import frozen, gpu
from lightmapper.gpu import AMD, NVIDIA, Adapter, DeviceError, choose, probe_choice
from lightmapper.settings import Settings


NV = Adapter('NVIDIA GeForce RTX 4060 Laptop GPU', NVIDIA, (1,), 'vulkan', True)
RADEON = Adapter('AMD Radeon(TM) 680M', AMD, (2,), 'vulkan', True)


def test_auto_cycles_chain_ends_with_oneapi():
    assert gpu.choose('auto', None).cycles == ('OPTIX', 'CUDA', 'HIP', 'ONEAPI')
    assert gpu.choose('auto', [gpu.Adapter('Arc', 0x8086, (9,), 'vulkan', True)]).cycles == ('OPTIX', 'CUDA', 'HIP', 'ONEAPI')


@pytest.mark.parametrize('value', ['ONEAPI', 'oneapi'])
def test_oneapi_setting_is_auto(value):
    assert gpu.normalise_device(value) == 'auto'
    assert Settings(device=value).device == 'auto'


def test_auto_prefers_nvidia_then_amd():
    assert choose('auto', [RADEON, NV]).adapter == NV
    c = choose('auto', [RADEON])
    assert (c.vendor, c.cycles, c.oidn, c.required) == ('amd', ('HIP',), 'HIP', False)
    c = choose('nvidia', [NV])
    assert (c.cycles, c.oidn, c.required) == (('OPTIX', 'CUDA'), 'CUDA', True)


def test_auto_without_ray_queries_keeps_the_cycles_gpu_chain_with_a_warning():
    c = choose('auto', [dataclasses.replace(NV, ray_query=False)])
    assert (c.adapter, c.cycles, c.oidn, c.required) == (None, ('OPTIX', 'CUDA', 'HIP', 'ONEAPI'), 'CPU', False)
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
    assert (c.vendor, c.adapter, c.cycles, c.oidn) == ('other', arc, ('OPTIX', 'CUDA', 'HIP', 'ONEAPI'), 'CPU')
    assert c.warnings


@pytest.mark.parametrize('setting,vendor,cycles,oidn,required', [
    ('auto', None, ('OPTIX', 'CUDA', 'HIP', 'ONEAPI'), 'CPU', False),
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
    monkeypatch.setattr(gpu, 'probe_adapters', lambda: pytest.fail('cpu probed the GPUs'))
    # the autouse fixture replaces the module attribute, not this original entry point
    assert probe_choice('cpu').cycles == ()


def test_probe_choice_uses_the_probed_adapters_and_failure_reason(monkeypatch):
    monkeypatch.setattr(gpu, 'probe_adapters', lambda: ([RADEON], None))
    assert probe_choice('HIP').adapter == RADEON
    monkeypatch.setattr(gpu, 'probe_adapters', lambda: ([], 'probe refused'))
    assert probe_choice('auto').warnings == (
        'no GPU with ray queries (probe refused); vertex pass on the CPU',)


def _stand_in(monkeypatch, code, timeout=None):
    monkeypatch.setattr(gpu, '_probe_command', lambda: [sys.executable, '-c', code])
    if timeout is not None:
        monkeypatch.setattr(gpu, 'PROBE_TIMEOUT_S', timeout)


CHILD_FAILURES = {
    'exit': ('import sys; sys.exit(3)', None, 'exit 3'),
    'access violation': ('import os; os._exit(0xC0000005)', None, 'exit'),
    'hang': ('import time; time.sleep(60)', 1, 'timed out'),
    'garbage': ('print("not json")', None, 'unreadable')}


@pytest.mark.parametrize('kind', CHILD_FAILURES)
def test_a_failing_probe_child_makes_auto_the_cpu_with_a_warning(monkeypatch, kind):
    code, timeout, text = CHILD_FAILURES[kind]
    _stand_in(monkeypatch, code, timeout)
    choice = probe_choice('auto')
    assert choice.adapter is None and choice.required is False
    assert len(choice.warnings) == 1 and 'probe' in choice.warnings[0] and text in choice.warnings[0]


@pytest.mark.parametrize('kind', CHILD_FAILURES)
def test_a_failing_probe_child_fails_a_named_vendor(monkeypatch, kind):
    code, timeout, text = CHILD_FAILURES[kind]
    _stand_in(monkeypatch, code, timeout)
    with pytest.raises(DeviceError, match='nvidia') as error:
        probe_choice('nvidia')
    assert text in str(error.value)


def test_the_probe_child_hands_back_the_adapters(monkeypatch):
    payload = json.dumps({'adapters': [dataclasses.asdict(NV)], 'reason': 'one refused'})
    _stand_in(monkeypatch, f'print("driver noise"); print({("lightmapper-gpu-probe: " + payload)!r})')
    adapters, reason = gpu.probe_adapters()
    assert adapters == [NV] and reason == 'one refused'


def test_a_per_adapter_probe_error_names_the_adapter_once(monkeypatch):
    def device(**kwargs):
        raise RuntimeError('adapter refused')
    device.enumerate_adapters = lambda kind: [NV]
    spy = SimpleNamespace(Device=device, DeviceType=SimpleNamespace(vulkan='vulkan'),
                          Feature=SimpleNamespace(ray_query='ray_query'))
    monkeypatch.setitem(sys.modules, 'slangpy', spy)
    _, reason = gpu.list_adapters()
    assert reason.count(NV.name) == 1 and reason.count('vulkan') == 1


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


def test_probe_reads_its_tagged_line_when_a_driver_prints_after_it(monkeypatch):
    line = 'lightmapper-gpu-probe: ' + json.dumps({'adapters': [{'name': 'X', 'vendor_id': 0x10DE, 'luid': [1, 2], 'backend': 'vulkan', 'ray_query': True}], 'reason': None})
    code = f'print("driver hello"); print({line!r}); print("driver bye")'
    monkeypatch.setattr(gpu, '_probe_command', lambda: [sys.executable, '-c', code])
    assert gpu.run_probe() == ([gpu.Adapter('X', 0x10DE, (1, 2), 'vulkan', True)], None)


def test_probe_ignores_stray_modules_in_the_working_folder(monkeypatch, tmp_path):
    (tmp_path / 'json.py').write_text('raise SystemExit(9)')
    (tmp_path / 'numpy.py').write_text('raise SystemExit(9)')
    monkeypatch.chdir(tmp_path)
    gpu.run_probe()


def test_a_dead_probe_child_is_a_probe_error_and_a_reason(monkeypatch):
    monkeypatch.setattr(gpu, '_probe_command', lambda: [sys.executable, '-c', 'raise SystemExit(3)'])
    with pytest.raises(gpu.ProbeError, match=r'exit 3'):
        gpu.run_probe()
    assert gpu.probe_adapters()[0] == [] and 'exit 3' in gpu.probe_adapters()[1]


def test_probe_starts_without_a_window(monkeypatch):
    seen = {}
    real = gpu.subprocess.run

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(gpu.subprocess, 'run', spy)
    gpu.probe_adapters()
    assert seen['creationflags'] == frozen.NO_WINDOW
