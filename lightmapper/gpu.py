"""Resolve one vendor for the bake stages before starting Blender.

SlangPy is optional for the CPU path and is imported only while probing or
opening a device. Adapter LUIDs keep subsequent passes on the probed GPU.
"""

import json
import subprocess
from dataclasses import asdict, dataclass

from . import frozen


NVIDIA, AMD = 0x10DE, 0x1002
# recon passed Vulkan ray queries on both GPUs; D3D12 buffer creation crashed the process
BACKEND_ORDER: dict[int, tuple[str, ...]] = {NVIDIA: ('vulkan',), AMD: ('vulkan',)}
DEFAULT_BACKENDS: tuple[str, ...] = ('vulkan',)
# recon's warmed 680M HIP bakes were not faster with HIP RT
HIPRT = False
# a healthy probe takes about 2 s; a driver that hangs while a device opens would otherwise hang the bake, so the
# child is cut off well before a user would give up and the bake goes on without the GPU
PROBE_TIMEOUT_S = 60.0
PROBE_PREFIX = 'lightmapper-gpu-probe: '


class ProbeError(RuntimeError):
    pass


class DeviceError(RuntimeError):
    pass


def normalise_device(value: str) -> str:
    names = {'auto': 'auto', 'nvidia': 'nvidia', 'amd': 'amd', 'cpu': 'cpu',
             'optix': 'nvidia', 'cuda': 'nvidia', 'hip': 'amd', 'oneapi': 'auto'}
    if isinstance(value, str) and value.lower() in names:
        return names[value.lower()]
    raise ValueError(f'device must be one of auto, nvidia, amd, cpu, not {value!r}')


@dataclass(frozen=True)
class Adapter:
    name: str
    vendor_id: int
    luid: tuple[int, ...]
    backend: str
    ray_query: bool


@dataclass(frozen=True)
class DeviceChoice:
    setting: str
    vendor: str | None
    adapter: Adapter | None
    cycles: tuple[str, ...]
    oidn: str
    required: bool
    warnings: tuple[str, ...]

    def summary(self) -> dict:
        return {'setting': self.setting, 'vendor': self.vendor,
                'adapter': self.adapter.name if self.adapter else None,
                'backend': self.adapter.backend if self.adapter else None,
                'cycles': list(self.cycles), 'oidn': self.oidn}


def dedupe_adapters(found: list[Adapter]) -> list[Adapter]:
    selected = {}
    ranks = {}
    for adapter in found:
        order = BACKEND_ORDER.get(adapter.vendor_id, DEFAULT_BACKENDS)
        rank = (not adapter.ray_query,
                order.index(adapter.backend) if adapter.backend in order else len(order))
        if adapter.luid not in selected or rank < ranks[adapter.luid]:
            selected[adapter.luid] = adapter
            ranks[adapter.luid] = rank
    return list(selected.values())


def choose(setting: str, adapters: list[Adapter] | None, reason: str | None = None) -> DeviceChoice:
    setting = normalise_device(setting)
    chains = {'nvidia': (('OPTIX', 'CUDA'), 'CUDA'), 'amd': (('HIP',), 'HIP'),
              'other': (('OPTIX', 'CUDA', 'HIP', 'ONEAPI'), 'CPU'), None: (('OPTIX', 'CUDA', 'HIP', 'ONEAPI'), 'CPU')}
    if setting == 'cpu':
        return DeviceChoice(setting, None, None, (), 'CPU', False, ())
    required = setting != 'auto'
    vendor = setting if required else None
    adapter = None
    warnings = ()
    if adapters is not None:
        seen = reason or ', '.join(a.name for a in adapters) or 'no adapters seen'
        vendor_order = (NVIDIA, AMD)
        vendors = {'nvidia': NVIDIA, 'amd': AMD}
        candidates = [a for a in adapters if a.ray_query and
                      (not required or a.vendor_id == vendors[setting])]
        if candidates:
            adapter = min(candidates, key=lambda a: vendor_order.index(a.vendor_id)
                          if a.vendor_id in vendor_order else len(vendor_order))
            vendor = {NVIDIA: 'nvidia', AMD: 'amd'}.get(adapter.vendor_id, 'other')
            if vendor == 'other':
                warnings = (f'{adapter.name}: unknown GPU vendor; Cycles uses its GPU chain, OIDN on the CPU',)
        elif required:
            raise DeviceError(f'{setting}: no GPU with ray queries ({seen})')
        else:
            warnings = (f'no GPU with ray queries ({seen}); vertex pass on the CPU',)
    cycles, oidn = chains[vendor]
    return DeviceChoice(setting, vendor, adapter, cycles, oidn, required, warnings)


def open_device(adapter: Adapter):
    try:
        import slangpy as spy
        return spy.Device(type=getattr(spy.DeviceType, adapter.backend), adapter_luid=list(adapter.luid),
                          enable_hot_reload=False, enable_rhi_validation=False)
    except Exception as error:
        raise DeviceError(f'{adapter.name} ({adapter.backend}): {error}') from error


def list_adapters() -> tuple[list[Adapter], str | None]:
    try:
        import slangpy as spy
    except ImportError as error:
        return [], f'slangpy: {error}'
    found = []
    errors = []
    # D3D12 can crash outside Python exception handling, so enumerate only recon-approved backends
    backends = dict.fromkeys(b for order in (*BACKEND_ORDER.values(), DEFAULT_BACKENDS) for b in order)
    for backend in backends:
        try:
            adapters = spy.Device.enumerate_adapters(getattr(spy.DeviceType, backend))
        except Exception as error:
            errors.append(f'{backend}: {error}')
            continue
        for info in adapters:
            adapter = Adapter(info.name, info.vendor_id, tuple(info.luid), backend, False)
            try:
                device = open_device(adapter)
                try:
                    ray_query = device.has_feature(spy.Feature.ray_query)
                finally:
                    device.close()
                found.append(Adapter(adapter.name, adapter.vendor_id, adapter.luid, backend, ray_query))
            except DeviceError as error:
                errors.append(str(error))
            except Exception as error:
                errors.append(f'{adapter.name} ({backend}): {error}')
    return dedupe_adapters(found), '; '.join(errors) or None


def _probe_command() -> list[str]:
    return frozen.child_command(frozen.PROBE_FLAG)


def run_probe() -> tuple[list[Adapter], str | None]:
    try:
        done = subprocess.run(_probe_command(), stdin=subprocess.DEVNULL, capture_output=True,
                              timeout=PROBE_TIMEOUT_S, creationflags=frozen.NO_WINDOW)
    except subprocess.TimeoutExpired:
        raise ProbeError(f'GPU probe timed out after {PROBE_TIMEOUT_S:g} s') from None
    except OSError as error:
        raise ProbeError(f'GPU probe could not start: {error}') from error
    if done.returncode:
        tail = ' | '.join(done.stderr.decode(errors='replace').strip().splitlines()[-3:])
        raise ProbeError(f'GPU probe process died (exit {done.returncode}){": " + tail if tail else ""}')
    lines = [line[len(PROBE_PREFIX):] for line in done.stdout.decode(errors='replace').splitlines()
             if line.startswith(PROBE_PREFIX)]
    if not lines:
        raise ProbeError('GPU probe output unreadable (no answer line)')
    try:
        data = json.loads(lines[-1])
        return [Adapter(**{**a, 'luid': tuple(a['luid'])}) for a in data['adapters']], data['reason']
    except (IndexError, ValueError, KeyError, TypeError) as error:
        raise ProbeError(f'GPU probe output unreadable ({error})') from error


def probe_adapters() -> tuple[list[Adapter], str | None]:
    """list_adapters in a child process, so a driver that crashes or hangs while a device opens is a reason string."""
    try:
        return run_probe()
    except ProbeError as error:
        return [], str(error)


def probe_choice(setting: str) -> DeviceChoice:
    setting = normalise_device(setting)
    return choose(setting, []) if setting == 'cpu' else choose(setting, *probe_adapters())


def probe_main() -> int:
    adapters, reason = list_adapters()
    print(PROBE_PREFIX + json.dumps({'adapters': [asdict(a) for a in adapters], 'reason': reason}))
    return 0
