import io
import struct
import subprocess

from lightmapper import denoise_worker, frozen, gpu

CPU_HINT = 'Bakes on the CPU, with fewer rays for Draft, Normal and High.'


def probe_denoiser():
    import numpy as np

    available = {}
    for kind in ('CUDA', 'HIP'):
        available[kind] = False
        try:
            # EOF after the ready frame exercises the bake's startup without denoising a page.
            done = subprocess.run(denoise_worker.worker_command(kind), input=b'', capture_output=True,
                                  timeout=5, creationflags=frozen.NO_WINDOW)
            if done.returncode:
                continue
            size = struct.unpack('<Q', done.stdout[:8])[0]
            payload = done.stdout[8:]
            if size != len(payload):
                continue
            ready = np.load(io.BytesIO(payload), allow_pickle=False)
            available[kind] = ready.shape == (1,) and ready.dtype == np.uint8 and bool(ready[0] == 1)
        except (OSError, subprocess.TimeoutExpired, ValueError, EOFError, struct.error):
            pass
    return available


def gpu_text(adapters, reason):
    if adapters is None:
        return 'Looking for GPUs…'
    names = [adapter.name for adapter in adapters if adapter.ray_query]
    if names:
        return 'GPU: ' + ', '.join(names)
    return 'No GPU with ray tracing found' + (f' ({reason})' if reason else '')


def device_hint(setting, adapters, reason):
    if adapters is None:
        return ''
    try:
        choice = gpu.choose(setting, adapters, reason)
    except gpu.DeviceError:
        name = {'nvidia': 'NVIDIA', 'amd': 'AMD'}[gpu.normalise_device(setting)]
        return f'No {name} GPU with ray tracing was found, so the bake would stop. Pick Auto or CPU only.'
    if choice.adapter:
        return f'Bakes on {choice.adapter.name}.'
    if choice.setting == 'cpu':
        return CPU_HINT
    return "The vertex pass runs on the CPU; pages still try Cycles' GPU backends."


def stage_rows(setting, adapters, reason, denoiser=None):
    stages = ('Page bake', 'Vertex pass', 'Denoise')
    if adapters is None:
        return [(stage, None, 'Looking for GPUs…') for stage in stages]
    try:
        choice = gpu.choose(setting, adapters, reason)
    except gpu.DeviceError:
        name = {'nvidia': 'NVIDIA', 'amd': 'AMD'}[gpu.normalise_device(setting)]
        return [(stage, None, f'No {name} GPU with ray tracing') for stage in stages]
    if choice.setting == 'cpu':
        return [(stage, 'CPU', '') for stage in stages]
    if choice.adapter:
        denoise = ('Denoise', 'CPU', '')
        if choice.oidn != 'CPU':
            if denoiser is None:
                denoise = ('Denoise', None, 'Checking the GPU denoiser…')
            elif denoiser.get(choice.oidn, False):
                denoise = ('Denoise', 'GPU', '')
            else:
                denoise = ('Denoise', 'CPU', 'the GPU denoiser could not start')
        return [('Page bake', 'GPU', ''), ('Vertex pass', 'GPU', ''),
                denoise]
    return [('Page bake', 'GPU', "Cycles picks a GPU when the bake starts, and falls back to the CPU without one"),
            ('Vertex pass', 'CPU', 'no GPU with ray queries'), ('Denoise', 'CPU', '')]


def stage_rows_used(used):
    """the same rows from a finished bake's summary, so they show what really happened."""
    def state(name, cpu):
        return 'CPU' if name == cpu else 'Off' if name == 'off' else 'GPU'
    return [('Page bake', state(used['page_bake'], 'CPU'), '' if used['page_bake'] == 'CPU' else used['page_bake']),
            ('Vertex pass', state(used['vertex_pass'], 'cpu'), ''),
            ('Denoise', state(used['denoise'], 'CPU'), '' if used['denoise'] in ('CPU', 'off') else used['denoise'])]


def result_line(devices, rays, warnings):
    pages = 'the CPU' if devices['page_bake'] == 'CPU' else f"the GPU ({devices['page_bake']})"
    vertex = 'the CPU' if devices['vertex_pass'] == 'cpu' else devices['vertex_pass']
    denoise = 'off' if devices['denoise'] == 'off' else f"on {'the CPU' if devices['denoise'] == 'CPU' else devices['denoise']}"
    parts = [f'Pages on {pages}, vertex pass on {vertex}, denoiser {denoise}.']
    if rays:
        parts.append(f"{rays['samples']} rays per texel, {rays['vertex_rays']} per vertex.")
    if warnings:
        parts.append(f"{len(warnings)} warning{'s' if len(warnings) != 1 else ''}, see the log.")
    return ' '.join(parts)
