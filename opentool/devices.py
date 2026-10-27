from lightmapper import gpu

CPU_HINT = 'Bakes on the CPU, with fewer rays for Draft, Normal and High.'


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
    return f'Bakes on {choice.adapter.name}.' if choice.adapter else CPU_HINT


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
