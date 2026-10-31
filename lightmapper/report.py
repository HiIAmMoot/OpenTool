"""The short report a bake prints when it finishes: stage times, page sizes, written file sizes and devices."""

import json
from pathlib import Path


def format_duration(seconds: float) -> str:
    whole = round(seconds)
    if whole < 60:
        return f'{whole} s'
    if whole < 3600:
        return f'{whole // 60} min {whole % 60} s'
    return f'{whole // 3600} h {whole % 3600 // 60} min'


def format_size(size: int) -> str:
    return f'{size / 1048576:.1f} MB'


def read_summary(path: Path) -> dict | None:
    try:
        summary = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    return summary if isinstance(summary, dict) else None


def bake_report(summary: dict) -> list[str]:
    seconds = summary.get('seconds')
    seconds = seconds if isinstance(seconds, dict) else {}
    lines = []

    def timed(label, *keys, required=1):
        if sum(key in seconds for key in keys) >= required:
            lines.append(f'{label}: {format_duration(sum(seconds.get(key, 0.0) for key in keys))}')

    timed('Total time', 'total')
    timed('Preparing', 'probe', 'gather', 'charts', 'scene', required=4)
    # (a bake without scenery lights has no lamp batches: the main bake is the blender stage alone)
    timed('Main bake', 'blender', 'blender_lamp_batches')
    timed('Vertex pass', 'vertex_pass')
    timed('Glow pass', 'glow')
    timed('Finishing', 'finish')
    timed('Denoising', 'denoise')
    timed('Writing', 'write')
    sizes = summary.get('sizes')
    if 'pages' in summary and isinstance(sizes, list):
        lines.append(f"Pages: {summary['pages']} ({', '.join(f'{w}x{h}' for w, h in sizes)})")
    written = summary.get('written')
    written = written if isinstance(written, dict) else {}
    if isinstance(written.get('bitmap'), int):
        lines.append(f"Bitmap written: {format_size(written['bitmap'])}")
    if isinstance(written.get('bsp'), int):
        lines.append(f"BSP written: {format_size(written['bsp'])}")
    devices = summary.get('devices')
    devices = devices if isinstance(devices, dict) else {}
    # (a skipped glow run is recorded as 'off')
    glow = devices.get('glow_pass')
    if glow in (None, '', 'off'):
        reason = devices.get('glow_reason')
        glow = f'skipped ({reason})' if reason else 'skipped'
    if {'page_bake', 'vertex_pass', 'denoise'} <= devices.keys():
        lines.append(f"Devices: page bake {devices['page_bake']}, vertex pass {devices['vertex_pass']}, "
                     f"glow pass {glow}, denoise {devices['denoise']}")
    return lines
