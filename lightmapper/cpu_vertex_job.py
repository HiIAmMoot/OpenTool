"""Launch and validate the compiled vertex child without importing its kernels."""

import json
from pathlib import Path
import subprocess
import zipfile

import numpy as np

from . import frozen
from .vertex_gpu import VertexPassResult


class CpuPassError(RuntimeError):
    pass


def run_in_child(work: Path, pages: list[np.ndarray], timeout: float | None) -> tuple[VertexPassResult, dict]:
    work = Path(work).resolve()
    np.savez(work / 'diffuse.npz', **{f'diffuse_{i}': p for i, p in enumerate(pages)})
    out = work / 'cpu_out.npz'
    # a crashed child must not leave an earlier run looking successful
    out.unlink(missing_ok=True)
    try:
        done = subprocess.run(frozen.child_command(frozen.CPU_VERTEX_FLAG, str(work)), cwd=work,
                              stdin=subprocess.DEVNULL, capture_output=True, timeout=timeout,
                              creationflags=frozen.NO_WINDOW)
    except subprocess.TimeoutExpired:
        raise CpuPassError(f'vertex: timed out after {timeout} s') from None
    except OSError as error:
        raise CpuPassError(f'vertex: process died (exit unavailable): {error}') from error
    tail = '\n'.join(done.stderr.decode(errors='replace').splitlines()[-40:])
    died = f'vertex: process died (exit {done.returncode}): {tail}'
    if done.returncode:
        raise CpuPassError(died)
    try:
        with np.load(out, allow_pickle=False) as data:
            result = VertexPassResult(np.array(data['vector'], np.float32), np.array(data['irradiance'], np.float32),
                                      np.array(data['sun_lit'], bool), np.array(data['edge_sun_lit'], bool),
                                      int(data['rays']), np.array(data['vertex_sun_throughput'], np.float32)
                                       if 'vertex_sun_throughput' in data.files else None,
                                      np.array(data['texel_rgb'], np.float32),
                                      np.array(data['vertex_light_flat'], np.float32)
                                       if 'vertex_light_flat' in data.files else None)
            timing = {'compile': float(data['seconds_compile']), 'trace': float(data['seconds_trace']),
                      'threading_layer': str(data['threading_layer']), 'threads': int(data['threads'])}
        job = json.loads((work / 'job.json').read_text())
        with np.load(work / job['scene'], allow_pickle=False) as data:
            q = len(data['query_pos'])
            edges = len(data['edge_pos']) if 'edge_pos' in data else 0
            texels = len(data['texel_pos']) if 'texel_pos' in data else 0
            needs_throughput = (job.get('ignore_normals_mode', 'tool') == 'tool'
                                and 'query_ignore_normals' in data and data['query_ignore_normals'].any())
        suns = len(job['suns'])
        if (result.vector.shape != (q, 3) or result.irradiance.shape != (q,)
                or result.sun_lit.shape != (q, suns) or result.edge_sun_lit.shape != (edges, suns)
                or (result.sun_throughput is not None) != bool(needs_throughput)
                or (result.sun_throughput is not None and result.sun_throughput.shape != (q, suns, 3))
                or (result.light_flat is not None
                    and (not needs_throughput or result.light_flat.shape != (q, 3)))
                or result.texel_rgb.shape != (texels, 3)):
            raise ValueError('short vertex arrays')
        return result, timing
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile) as error:
        raise CpuPassError(f'{died}\n{error}') from error
