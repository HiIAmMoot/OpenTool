"""The file exchange with the Blender bake process.

Blender is GPL, so it only ever runs as a separate process: we write job.json,
scene.npz and textures.npz, run it, and read out.npz back. Nothing in here imports
bpy. A failed run raises before the caller has written anything to a tag.
"""

from dataclasses import dataclass, fields, replace
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from .memory import SceneCounts, free_physical_bytes, out_of_memory_message
from .settings import Settings
from .gpu import DeviceChoice, HIPRT, choose
from .vertex_gpu import VertexPassResult
from .twins import reversed_twins
from . import frozen


BAKE_SCRIPT = Path(__file__).with_name('blender') / 'bake.py'
LOG_TAIL_LINES = 40
SOURCES = ('all', 'sun', 'emit')
# what Windows reports when Blender dies on a failed allocation: unsigned, or signed as Python sees it elsewhere
ACCESS_VIOLATION = (3221225477, -1073741819)


@dataclass
class Scene:
    positions: np.ndarray
    normals: np.ndarray
    lm_uv: np.ndarray
    tex_uv: np.ndarray
    # counter-clockwise relative to the normals, the opposite of Halo's winding
    tris: np.ndarray
    tri_page: np.ndarray
    tri_class: np.ndarray
    tri_casts_shadow: np.ndarray
    tri_emission: np.ndarray
    tri_texture: np.ndarray
    tri_cutout: np.ndarray
    # the colour of light that crosses a see-through triangle; a solid one ignores it
    tri_tint: np.ndarray
    occ_positions: np.ndarray
    occ_uv: np.ndarray
    occ_tris: np.ndarray
    occ_tri_texture: np.ndarray
    occ_tri_cutout: np.ndarray
    occ_tri_casts_shadow: np.ndarray
    occ_tri_emission: np.ndarray
    occ_tri_tint: np.ndarray
    query_pos: np.ndarray
    query_normal: np.ndarray
    query_class: np.ndarray
    query_inverse: np.ndarray | None = None
    tri_ignore_normals: np.ndarray | None = None
    query_ignore_normals: np.ndarray | None = None
    texel_pos: np.ndarray | None = None
    texel_normal: np.ndarray | None = None
    texel_class: np.ndarray | None = None
    texel_page: np.ndarray | None = None
    texel_xy: np.ndarray | None = None


_DTYPES = {
    'positions': np.float32, 'normals': np.float32, 'lm_uv': np.float32, 'tex_uv': np.float32,
    'tris': np.int32, 'tri_page': np.int16, 'tri_class': np.int16, 'tri_casts_shadow': np.bool_,
    'tri_emission': np.float32, 'tri_texture': np.int32, 'tri_cutout': np.bool_, 'tri_tint': np.float32,
    'occ_positions': np.float32, 'occ_uv': np.float32, 'occ_tris': np.int32,
    'occ_tri_texture': np.int32, 'occ_tri_cutout': np.bool_, 'occ_tri_casts_shadow': np.bool_,
    'occ_tri_emission': np.float32, 'occ_tri_tint': np.float32,
    'query_pos': np.float32, 'query_normal': np.float32, 'query_class': np.int16,
    'query_inverse': np.int32,
    'tri_ignore_normals': np.bool_, 'query_ignore_normals': np.bool_,
    'texel_pos': np.float32, 'texel_normal': np.float32, 'texel_class': np.int16,
    'texel_page': np.int16, 'texel_xy': np.int32,
}


class BlenderError(RuntimeError):
    pass


class BlenderOutOfMemory(BlenderError):
    def __init__(self, message, tail):
        self.message = message
        super().__init__(f'{message}\n{tail}')


@dataclass
class BakeResult:
    pages: list
    vertex_vector: np.ndarray | None
    vertex_irradiance: np.ndarray | None
    device: str
    written: list
    # per query and job sun: the vertex pass lit it (None when the run predates the output)
    vertex_sun_lit: np.ndarray | None = None
    edge_sun_lit: np.ndarray | None = None
    diffuse_pages: list | None = None
    vertex_rays_cast: int | None = None
    stages: dict | None = None
    samples_used: int | None = None
    texel_samples_used: int | None = None
    vertex_sun_throughput: np.ndarray | None = None
    texel_rgb: np.ndarray | None = None


def write_job(work: Path, scene: Scene, pages: list, suns: list, ambient: dict,
              s: Settings, sources: str = 'all', textures: list | None = None,
              choice: DeviceChoice | None = None, vertex_pass: str = 'cpu') -> Path:
    if sources not in SOURCES:
        raise ValueError(f'sources must be one of {SOURCES}, not {sources!r}')
    if vertex_pass not in ('cpu', 'skip'):
        raise ValueError(f'vertex_pass must be cpu or skip, not {vertex_pass!r}')
    if choice is None:
        choice = choose(s.device, None)
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)

    arrays = {f.name: np.ascontiguousarray(getattr(scene, f.name), _DTYPES[f.name])
              for f in fields(Scene) if getattr(scene, f.name) is not None}
    for name, count in (('tri_ignore_normals', len(scene.tris)), ('query_ignore_normals', len(scene.query_pos))):
        if name not in arrays:
            arrays[name] = np.zeros(count, np.bool_)
    from .raster import texel_queries
    queries = [texel_queries(scene, i, w, h) for i, (w, h) in enumerate(pages)] if s.ignore_normals_mode == 'enhanced' else []
    for name, shape in (('texel_pos', (0, 3)), ('texel_normal', (0, 3)), ('texel_class', (0,)),
                        ('texel_page', (0,)), ('texel_xy', (0, 2))):
        if s.ignore_normals_mode == 'tool':
            arrays[name] = np.empty(shape, _DTYPES[name])
        elif name not in arrays:
            arrays[name] = (np.concatenate([q[name] for q in queries]) if queries
                            else np.empty(shape, _DTYPES[name]))
    corners = np.concatenate([arrays['positions'][arrays['tris']], arrays['occ_positions'][arrays['occ_tris']]])
    arrays['tri_twin'], arrays['tri_twin_perm'], ambiguous_twins = reversed_twins(corners)
    # an unlightmapped emitter is transparent to diffuse rays whatever the weights leave of its emission
    arrays['occ_tri_emit_only'] = (~arrays['occ_tri_casts_shadow'] & arrays['occ_tri_emission'].any(axis=1))
    if sources == 'sun':
        arrays['tri_emission'] = np.zeros_like(arrays['tri_emission'])
        arrays['occ_tri_emission'] = np.zeros_like(arrays['occ_tri_emission'])
    else:
        arrays['tri_emission'] = (arrays['tri_emission'] * s.w_emit).astype(np.float32)
        arrays['occ_tri_emission'] = (arrays['occ_tri_emission'] * s.w_emit).astype(np.float32)
    np.savez(work / 'textures.npz',
             **{f'tex_{k}': np.asarray(t, np.uint8) for k, t in enumerate(textures or [])})

    job_suns = []
    if sources != 'emit':
        for sun in suns:
            # a sun that reaches no class lights nothing, so Blender never sees it
            if not sun.classes:
                continue
            job_suns.append({
                'direction': [float(v) for v in sun.direction],
                **({'centre': [float(v) for v in sun.centre]} if sun.centre is not None else {}),
                'angle': float(sun.angle),
                'color': [float(v) for v in sun.color],
                'strength': float(sun.power * s.w_sun),
                'classes': [int(c) for c in sun.classes],
            })
    # calibration treats ambient analytically, so only a full bake hands it over
    job_ambient = {}
    if sources == 'all':
        job_ambient = {str(k): [float(x) * s.w_amb for x in v] for k, v in ambient.items()}

    from .terminator import edge_queries
    arrays.update(edge_queries(scene, job_suns, s.reach))
    np.savez(work / 'scene.npz', **arrays)

    job = {
        'version': 1,
        'scene': 'scene.npz',
        'out': 'out.npz',
        'pages': [[int(w), int(h)] for w, h in pages],
        'suns': job_suns,
        'ambient': job_ambient,
        'samples': s.samples,
        'texel_samples': s.resolved_texel_samples(),
        'cpu_samples': s.resolved_cpu_samples(),
        'cpu_texel_samples': s.resolved_cpu_texel_samples(),
        'cpu_vertex_rays': s.resolved_cpu_vertex_rays(),
        'bounces': s.bounces,
        'albedo_scale': s.albedo_scale,
        'albedo_fallback': s.albedo_fallback,
        'vertex_rays': s.vertex_rays,
        'seed': s.seed,
        'device': choice.setting,
        'cycles_devices': list(choice.cycles),
        'cycles_required': choice.required,
        'hiprt': HIPRT,
        'vertex_pass': vertex_pass,
        'ignore_normals_mode': s.ignore_normals_mode,
        'ambiguous_twin_groups': ambiguous_twins,
    }
    path = work / 'job.json'
    path.write_text(json.dumps(job, indent=2))
    return path


def _log_tail(log: Path) -> str:
    if not log.is_file():
        return '(no log)'
    return '\n'.join(log.read_text(errors='replace').splitlines()[-LOG_TAIL_LINES:])


def _run_blender(work: Path, s: Settings, job_name: str, log_name: str):
    work = Path(work)
    job = json.loads((work / job_name).read_text())
    out = work / job['out']
    # a result left by an earlier run must not pass for this one's
    out.unlink(missing_ok=True)
    log = work / log_name
    command = [s.blender, '-b', '--factory-startup', '--python-exit-code', '1',
               '--python', str(BAKE_SCRIPT), '--', str(work / job_name)]
    environment = dict(os.environ)
    environment['LIGHTMAPPER_LAUNCH_TIME'] = str(time.time())
    # (read before Blender starts: once it has died its memory is free again and the message would contradict itself)
    free_before = free_physical_bytes()
    try:
        with open(log, 'wb') as handle:
            done = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=handle,
                                  stderr=subprocess.STDOUT, timeout=s.timeout_s, env=environment,
                                  creationflags=frozen.NO_WINDOW)
    except subprocess.TimeoutExpired:
        raise BlenderError(f'blender timed out after {s.timeout_s} s\n{_log_tail(log)}') from None
    except OSError as error:
        raise BlenderError(f'blender could not start: {error}') from None
    if done.returncode != 0:
        tail = _log_tail(log)
        # an access violation alone is any crash; only Blender's own failed-allocation line makes it memory
        if done.returncode in ACCESS_VIOLATION and ('Calloc returns null' in tail or 'Malloc returns null' in tail):
            message = out_of_memory_message(SceneCounts.from_npz(work / job['scene']),
                                            [tuple(p) for p in job['pages']], s.page_max, free_before)
            raise BlenderOutOfMemory(message, tail)
        raise BlenderError(f'blender exited with {done.returncode}\n{tail}')
    if not out.is_file():
        raise BlenderError(f'blender wrote no {out.name}\n{_log_tail(log)}')
    return job, out


def run_blender(work: Path, s: Settings) -> BakeResult:
    job, out = _run_blender(work, s, 'job.json', 'blender.log')

    with np.load(out) as data:
        pages = [np.array(data[f'page_{i}'], np.float32) for i in range(len(job['pages']))]
        return BakeResult(pages, np.array(data['vertex_vector'], np.float32) if 'vertex_vector' in data else None,
                          np.array(data['vertex_irradiance'], np.float32) if 'vertex_irradiance' in data else None,
                          str(data['device']),
                          [np.array(data[f'written_{i}'], np.bool_) for i in range(len(job['pages']))],
                          np.array(data['vertex_sun_lit'], np.bool_) if 'vertex_sun_lit' in data.files else None,
                          np.array(data['edge_sun_lit'], np.bool_) if 'edge_sun_lit' in data.files else None,
                          [np.array(data[f'diffuse_{i}'], np.float32) for i in range(len(job['pages']))],
                          int(data['vertex_rays_cast']),
                          ({'start': float(data['seconds_start']), 'scene': float(data['seconds_scene']),
                            'pages': float(data['seconds_pages']), 'vertex': float(data['seconds_vertex'])}
                           if 'seconds_start' in data.files else None),
                          int(data['samples_used']) if 'samples_used' in data.files else None,
                           int(data['texel_samples_used']) if 'texel_samples_used' in data.files else None,
                           np.array(data['vertex_sun_throughput'], np.float32)
                            if 'vertex_sun_throughput' in data.files else None,
                            np.array(data['texel_rgb'], np.float32) if 'texel_rgb' in data.files else None)


def run_blender_vertex_only(work: Path, s: Settings, pages: list[np.ndarray]) -> VertexPassResult:
    work = Path(work)
    np.savez(work / 'diffuse.npz', **{f'diffuse_{i}': p for i, p in enumerate(pages)})
    job = json.loads((work / 'job.json').read_text())
    job.update(vertex_only=True, out='vertex_out.npz', diffuse='diffuse.npz')
    (work / 'job_vertex.json').write_text(json.dumps(job, indent=2))
    _, out = _run_blender(work, s, 'job_vertex.json', 'blender_vertex.log')
    with np.load(out) as data:
        return VertexPassResult(np.array(data['vertex_vector'], np.float32),
                                np.array(data['vertex_irradiance'], np.float32),
                                np.array(data['vertex_sun_lit'], bool),
                                np.array(data['edge_sun_lit'], bool), int(data['vertex_rays_cast']),
                                np.array(data['vertex_sun_throughput'], np.float32)
                                 if 'vertex_sun_throughput' in data.files else None,
                                np.array(data['texel_rgb'], np.float32) if 'texel_rgb' in data.files else None)


def with_vertex_pass(result: BakeResult, vp: VertexPassResult) -> BakeResult:
    return replace(result, vertex_vector=vp.vector, vertex_irradiance=vp.irradiance,
                   vertex_sun_lit=vp.sun_lit, edge_sun_lit=vp.edge_sun_lit, vertex_rays_cast=vp.rays,
                   vertex_sun_throughput=vp.sun_throughput, texel_rgb=vp.texel_rgb)


def run_blender_glow(work: Path, s: Settings, pages: list[np.ndarray]) -> list[np.ndarray]:
    work = Path(work)
    np.savez(work / 'glow_pages.npz', **{f'page_{i}': p for i, p in enumerate(pages)})
    job = json.loads((work / 'job.json').read_text())
    job.update(glow=True, out='glow_out.npz', glow_pages='glow_pages.npz', suns=[], ambient={},
               bounces=0, vertex_pass='skip')
    (work / 'job_glow.json').write_text(json.dumps(job, indent=2))
    _, out = _run_blender(work, s, 'job_glow.json', 'blender_glow.log')
    with np.load(out) as data:
        return [np.array(data[f'page_{i}'], np.float32) for i in range(len(job['pages']))]
