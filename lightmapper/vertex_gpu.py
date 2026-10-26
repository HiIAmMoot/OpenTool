"""The per-vertex incident radiosity pass outside Blender: the hemisphere rays, query frames, ray sums and the sun and
edge ray lists that blender/bake.py's VertexPass builds, as float64 numpy so the GPU pass casts exactly what the CPU
pass does, and the pass itself, whose ray casting and shading run as ray queries in shaders/vertex_pass.slang.

The constants are copies (bake.py cannot be imported outside Blender); a test parses bake.py and holds them equal.
"""

from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .sun_facing import sun_facing_queries


RAY_LENGTH, RAY_EPSILON, PASS_LIMIT = 1e5, 1e-3, 8
LUMA = np.float64([0.2126, 0.7152, 0.0722])
GOLDEN_FRACTION = 0.618034
SEED_SPIN = 0.7548776662466927
GROUP_SIZE = 64
# the 680M gather kernel on putput's scene.npz at 4096 rays: 266M rays/s (slowest of 10 runs, 0.16-0.22 s each) x 0.25 s
RAYS_PER_DISPATCH = 66600960
SHADERS = Path(__file__).with_name('shaders')


@dataclass
class VertexPassResult:
    vector: np.ndarray
    irradiance: np.ndarray
    # per query and sun, per edge probe and sun: the shadow ray reached the sun
    sun_lit: np.ndarray
    edge_sun_lit: np.ndarray
    rays: int


def radical_inverse(i):
    result, scale = 0.0, 0.5
    while i:
        result += scale * (i & 1)
        i >>= 1
        scale *= 0.5
    return result


def hemisphere(count):
    """Cosine-weighted directions about +z from a Hammersley set, with the polar radius and azimuth they came from."""
    i = np.arange(count)
    u1 = (i + 0.5) / count
    phi = 2 * np.pi * np.array([radical_inverse(int(k)) for k in i])
    r = np.sqrt(u1)
    return np.stack([r * np.cos(phi), r * np.sin(phi), np.sqrt(1 - u1)], axis=1), phi, r


def query_frames(normals, seed):
    """(usable, n, tangent, bitangent, spin) per query; the spin turns every query's hemisphere by its own angle."""
    nrm = np.asarray(normals).astype(np.float64)
    length = np.linalg.norm(nrm, axis=1)
    # a zero or non-finite normal has no hemisphere to cast into: that query stays at zero
    usable = np.isfinite(length) & (length > 1e-12)
    n = np.where(usable[:, None], nrm / np.where(usable, length, 1.0)[:, None], [0.0, 0.0, 1.0])
    axis = np.where((np.abs(n[:, 0]) < 0.9)[:, None], [[1.0, 0, 0]], [[0, 1.0, 0]])
    tangent = np.cross(n, axis)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    bitangent = np.cross(n, tangent)
    ids = np.arange(len(n))
    spin = 2 * np.pi * ((ids * GOLDEN_FRACTION + seed * SEED_SPIN) % 1.0)
    return usable, n, tangent, bitangent, spin


def ambient_table(job, query_class, tri_class):
    """The class ambients indexed by class + 1: class -1 (indoor) has its own and must not wrap onto the last outdoor slot."""
    classes = [int(k) for k in job['ambient']] + [int(np.max(query_class, initial=0)), int(np.max(tri_class, initial=0))]
    table = np.zeros((max(classes) + 2, 3))
    for key, rgb in job['ambient'].items():
        table[int(key) + 1] = rgb
    return table


def gather_sums(lum, phi, r, h, n, tangent, bitangent, spin):
    """pi / rays x the sum of lum x direction and of lum, without building a direction per ray: the ray's azimuth is
    phi + spin, so its sine and cosine split into the spin-free sums over r cos phi and r sin phi."""
    rays = lum.shape[1]
    a, b, c = lum @ (r * np.cos(phi)), lum @ (r * np.sin(phi)), lum @ h
    cos_s, sin_s = np.cos(spin), np.sin(spin)
    vector = np.pi / rays * (((a * cos_s - b * sin_s)[:, None] * tangent)
                             + ((a * sin_s + b * cos_s)[:, None] * bitangent) + c[:, None] * n)
    return vector, np.pi / rays * lum.sum(axis=1)


def _unit(direction):
    d = np.float64(direction)
    return d / np.linalg.norm(d)


def sun_rays(scene, job, usable, n):
    """(query, sun, origin, direction, cosine) of every sun ray the CPU pass casts, ordered by sun then query."""
    pos = scene['query_pos'].astype(np.float64)
    visible = sun_facing_queries(scene, job['suns'])
    query, sun, cosine, direction = [], [], [], []
    for k, s in enumerate(job['suns']):
        d = _unit(s['direction'])
        c = n @ d
        rows = np.flatnonzero(usable & (c > 0) & np.isin(scene['query_class'], s['classes']) & visible[:, k])
        query.append(rows)
        sun.append(np.full(len(rows), k, np.int64))
        cosine.append(c[rows])
        direction.append(np.tile(d, (len(rows), 1)))
    query, sun = np.concatenate(query + [np.empty(0, np.int64)]), np.concatenate(sun + [np.empty(0, np.int64)])
    direction = np.concatenate(direction + [np.empty((0, 3))])
    return (query, sun, pos[query] + RAY_EPSILON * direction, direction,
            np.concatenate(cosine + [np.empty(0)]))


def edge_rays(scene, job):
    """(edge, sun, origin, direction) of every edge probe's sun ray the CPU pass casts, ordered by sun then edge."""
    edge_pos = scene.get('edge_pos', np.empty((0, 3), np.float32)).astype(np.float64)
    normals = scene.get('edge_normal', np.empty((0, 3), np.float32)).astype(np.float64)
    length = np.linalg.norm(normals, axis=1)
    usable = np.isfinite(length) & (length > 1e-12)
    normals = normals / np.where(usable, length, 1)[:, None]
    edge, sun, direction = [], [], []
    for k, s in enumerate(job['suns']):
        d = _unit(s['direction'])
        rows = np.flatnonzero(usable & (normals @ d > 0) & scene['edge_active'][:, k]
                              & np.isin(scene['edge_class'], s['classes']))
        edge.append(rows)
        sun.append(np.full(len(rows), k, np.int64))
        direction.append(np.tile(d, (len(rows), 1)))
    edge, sun = np.concatenate(edge + [np.empty(0, np.int64)]), np.concatenate(sun + [np.empty(0, np.int64)])
    direction = np.concatenate(direction + [np.empty((0, 3))])
    return edge, sun, edge_pos[edge] + RAY_EPSILON * direction, direction


# ---------- the pass on the GPU

class GpuPassError(RuntimeError):
    pass


@contextmanager
def _stage(name):
    try:
        yield
    except GpuPassError:
        raise
    except Exception as error:
        raise GpuPassError(f'{name}: {error}') from error


def dispatch_cap(device):
    return min(RAYS_PER_DISPATCH, device.info.limits.max_compute_dispatch_thread_groups.x * GROUP_SIZE)


def _create_buffer(device, **kwargs):
    return device.create_buffer(**kwargs)


def _dispatch(kernel, thread_count, **kwargs):
    kernel.dispatch(thread_count=[thread_count, 1, 1], **kwargs)


def _upload(device, array, dtype, writable=False):
    """A structured buffer of array's rows; an empty array becomes one zero row, which the shader never reads."""
    import slangpy as spy
    array = np.ascontiguousarray(array, dtype)
    if array.size == 0:
        array = np.zeros((1,) + array.shape[1:], dtype)
    usage = spy.BufferUsage.shader_resource | (spy.BufferUsage.unordered_access if writable else 0)
    return _create_buffer(device, element_count=len(array), struct_size=array.nbytes // len(array), usage=usage,
                          data=array)


def _read(device, buffer, dtype):
    device.wait_for_idle()
    return buffer.to_numpy().view(dtype)


def _scene_arrays(scene, textures, pages):
    """The triangles of the receivers then the occluders as the arrays the shader reads (one triangle id space)."""
    n_vertices = len(scene['positions'])
    n_recv, n_occ = len(scene['tris']), len(scene['occ_tris'])
    positions = np.concatenate([scene['positions'], scene['occ_positions']]).astype(np.float64)
    tris = np.concatenate([scene['tris'], scene['occ_tris'] + n_vertices]).astype(np.int64)
    corners = positions[tris]
    emission = np.concatenate([scene['tri_emission'], scene['occ_tri_emission']])
    casts = np.concatenate([scene['tri_casts_shadow'], scene['occ_tri_casts_shadow']])
    cutout = np.concatenate([scene['tri_cutout'], scene['occ_tri_cutout']])
    # a see-through emitter among the occluders is still seen by the rays that gather light (a shadow ray passes it)
    glows = np.concatenate([np.zeros(n_recv, bool), emission[n_recv:].any(axis=1)])
    flags = cutout * 1 + casts * 2 + glows * 4 + (np.arange(n_recv + n_occ) < n_recv) * 8
    class_of = np.concatenate([scene['tri_class'], np.full(n_occ, -1)]).astype(np.int64)
    info = np.stack([np.concatenate([scene['tri_texture'], scene['occ_tri_texture']]), flags,
                     np.concatenate([scene['tri_page'], np.full(n_occ, -1)]), class_of], axis=1)
    texels = [(np.asarray(t, np.float32) / 255.0).reshape(-1, 4) for t in textures]
    page_texels = [np.asarray(p, np.float32).reshape(-1, 3) for p in pages]

    def table(images, flat):
        offsets = np.cumsum([0] + [len(a) for a in flat])[:len(flat)]
        return np.array([[o, im.shape[1], im.shape[0], 0] for o, im in zip(offsets, images)], np.int32).reshape(-1, 4)
    return dict(
        positions=positions, tris=tris, info=info, class_of=class_of,
        tint=np.concatenate([scene['tri_tint'], scene['occ_tri_tint']]), emission=emission,
        normal=np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]),
        tex_uv=np.concatenate([scene['tex_uv'], scene['occ_uv']]),
        lm_uv=np.concatenate([scene['lm_uv'], np.zeros((len(scene['occ_positions']), 2))]),
        texture_info=table(textures, texels), texture_texels=np.concatenate(texels) if texels else np.empty((0, 4)),
        page_info=table(pages, page_texels),
        page_texels=np.concatenate(page_texels) if page_texels else np.empty((0, 3)))


def _build_world(device, vertices, indices, vertex_count, index_count):
    """The bottom level structure over every triangle, opaque and built to trace fast, under one top level instance."""
    import slangpy as spy
    triangles = spy.AccelerationStructureBuildInputTriangles({
        'vertex_buffers': [vertices], 'vertex_format': spy.Format.rgb32_float, 'vertex_count': vertex_count,
        'vertex_stride': 12, 'index_buffer': indices, 'index_format': spy.IndexFormat.uint32,
        'index_count': index_count, 'flags': spy.AccelerationStructureGeometryFlags.opaque})
    bottom_desc = spy.AccelerationStructureBuildDesc({'inputs': [triangles],
                                                      'flags': spy.AccelerationStructureBuildFlags.prefer_fast_trace})
    sizes = device.get_acceleration_structure_sizes(bottom_desc)
    bottom = device.create_acceleration_structure(kind=spy.AccelerationStructureKind.bottom_level,
                                                  size=sizes.acceleration_structure_size)
    scratch = _create_buffer(device, size=sizes.scratch_size, usage=spy.BufferUsage.unordered_access)
    encoder = device.create_command_encoder()
    encoder.build_acceleration_structure(desc=bottom_desc, dst=bottom, src=None, scratch_buffer=scratch)
    device.submit_command_buffer(encoder.finish())
    device.wait_for_idle()
    instances = device.create_acceleration_structure_instance_list(1)
    instances.write(0, {'transform': spy.float3x4.identity(), 'instance_id': 0, 'instance_mask': 0xff,
                        'instance_contribution_to_hit_group_index': 0,
                        'flags': spy.AccelerationStructureInstanceFlags.none,
                        'acceleration_structure': bottom.handle})
    top_desc = spy.AccelerationStructureBuildDesc({'inputs': [instances.build_input_instances()]})
    sizes = device.get_acceleration_structure_sizes(top_desc)
    top = device.create_acceleration_structure(kind=spy.AccelerationStructureKind.top_level,
                                               size=sizes.acceleration_structure_size)
    scratch_top = _create_buffer(device, size=sizes.scratch_size, usage=spy.BufferUsage.unordered_access)
    encoder = device.create_command_encoder()
    encoder.build_acceleration_structure(desc=top_desc, dst=top, src=None, scratch_buffer=scratch_top)
    device.submit_command_buffer(encoder.finish())
    device.wait_for_idle()
    # (the top level structure refers to the bottom one, so the bottom one and what built them live as long as it does)
    return top, (bottom, scratch, scratch_top, instances)


def _kernels(device):
    import slangpy as spy
    defines = {'GROUP_SIZE': str(GROUP_SIZE), 'RAY_LENGTH': repr(float(RAY_LENGTH)),
               'RAY_EPSILON': repr(float(RAY_EPSILON)), 'PASS_LIMIT': str(PASS_LIMIT),
               'LUMA': 'float3(' + ', '.join(repr(float(x)) for x in LUMA) + ')'}
    session = device.create_slang_session(spy.SlangCompilerOptions({'include_paths': [str(SHADERS)],
                                                                      'defines': defines}))
    return [device.create_compute_kernel(session.load_program('vertex_pass', [name])) for name in ('gather', 'shadow')]


def _ambient_of(table, cls):
    cls = np.asarray(cls)
    return table[np.clip(cls + 1, 0, None)] * (cls >= -1)[:, None]


def _trace(device, arrays, table, job, hemi, queries, shadow_rays, reduce):
    """(carried rgb, escaped) per shadow ray from the kernels; each gather dispatch's luminances go to
    reduce(chunk, lum) as they land, so no matrix of every ray's luminance is ever held."""
    rays = int(job['vertex_rays'])
    ids, pos, n, tangent, bitangent, spin, cls = queries
    origins, dirs = shadow_rays
    carried, escaped = np.ones((len(origins), 3)), np.ones(len(origins), bool)
    if not len(arrays['tris']):
        return carried, escaped
    cap = dispatch_cap(device)
    if rays > cap:
        raise GpuPassError(f'gather: vertex_rays {rays} exceeds the dispatch cap {cap}')
    with _stage('upload'):
        vertices = _upload(device, arrays['positions'], np.float32)
        indices = _upload(device, arrays['tris'].reshape(-1), np.uint32)
        shared = dict(
            tri_info=_upload(device, arrays['info'], np.int32), tri_tint=_upload(device, arrays['tint'], np.float32),
            tri_emission=_upload(device, arrays['emission'], np.float32),
            tri_normal=_upload(device, arrays['normal'], np.float32), tri_vertices=indices,
            vertex_tex_uv=_upload(device, arrays['tex_uv'], np.float32),
            vertex_lm_uv=_upload(device, arrays['lm_uv'], np.float32),
            texture_info=_upload(device, arrays['texture_info'], np.int32),
            texture_texels=_upload(device, arrays['texture_texels'], np.float32),
            page_info=_upload(device, arrays['page_info'], np.int32),
            page_texels=_upload(device, arrays['page_texels'], np.float32),
            ambient=_upload(device, table, np.float32), ambient_rows=len(table),
            albedo_scale=float(job['albedo_scale']), albedo_fallback=float(job['albedo_fallback']))
    with _stage('build'):
        shared['world'], keep = _build_world(device, vertices, indices, len(arrays['positions']), arrays['tris'].size)
        gather, shadow = _kernels(device)
    with _stage('gather'):
        hemisphere_buffers = dict(hemi_phi=_upload(device, hemi[0], np.float32),
                                  hemi_r=_upload(device, hemi[1], np.float32),
                                  hemi_h=_upload(device, hemi[2], np.float32), rays=rays)
    step = cap // rays
    for first in range(0, len(ids), step):
        chunk = slice(first, first + step)
        with _stage('gather'):
            out = _upload(device, np.zeros(len(ids[chunk]) * rays), np.float32, writable=True)
            _dispatch(gather, len(ids[chunk]) * rays, vars=dict(
                shared, **hemisphere_buffers, query_pos=_upload(device, pos[chunk], np.float32),
                query_n=_upload(device, n[chunk], np.float32),
                query_tangent=_upload(device, tangent[chunk], np.float32),
                query_bitangent=_upload(device, bitangent[chunk], np.float32),
                query_spin=_upload(device, spin[chunk], np.float32), query_class=_upload(device, cls[chunk], np.int32),
                gather_count=len(ids[chunk]) * rays, lum_out=out))
            lum = _read(device, out, np.float32).reshape(-1, rays)[:len(ids[chunk])]
        reduce(chunk, lum)
    for first in range(0, len(origins), cap):
        chunk = slice(first, first + cap)
        with _stage('shadow'):
            out = _upload(device, np.zeros((len(origins[chunk]), 4)), np.float32, writable=True)
            _dispatch(shadow, len(origins[chunk]), vars=dict(
                shared, shadow_origin=_upload(device, origins[chunk], np.float32),
                shadow_dir=_upload(device, dirs[chunk], np.float32), shadow_count=len(origins[chunk]), shadow_out=out))
            result = _read(device, out, np.float32).reshape(-1, 4)[:len(origins[chunk])]
        carried[chunk], escaped[chunk] = result[:, :3], result[:, 3] > 0.5
    del keep
    return carried, escaped


def run_vertex_pass(scene, job, textures, pages, device):
    """The CPU pass's vector, irradiance, sun and edge reach and ray count: the rays are traced and shaded on the GPU
    and every float64 sum over them is added here."""
    rays = int(job['vertex_rays'])
    queries = len(scene['query_pos'])
    suns = job['suns']
    local, phi, r = hemisphere(rays)
    usable, n, tangent, bitangent, spin = query_frames(scene['query_normal'], job.get('seed', 0))
    qcls = scene['query_class'].astype(np.int64)
    arrays = _scene_arrays(scene, textures, pages)
    table = ambient_table(job, qcls, arrays['class_of'])
    sun_query, sun_index, sun_origin, sun_dir, sun_cos = sun_rays(scene, job, usable, n)
    if len(scene.get('edge_pos', ())):
        edge, edge_sun, edge_origin, edge_dir = edge_rays(scene, job)
    else:
        edge = edge_sun = np.empty(0, np.int64)
        edge_origin = edge_dir = np.empty((0, 3))
    ids = np.flatnonzero(usable)
    vector, irradiance = np.zeros((queries, 3)), np.zeros(queries)

    def reduce(chunk, lum):
        rows = ids[chunk]
        vector[rows], irradiance[rows] = gather_sums(lum, phi, r, local[:, 2], n[rows], tangent[rows],
                                                     bitangent[rows], spin[rows])
    carried, escaped = _trace(
        device, arrays, table, job, (phi, r, local[:, 2]),
        (ids, scene['query_pos'][ids], n[ids], tangent[ids], bitangent[ids], spin[ids], qcls[ids]),
        (np.concatenate([sun_origin, edge_origin]), np.concatenate([sun_dir, edge_dir])), reduce)
    # (the pass works in pi x tool units, the sun's strength carries pi and a gathered hit has no 1/pi, so the
    # ambient that tool adds as lum x normal takes pi too)
    flat = np.pi * (_ambient_of(table, qcls) @ LUMA) * usable
    vector += flat[:, None] * n
    irradiance += flat
    sun_lit = np.zeros((queries, len(suns)), bool)
    for k, sun in enumerate(suns):
        rows = np.flatnonzero((sun_index == k) & escaped[:len(sun_query)])
        strength = ((np.float64(sun['color']) * carried[rows]) @ LUMA) * float(sun['strength'])
        vector[sun_query[rows]] += (strength * sun_cos[rows])[:, None] * sun_dir[rows]
        irradiance[sun_query[rows]] += strength * sun_cos[rows]
        sun_lit[sun_query[rows], k] = True
    edge_sun_lit = np.zeros((len(scene.get('edge_pos', ())), len(suns)), bool)
    edge_sun_lit[edge, edge_sun] = escaped[len(sun_query):]
    total = len(ids) * rays + len(sun_query) + len(edge)
    return VertexPassResult(vector.astype(np.float32), irradiance.astype(np.float32), sun_lit, edge_sun_lit, total)


def load_and_run(work, pages, adapter):
    """The pass on the arrays a job folder holds (job.json, scene.npz, textures.npz), on a device opened for adapter."""
    from .gpu import open_device
    work = Path(work)
    job = json.loads((work / 'job.json').read_text())
    with np.load(work / 'scene.npz') as data:
        scene = {key: data[key] for key in data.files}
    with np.load(work / 'textures.npz') as data:
        textures = [data[f'tex_{k}'] for k in range(len(data.files))]
    device = open_device(adapter)
    try:
        return run_vertex_pass(scene, job, textures, pages, device)
    finally:
        device.close()
