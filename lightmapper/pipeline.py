"""Orchestrate stock-tag baking without loading Blender into the tag-writing process.

All expensive stages finish before any tag is written. Diagnostic source passes
and dry runs keep their work products but never replace a tag.

Ignore-normals pages take their direct sun from RGB corner throughputs, with no cosine. Blender restores
their two-closure diffuse share before averaging sub-texels, keeping the vertex pass's diffuse page unchanged.
"""

from dataclasses import asdict, dataclass, fields
from datetime import datetime
from typing import Callable
import hashlib
import json
import logging
from pathlib import Path, PureWindowsPath
import tempfile
from time import perf_counter

import numpy as np

from . import gpu, memory, vertex_gpu
from .blender_job import Scene, run_blender, run_blender_vertex_only, run_blender_glow, with_vertex_pass, write_job
from .charts import ChartSettings, chart_and_pack, load_charts, save_charts
from .errors import BakeError
from .formats.bitmap import write_bitmap
from .formats.bsp import apply_layout, read_geometry
from .formats.scenery import gather_occluders
from .formats.vertices import ENV, unpack_uncompressed
from .gather import BASE_ONLY_GROUPS, IGNORE_NORMALS, LAYOUT_GROUPS, SIMPLE_PARAMETERIZATION, TRANSPARENT_GROUPS, emission, gather_lighting, read_radiosity, surface_classes
from .pages import DenoiseWorker, finish_pages, quantise
from .radiosity import finalize, query_points
from .raster import coverage, texel_contribution
from .settings import Settings
from .terminator import flat_sun_term, prepare_boundaries, sun_term
from .tagio.tagfile import Tag, read_tag, resolve_reference_path, write_tag
from .textures import TextureSet, shader_texture
from .twins import ambiguous_warning


MAX_BSP_SURFACES = 131072
logger = logging.getLogger(__name__)


@dataclass
class BakeOutput:
    bsp_tag: Tag
    bitmap_tag: Tag
    float_pages: list
    new_to_input: np.ndarray
    work: Path


def find_bsp(tags: Path, scenario: str, name: str):
    tag = read_tag(resolve_reference_path(tags, 'scnr', scenario))
    matches = [(i, entry.reference('structure_bsp')[1])
               for i, entry in enumerate(tag.root.block('structure_bsps'))
               if PureWindowsPath(entry.reference('structure_bsp')[1]).name == name]
    if len(matches) != 1:
        raise ValueError(f'expected one structure BSP named {name!r}, found {len(matches)}')
    index, path = matches[0]
    return tag, index, path


def chart_cache_key(input_file: Path, chart_s: ChartSettings, radiosity) -> str:
    digest = hashlib.sha256(Path(input_file).read_bytes())
    digest.update(json.dumps(asdict(chart_s), sort_keys=True).encode())
    # (the flags word is hashed whole, so a shader edit to any flag bit makes new charts; group order)
    digest.update(json.dumps([[int(r.flags), int(r.detail_level)] for r in radiosity]).encode())
    return digest.hexdigest()


def _write(tags, group, path, tag, backups=True):
    target = resolve_reference_path(tags, group, path)
    target.parent.mkdir(parents=True, exist_ok=True)
    write_tag(tag, target, backup=backups)


def copy_tags(tags: Path, scenario: str, bsp: str, out_bsp: str, out_scenario: str, backups=True):
    tags = Path(tags)
    scenario_tag, index, path = find_bsp(tags, scenario, bsp)
    bsp_tag = read_tag(resolve_reference_path(tags, 'sbsp', path))
    group, bitmap_path = bsp_tag.root.reference('lightmaps_bitmap')
    bitmap_tag = read_tag(resolve_reference_path(tags, group, bitmap_path))
    bsp_tag.root.set_reference('lightmaps_bitmap', 'bitm', out_bsp)
    scenario_tag.root.block('structure_bsps')[index].set_reference('structure_bsp', 'sbsp', out_bsp)
    _write(tags, 'bitm', out_bsp, bitmap_tag, backups)
    _write(tags, 'sbsp', out_bsp, bsp_tag, backups)
    _write(tags, 'scnr', out_scenario, scenario_tag, backups)


def _unlit_emitters(tags, geometry, textures):
    """Triangles of the final group's materials whose shader emits, as occluder arrays: they light the
    level but take no lightmap, and (unlike the group's solid surfaces) block nothing."""
    positions, uvs, texture_ids, cutouts, emissions, tints = [], [], [], [], [], []
    cache = {}
    for material in geometry.passthrough.block('materials'):
        group, path = material.reference('shader')
        if not path or group not in LAYOUT_GROUPS + BASE_ONLY_GROUPS:
            continue
        if (group, path) not in cache:
            radiosity = read_radiosity(tags, group, path)
            shader = shader_texture(tags, group, path)
            cache[group, path] = (radiosity, shader, textures.index(shader.albedo) if shader.albedo else -1)
        radiosity, shader, texture = cache[group, path]
        if radiosity.power <= 0:
            continue
        env, _ = unpack_uncompressed(material.data('uncompressed_vertices'),
                                     material.get('rendered_vertices.vertex_count'),
                                     material.get('lightmap_vertices.vertex_count'))
        start = material.get('surfaces')
        tris = geometry.surface_vertices[start:start + material.get('surface_count')]
        # (the same clockwise-to-counter-clockwise swap as the receivers)
        corners = tris[:, [0, 2, 1]]
        count = len(tris)
        positions.append(env['position'][corners].reshape(-1, 3))
        uvs.append((env['texcoord'][corners] * shader.uv_scale).reshape(-1, 2))
        texture_ids.append(np.full(count, texture, np.int32))
        cutouts.append(np.full(count, shader.cutout, np.bool_))
        emissions.append(np.tile(emission(radiosity), (count, 1)))
        tints.append(np.tile(radiosity.tint if group in TRANSPARENT_GROUPS else np.ones(3), (count, 1)))
    return positions, uvs, texture_ids, cutouts, emissions, tints


def _scene(tags, scenario, bsp_index, geometry, pages, radiosity, classes, scenery_emission=True):
    textures = TextureSet(tags)
    shaders = [shader_texture(tags, *group.key[:2]) for group in geometry.groups]
    texture_ids = [textures.index(shader.albedo) if shader.albedo else -1 for shader in shaders]
    envs, lm_uvs, tex_uvs, triangles = [], [], [], []
    tri_pages, tri_classes, shadows, emissions, tex_ids, cutouts, tints = [], [], [], [], [], [], []
    ignore_normals = []
    base = 0
    for p, page in enumerate(pages):
        for material in page.materials:
            group = material.group
            group_key = geometry.groups[group].key
            shader = shaders[group]
            count = len(material.tris)
            envs.append(material.env)
            lm_uvs.append(material.lm['texcoord'])
            tex_uvs.append(material.env['texcoord'] * shader.uv_scale)
            # (Halo winds clockwise relative to its normals and Blender expects counter-clockwise, so Blender gets (a, c, b); the tag keeps tool's order)
            triangles.append(material.tris[:, [0, 2, 1]].astype(np.int32) + base)
            base += len(material.env)
            tri_pages.append(np.full(count, p, np.int16))
            tri_classes.append(classes[material.surface_ids])
            ignore_normals.append(np.full(count, bool(radiosity[group].flags & IGNORE_NORMALS), np.bool_))
            shadows.append(np.full(count, shader.casts_shadow, np.bool_))
            emissions.append(np.tile(emission(radiosity[group]), (count, 1)))
            tex_ids.append(np.full(count, texture_ids[group], np.int32))
            cutouts.append(np.full(count, shader.cutout, np.bool_))
            # (a solid surface blocks outright, so only a see-through one keeps its shader's tint)
            see_through = group_key[0] in TRANSPARENT_GROUPS and not shader.casts_shadow
            tints.append(np.tile(radiosity[group].tint if see_through else np.ones(3), (count, 1)))

    def join(items, shape, dtype):
        return np.concatenate(items).astype(dtype, copy=False) if items else np.empty(shape, dtype)

    env = join(envs, (0,), ENV)
    tris = join(triangles, (0, 3), np.int32)
    tri_class = join(tri_classes, (0,), np.int16)
    pos, normal, query_class, inverse = query_points(env, tris, tri_class)
    tri_ignore_normals = join(ignore_normals, (0,), np.bool_)
    query_ignore_normals = np.zeros(len(pos), np.bool_)
    # (a welded query takes the flag from any owner, not only the first triangle)
    query_ignore_normals[inverse[tris[tri_ignore_normals]].ravel()] = True
    occluders = gather_occluders(scenario, tags, bsp_index)
    occ_shaders = [shader_texture(tags, group, path, (u, v))
                   for group, path, u, v in occluders.shaders]
    occ_ids = np.array([textures.index(s.albedo) if s.albedo else -1 for s in occ_shaders], np.int32)
    occ_radiosity = {}
    occ_emission = []
    for group, path, _, _ in occluders.shaders:
        if scenery_emission and path and group in LAYOUT_GROUPS + BASE_ONLY_GROUPS:
            key = (group, path)
            if key not in occ_radiosity:
                occ_radiosity[key] = read_radiosity(tags, group, path)
            occ_emission.append(emission(occ_radiosity[key]))
        else:
            occ_emission.append(np.zeros(3))
    occ_emission = np.asarray(occ_emission, np.float32).reshape(-1, 3)
    # corners can share a model vertex but use shaders with different UV scales
    # (model strips are clockwise relative to their normals too, as the BSP is)
    occ_corners = occluders.tris[:, [0, 2, 1]]
    occ_positions = occluders.positions[occ_corners].reshape(-1, 3)
    scales = np.array([s.uv_scale for s in occ_shaders], np.float32).reshape(-1, 2)
    occ_uv = (occluders.uv[occ_corners] * scales[occluders.tri_shader, None, :]).reshape(-1, 2)
    unlit = _unlit_emitters(tags, geometry, textures)
    emit_positions, emit_uvs, emit_textures, emit_cutouts, emit_emission, emit_tints = unlit
    occ_positions = join([occ_positions, *emit_positions], (0, 3), np.float32)
    occ_uv = join([occ_uv, *emit_uvs], (0, 2), np.float32)
    emit_count = sum(len(e) for e in emit_emission)
    scenery_count = len(occluders.tris)
    scene = Scene(
        positions=env['position'], normals=env['normal'],
        lm_uv=join(lm_uvs, (0, 2), np.float32), tex_uv=join(tex_uvs, (0, 2), np.float32),
        tris=tris, tri_page=join(tri_pages, (0,), np.int16), tri_class=tri_class,
        tri_casts_shadow=join(shadows, (0,), np.bool_),
        tri_emission=join(emissions, (0, 3), np.float32),
        tri_texture=join(tex_ids, (0,), np.int32), tri_cutout=join(cutouts, (0,), np.bool_),
        tri_tint=join(tints, (0, 3), np.float32),
        occ_positions=occ_positions, occ_uv=occ_uv,
        occ_tris=np.arange(len(occ_positions), dtype=np.int32).reshape(-1, 3),
        occ_tri_texture=join([occ_ids[occluders.tri_shader], *emit_textures], (0,), np.int32),
        occ_tri_cutout=join([np.array([s.cutout for s in occ_shaders], np.bool_)[occluders.tri_shader],
                             *emit_cutouts], (0,), np.bool_),
        occ_tri_casts_shadow=np.concatenate([
            np.array([s.casts_shadow for s in occ_shaders], np.bool_)[occluders.tri_shader],
            np.zeros(emit_count, np.bool_)]),
        occ_tri_emission=join([occ_emission[occluders.tri_shader], *emit_emission], (0, 3), np.float32),
        occ_tri_tint=join([np.ones((scenery_count, 3)), *emit_tints], (0, 3), np.float32),
        query_pos=pos, query_normal=normal, query_class=query_class, query_inverse=inverse,
        tri_ignore_normals=tri_ignore_normals, query_ignore_normals=query_ignore_normals)
    return scene, inverse, textures


def _charts(input_file: Path, geometry, radiosity, s: Settings, cache: Path) -> list:
    chart_s = ChartSettings(**{f.name: getattr(s, f.name) for f in fields(ChartSettings)})
    key = chart_cache_key(input_file, chart_s, radiosity)
    pages = load_charts(cache, key)
    if pages is None:
        pages = chart_and_pack(geometry.groups, radiosity, chart_s)
        save_charts(cache, pages, key)
    return pages


@dataclass
class SceneEstimate:
    counts: memory.SceneCounts
    pages: list[tuple[int, int]]


def estimate_scene(tags: Path, scenario: str, bsp: str, s: Settings, charts: Path) -> SceneEstimate:
    tags = Path(tags)
    scenario_tag, bsp_index, input_path = find_bsp(tags, scenario, bsp)
    input_file = resolve_reference_path(tags, 'sbsp', input_path)
    bsp_tag = read_tag(input_file)
    count = len(bsp_tag.root.block('surfaces'))
    if count > MAX_BSP_SURFACES:
        raise ValueError(f'{count} BSP surfaces exceed stock cap {MAX_BSP_SURFACES}')
    geometry = read_geometry(bsp_tag)
    radiosity = [read_radiosity(tags, *group.key[:2]) for group in geometry.groups]
    classes = surface_classes(geometry, len(scenario_tag.root.block('skies')))
    pages = _charts(input_file, geometry, radiosity, s, Path(charts))
    scene, _, _ = _scene(tags, scenario_tag, bsp_index, geometry, pages, radiosity, classes,
                         s.scenery_emission)
    return SceneEstimate(memory.SceneCounts.from_scene(scene), [(p.width, p.height) for p in pages])


def bake(tags: Path, scenario: str, bsp: str, s: Settings, out_bsp: str | None = None,
          out_scenario: str | None = None, work: Path | None = None, sources: str = 'all',
          charts: Path | None = None, dry_run: bool = False,
          progress: Callable[[str], None] | None = None, backups: bool = True) -> BakeOutput:
    tags = Path(tags)
    start = last = perf_counter()
    seconds = {}
    choice = gpu.probe_choice(s.device)
    logger.info('devices: %s', choice.summary())
    warnings = list(choice.warnings)
    for warning in warnings:
        logger.warning('%s', warning)
    vertex_device = 'cpu'

    def stage(name):
        nonlocal last
        now = perf_counter()
        seconds[name] = now - last
        last = now

    def begin(name):
        logger.info('stage: %s', name)
        if progress is not None:
            progress(name)

    stage('probe')
    begin('gather')
    scenario_tag, bsp_index, input_path = find_bsp(tags, scenario, bsp)
    input_file = resolve_reference_path(tags, 'sbsp', input_path)
    bsp_tag = read_tag(input_file)
    count = len(bsp_tag.root.block('surfaces'))
    over_cap = count > MAX_BSP_SURFACES
    if over_cap:
        message = f'{count} BSP surfaces exceed stock cap {MAX_BSP_SURFACES}'
        if not dry_run:
            raise ValueError(message)
        logger.warning('%s; dry run continues for timing only', message)
    output_path = out_bsp or input_path
    work = (Path(work) if work is not None else Path(tempfile.gettempdir()) / 'halo_lightmapper'
            / f'{bsp}-{datetime.now():%Y%m%d-%H%M%S}')
    work.mkdir(parents=True, exist_ok=True)
    geometry = read_geometry(bsp_tag)
    lighting = gather_lighting(scenario_tag, tags, s.sky_grid)
    radiosity = [read_radiosity(tags, *group.key[:2]) for group in geometry.groups]
    classes = surface_classes(geometry, len(scenario_tag.root.block('skies')))
    group, path = bsp_tag.root.reference('lightmaps_bitmap')
    template = read_tag(resolve_reference_path(tags, group, path)) if group == 'bitm' and path else None
    stage('gather')
    begin('charts')
    pages = _charts(input_file, geometry, radiosity, s,
                    Path(charts) if charts is not None else work / 'charts.npz')
    stage('charts')
    begin('scene')
    scene, inverse, textures = _scene(tags, scenario_tag, bsp_index, geometry, pages, radiosity, classes,
                                      s.scenery_emission)
    try:
        check = memory.check_bake_memory(memory.SceneCounts.from_scene(scene),
                                         [(p.width, p.height) for p in pages], s.page_max)
    except memory.BakeMemoryError as error:
        if not dry_run:
            raise
        logger.warning('%s; dry run continues for timing only', error)
        warnings.append(str(error))
        check = memory.MemoryCheck(error.estimate, error.free)
    write_job(work, scene, [(p.width, p.height) for p in pages], lighting.suns,
              lighting.ambient, s, sources=sources, textures=textures.arrays(), choice=choice,
              vertex_pass='skip' if choice.adapter else 'cpu')
    stage('scene')
    begin('blender')
    result = run_blender(work, s)
    stage('blender')
    blender_stages = result.stages or {}
    seconds['blender_start'] = blender_stages.get('start', 0.0)
    seconds['blender_scene'] = blender_stages.get('scene', 0.0)
    seconds['page_bake'] = blender_stages.get('pages', 0.0)
    seconds['blender_vertex'] = blender_stages.get('vertex', 0.0)
    if choice.cycles and result.device == 'CPU':
        warning = 'cycles fell back to the CPU'
        logger.warning('%s', warning)
        warnings.append(warning)
    if choice.adapter:
        begin('vertex')
        try:
            vp = vertex_gpu.run_in_child(work, result.diffuse_pages, choice.adapter, s.timeout_s)
        except (vertex_gpu.GpuPassError, gpu.DeviceError) as error:
            if choice.required:
                raise BakeError(f'{choice.vendor} vertex pass failed: {error}') from error
            warning = f'GPU vertex pass failed ({error}); falling back to the CPU pass'
            logger.warning('%s', warning)
            warnings.append(warning)
            vp = run_blender_vertex_only(work, s, result.diffuse_pages)
        else:
            vertex_device = f'{choice.adapter.name} ({choice.adapter.backend})'
        result = with_vertex_pass(result, vp)
    stage('vertex_pass')
    begin('finish')
    ambient = {k: np.asarray(v) * s.w_amb for k, v in lighting.ambient.items()} if sources == 'all' else {}
    raw_pages, glow_pages, masks, class_maps, utilisation, clipped = [], [], [], [], [], []
    job = json.loads((work / 'job.json').read_text())
    job_suns = job['suns']
    enhanced = s.ignore_normals_mode == 'enhanced' and scene.tri_ignore_normals is not None and scene.tri_ignore_normals.any()
    texel_scene = None
    if enhanced:
        with np.load(work / 'scene.npz') as data:
            texel_scene = {key: data[key] for key in data.files}
    if job.get('ambiguous_twin_groups'):
        warnings.append(ambiguous_warning(job['ambiguous_twin_groups']))
    boundaries = (prepare_boundaries(scene, job_suns, s.reach, result.edge_sun_lit)
                  if result.vertex_sun_lit is not None else None)
    for i, page in enumerate(pages):
        rows = scene.tri_page == i
        mask, cls = coverage(scene.lm_uv, scene.tris[rows], scene.tri_class[rows], page.width, page.height)
        raw = result.pages[i]
        if result.vertex_sun_lit is not None:
            raw = raw + sun_term(scene, job_suns, result.vertex_sun_lit, i, page.width, page.height,
                                 s.reach, edge_sun_lit=result.edge_sun_lit, boundaries=boundaries)
        if enhanced:
            gathered = texel_contribution(texel_scene, i, page.width, page.height, result.texel_rgb)[0]
            raw = raw + gathered
            from .raster import flagged_source
            glow_pages.append(flagged_source(texel_scene, i, page.width, page.height, result.texel_rgb))
        elif scene.tri_ignore_normals is not None and scene.tri_ignore_normals[rows].any():
            raw = raw + flat_sun_term(scene, job_suns, result.vertex_sun_throughput,
                                     i, page.width, page.height)
        raw_pages.append(raw)
        masks.append(mask)
        class_maps.append(cls)
        utilisation.append(float(mask.mean()))
    if enhanced and s.bounces > 0:
        # (the glow pages are the light the flagged surfaces received: `raw_pages` hold their own emission too, which
        # would come back off them as albedo times emission)
        started = perf_counter()
        glow = run_blender_glow(work, s, glow_pages)
        seconds['blender_glow'] = perf_counter() - started
        # (the glow bake already zeroes the flagged sub-texels, so a mixed texel gets only its unflagged share)
        raw_pages = [raw + added for raw, added in zip(raw_pages, glow)]
    # (OIDN's HIP device cannot open on the Radeon 680M, so AMD denoises on the CPU with a warning even when `amd` is
    # named; owner decision 2026-10-26)
    denoise_worker = DenoiseWorker.start(choice.oidn, s.timeout_s,
                                         strict=choice.required and choice.oidn != 'HIP') if s.denoise else None
    denoise_device = 'off' if denoise_worker is None else denoise_worker.device
    denoise_seconds = 0.0
    try:
        float_pages = finish_pages(pages, raw_pages, masks, class_maps, ambient, s, written=result.written,
                                   joined_groups={i for i, r in enumerate(radiosity) if r.flags & SIMPLE_PARAMETERIZATION},
                                   worker=denoise_worker)
    finally:
        if denoise_worker is not None:
            denoise_worker.close()
            denoise_seconds = denoise_worker.seconds
            denoise_device = denoise_worker.device
            warnings.extend(denoise_worker.warnings)
    pixels = [quantise(finished, s) for finished in float_pages]
    for finished, mask in zip(float_pages, masks):
        clipped.append(float(np.mean(finished[mask].max(axis=-1) > 1)) if mask.any() else 0.)
    vectors = finalize(result.vertex_vector, result.vertex_irradiance, scene.query_normal,
                       s.radiosity_mode, s.radiosity_scale)[inverse]
    offset = 0
    for page in pages:
        for material in page.materials:
            material.lm['incident_radiosity'] = vectors[offset:offset + len(material.env)]
            offset += len(material.env)
    stage('finish')
    seconds['denoise'] = denoise_seconds
    seconds['finish'] -= denoise_seconds
    begin('write')
    if over_cap:
        # apply_layout enforces the real cap too; a timing-only run cannot make a stock tag
        provenance = [m.surface_ids for p in pages for m in p.materials]
        provenance.extend(np.arange(m.get('surfaces'), m.get('surfaces') + m.get('surface_count'), dtype=np.int32)
                          for m in geometry.passthrough.block('materials'))
        mapping = np.concatenate(provenance) if provenance else np.empty(0, np.int32)
    else:
        mapping = apply_layout(bsp_tag, geometry, pages, output_path)
    bitmap_tag = write_bitmap(pixels, template)
    np.save(work / 'new_to_input.npy', mapping)
    np.savez(work / 'float_pages.npz', **{f'page_{i}': page for i, page in enumerate(float_pages)})
    if sources == 'all' and not dry_run:
        _write(tags, 'bitm', output_path, bitmap_tag, backups)
        _write(tags, 'sbsp', output_path, bsp_tag, backups)
        if out_scenario is not None:
            scenario_tag.root.block('structure_bsps')[bsp_index].set_reference('structure_bsp', 'sbsp', output_path)
            _write(tags, 'scnr', out_scenario, scenario_tag, backups)
    stage('write')
    seconds['total'] = perf_counter() - start
    (work / 'summary.json').write_text(json.dumps({
        'pages': len(pages), 'sizes': [[p.width, p.height] for p in pages],
        'utilisation': utilisation, 'texels_per_unit': s.texels_per_unit,
        'clipped_fraction': clipped, 'device': result.device, 'seconds': seconds,
        'devices': {**choice.summary(), 'page_bake': result.device, 'vertex_pass': vertex_device,
                    'denoise': denoise_device},
        'warnings': warnings, 'vertex_rays_cast': result.vertex_rays_cast,
        'memory': {'estimate_gb': round(check.estimate / memory.GB, 2),
                   'free_gb': round(check.free / memory.GB, 2) if check.free is not None else None},
        'rays': {'samples': result.samples_used, 'texel_samples': result.texel_samples_used,
                 'vertex_rays': s.vertex_rays if vertex_device != 'cpu' else s.resolved_cpu_vertex_rays()},
        'texel_samples': s.resolved_texel_samples(), 'settings': asdict(s)}, indent=2))
    return BakeOutput(bsp_tag, bitmap_tag, float_pages, mapping, work)
