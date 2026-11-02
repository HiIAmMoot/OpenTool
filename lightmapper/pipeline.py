"""Orchestrate stock-tag baking without loading Blender into the tag-writing process.

All expensive stages finish before any tag is written. Diagnostic source passes
and dry runs keep their work products but never replace a tag.

Ignore-normals pages take their direct sun from RGB corner throughputs, with no cosine. Blender restores
their two-closure diffuse share before averaging sub-texels, keeping the vertex pass's diffuse page unchanged.
"""

from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime
from typing import Callable
import hashlib
import json
import logging
from pathlib import Path, PureWindowsPath
import shutil
import tempfile
from time import perf_counter

import numpy as np

from . import cpu_vertex_job, gpu, kept, memory, vertex_gpu
from .blender_job import Scene, run_blender, run_blender_lamp_batch, run_blender_vertex_only, run_blender_glow, with_vertex_pass, write_job
from .charts import ChartSettings, chart_and_pack, load_charts, save_charts
from .combine import chart_blocks, plan_repack, repack_arrays, repack_layout, written_shape
from .errors import BakeError, KeptBakeError
from .formats.bitmap import write_bitmap
from .formats.bsp import apply_layout, point_clusters, read_geometry
from .formats.lights import SceneLight
from .formats.scenery import gather_lights, gather_occluders
from .formats.vertices import ENV, unpack_uncompressed
from .gather import BASE_ONLY_GROUPS, IGNORE_NORMALS, LAYOUT_GROUPS, SIMPLE_PARAMETERIZATION, TRANSPARENT_GROUPS, emission, gather_lighting, read_radiosity, surface_classes
from .pages import DenoiseWorker, finish_pages, quantise
from .progress import GLOW_PLANNED
from .radiosity import finalize, query_points
from .raster import coverage, texel_contribution, triangle_weights
from .settings import Settings
from .terminator import flat_light_term, flat_sun_term, prepare_boundaries, sun_term
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


def bsp_index(scenario_tag: Tag, name: str) -> tuple[int, str]:
    matches = [(i, entry.reference('structure_bsp')[1])
               for i, entry in enumerate(scenario_tag.root.block('structure_bsps'))
               if PureWindowsPath(entry.reference('structure_bsp')[1]).name == name]
    if len(matches) != 1:
        raise ValueError(f'expected one structure BSP named {name!r}, found {len(matches)}')
    return matches[0]


def find_bsp(tags: Path, scenario: str, name: str):
    tag = read_tag(resolve_reference_path(tags, 'scnr', scenario))
    index, path = bsp_index(tag, name)
    return tag, index, path


def chart_cache_key(input_file: Path, chart_s: ChartSettings, radiosity, nest: bool) -> str:
    digest = hashlib.sha256(Path(input_file).read_bytes())
    digest.update(json.dumps(asdict(chart_s), sort_keys=True).encode())
    # (a nested layout and a rectangle one of the same level are different pages)
    digest.update(b'nest' if nest else b'flat')
    # (the flags word is hashed whole, so a shader edit to any flag bit makes new charts; group order)
    digest.update(json.dumps([[int(r.flags), int(r.detail_level)] for r in radiosity]).encode())
    return digest.hexdigest()


def _write(tags, group, path, tag, backups=True):
    target = resolve_reference_path(tags, group, path)
    target.parent.mkdir(parents=True, exist_ok=True)
    write_tag(tag, target, backup=backups)
    return target


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
            cache[group, path] = (radiosity, shader, textures.texture_of(shader))
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


# the scenery lights tool would bake: one outside every cluster of the BSP lights nothing and is dropped
def scene_lights(scenario: Tag, tags: Path, bsp_index: int, bsp_tag: Tag) -> tuple[list[SceneLight], dict[str, int]]:
    lights, counts = gather_lights(scenario, tags, bsp_index)
    clusters = point_clusters(bsp_tag, np.array([l.position for l in lights], np.float64).reshape(-1, 3))
    lights = [l for l, cluster in zip(lights, clusters) if cluster >= 0]
    counts = {**counts, 'outside_bsp': len(clusters) - len(lights), 'baked': len(lights)}
    logger.info('scenery lights: %s', ', '.join(f'{k} {v}' for k, v in counts.items()))
    return lights, counts


def _scene(tags, scenario, bsp_index, geometry, pages, radiosity, classes, scenery_emission=True):
    textures = TextureSet(tags)
    shaders = [shader_texture(tags, *group.key[:2]) for group in geometry.groups]
    texture_ids = [textures.texture_of(shader) for shader in shaders]
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
    occ_ids = np.array([textures.texture_of(s) for s in occ_shaders], np.int32)
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
        occ_tri_owner=join([occluders.tri_owner, np.full(emit_count, -1, np.int32)], (0,), np.int32),
        query_pos=pos, query_normal=normal, query_class=query_class, query_inverse=inverse,
        tri_ignore_normals=tri_ignore_normals, query_ignore_normals=query_ignore_normals)
    return scene, inverse, textures


def _chart_settings(s: Settings) -> ChartSettings:
    return ChartSettings(**{f.name: getattr(s, f.name) for f in fields(ChartSettings)})


def _estimate_cache(cache: Path) -> Path:
    # (the estimate never nests: it must not overwrite, or be served from, the bake's nested layout at the same path)
    return cache.with_name(cache.stem + '.estimate' + cache.suffix)


def _charts(input_file: Path, geometry, radiosity, s: Settings, cache: Path, nest: bool) -> list:
    chart_s = _chart_settings(s)
    key = chart_cache_key(input_file, chart_s, radiosity, nest)
    pages = load_charts(cache, key)
    if pages is None:
        pages = chart_and_pack(geometry.groups, radiosity, chart_s, nest=nest)
        save_charts(cache, pages, key)
    return pages


def written_plan(pages, page_max_resolved: int, s: Settings):
    # the repack the write applies, a function of the bake's layout and settings alone, so a write again repeats it
    chart_s = _chart_settings(s)
    blocks = chart_blocks(pages, chart_s.padding)
    shape = written_shape(blocks, page_max_resolved, s.page_max == 'auto')
    return plan_repack(blocks, shape, chart_s.page_min, s.max_materials_per_page, s.max_pages), shape


def _kept_sources(work: Path, manifest: kept.Manifest, name: str):
    folder = kept.kept_dir(work)
    try:
        scenario_tag = read_tag(folder / kept.SCENARIO)
        bsp_tag = read_tag(folder / kept.SOURCE)
        template = read_tag(folder / kept.TEMPLATE) if manifest.has_template else None
        number, path = bsp_index(scenario_tag, name)
    except kept.DAMAGED as error:
        raise kept.refusal(work) from error
    return scenario_tag, bsp_tag, template, number, path


def _kept_baked(work: Path, manifest: kept.Manifest, page_count: int):
    # everything the finish reads from the kept bake, loaded before anything is written
    folder = kept.kept_dir(work)
    try:
        with np.load(folder / kept.SCENE) as data:
            scene = Scene(**{f.name: data[f.name] for f in fields(Scene) if f.name in data.files})
        job = json.loads((folder / kept.JOB).read_text())
        glow = kept.load_glow(work) if manifest.glow_device != 'off' else None
    except kept.DAMAGED as error:
        raise kept.refusal(work) from error
    result = kept.load_result(work)
    if len(result.pages) != page_count or (glow is not None and len(glow) != page_count):
        raise kept.refusal(work)
    return scene, job, result, glow


@dataclass
class SceneEstimate:
    counts: memory.SceneCounts
    pages: list[tuple[int, int]]
    # 9 * w * h over the pages with any triangle that ignores normals
    flagged_sub_texels: int = 0
    device: str = 'gpu'


def stage_inputs(estimate: SceneEstimate, ignore_normals_mode: str, machine, bounces: int = 1) -> memory.StageInputs:
    # (the glow run is skipped without bounces)
    glow = estimate.flagged_sub_texels if ignore_normals_mode == 'enhanced' and bounces > 0 else 0
    inputs = memory.StageInputs(estimate.counts, estimate.pages, estimate.device, glow)
    return replace(inputs, whole_page_tiles=memory.whole_page_tiles(inputs, machine))


def _scene_estimate(scene, pages, device: str) -> SceneEstimate:
    sizes = [(p.width, p.height) for p in pages]
    return SceneEstimate(memory.SceneCounts.from_scene(scene), sizes,
                         memory.flagged_sub_texels(scene.tri_page, scene.tri_ignore_normals, sizes), device)


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
    pages = _charts(input_file, geometry, radiosity, s, _estimate_cache(Path(charts)), nest=False)
    scene, _, _ = _scene(tags, scenario_tag, bsp_index, geometry, pages, radiosity, classes,
                         s.scenery_emission)
    # (no GPU probe: the setting alone says whether the CPU constants apply)
    return _scene_estimate(scene, pages, 'cpu' if s.device == 'cpu' else 'gpu')


def bake(tags: Path, scenario: str, bsp: str, s: Settings, out_bsp: str | None = None,
          out_scenario: str | None = None, work: Path | None = None, sources: str = 'all',
          charts: Path | None = None, dry_run: bool = False,
          progress: Callable[[str], None] | None = None, backups: bool = True, keep: bool = False,
          reuse: bool = False) -> BakeOutput:
    if keep and (sources != 'all' or dry_run):
        raise ValueError('a kept bake must be a full bake that writes tags')
    if keep and work is None:
        # (a kept bake in the temp folder is one latest() never finds)
        raise ValueError('a kept bake needs a work folder')
    if reuse and (keep or sources != 'all' or dry_run or work is None):
        raise ValueError('a kept bake is written again whole, from its own folder: no keep, dry run or partial sources')
    tags = Path(tags)
    start = last = perf_counter()
    seconds = {}
    warnings = []
    try:
        choice = gpu.probe_choice(s.device)
    except gpu.DeviceError as error:
        if not reuse:
            raise
        # (a write again runs no GPU stage: the device only picks its denoiser, so a vendor this machine lacks is no
        # reason to refuse)
        choice = gpu.probe_choice('cpu')
        warnings.append(f'{error} The CPU denoiser is used for this write.')
    # (a write again runs no GPU stage, so this machine's ray-query warnings say nothing about it)
    if not reuse:
        warnings.extend(choice.warnings)
    if not reuse:
        logger.info('devices: %s', choice.summary())
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
    if reuse:
        work = Path(work)
        manifest = kept.read_manifest(work)
        if (scenario, bsp) != (manifest.scenario, manifest.bsp):
            raise KeptBakeError('This kept bake is of another level.')
        differences = kept.setting_differences(manifest.settings, s)
        if differences:
            raise KeptBakeError("This bake was kept with other settings, so it can't be written again with these: "
                                + '; '.join(differences) + '. Bake again, or change them back.')
        # (the kept copies, not the tags as they are now: the pages belong to the BSP they were baked for)
        scenario_tag, bsp_tag, template, bsp_number, input_path = _kept_sources(work, manifest, bsp)
        input_file = kept.kept_dir(work) / kept.SOURCE
        vertex_device = manifest.vertex_device
    else:
        scenario_tag, bsp_number, input_path = find_bsp(tags, scenario, bsp)
        input_file = resolve_reference_path(tags, 'sbsp', input_path)
        bsp_tag = read_tag(input_file)
    count = len(bsp_tag.root.block('surfaces'))
    over_cap = count > MAX_BSP_SURFACES
    if over_cap:
        message = f'{count} BSP surfaces exceed stock cap {MAX_BSP_SURFACES}'
        if not dry_run:
            raise ValueError(message)
        logger.warning('%s; dry run continues for timing only', message)
    output_path = out_bsp or (manifest.output_path if reuse else input_path)
    work = (Path(work) if work is not None else Path(tempfile.gettempdir()) / 'halo_lightmapper'
            / f'{bsp}-{datetime.now():%Y%m%d-%H%M%S}')
    work.mkdir(parents=True, exist_ok=True)
    try:
        geometry = read_geometry(bsp_tag)
    except kept.DAMAGED as error:
        if not reuse:
            raise
        raise kept.refusal(work) from error
    radiosity = [read_radiosity(tags, *group.key[:2]) for group in geometry.groups]
    if not reuse:
        lighting = gather_lighting(scenario_tag, tags, s.sky_grid)
        classes = surface_classes(geometry, len(scenario_tag.root.block('skies')))
        group, path = bsp_tag.root.reference('lightmaps_bitmap')
        template = read_tag(resolve_reference_path(tags, group, path)) if group == 'bitm' and path else None
    stage('gather')
    begin('charts')
    if reuse:
        pages = load_charts(kept.kept_dir(work) / kept.CHARTS, chart_cache_key(input_file, _chart_settings(s), radiosity, True))
        if pages is None:
            # (the key hashes the shaders' flags, so a shader edit since the bake leaves the kept charts not fitting)
            raise KeptBakeError("The level's shaders changed since this bake was kept, so it can't be written again. "
                                "Bake it again.")
    else:
        pages = _charts(input_file, geometry, radiosity, s,
                        Path(charts) if charts is not None else work / 'charts.npz', nest=True)
    stage('charts')
    if keep:
        charts_file = Path(charts) if charts is not None else work / 'charts.npz'
        spare = work / 'charts-spare.npz'
        # (a charts file inside the kept folder goes with the folder, before it can be copied back in)
        if charts_file.resolve().is_relative_to(kept.kept_dir(work).resolve()):
            shutil.copyfile(charts_file, spare)
            charts_file = spare
        try:
            # (a folder that held a kept bake must not keep any of it: a bake that fails before its manifest would
            # otherwise leave the old manifest and result beside the new sources, and an old glow or template would
            # outlive a bake that has none)
            (work / kept.MANIFEST).unlink(missing_ok=True)
            shutil.rmtree(kept.kept_dir(work), ignore_errors=True)
            try:
                kept.kept_dir(work).mkdir()
            except FileExistsError as error:
                raise KeptBakeError(f'The old kept folder {kept.kept_dir(work)} could not be cleared '
                                    '(a file in it may be open in another program).') from error
            template_file = resolve_reference_path(tags, group, path) if template is not None else None
            kept.save_sources(work, input_file, template_file, resolve_reference_path(tags, 'scnr', scenario),
                              charts_file)
        finally:
            spare.unlink(missing_ok=True)
    begin('scene')
    if reuse:
        scene, job, result, kept_glow = _kept_baked(work, manifest, len(pages))
        inverse = scene.query_inverse
        main_job = job
        batches = 1 + max((light['batch'] for light in main_job['lights']), default=0)
        check, lights, light_counts = None, [], {}
    else:
        scene, inverse, textures = _scene(tags, scenario_tag, bsp_number, geometry, pages, radiosity, classes,
                                          s.scenery_emission)
        sizes = [(p.width, p.height) for p in pages]
        machine = memory.read_machine()
        stages_in = stage_inputs(_scene_estimate(scene, pages, 'cpu' if choice.setting == 'cpu' else 'gpu'),
                                 s.ignore_normals_mode, machine, s.bounces)
        check = memory.check_bake_memory(stages_in, machine)
        if not check.fits:
            logger.warning('%s', check.message)
            warnings.append(check.message)
        # (a calibration run bakes no lamps, so it neither reads nor counts them)
        lights, light_counts = scene_lights(scenario_tag, tags, bsp_number, bsp_tag) if sources == 'all' else ([], {})
        write_job(work, scene, sizes, lighting.suns,
                  lighting.ambient, s, sources=sources, textures=textures.arrays(), choice=choice,
                  vertex_pass='skip', lights=lights, whole_page_tiles=stages_in.whole_page_tiles,
                  glow_sub_texels=stages_in.glow_sub_texels)
    stage('scene')
    # (decided here, not at the page assembly, because the app lays its progress bar out for the glow run before the
    # Blender stage starts)
    enhanced = s.ignore_normals_mode == 'enhanced' and scene.tri_ignore_normals is not None and scene.tri_ignore_normals.any()
    glow_runs = enhanced and s.bounces > 0 and not reuse
    if not reuse:
        if glow_runs:
            begin(GLOW_PLANNED)
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

        main_job = json.loads((work / 'job.json').read_text())
        batches = 1 + max((light['batch'] for light in main_job['lights']), default=0)
        if batches > 1:
            # (the vertex passes read the pages: they must come after every batch, and a pass inside the main run would miss
            # the bounce of the others)
            assert main_job['vertex_pass'] == 'skip'
            for k in range(1, batches):
                begin(f'lamp batch {k} of {batches - 1}')
                added, added_diffuse, written = run_blender_lamp_batch(work, s, k)
                if any(not np.array_equal(a, b) for a, b in zip(written, result.written)):
                    warning = f'lamp batch {k} wrote other texels than the main bake'
                    logger.warning('%s', warning)
                    warnings.append(warning)
                result = replace(result, pages=[a + b for a, b in zip(result.pages, added)],
                                 diffuse_pages=[a + b for a, b in zip(result.diffuse_pages, added_diffuse)])
            stage('blender_lamp_batches')

        def cpu_pass():
            if s.cpu_vertex_backend == 'blender':
                return run_blender_vertex_only(work, s, result.diffuse_pages)
            try:
                vp, info = cpu_vertex_job.run_in_child(work, result.diffuse_pages, s.timeout_s)
            except cpu_vertex_job.CpuPassError as error:
                warning = f"compiled CPU vertex pass failed ({error}); falling back to Blender's"
                logger.warning('%s', warning)
                warnings.append(warning)
                return run_blender_vertex_only(work, s, result.diffuse_pages)
            seconds['vertex_compile'] = info['compile']
            logger.info('compiled CPU vertex pass: %s threads, %s layer, %.1f s compile',
                        info['threads'], info['threading_layer'], info['compile'])
            return vp

        begin('vertex')
        if choice.adapter:
            try:
                vp = vertex_gpu.run_in_child(work, result.diffuse_pages, choice.adapter, s.timeout_s)
            except (vertex_gpu.GpuPassError, gpu.DeviceError) as error:
                if choice.required:
                    raise BakeError(f'{choice.vendor} vertex pass failed: {error}') from error
                warning = f'GPU vertex pass failed ({error}); falling back to the CPU pass'
                logger.warning('%s', warning)
                warnings.append(warning)
                vp = cpu_pass()
            else:
                vertex_device = f'{choice.adapter.name} ({choice.adapter.backend})'
            result = with_vertex_pass(result, vp)
        else:
            result = with_vertex_pass(result, cpu_pass())
        # (the pass has its copy and the finish never reads these)
        result = replace(result, diffuse_pages=None)
        stage('vertex_pass')
        if keep:
            kept.save_baked(work)
            kept.save_result(work, result)
    begin('glow' if glow_runs else 'finish')
    raw_pages, glow_pages, masks, class_maps = [], [], [], []
    if reuse:
        # (the job's ambient is already weighted by w_amb)
        ambient = {int(k): np.asarray(v) for k, v in job['ambient'].items()}
    else:
        ambient = {k: np.asarray(v) * s.w_amb for k, v in lighting.ambient.items()} if sources == 'all' else {}
        job = json.loads((work / 'job.json').read_text())
    job_suns = job['suns']
    texel_scene = None
    if enhanced:
        with np.load((kept.kept_dir(work) if reuse else work) / 'scene.npz') as data:
            texel_scene = {key: data[key] for key in data.files}
    if job.get('ambiguous_twin_groups'):
        warnings.append(ambiguous_warning(job['ambiguous_twin_groups']))
    boundaries = (prepare_boundaries(scene, job_suns, s.reach, result.edge_sun_lit)
                  if result.vertex_sun_lit is not None else None)
    # (the glow scene is the bake scene's own charts: one page's owner map then serves the sun term and both gathers)
    one_grid = texel_scene is not None and all(
        np.array_equal(texel_scene[key], getattr(scene, name)) for key, name in
        (('lm_uv', 'lm_uv'), ('tris', 'tris'), ('tri_page', 'tri_page')))
    for i, page in enumerate(pages):
        rows = scene.tri_page == i
        mask, cls = coverage(scene.lm_uv, scene.tris[rows], scene.tri_class[rows], page.width, page.height)
        raw = result.pages[i]
        grid = triangle_weights(scene.lm_uv, scene.tris[rows], 3 * page.width, 3 * page.height) if one_grid else None
        if result.vertex_sun_lit is not None:
            raw = raw + sun_term(scene, job_suns, result.vertex_sun_lit, i, page.width, page.height,
                                 s.reach, edge_sun_lit=result.edge_sun_lit, boundaries=boundaries, grid=grid)
        if enhanced:
            owner = grid[0] if one_grid else None
            # (the weights are 450 MB on a big page and nothing below reads them)
            grid = None
            gathered = texel_contribution(texel_scene, i, page.width, page.height, result.texel_rgb, owner=owner)[0]
            raw = raw + gathered
            from .raster import flagged_source
            if not reuse:
                glow_pages.append(flagged_source(texel_scene, i, page.width, page.height, result.texel_rgb, owner=owner))
            owner = None
        elif scene.tri_ignore_normals is not None and scene.tri_ignore_normals[rows].any():
            raw = raw + flat_sun_term(scene, job_suns, result.vertex_sun_throughput,
                                     i, page.width, page.height)
            if result.light_flat is not None:
                raw = raw + flat_light_term(scene, result.light_flat, i, page.width, page.height)
        raw_pages.append(raw)
        masks.append(mask)
        class_maps.append(cls)
    # (nothing below reads the baked pages, and the loop's last page and gather would stay in these names)
    result = replace(result, pages=None)
    raw = gathered = texel_scene = None
    glow_device = manifest.glow_device if reuse else 'off'
    if reuse and kept_glow is not None:
        raw_pages = [raw + added for raw, added in zip(raw_pages, kept_glow)]
    if glow_runs:
        # (the glow pages are the light the flagged surfaces received: `raw_pages` hold their own emission too, which
        # would come back off them as albedo times emission)
        np.savez(work / 'raw_pages.npz', **{f'page_{i}': raw for i, raw in enumerate(raw_pages)})
        raw_pages = None
        started = perf_counter()
        glow, glow_device = run_blender_glow(work, s, glow_pages)
        glow_pages = None
        if keep:
            kept.save_glow(work, glow)
        seconds['blender_glow'] = perf_counter() - started
        with np.load(work / 'raw_pages.npz') as data:
            raw_pages = [data[f'page_{i}'] for i in range(len(pages))]
        (work / 'raw_pages.npz').unlink()
        # (the glow bake already zeroes the flagged sub-texels, so a mixed texel gets only its unflagged share)
        raw_pages = [raw + added for raw, added in zip(raw_pages, glow)]
        stage('glow')
        begin('finish')
    if reuse:
        glow_reason = manifest.glow_reason
    elif s.ignore_normals_mode == 'tool':
        glow_reason = 'Traditional ignore normals'
    elif not enhanced:
        glow_reason = 'no ignore-normals surfaces in this level'
    elif s.bounces == 0:
        glow_reason = 'bounces set to 0'
    else:
        glow_reason = ''
    if keep:
        # (before the finish and the write: a bake that fails there is still one to write again)
        kept.write_manifest(work, kept.Manifest(
            tags=str(Path(tags).resolve()), scenario=scenario, bsp=bsp, input_path=input_path,
            has_template=template is not None, vertex_device=vertex_device, glow_device=glow_device,
            glow_reason=glow_reason, settings=asdict(s), kept_at=datetime.now().isoformat(timespec='seconds'),
            output_path=output_path, output_scenario=out_scenario or ''))
        # (only a folder inside the level's kept folder holds older kept bakes: a work folder the caller named
        # may sit beside anything)
        if work.parent.name == kept.level_folder(Path(), tags, scenario, bsp).name:
            kept.prune(work)
        kept.drop_superseded(work)
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
    vectors = finalize(result.vertex_vector, result.vertex_irradiance, scene.query_normal,
                       s.radiosity_mode, s.radiosity_scale)[inverse]
    offset = 0
    for page in pages:
        for material in page.materials:
            material.lm['incident_radiosity'] = vectors[offset:offset + len(material.env)]
            offset += len(material.env)
    # (the bake ran on its own pages; the written bitmap holds the charts repacked, and everything the write and the
    # summary report about pages describes those)
    page_max_resolved = getattr(pages, 'page_max', 0) or (
        s.page_max if isinstance(s.page_max, int) else max(max(p.width, p.height) for p in pages))
    plan, page_shape = written_plan(pages, page_max_resolved, s)
    bake_sizes = [[p.width, p.height] for p in pages]
    float_pages = repack_arrays(float_pages, plan)
    # (quantised after the move: the dither follows the written position, so the bitmap is the written float page's)
    pixels = [quantise(finished, s) for finished in float_pages]
    masks = repack_arrays(masks, plan)
    pages = repack_layout(pages, plan)
    utilisation = [float(mask.mean()) for mask in masks]
    clipped = [float(np.mean(finished[mask].max(axis=-1) > 1)) if mask.any() else 0.
               for finished, mask in zip(float_pages, masks)]
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
    written, written_paths = {}, {}
    if sources == 'all' and not dry_run:
        written['bitmap'] = _write(tags, 'bitm', output_path, bitmap_tag, backups).stat().st_size
        written['bsp'] = _write(tags, 'sbsp', output_path, bsp_tag, backups).stat().st_size
        written_paths = {'bitmap': output_path, 'bsp': output_path}
        if out_scenario is not None:
            scenario_tag.root.block('structure_bsps')[bsp_number].set_reference('structure_bsp', 'sbsp', output_path)
            _write(tags, 'scnr', out_scenario, scenario_tag, backups)
    stage('write')
    seconds['total'] = perf_counter() - start
    (work / ('write_summary.json' if reuse else 'summary.json')).write_text(json.dumps({
        'pages': len(pages), 'sizes': [[p.width, p.height] for p in pages],
        'bake_sizes': bake_sizes,
        'page_max_resolved': page_max_resolved, 'page_shape': list(page_shape),
        'utilisation': utilisation, 'texels_per_unit': s.texels_per_unit,
        'clipped_fraction': clipped, 'device': result.device, 'seconds': seconds, 'written': written,
        'written_paths': written_paths,
        'devices': {**({} if reuse else choice.summary()), 'page_bake': result.device, 'vertex_pass': vertex_device,
                    'denoise': denoise_device, 'glow_pass': glow_device, 'glow_reason': glow_reason},
        'lights': {**light_counts, 'baked': len(main_job['lights']), 'batches': batches},
        'whole_page_tiles': bool(main_job.get('whole_page_tiles', False)),
        'warnings': warnings, 'vertex_rays_cast': result.vertex_rays_cast,
        'memory': None if check is None else memory.summary(stages_in, check),
        'rays': {'samples': result.samples_used, 'texel_samples': result.texel_samples_used,
                 'vertex_rays': s.vertex_rays if vertex_device != 'cpu' else s.resolved_cpu_vertex_rays()},
        'texel_samples': s.resolved_texel_samples(), 'settings': asdict(s)}, indent=2))
    return BakeOutput(bsp_tag, bitmap_tag, float_pages, mapping, work)
