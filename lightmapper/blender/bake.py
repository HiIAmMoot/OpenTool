"""Runs inside Blender only (blender -b --python bake.py -- job.json): builds the job's scene and bakes the lightmap pages.

This script imports Blender's Python API and needs a GPL-compatible license. It exchanges data with the
pipeline through job.json, scene.npz, textures.npz and out.npz. The page bakes are Cycles diffuse + emission
passes into one float image per page; rows come back in our convention (row 0 = v 0). The per-vertex pass
runs here only on the CPU path or a vertex-only relaunch: Blender's BVH tree casts the rays and numpy
shades the hits in batches.

Ignore-normals queries gather uniformly over the whole sphere with weight 4pi/N and take reached suns
at full strength from either side. Their ambient stays the same as an ordinary query's.

The two-closure diffuse bake is restored on flagged sub-texels before averaging, including edge top-ups.
The vertex pass still receives the uncorrected diffuse page; mixed texels never scale a neighbour's light.
"""

from dataclasses import dataclass, replace
import json
import math
import os
import sys
import time

import bpy
import mathutils
from mathutils.bvhtree import BVHTree
import numpy as np


DEVICE_TYPES = ('OPTIX', 'CUDA', 'HIP', 'ONEAPI')
UV_LIGHTMAP = 'lightmap'
UV_TEXTURE = 'texture'
RAY_LENGTH = 1e5
RAY_EPSILON = 1e-3
# coincident reverse faces self-shadow in Cycles; separate only bake copies, including at far-out map positions
TWIN_SEPARATION = 1e-4
# the dominant normal axis carries at least 1/sqrt(3) of the move, giving over two float32 steps of rounding margin
TWIN_SEPARATION_STEPS = 4
PASS_LIMIT = 8
# owner triangles a lamp's shadow ray walks through before the ray ends as a miss
SKIP_LIMIT = 256
# lamp shadow rays are cast one by one here, so a block is small
LAMP_RAY_CHUNK = 2 ** 14
LUMA = np.float64([0.2126, 0.7152, 0.0722])
GOLDEN_FRACTION = 0.618034
SEED_STRIDE = 1000
# a lamp batch's bake takes its own noise: summed batches sharing a seed would share their noise
LAMP_BATCH_SEED_STEP = 100
SEED_SPIN = 0.7548776662466927
# queries per batch: bounds the hit arrays held before they are shaded
QUERY_CHUNK = 256
# measured Diffuse + Translucent / Diffuse-only bake ratio: 0.5011076 (Blender 5.2)
BACK_SIDE_SHARE = 0.5
# a radius-0 point light of power P bakes P cos / (4 pi^2 d^2) (1 W measured, Blender 5.2); 4 pi W makes
# strength cos / (pi d^2), the same units as a sun of that strength
LAMP_WATTS = 4 * math.pi


def log(message):
    print(f'[bake] {message}', flush=True)


def pick_device(kinds: list[str], required: bool, hiprt: bool) -> str:
    """Enable every device of the first working GPU type and give its name; 'CPU' when none, or raise when required."""
    prefs = bpy.context.preferences.addons['cycles'].preferences
    for kind in kinds:
        try:
            prefs.compute_device_type = kind
        except TypeError:
            continue
        prefs.refresh_devices()
        devices = [d for d in prefs.devices if d.type == kind]
        if not devices:
            continue
        for device in prefs.devices:
            device.use = device.type == kind
        if kind == 'HIP' and hasattr(prefs, 'use_hiprt'):
            prefs.use_hiprt = hiprt
        bpy.context.scene.cycles.device = 'GPU'
        return kind
    if required and kinds:
        raise RuntimeError(f'cycles: none of {kinds} is available')
    if kinds:
        log('WARNING: no GPU device found, baking on the CPU')
    bpy.context.scene.cycles.device = 'CPU'
    return 'CPU'


def cpu_counts(job: dict, device: str) -> dict:
    counts = job.copy()
    if device == 'CPU':
        for name in ('samples', 'texel_samples'):
            if f'cpu_{name}' in job:
                counts[name] = job[f'cpu_{name}']
    return counts


def setup_scene(job):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    cycles = scene.cycles
    wanted = job.get('device', 'OPTIX')
    kinds = list(DEVICE_TYPES[DEVICE_TYPES.index(wanted):]) if wanted in DEVICE_TYPES else []
    device = pick_device(job.get('cycles_devices', kinds), job.get('cycles_required', False),
                         job.get('hiprt', False))
    job.update(cpu_counts(job, device))
    log(f'samples {job["samples"]}, texel samples {job["texel_samples"]}')
    cycles.samples = max(1, round(job['samples'] / 9))
    cycles.seed = job.get('seed', 0) * SEED_STRIDE + LAMP_BATCH_SEED_STEP * job.get('lamp_batch', 0)
    log(f'cycles seed {cycles.seed}')
    cycles.max_bounces = cycles.diffuse_bounces = job['bounces']
    cycles.glossy_bounces = cycles.transmission_bounces = cycles.volume_bounces = 0
    cycles.use_denoising = False
    cycles.use_adaptive_sampling = bool(job.get('adaptive_sampling', True))
    # (the factory clamp of 10 clips bright bounce near emitters and lamps, and would make summed lamp batches differ from one bake)
    cycles.sample_clamp_direct = 0
    # test_clamp_indirect is a test-only override, never written by write_job
    cycles.sample_clamp_indirect = job.get('test_clamp_indirect', 0)
    world = bpy.data.worlds.new('world')
    world.use_nodes = True
    background = world.node_tree.nodes['Background']
    background.inputs['Color'].default_value = (0, 0, 0, 1)
    background.inputs['Strength'].default_value = 0
    scene.world = world
    return scene, device


def make_images(textures_path, pages, masked=False):
    textures = []
    with np.load(textures_path) as data:
        for k in range(len(data.files)):
            rgba = np.asarray(data[f'tex_{k}'], np.float32) / 255.0
            h, w = rgba.shape[:2]
            image = bpy.data.images.new(f'tex_{k}', w, h, alpha=True)
            # tool reads the 8-bit value over 255 as the colour, with no sRGB curve
            image.colorspace_settings.name = 'Non-Color'
            image.alpha_mode = 'STRAIGHT'
            # Blender's row 0 is the bottom, ours is v = 0
            image.pixels.foreach_set(np.ascontiguousarray(rgba[::-1]).ravel())
            textures.append(image)
    page_images = []
    for i, (w, h) in enumerate(pages):
        # the 3x float buffer is only allocated while its page bakes (the 3x image alone is about 151 MB per 1024 page; the whole bake's peak is estimated in lightmapper/memory.py)
        image = bpy.data.images.new(f'page_{i}', 1, 1, alpha=False, float_buffer=True, is_data=True)
        image.colorspace_settings.name = 'Non-Color'
        page_images.append(image)
    mask_images = []
    if masked:
        for i in range(len(pages)):
            image = bpy.data.images.new(f'mask_{i}', 1, 1, alpha=False, float_buffer=True, is_data=True)
            image.colorspace_settings.name = 'Non-Color'
            reset_mask(image)
            mask_images.append(image)
    return textures, page_images, mask_images


def make_atlas_image():
    # one atlas image for the run, one texel until a tier sizes it
    image = bpy.data.images.new('atlas', 1, 1, alpha=False, float_buffer=True, is_data=True)
    image.colorspace_settings.name = 'Non-Color'
    return image


def size_page_image(image, w, h):
    image.generated_width, image.generated_height = 3 * w, 3 * h


def release_page_image(image):
    image.generated_width = image.generated_height = 1


def reset_mask(image):
    # one white texel reads as all ones everywhere (the node extends it), so a mask costs memory only while a top-up
    # pass needs a real one; a resized generated image comes back black, so the texel is written again
    image.generated_width = image.generated_height = 1
    image.pixels.foreach_set(np.ones(4, np.float32))


def make_material(name, texture, cutout, emission, job, page_image, tint=(), emit_only=False, mask_image=None,
                  both_sides=False, zero_albedo=False, glow_image=None, glow_transform=None, back_only=False):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes, links = material.node_tree.nodes, material.node_tree.links
    nodes.clear()
    output = nodes.new('ShaderNodeOutputMaterial')
    diffuse = nodes.new('ShaderNodeBsdfDiffuse')
    alpha_socket = None
    if texture is None:
        fallback = job['albedo_fallback']
        diffuse.inputs['Color'].default_value = (fallback, fallback, fallback, 1)
    else:
        uv = nodes.new('ShaderNodeUVMap')
        uv.uv_map = UV_TEXTURE
        sampled = nodes.new('ShaderNodeTexImage')
        sampled.image = texture
        sampled.extension = 'REPEAT'
        links.new(uv.outputs['UV'], sampled.inputs['Vector'])
        scale = nodes.new('ShaderNodeVectorMath')
        scale.operation = 'MULTIPLY'
        scale.inputs[1].default_value = (job['albedo_scale'],) * 3
        links.new(sampled.outputs['Color'], scale.inputs[0])
        links.new(scale.outputs['Vector'], diffuse.inputs['Color'])
        alpha_socket = sampled.outputs['Alpha']
    if zero_albedo:
        for link in list(diffuse.inputs['Color'].links):
            links.remove(link)
        diffuse.inputs['Color'].default_value = (0, 0, 0, 1)
    shader = diffuse.outputs['BSDF']
    if both_sides:
        back = nodes.new('ShaderNodeBsdfTranslucent')
        colour = diffuse.inputs['Color']
        if colour.is_linked:
            links.new(colour.links[0].from_socket, back.inputs['Color'])
        else:
            back.inputs['Color'].default_value = colour.default_value
        if back_only:
            # (the atlas bakes the behind half on its own copy: the DIFFUSE pass divides by the summed closure colour,
            # so the page's two-closure value is the mean of the halves)
            shader = back.outputs['BSDF']
        else:
            add = nodes.new('ShaderNodeAddShader')
            links.new(shader, add.inputs[0])
            links.new(back.outputs['BSDF'], add.inputs[1])
            shader = add.outputs['Shader']
    if emit_only:
        # unlightmapped emitters have no diffuse bounce: their back face must pass gather rays
        shader = nodes.new('ShaderNodeBsdfTransparent').outputs['BSDF']
    if max(emission) > 0:
        glow = nodes.new('ShaderNodeEmission')
        glow.inputs['Color'].default_value = (*[float(c) for c in emission], 1)
        # tool's emitters shine from their front side only (the side the vertex normals face)
        front = nodes.new('ShaderNodeMath')
        front.operation = 'SUBTRACT'
        front.inputs[0].default_value = 1
        links.new(nodes.new('ShaderNodeNewGeometry').outputs['Backfacing'], front.inputs[1])
        links.new(front.outputs['Value'], glow.inputs['Strength'])
        add = nodes.new('ShaderNodeAddShader')
        links.new(shader, add.inputs[0])
        links.new(glow.outputs['Emission'], add.inputs[1])
        shader = add.outputs['Shader']
    if glow_image is not None:
        uv = nodes.new('ShaderNodeUVMap')
        uv.uv_map = UV_LIGHTMAP
        image = nodes.new('ShaderNodeTexImage')
        image.image = glow_image
        # (outside the crop the page held zeros, which the cropped image must read as black)
        image.extension = 'CLIP'
        # each source pixel is one receiver sample, not a mixture of the materials next to it
        image.interpolation = 'Closest'
        if glow_transform is None:
            links.new(uv.outputs['UV'], image.inputs['Vector'])
        else:
            mapping = nodes.new('ShaderNodeMapping')
            mapping.vector_type = 'POINT'
            mapping.inputs['Scale'].default_value = (*glow_transform[0], 1)
            mapping.inputs['Location'].default_value = (*glow_transform[1], 0)
            links.new(uv.outputs['UV'], mapping.inputs['Vector'])
            links.new(mapping.outputs['Vector'], image.inputs['Vector'])
        multiply = nodes.new('ShaderNodeVectorMath')
        multiply.operation = 'MULTIPLY'
        links.new(image.outputs['Color'], multiply.inputs[0])
        colour = diffuse.inputs['Color']
        if colour.is_linked:
            links.new(colour.links[0].from_socket, multiply.inputs[1])
        else:
            multiply.inputs[1].default_value = colour.default_value[:3]
        glow = nodes.new('ShaderNodeEmission')
        links.new(multiply.outputs['Vector'], glow.inputs['Color'])
        # no backface gate: enhanced receivers send their gathered light to both sides
        shader = glow.outputs['Emission']
    # the hole cuts the emission too: a transparent texel must not glow
    if cutout and alpha_socket is not None:
        mix = nodes.new('ShaderNodeMixShader')
        hole = nodes.new('ShaderNodeMath')
        hole.operation = 'LESS_THAN'
        hole.inputs[1].default_value = 0.5
        links.new(alpha_socket, hole.inputs[0])
        links.new(hole.outputs['Value'], mix.inputs['Fac'])
        links.new(shader, mix.inputs[1])
        links.new(nodes.new('ShaderNodeBsdfTransparent').outputs['BSDF'], mix.inputs[2])
        shader = mix.outputs['Shader']
    if tint:
        # a shadow ray crosses the pane in the tint's colour, every other ray sees the pane's diffuse
        switch = nodes.new('ShaderNodeMixShader')
        glass = nodes.new('ShaderNodeBsdfTransparent')
        glass.inputs['Color'].default_value = (*tint, 1)
        links.new(nodes.new('ShaderNodeLightPath').outputs['Is Shadow Ray'], switch.inputs['Fac'])
        links.new(shader, switch.inputs[1])
        links.new(glass.outputs['BSDF'], switch.inputs[2])
        shader = switch.outputs['Shader']
    if mask_image is not None:
        # a texel the mask leaves out is not shaded for the ray the bake starts from, which is cheaper than shading
        # and sampling the full surface again
        uv = nodes.new('ShaderNodeUVMap')
        uv.uv_map = UV_LIGHTMAP
        chosen = nodes.new('ShaderNodeTexImage')
        chosen.image = mask_image
        chosen.interpolation = 'Closest'
        chosen.extension = 'EXTEND'
        links.new(uv.outputs['UV'], chosen.inputs['Vector'])
        skip = nodes.new('ShaderNodeMath')
        skip.operation = 'SUBTRACT'
        skip.inputs[0].default_value = 1
        red = nodes.new('ShaderNodeSeparateColor')
        links.new(chosen.outputs['Color'], red.inputs[0])
        links.new(red.outputs['Red'], skip.inputs[1])
        # only the ray the bake starts from is cut: every other ray still meets and is shadowed by this surface
        primary = nodes.new('ShaderNodeMath')
        primary.operation = 'MULTIPLY'
        links.new(skip.outputs['Value'], primary.inputs[0])
        links.new(nodes.new('ShaderNodeLightPath').outputs['Is Camera Ray'], primary.inputs[1])
        gate = nodes.new('ShaderNodeMixShader')
        links.new(primary.outputs['Value'], gate.inputs['Fac'])
        links.new(shader, gate.inputs[1])
        links.new(nodes.new('ShaderNodeBsdfTransparent').outputs['BSDF'], gate.inputs[2])
        shader = gate.outputs['Shader']
    links.new(shader, output.inputs['Surface'])
    if page_image is not None:
        # bakes go to the active image node, and it must stay unlinked
        target = nodes.new('ShaderNodeTexImage')
        target.image = page_image
        nodes.active = target
    return material


def make_object(name, positions, uv_layers, tris, materials, material_index, shadow, normals=None):
    """positions (N,3), tris (T,3) into them, uv_layers {name: (N,2)}, materials a list, normals (N,3) or flat."""
    mesh = bpy.data.meshes.new(name)
    count = len(tris)
    mesh.vertices.add(len(positions))
    mesh.vertices.foreach_set('co', np.ascontiguousarray(positions, np.float32).ravel())
    mesh.loops.add(count * 3)
    mesh.loops.foreach_set('vertex_index', np.ascontiguousarray(tris, np.int32).ravel())
    mesh.polygons.add(count)
    mesh.polygons.foreach_set('loop_start', np.arange(count, dtype=np.int32) * 3)
    mesh.polygons.foreach_set('loop_total', np.full(count, 3, np.int32))
    mesh.polygons.foreach_set('material_index', np.ascontiguousarray(material_index, np.int32))
    for uv_name, uv in uv_layers.items():
        layer = mesh.uv_layers.new(name=uv_name)
        per_loop = np.asarray(uv, np.float32)[np.asarray(tris).ravel()]
        layer.data.foreach_set('uv', np.ascontiguousarray(per_loop).ravel())
    if UV_LIGHTMAP in uv_layers:
        mesh.uv_layers.active = mesh.uv_layers[UV_LIGHTMAP]
    for material in materials:
        mesh.materials.append(material)
    mesh.update()
    if normals is not None:
        # flat shading would facet every curved surface of the BSP, whose vertex normals are the smooth truth
        mesh.polygons.foreach_set('use_smooth', np.ones(count, np.bool_))
        mesh.normals_split_custom_set_from_vertices(np.ascontiguousarray(normals, np.float32).tolist())
    obj = bpy.data.objects.new(name, mesh)
    obj.visible_shadow = bool(shadow)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def flip_v(uv):
    uv = np.array(uv, np.float32)
    uv[:, 1] = 1 - uv[:, 1]
    return uv


def sun_facing_planes(scene, suns):
    corners = scene['positions'][scene['tris']].astype(np.float64)
    plane = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    smooth = scene['normals'][scene['tris']].astype(np.float64).sum(axis=1)
    plane *= np.where((plane * smooth).sum(axis=1) < 0, -1, 1)[:, None]
    # a sky light's grid samples all use the light's centre direction, so they share one receiver
    # set (and one light-linking collection). tool tests each sample, so a light wider than the
    # small diameters seen in shipped skies would, near the terminator, light a face for all of
    # its samples here where tool lights it for only some of them
    if not suns:
        return np.empty((len(plane), 0), bool)
    return np.stack([plane @ np.float64(sun.get('centre', sun['direction'])) > 0 for sun in suns], axis=1)


def sun_facing_queries(scene, suns):
    count = len(scene['query_pos'])
    if 'query_inverse' not in scene:
        # standalone point probes have no owning mesh faces; pipeline jobs always supply the relation
        return np.ones((count, len(suns)), bool)
    visible = np.zeros((count, len(suns)), bool)
    corners = scene['query_inverse'][scene['tris']]
    facing = sun_facing_planes(scene, suns)
    for n in range(len(suns)):
        # (assigning True through repeated indices is the same as an or over them)
        visible[corners.ravel()[np.repeat(facing[:, n], 3)], n] = True
    return visible


def receiver_sun_bits(scene, suns):
    # a sun's plane bit only matters to receivers whose class it lights; leaving the rest out
    # keeps them in one object instead of splitting every page by every sun
    facing = sun_facing_planes(scene, suns)
    for n, sun in enumerate(suns):
        facing[:, n] &= np.isin(scene['tri_class'], sun['classes'])
    # flagged receivers take their full-strength sun from the vertex pass, never twice
    facing[scene.get('tri_ignore_normals', np.zeros(len(facing), bool))] = False
    return facing


def glow_uv_transform(box):
    x0, y0, x1, y1, w, h = (int(v) for v in box)
    cw, ch = x1 - x0, y1 - y0
    # (the page's v runs up, its rows down: the crop's top edge is row y0 and the page's is row 0)
    return (w / cw, h / ch), (-x0 / cw, (y1 - h) / ch)


def new_glow_image(name, source):
    h, w = source.shape[:2]
    image = bpy.data.images.new(name, w, h, alpha=False, float_buffer=True, is_data=True)
    image.colorspace_settings.name = 'Non-Color'
    # (built per image and dropped here: the temporary is as large as the image itself)
    rgba = np.ones((h, w, 4), np.float32)
    rgba[..., :3] = source[::-1]
    image.pixels.foreach_set(rgba.ravel())
    del rgba
    return image


def load_glow_images(path, page_count, uncropped=False):
    images, transforms = [], []
    black = None
    with np.load(path) as data:
        for i in range(page_count):
            if f'page_{i}' not in data.files:
                if black is None:
                    black = new_glow_image('glow_black', np.zeros((1, 1, 3), np.float32))
                images.append(black)
                transforms.append(None)
                continue
            source, box = data[f'page_{i}'], data[f'box_{i}'].tolist()
            if uncropped:
                # (test-only, never written by write_job: the page as a full image, as before cropping)
                full = np.zeros((box[5], box[4], 3), np.float32)
                full[box[1]:box[3], box[0]:box[2]] = source
                images.append(new_glow_image(f'glow_{i}', full))
                transforms.append(None)
                continue
            images.append(new_glow_image(f'glow_{i}', source))
            transforms.append(glow_uv_transform(box))
    return images, transforms


def lamp_free_run(job):
    # the glow run has no lamps in its job (blender_job.run_blender_glow)
    return bool(job.get('glow')) or not job['lights']


def receiver_material(name, prop, job, textures, page, image, mask_image=None, glow_images=(), glow_transforms=(),
                      **options):
    tex, cut, emit, tint, flagged = prop
    enhanced = job.get('ignore_normals_mode') == 'enhanced'
    settings = dict(both_sides=flagged and not enhanced,
                    zero_albedo=flagged and enhanced and not glow_images,
                    glow_image=glow_images[page] if glow_images and flagged else None,
                    glow_transform=glow_transforms[page] if glow_transforms and flagged else None)
    settings.update(options)
    return make_material(name, textures[tex] if tex >= 0 else None, cut, emit, job, image, tint,
                         mask_image=mask_image, **settings)


@dataclass
class AtlasSource:
    # a receiver object with what an atlas copy of any of its triangles needs: the scene triangles it holds (ascending)
    # and each one's material key
    obj: object
    picked: np.ndarray
    tri_props: list
    page: int


@dataclass
class AtlasInputs:
    scene: dict
    page: int
    sources: list
    textures: list
    image: object
    two_sided: np.ndarray
    job: dict = None
    glow_images: tuple = ()
    glow_transforms: tuple = ()


def atlas_inputs(scene_data, job, page, sources, textures, image, glow_images=(), glow_transforms=()):
    # test_atlas_page_path is test-only, never written by write_job
    if job.get('test_atlas_page_path'):
        return None
    flags = scene_data.get('tri_ignore_normals', np.zeros(len(scene_data['tris']), bool))
    return AtlasInputs(scene_data, page, [s for s in sources if s.page == page], textures, image,
                       flags & (job.get('ignore_normals_mode') != 'enhanced'), job, glow_images, glow_transforms)


def build_objects(scene_data, job, textures, page_images, mask_images=(), glow_images=(), glow_transforms=(),
                  sources=None):
    """Receivers per page, class, shadow and sun visibility; occluders per material and shadow."""
    positions, tris = scene_data['positions'], scene_data['tris']
    receivers = []
    flags = scene_data.get('tri_ignore_normals', np.zeros(len(tris), bool))
    # only add_lamps tells flagged receivers apart (it keeps Cycles lamps off them), so a run without lamps
    # shades the flag per triangle and shares one object; test_split_flagged is a test-only override, never written by write_job
    split = not lamp_free_run(job) or bool(job.get('test_split_flagged'))
    sun_from = 4 if split else 3
    # a see-through triangle still has to meet shadow rays when its tint dims them
    tinted = ~scene_data['tri_casts_shadow'] & (scene_data['tri_tint'] < 1).any(axis=1)
    keys = np.column_stack([scene_data['tri_page'], scene_data['tri_class'],
                            scene_data['tri_casts_shadow'] | tinted,
                            *([flags] if split else []),
                            receiver_sun_bits(scene_data, job['suns'])])
    distinct_keys, labels = np.unique(keys, axis=0, return_inverse=True)
    order = np.argsort(labels, kind='stable')
    groups = np.split(order, np.flatnonzero(np.diff(labels[order])) + 1) if len(order) else []
    for key, picked in zip(distinct_keys, groups):
        page, cls, casts = int(key[0]), int(key[1]), bool(key[2])
        used, local = np.unique(tris[picked].ravel(), return_inverse=True)
        local = local.reshape(-1, 3)
        paired = scene_data['tri_twin'][picked] >= 0
        copied = tris[picked[paired]].ravel()
        corners = positions[copied].astype(np.float64).reshape(-1, 3, 3)
        normal = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
        length = np.linalg.norm(normal, axis=1, keepdims=True)
        normal /= np.where(length > 0, length, 1)
        spacing = np.spacing(np.abs(corners).astype(np.float32)).max(axis=(1, 2))
        separation = np.maximum(TWIN_SEPARATION, TWIN_SEPARATION_STEPS * spacing.astype(np.float64))
        moved = corners + separation[:, None, None] * normal[:, None]
        local[paired] = np.arange(len(used), len(used) + len(copied)).reshape(-1, 3)
        bake_positions = np.concatenate([positions[used], moved.reshape(-1, 3)]).astype(np.float32)
        used = np.concatenate([used, copied])
        tri_props = [(int(scene_data['tri_texture'][t]), bool(scene_data['tri_cutout'][t]),
                      tuple(float(c) for c in scene_data['tri_emission'][t]),
                      tuple(float(c) for c in scene_data['tri_tint'][t]) if tinted[t] else (), bool(flags[t]))
                     for t in picked]
        distinct = sorted(set(tri_props))
        materials = [receiver_material(f'recv_{page}_{cls}_{n}', prop, job, textures, page, page_images[page],
                                       mask_images[page] if mask_images else None, glow_images, glow_transforms)
                     for n, prop in enumerate(distinct)]
        obj = make_object(f'recv_{page}_{cls}_{int(casts)}', bake_positions,
                          {UV_TEXTURE: flip_v(scene_data['tex_uv'][used]),
                           UV_LIGHTMAP: flip_v(scene_data['lm_uv'][used])},
                          local, materials, [distinct.index(p) for p in tri_props], casts,
                          scene_data['normals'][used])
        obj['sun_facing'] = key[sun_from:].tolist()
        obj['flagged'] = bool(flags[picked].any())
        receivers.append((obj, page, cls))
        if sources is not None:
            sources.append(AtlasSource(obj, picked, tri_props, page))

    occluders = []
    occ_tris = scene_data['occ_tris']
    # a see-through occluder meets shadow rays too when its tint dims them
    otinted = ~scene_data['occ_tri_casts_shadow'] & (scene_data['occ_tri_tint'] < 1).any(axis=1)
    okeys = np.concatenate([
        np.stack([scene_data['occ_tri_texture'], scene_data['occ_tri_cutout'],
                  scene_data['occ_tri_casts_shadow'] | otinted], axis=1).astype(np.float64),
        scene_data['occ_tri_emission'].astype(np.float64),
        emit_only_of(scene_data)[:, None].astype(np.float64),
        np.where(otinted[:, None], scene_data['occ_tri_tint'], 1).astype(np.float64),
        lamp_owner_of(scene_data, job)[:, None].astype(np.float64)], axis=1)
    for key in np.unique(okeys, axis=0) if len(occ_tris) else []:
        tex, cut, casts = int(key[0]), bool(key[1]), bool(key[2])
        emit, tint = tuple(float(c) for c in key[3:6]), tuple(float(c) for c in key[7:10])
        picked = np.flatnonzero((okeys == key).all(axis=1))
        used, local = np.unique(occ_tris[picked].ravel(), return_inverse=True)
        material = make_material(f'occ_{tex}_{int(cut)}', textures[tex] if tex >= 0 else None, cut,
                                  emit, job, None, tint if tint != (1.0, 1.0, 1.0) else (),
                                  emit_only=bool(key[6]))
        owner = int(key[10])
        occ = make_object(f'occ_{tex}_{int(cut)}_{int(casts)}' + (f'_owner{owner}' if owner >= 0 else ''),
                          scene_data['occ_positions'][used],
                          {UV_TEXTURE: flip_v(scene_data['occ_uv'][used])},
                          local.reshape(-1, 3), [material], np.zeros(len(picked), np.int32), casts)
        occ['owner'] = owner
        occluders.append(occ)
    return receivers, occluders


def lamp_owner_of(scene_data, job):
    # only a placement that carries a lamp gets objects of its own, so a map without lamps groups exactly as before
    owner = np.asarray(scene_data.get('occ_tri_owner', np.full(len(scene_data['occ_tris']), -1)))
    return np.where(np.isin(owner, [lamp['owner'] for lamp in job['lights']]), owner, -1)


def emit_only_of(scene_data):
    # decided by write_job before the source weights zero the emission, so a pane stays transparent without it
    return np.asarray(scene_data['occ_tri_emit_only'], np.bool_)


# Cycles keeps 63 non-default membership patterns per side, light and shadow counted apart, and past that warns on
# every link and bakes a hundred times slower, so more distinct sets than this is refused rather than linked wrongly
MAX_LINK_SETS = 63


def link_groups(suns, receivers):
    """Suns grouped by the exact set of receivers they light: {receiver indices: [sun indices]}."""
    groups = {}
    for n, sun in enumerate(suns):
        lit = tuple(i for i, (recv, _, cls) in enumerate(receivers)
                    if cls in sun['classes'] and recv['sun_facing'][n])
        groups.setdefault(lit, []).append(n)
    if len(groups) > MAX_LINK_SETS:
        raise RuntimeError(f'{len(groups)} distinct sun receiver sets exceed the {MAX_LINK_SETS} '
                           f'light-linking sets Cycles supports')
    return groups


def add_suns(job, receivers, occluders):
    suns = job['suns']
    for lit, shining in link_groups(suns, receivers).items():
        linked = bpy.data.collections.new(f'sun_receivers_{shining[0]}')
        for i in lit:
            linked.objects.link(receivers[i][0])
        for occ in occluders:
            linked.objects.link(occ)
        for n in shining:
            add_sun(n, suns[n], linked)


def link_patterns(collections, objects):
    """How many distinct non-default membership patterns the objects have across the collections."""
    members = [{o.name for o in c.objects} for c in collections]
    patterns = {tuple(o.name in m for m in members) for o in objects}
    return len(patterns - {(False,) * len(collections)})


def add_lamps(job, receivers, occluders, batch):
    lamps = [(n, lamp) for n, lamp in enumerate(job['lights']) if lamp['batch'] == batch]
    if not lamps:
        return
    objects = [recv for recv, _, _ in receivers] + list(occluders)
    shared = bpy.data.collections.new('lamp_receivers')
    # a flagged receiver is lit by the vertex pass's lamp rays, never by Cycles
    for obj in [recv for recv, _, _ in receivers if not recv['flagged']] + list(occluders):
        shared.objects.link(obj)
    blockers = {}
    for owner in dict.fromkeys(lamp['owner'] for _, lamp in lamps):
        blocking = bpy.data.collections.new(f'lamp_blockers_{owner}')
        for obj in objects:
            if obj.get('owner', -1) != owner:
                blocking.objects.link(obj)
        blockers[owner] = blocking
    sun_sets = [c for c in bpy.data.collections if c.name.startswith('sun_receivers_')]
    light_count = link_patterns(sun_sets + [shared], objects)
    shadow_count = link_patterns(list(blockers.values()), objects)
    if max(light_count, shadow_count) > MAX_LINK_SETS:
        raise RuntimeError(f'{light_count} light and {shadow_count} shadow link patterns exceed the '
                           f'{MAX_LINK_SETS} Cycles supports per side')
    for n, lamp in lamps:
        obj = make_lamp(f'lamp_{n}', lamp)
        obj.light_linking.receiver_collection = shared
        obj.light_linking.blocker_collection = blockers[lamp['owner']]


def add_sun(n, sun, receiver_collection):
    light = bpy.data.lights.new(f'sun_{n}', 'SUN')
    light.angle = sun['angle']
    light.color = sun['color']
    light.energy = sun['strength']
    obj = bpy.data.objects.new(f'sun_{n}', light)
    bpy.context.scene.collection.objects.link(obj)
    # a sun shines along its -Z axis and the job's direction points at the sun
    obj.rotation_mode = 'QUATERNION'
    obj.rotation_quaternion = mathutils.Vector((0, 0, 1)).rotation_difference(
        mathutils.Vector(sun['direction']))
    obj.light_linking.receiver_collection = receiver_collection


def make_lamp(name, lamp):
    light = bpy.data.lights.new(name, 'POINT')
    light.shadow_soft_size = 0
    strength = max(lamp['strength'])
    light.energy = LAMP_WATTS * strength
    light.color = [c / strength for c in lamp['strength']]
    light.use_nodes = True
    tree = light.node_tree
    tree.nodes.clear()
    output = tree.nodes.new('ShaderNodeOutputLight')
    emission = tree.nodes.new('ShaderNodeEmission')
    tree.links.new(emission.outputs['Emission'], output.inputs['Surface'])
    falloff = tree.nodes.new('ShaderNodeLightFalloff')
    falloff.inputs['Strength'].default_value = 1
    falloff.inputs['Smooth'].default_value = 0
    cutoff = tree.nodes.new('ShaderNodeMath')
    cutoff.operation = 'LESS_THAN'
    cutoff.inputs[1].default_value = lamp['reach']
    tree.links.new(falloff.outputs['Linear'], cutoff.inputs[0])
    gain = cutoff.outputs['Value']
    if lamp['kind'] == 3:
        # in a light's node tree the Normal is the light-local direction it emits along, -Z at rest
        direction = tree.nodes.new('ShaderNodeTexCoord')
        along = tree.nodes.new('ShaderNodeVectorMath')
        along.operation = 'DOT_PRODUCT'
        along.inputs[1].default_value = (0, 0, -1)
        tree.links.new(direction.outputs['Normal'], along.inputs[0])
        if lamp['cos_falloff'] <= lamp['cos_cutoff']:
            ramp = tree.nodes.new('ShaderNodeMath')
            ramp.operation = 'GREATER_THAN'
            ramp.inputs[1].default_value = lamp['cos_cutoff']
            tree.links.new(along.outputs['Value'], ramp.inputs[0])
            ramp_out = ramp.outputs['Value']
        else:
            ramp = tree.nodes.new('ShaderNodeMapRange')
            ramp.interpolation_type = 'LINEAR'
            ramp.clamp = True
            ramp.inputs['From Min'].default_value = lamp['cos_cutoff']
            ramp.inputs['From Max'].default_value = lamp['cos_falloff']
            ramp.inputs['To Min'].default_value = 0
            ramp.inputs['To Max'].default_value = 1
            tree.links.new(along.outputs['Value'], ramp.inputs['Value'])
            ramp_out = ramp.outputs['Result']
        both = tree.nodes.new('ShaderNodeMath')
        both.operation = 'MULTIPLY'
        tree.links.new(gain, both.inputs[0])
        tree.links.new(ramp_out, both.inputs[1])
        gain = both.outputs['Value']
    tree.links.new(gain, emission.inputs['Strength'])
    obj = bpy.data.objects.new(name, light)
    bpy.context.scene.collection.objects.link(obj)
    obj.location = lamp['position']
    obj.rotation_mode = 'QUATERNION'
    obj.rotation_quaternion = mathutils.Vector((0, 0, -1)).rotation_difference(mathutils.Vector(lamp['axis']))
    return obj


def clear_page(image, buffer):
    # (a bake leaves a texel it does not write as it was, so the sentinel has to be put back before each one)
    buffer.fill(-1)
    image.pixels.foreach_set(buffer)


def read_page(image, buffer, h, w):
    # a view of the buffer: the next clear_page overwrites it, so the caller reduces it before baking again
    image.pixels.foreach_get(buffer)
    return buffer.reshape(h, w, 4)[:, :, :3]


def receiver_diffuse_scale(scene, page, width, height, enhanced=False, coverage=False):
    rows = np.flatnonzero(scene['tri_page'] == page)
    flags = scene.get('tri_ignore_normals', np.zeros(len(scene['tris']), bool))[rows]
    if not flags.any() and not coverage:
        return None
    width, height = 3 * width, 3 * height
    scale = np.full((height, width), -1 if coverage else 1, np.float32)
    flagged_value = 1 if coverage else (0 if enhanced else 1 / BACK_SIDE_SHARE)
    if flags.all() and not coverage:
        scale.fill(flagged_value)
        return scale
    occupied = np.zeros((height, width), bool)
    # (Blender's rows and raster point, as subtexel_owner, so the scale covers exactly the sub-texels the bake
    # writes; its own bool claim map keeps the peak below subtexel_owner's int32 one)
    uv = flip_v(scene['lm_uv']).astype(np.float64) * [width, height]
    for tri, flagged in zip(scene['tris'][rows], flags):
        a, b, c = uv[tri]
        x0 = max(0, math.ceil(min(a[0], b[0], c[0]) - 1))
        x1 = min(width - 1, math.floor(max(a[0], b[0], c[0])))
        y0 = max(0, math.ceil(min(a[1], b[1], c[1]) - 1))
        y1 = min(height - 1, math.floor(max(a[1], b[1], c[1])))
        if x1 < x0 or y1 < y0:
            continue
        x = np.arange(x0, x1 + 1, dtype=np.float64)[None, :] + RASTER_POINT[0]
        # row strips bound the edge-function temporaries even for a triangle spanning a whole page
        for first in range(y0, y1 + 1, 64):
            end = min(first + 64, y1 + 1)
            y = np.arange(first, end, dtype=np.float64)[:, None] + RASTER_POINT[1]
            ab = (b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0])
            bc = (c[0] - b[0]) * (y - b[1]) - (c[1] - b[1]) * (x - b[0])
            ca = (a[0] - c[0]) * (y - c[1]) - (a[1] - c[1]) * (x - c[0])
            claimed = occupied[first:end, x0:x1 + 1]
            inside = (((ab >= 0) & (bc >= 0) & (ca >= 0)) | ((ab <= 0) & (bc <= 0) & (ca <= 0))) & ~claimed
            scale[first:end, x0:x1 + 1][inside] = flagged_value if flagged else (0 if coverage else 1)
            claimed[inside] = True
    return scale


# texel rows per strip: bounds the page loop's temporaries to a slice of a page instead of several whole ones
STRIP_ROWS = 256


def strips(texel_rows):
    for first in range(0, texel_rows, STRIP_ROWS):
        yield first, min(first + STRIP_ROWS, texel_rows)


def downsample_and_owned(rgb, scale=None):
    """Box-average 3x3 sub-texels, excluding the negative unwritten sentinel, not black samples.

    Gives (mean, texels written, sub-texels written per texel)."""
    h, w = rgb.shape[:2]
    mean = np.empty((h // 3, w // 3, 3), np.float32)
    owned = np.empty((h // 3, w // 3), np.int64)
    masked = np.empty((3 * min(STRIP_ROWS, h // 3), w, 3), np.float32)
    for first, end in strips(h // 3):
        part = rgb[3 * first:3 * end]
        n = end - first
        written = (part >= 0).all(axis=2)
        count = written.reshape(n, 3, w // 3, 3).sum(axis=(1, 3))
        values = part if scale is None else part * scale[3 * first:3 * end, :, None]
        out = masked[:3 * n]
        # (reused across strips, so a page allocates one strip-sized work array, not one per strip)
        out.fill(0)
        np.copyto(out, values, where=written[:, :, None])
        total = out.reshape(n, 3, w // 3, 3, 3).sum(axis=(1, 3))
        mean[first:end] = total / np.maximum(count, 1)[:, :, None]
        owned[first:end] = count
    return mean, owned > 0, owned


def downsample_written(rgb, scale=None):
    mean, written, _ = downsample_and_owned(rgb, scale)
    return mean, written


# an edge texel owns few of its 9 sub-texels, so with the same samples per sub-texel it is a noisier
# estimate than an interior one; a texel is topped up unless it already has this share of the target
ENOUGH_SHARE = 0.95
# a tier tops its texels up for its smallest owned count, so the largest in it overshoots by about this
TIER_SPAN = 1.5


def owned_subsamples(rgb):
    """How many of each texel's 3x3 sub-texels the bake wrote."""
    h, w = rgb.shape[:2]
    owned = np.empty((h // 3, w // 3), np.int64)
    for first, end in strips(h // 3):
        written = (rgb[3 * first:3 * end] >= 0).all(axis=2)
        owned[first:end] = written.reshape(end - first, 3, w // 3, 3).sum(axis=(1, 3))
    return owned


def plan_extra_passes(owned, base_spp, target):
    """[(extra samples per sub-texel, texel mask)] that bring every written texel near target samples in all."""
    if target <= 0:
        return []
    tiers = []
    smallest = 1
    while smallest * base_spp < ENOUGH_SHARE * target:
        largest = min(int(smallest * TIER_SPAN), math.ceil(ENOUGH_SHARE * target / base_spp) - 1)
        mask = (owned >= smallest) & (owned <= largest)
        spp = -(-target // smallest) - base_spp
        if mask.any():
            tiers.append((spp, mask))
        smallest = largest + 1
    return tiers


def effective_samples(owned, base_spp, tiers):
    return owned * (base_spp + sum(np.where(mask, spp, 0) for spp, mask in tiers))


def merge_extra(base, owned, base_spp, tiers, extra):
    """Mean of the base and extra bakes of each texel, weighted by the samples each put in."""
    merged = np.empty(base.shape, np.float32)
    for first, end in strips(base.shape[0]):
        rows = slice(first, end)
        part = owned[rows]
        total = base[rows] * (part * base_spp)[:, :, None]
        for (spp, mask), image in zip(tiers, extra):
            # (a texel outside the mask may hold inf or NaN, and a zero weight would not clear it)
            total = total + np.where(mask[rows, :, None], image[rows], 0) * (part * spp)[:, :, None]
        weight = effective_samples(part, base_spp, [(spp, mask[rows]) for spp, mask in tiers])
        merged[rows] = total / np.maximum(weight, 1)[:, :, None]
    return merged


# Blender's bake raster samples a pixel at this offset from its corner (bake.cc subtracts 0.501, 0.502 before scan conversion)
RASTER_POINT = (0.501, 0.502)
# widens a slot's footprint square to hold the jitter footprint the raster offset shifts (b9 spike report)
ATLAS_EPS = 0.02
# slot pixels sit on a grid this far apart, from this corner, so the triangle around each has room (b9 spike report)
ATLAS_STRIDE = 4
ATLAS_OFFSET = 2
# how far, in pixels, a triangle may fall short of a pixel centre and still count as covering it
ATLAS_SLACK = 0.01
MAX_ATLAS = 8192


def subtexel_owner(scene, page, width, height):
    width, height = 3 * width, 3 * height
    owner = np.full((height, width), -1, np.int32)
    uv = flip_v(scene['lm_uv']).astype(np.float64) * [width, height]
    for t in np.flatnonzero(scene['tri_page'] == page):
        a, b, c = uv[scene['tris'][t]]
        x0 = max(0, math.ceil(min(a[0], b[0], c[0]) - 1))
        x1 = min(width - 1, math.floor(max(a[0], b[0], c[0])))
        y0 = max(0, math.ceil(min(a[1], b[1], c[1]) - 1))
        y1 = min(height - 1, math.floor(max(a[1], b[1], c[1])))
        if x1 < x0 or y1 < y0:
            continue
        x = np.arange(x0, x1 + 1, dtype=np.float64)[None, :] + RASTER_POINT[0]
        # row strips bound the edge-function temporaries even for a triangle spanning a whole page
        for first in range(y0, y1 + 1, 64):
            end = min(first + 64, y1 + 1)
            y = np.arange(first, end, dtype=np.float64)[:, None] + RASTER_POINT[1]
            ab = (b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0])
            bc = (c[0] - b[0]) * (y - b[1]) - (c[1] - b[1]) * (x - b[0])
            ca = (a[0] - c[0]) * (y - c[1]) - (a[1] - c[1]) * (x - c[0])
            inside = ((ab >= 0) & (bc >= 0) & (ca >= 0)) | ((ab <= 0) & (bc <= 0) & (ca <= 0))
            block = owner[first:end, x0:x1 + 1]
            block[inside & (block < 0)] = t
    return owner


@dataclass
class Atlas:
    slots: np.ndarray
    owner: np.ndarray
    role: np.ndarray
    twin: np.ndarray
    triangles: np.ndarray | None = None
    needed: np.ndarray | None = None
    pixels: np.ndarray | None = None
    side: int = 0


def plan_atlas(written, mask, owner, two_sided):
    chosen = written & np.repeat(np.repeat(mask, 3, axis=0), 3, axis=1)
    rows, cols = np.nonzero(chosen)
    own = owner[rows, cols]
    # (a written sub-texel no triangle owns has no geometry to copy: the caller bakes the page instead)
    if (own < 0).any():
        return None
    flagged = two_sided[own]
    front = np.flatnonzero(flagged)
    count, extra = len(own), len(front)
    twin = np.full(count + extra, -1, np.int64)
    twin[front] = count + np.arange(extra)
    twin[count:] = front
    xy = np.column_stack([cols, rows]).astype(np.int64)
    return Atlas(slots=np.concatenate([xy, xy[front]]), owner=np.concatenate([own, own[front]]).astype(np.int32),
                 role=np.concatenate([flagged.astype(np.int8), np.full(extra, 2, np.int8)]), twin=twin)


def _clip_squares(squares, tris):
    # Sutherland-Hodgman of one convex polygon per row against that row's triangle (either orientation); rows hold up to
    # 8 vertices (a square cut by three lines keeps at most 7) and the count says how many are real
    count = len(tris)
    sign = np.sign((tris[:, 1, 0] - tris[:, 0, 0]) * (tris[:, 2, 1] - tris[:, 0, 1])
                   - (tris[:, 1, 1] - tris[:, 0, 1]) * (tris[:, 2, 0] - tris[:, 0, 0]))
    poly = np.zeros((count, 8, 2))
    poly[:, :4] = squares
    size = np.full(count, 4)
    index = np.arange(8)[None, :]
    for i in range(3):
        a, b = tris[:, i], tris[:, (i + 1) % 3]
        s = sign[:, None] * ((b[:, None, 0] - a[:, None, 0]) * (poly[..., 1] - a[:, None, 1])
                             - (b[:, None, 1] - a[:, None, 1]) * (poly[..., 0] - a[:, None, 0]))
        valid = index < size[:, None]
        following = np.where(index + 1 >= size[:, None], 0, index + 1)
        s_next = np.take_along_axis(s, following, axis=1)
        keep = valid & (s >= 0)
        cross = valid & ((s >= 0) != (s_next >= 0))
        with np.errstate(divide='ignore', invalid='ignore'):
            q = np.take_along_axis(poly, following[..., None], axis=1)
            cut = poly + (s / (s - s_next))[..., None] * (q - poly)
        before = np.cumsum(keep, axis=1) + np.cumsum(cross, axis=1) - keep - cross
        out = np.zeros_like(poly)
        rows, cols = np.nonzero(keep)
        out[rows, before[rows, cols]] = poly[rows, cols]
        rows, cols = np.nonzero(cross)
        out[rows, before[rows, cols] + keep[rows, cols]] = cut[rows, cols]
        poly, size = out, keep.sum(axis=1) + cross.sum(axis=1)
    return poly, size


def _needed_edges(poly, size, tris):
    # per row, which owner edges a side of the clipped polygon lies along (a corner touching an edge is not enough)
    index = np.arange(8)[None, :]
    valid = index < size[:, None]
    following = np.where(index + 1 >= size[:, None], 0, index + 1)
    nxt = np.take_along_axis(poly, following[..., None], axis=1)
    long_enough = np.hypot(nxt[..., 0] - poly[..., 0], nxt[..., 1] - poly[..., 1]) > 1e-7
    scale = np.maximum(1.0, np.abs(tris).max(axis=(1, 2)))
    need = np.zeros((len(tris), 3), bool)
    for i in range(3):
        a, b = tris[:, i], tris[:, (i + 1) % 3]
        d = b - a
        reach = (1e-9 * scale * np.hypot(d[:, 0], d[:, 1]))[:, None]
        on_p = np.abs(d[:, None, 0] * (poly[..., 1] - a[:, None, 1]) - d[:, None, 1] * (poly[..., 0] - a[:, None, 0])) <= reach
        on_r = np.abs(d[:, None, 0] * (nxt[..., 1] - a[:, None, 1]) - d[:, None, 1] * (nxt[..., 0] - a[:, None, 0])) <= reach
        need[:, i] = (valid & long_enough & on_p & on_r).any(axis=1)
    return need


def _slot_triangles(slots, tris):
    # one triangle T' per slot that equals the owner on the slot's widened footprint square and reaches no further than it
    # must: two triangles would sample only part of the footprint (Cycles keeps a jittered sample inside the triangle
    # that holds the pixel's centre), and a square past the owner's edges would sample area the page never does
    count = len(slots)
    col, row = slots[:, 0].astype(np.float64), slots[:, 1].astype(np.float64)
    squares = np.stack([np.stack([col - ATLAS_EPS, row - ATLAS_EPS], axis=1), np.stack([col + 1 + ATLAS_EPS, row - ATLAS_EPS], axis=1),
                        np.stack([col + 1 + ATLAS_EPS, row + 1 + ATLAS_EPS], axis=1),
                        np.stack([col - ATLAS_EPS, row + 1 + ATLAS_EPS], axis=1)], axis=1)
    poly, size = _clip_squares(squares, tris)
    need = _needed_edges(poly, size, tris)
    needed = need.sum(axis=1).astype(np.int8)
    # (the widened footprint lies inside the owner, so the smallest right triangle around it does)
    side = 1 + 2 * ATLAS_EPS
    x0, y0 = col - ATLAS_EPS, row - ATLAS_EPS
    out = np.stack([np.stack([x0, y0], axis=1), np.stack([x0 + 2 * side + 0.01, y0], axis=1),
                    np.stack([x0, y0 + 2 * side + 0.01], axis=1)], axis=1)
    whole = needed == 3
    out[whole] = tris[whole]
    two = np.flatnonzero(needed == 2)
    if len(two):
        edges = need[two]
        # the two needed edges meet at the owner corner not on the third edge
        shared = np.where(edges[:, 0] & edges[:, 1], 1, np.where(edges[:, 1] & edges[:, 2], 2, 0))
        first = np.where(shared == 0, 1, 0)
        second = np.where(shared == 2, 1, 2)
        t2 = tris[two]
        rows = np.arange(len(two))
        corner = t2[rows, shared]
        da, db = t2[rows, first] - corner, t2[rows, second] - corner
        ab = np.linalg.solve(np.stack([da, db], axis=2), np.swapaxes(poly[two] - corner[:, None], 1, 2))
        total = np.where(np.arange(8)[None, :] < size[two][:, None], ab.sum(axis=1), -np.inf)
        # (the margin keeps the footprint's far corners inside T', whose far side then stays inside the owner)
        t = total.max(axis=1) * (1 + 1e-6) + 1e-6
        out[two] = np.stack([corner, corner + t[:, None] * da, corner + t[:, None] * db], axis=1)
    one = np.flatnonzero(needed == 1)
    if len(one):
        edge = need[one].argmax(axis=1)
        t1 = tris[one]
        rows = np.arange(len(one))
        a, b, third = t1[rows, edge], t1[rows, (edge + 1) % 3], t1[rows, (edge + 2) % 3]
        d = (b - a) / np.hypot(b[:, 0] - a[:, 0], b[:, 1] - a[:, 1])[:, None]
        m = np.stack([-d[:, 1], d[:, 0]], axis=1)
        m = np.where(((third[:, 0] - a[:, 0]) * m[:, 0] + (third[:, 1] - a[:, 1]) * m[:, 1] < 0)[:, None], -m, m)
        rel = poly[one] - a[:, None]
        valid = np.arange(8)[None, :] < size[one][:, None]
        along = np.where(valid, rel[..., 0] * d[:, None, 0] + rel[..., 1] * d[:, None, 1], np.nan)
        height = np.where(valid, rel[..., 0] * m[:, None, 0] + rel[..., 1] * m[:, None, 1], np.nan)
        lo, hi = np.nanmin(along, axis=1)[:, None], np.nanmax(along, axis=1)[:, None]
        top = np.maximum(np.nanmax(height, axis=1), 0)[:, None] + 1e-3
        out[one] = np.stack([a + (lo - top) * d, a + (hi + top) * d,
                             a + 0.5 * (lo + hi) * d + (0.5 * (hi - lo) + top) * m], axis=1)
    return out, needed


def _page_corners(scene, width, height):
    # the lightmap UVs as Blender's raster sees them: float32 after the flip, then scaled to sub-texel pixels
    return flip_v(scene['lm_uv']).astype(np.float64)[scene['tris']] * [3 * width, 3 * height]


def _depth(tri, x, y):
    # how far inside its tightest edge line the pixel centre (x, y) lies, negative outside; tri broadcasts against x, y
    sign = np.sign((tri[..., 1, 0] - tri[..., 0, 0]) * (tri[..., 2, 1] - tri[..., 0, 1])
                   - (tri[..., 1, 1] - tri[..., 0, 1]) * (tri[..., 2, 0] - tri[..., 0, 0]))
    depth = np.inf
    for i in range(3):
        p, q = tri[..., i, :], tri[..., (i + 1) % 3, :]
        depth = np.minimum(depth, sign * ((q[..., 0] - p[..., 0]) * (y - p[..., 1]) - (q[..., 1] - p[..., 1]) * (x - p[..., 0]))
                           / np.hypot(q[..., 0] - p[..., 0], q[..., 1] - p[..., 1]))
    return depth


def _land(tri, n_cols, cursor, covered, occupied):
    """Where a triangle given relative to its slot lands under the greedy rule, and the cells it then covers. Each
    candidate cell is decided on the triangle shifted to that cell, as the pixels are laid out, a chunk of free cells at a
    time (nothing changes while one slot looks, so the first clear cell of a chunk is its cell). Only a slot pixel can
    clash with a triangle, so only the grid cells around the triangle's bounding box are tested."""
    low, high = tri.min(axis=0), tri.max(axis=0)
    columns = np.arange(int(np.floor((low[0] - 2) / ATLAS_STRIDE)) - 1, int(np.ceil((high[0] + 2) / ATLAS_STRIDE)) + 2)
    rows = np.arange(int(np.floor((low[1] - 2) / ATLAS_STRIDE)) - 1, int(np.ceil((high[1] + 2) / ATLAS_STRIDE)) + 2)
    chunk = 4
    while True:
        span = 8 * chunk
        reserve = cursor + span + (rows[-1] + 2) * n_cols
        if reserve > len(covered):
            covered.extend(bytes(reserve - len(covered)))
            occupied.extend(bytes(reserve - len(occupied)))
        free = np.flatnonzero(np.frombuffer(bytes(covered[cursor:cursor + span]), np.uint8) == 0)[:chunk]
        if not len(free):
            cursor += span
            continue
        cells = cursor + free
        column, row = cells % n_cols, cells // n_cols
        ax, ay = ATLAS_OFFSET + ATLAS_STRIDE * column, ATLAS_OFFSET + ATLAS_STRIDE * row
        shifted = tri + np.stack([ax, ay], axis=1)[:, None]
        x = ax[:, None] + ATLAS_STRIDE * columns[None, :]
        y = ay[:, None] + ATLAS_STRIDE * rows[None, :]
        low_pixel = np.floor(shifted.min(axis=1) - 1).astype(int)
        high_pixel = np.ceil(shifted.max(axis=1) + 1).astype(int)
        hit = ((_depth(shifted[:, None, None], x[:, None, :] + 0.5, y[:, :, None] + 0.5) >= -ATLAS_SLACK)
               & (x >= low_pixel[:, :1])[:, None, :] & (x <= high_pixel[:, :1])[:, None, :]
               & (y >= low_pixel[:, 1:])[:, :, None] & (y <= high_pixel[:, 1:])[:, :, None]
               & ((column[:, None] + columns[None, :] >= 0) & (column[:, None] + columns[None, :] < n_cols))[:, None, :]
               & (row[:, None] + rows[None, :] >= 0)[:, :, None])
        ids = cells[:, None, None] + rows[None, :, None] * n_cols + columns[None, None, :]
        taken = np.zeros(ids.shape, bool)
        if hit.any():
            bottom, top = int(ids[hit].min()), int(ids[hit].max())
            if top >= len(occupied):
                covered.extend(bytes(top + 1 - len(covered)))
                occupied.extend(bytes(top + 1 - len(occupied)))
            taken[hit] = np.frombuffer(bytes(occupied[bottom:top + 1]), np.uint8)[ids[hit] - bottom] != 0
        # (a cell under an earlier triangle would bake that slot's neighbour; one whose triangle reaches an earlier
        # slot's centre would overwrite it)
        blocked = (taken & (ids != cells[:, None, None])).any(axis=(1, 2))
        if not blocked.all():
            chosen = int(np.argmin(blocked))
            return int(cells[chosen]), ids[chosen][hit[chosen]].tolist()
        cursor = int(cells[-1]) + 1
        chunk = min(64, 2 * chunk)


# cells either side of a slot's own whose centre the vectorised pass tests; a triangle that could reach further is decided
# cell by cell instead
WINDOW = 2
# a covered/not-covered verdict closer than this to its threshold is recomputed exactly where the slot lands (the pass
# works on the triangle relative to its slot, the layout on the triangle shifted to the cell it lands in)
MARGIN = 1e-7


def _window_masks(relative):
    """Per slot, the cells of the window around its own whose pixel centre T' covers, as bits (row * 5 + column of the
    window), whether the slot must be decided exactly once it has a cell, and whether its triangle reaches past the window."""
    count = len(relative)
    masks = np.zeros(count, np.int64)
    exact = np.zeros(count, bool)
    outside = np.zeros(count, bool)
    steps = np.arange(-WINDOW, WINDOW + 1) * ATLAS_STRIDE
    gx, gy = steps[None, None, :], steps[None, :, None]
    bit = (1 << np.arange((2 * WINDOW + 1) ** 2)).reshape(1, 2 * WINDOW + 1, 2 * WINDOW + 1)
    for start in range(0, count, 32768):
        tri = relative[start:start + 32768]
        low, high = tri.min(axis=1), tri.max(axis=1)
        depth = _depth(tri[:, None, None], gx + 0.5, gy + 0.5)
        near = depth >= -ATLAS_SLACK - MARGIN
        # (the layout only considers pixels within one pixel of the bounding box: a centre outside that is not covered)
        inside = ((gx > low[:, None, None, 0] - 2 + MARGIN) & (gx < high[:, None, None, 0] + 2 - MARGIN)
                  & (gy > low[:, None, None, 1] - 2 + MARGIN) & (gy < high[:, None, None, 1] + 2 - MARGIN))
        unsure = near & ((np.abs(depth + ATLAS_SLACK) < MARGIN)
                         | (np.abs(gx - (low[:, None, None, 0] - 2)) < MARGIN) | (np.abs(gx - (high[:, None, None, 0] + 2)) < MARGIN)
                         | (np.abs(gy - (low[:, None, None, 1] - 2)) < MARGIN) | (np.abs(gy - (high[:, None, None, 1] + 2)) < MARGIN))
        edges = np.hypot(tri[:, [1, 2, 0], 0] - tri[:, :, 0], tri[:, [1, 2, 0], 1] - tri[:, :, 1])
        area = np.abs((tri[:, 1, 0] - tri[:, 0, 0]) * (tri[:, 2, 1] - tri[:, 0, 1]) - (tri[:, 1, 1] - tri[:, 0, 1]) * (tri[:, 2, 0] - tri[:, 0, 0]))
        far = (low < -(ATLAS_STRIDE * WINDOW - 2)).any(axis=1) | (high > ATLAS_STRIDE * WINDOW - 2).any(axis=1)
        masks[start:start + 32768] = (((depth >= -ATLAS_SLACK) & inside) * bit).sum(axis=(1, 2))
        exact[start:start + 32768] = unsure.any(axis=(1, 2)) | far | (edges.min(axis=1) < 1e-3) | (area < 1e-6)
        outside[start:start + 32768] = far
    return masks, exact, outside


def _reach(tri):
    """The cells, as (columns, rows) away from the slot's own, whose pixel centre a triangle given relative to its slot
    covers, with whether any verdict is too close to call (then the cell it lands in decides, see MARGIN)."""
    low, high = tri.min(axis=0), tri.max(axis=0)
    columns = np.arange(int(np.floor((low[0] - 2 - MARGIN) / ATLAS_STRIDE)), int(np.ceil((high[0] + 2 + MARGIN) / ATLAS_STRIDE)) + 1)
    rows = np.arange(int(np.floor((low[1] - 2 - MARGIN) / ATLAS_STRIDE)), int(np.ceil((high[1] + 2 + MARGIN) / ATLAS_STRIDE)) + 1)
    x, y = ATLAS_STRIDE * columns[None, :], ATLAS_STRIDE * rows[:, None]
    depth = _depth(tri, x + 0.5, y + 0.5)
    inside = (x > low[0] - 2 + MARGIN) & (x < high[0] + 2 - MARGIN) & (y > low[1] - 2 + MARGIN) & (y < high[1] + 2 - MARGIN)
    unsure = ((depth >= -ATLAS_SLACK - MARGIN)
              & ((np.abs(depth + ATLAS_SLACK) < MARGIN) | (np.abs(x - (low[0] - 2)) < MARGIN) | (np.abs(x - (high[0] + 2)) < MARGIN)
                 | (np.abs(y - (low[1] - 2)) < MARGIN) | (np.abs(y - (high[1] + 2)) < MARGIN))).any()
    edges = np.hypot(*(tri[[1, 2, 0]] - tri).T)
    area = abs((tri[1, 0] - tri[0, 0]) * (tri[2, 1] - tri[0, 1]) - (tri[1, 1] - tri[0, 1]) * (tri[2, 0] - tri[0, 0]))
    rows_hit, columns_hit = np.nonzero((depth >= -ATLAS_SLACK) & inside)
    return list(zip(columns[columns_hit].tolist(), rows[rows_hit].tolist())), bool(unsure or edges.min() < 1e-3 or area < 1e-6)


def _place(triangles, slots, n_cols):
    """The greedy layout: each slot takes the first cell after the previous slot's that no accepted triangle covers
    and where its own triangle covers no accepted slot's pixel. Returns the slot pixels, or None past MAX_ATLAS."""
    count = len(slots)
    relative = triangles - slots[:, None]
    masks, exact, outside = _window_masks(relative)
    pixels = np.empty((count, 2), np.int64)
    covered = bytearray(count + 8 * n_cols)
    occupied = bytearray(len(covered))
    reach = {}
    cursor = 0
    for k, (mask, decide, far) in enumerate(zip(masks.tolist(), exact.tolist(), outside.tolist())):
        # (a triangle reaching past the window is decided cell by cell, so no rounding bound has to hold for it)
        if far:
            decide = True
        elif decide:
            offsets, decide = _reach(relative[k])
        else:
            if mask not in reach:
                reach[mask] = [(i % 5 - WINDOW, i // 5 - WINDOW) for i in range(25) if mask >> i & 1]
            offsets = reach[mask]
        if decide:
            cursor, cells = _land(relative[k], n_cols, cursor, covered, occupied)
        else:
            while True:
                if len(covered) < cursor + (WINDOW + 3) * n_cols:
                    covered.extend(bytes(len(covered)))
                    occupied.extend(bytes(len(occupied)))
                if covered[cursor]:
                    cursor += 1
                    continue
                column, row = cursor % n_cols, cursor // n_cols
                cells = [cursor + dy * n_cols + dx for dx, dy in offsets if 0 <= column + dx < n_cols and row + dy >= 0]
                if cells and max(cells) >= len(covered):
                    covered.extend(bytes(max(cells) + 1 - len(covered)))
                    occupied.extend(bytes(len(covered) - len(occupied)))
                # (a cell under an earlier triangle would bake that slot's neighbour; one whose triangle reaches an
                # earlier slot's centre would overwrite it)
                if not any(occupied[cell] for cell in cells if cell != cursor):
                    break
                cursor += 1
        column, row = cursor % n_cols, cursor // n_cols
        occupied[cursor] = 1
        for cell in cells:
            covered[cell] = 1
        pixels[k] = ATLAS_OFFSET + ATLAS_STRIDE * column, ATLAS_OFFSET + ATLAS_STRIDE * row
        if max(pixels[k]) + ATLAS_STRIDE + ATLAS_OFFSET + 4 > MAX_ATLAS:
            return None
        cursor += 1
    return pixels


def place_atlas(scene, atlas, width, height):
    corners = _page_corners(scene, width, height)
    area = ((corners[:, 1, 0] - corners[:, 0, 0]) * (corners[:, 2, 1] - corners[:, 0, 1])
            - (corners[:, 1, 1] - corners[:, 0, 1]) * (corners[:, 2, 0] - corners[:, 0, 0]))
    # (a zero-area owner has no barycentrics to place the slot's corners by)
    if (area[atlas.owner] == 0).any():
        log('atlas: an owner has a zero-area lightmap triangle, so the page is baked instead')
        return None
    count = len(atlas.slots)
    triangles = np.empty((count, 3, 2))
    needed = np.empty(count, np.int8)
    # (a back copy repeats its front copy's footprint)
    own = np.flatnonzero(atlas.role != 2)
    triangles[own], needed[own] = _slot_triangles(atlas.slots[own], corners[atlas.owner[own]])
    back = np.flatnonzero(atlas.role == 2)
    triangles[back], needed[back] = triangles[atlas.twin[back]], needed[atlas.twin[back]]
    n_cols = int(math.ceil(math.sqrt(1.3 * count)))
    pixels = _place(triangles, atlas.slots, n_cols)
    if pixels is None:
        log(f'atlas: {count} slots need an image over MAX_ATLAS ({MAX_ATLAS}) pixels, so the page is baked instead')
        return None
    side = int(pixels.max()) + ATLAS_STRIDE + ATLAS_OFFSET + 4 if count else 0
    return replace(atlas, triangles=triangles, needed=needed, pixels=pixels, side=side)


def _face_normal(corners):
    normal = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    length = np.linalg.norm(normal, axis=1, keepdims=True)
    return normal / np.where(length > 0, length, 1)


def _separation(corners):
    spacing = np.spacing(np.abs(corners).astype(np.float32)).max(axis=(1, 2))
    return np.maximum(TWIN_SEPARATION, TWIN_SEPARATION_STEPS * spacing.astype(np.float64))


def _bake_corners(scene):
    # the corners as build_objects leaves them on the bake mesh: twins moved off their partner, all rounded to float32
    tris = scene['tris']
    corners = scene['positions'][tris].astype(np.float64)
    paired = scene['tri_twin'][:len(tris)] >= 0
    moved = corners[paired] + _separation(corners[paired])[:, None, None] * _face_normal(corners[paired])[:, None]
    corners = corners.astype(np.float32).astype(np.float64)
    corners[paired] = moved.astype(np.float32).astype(np.float64)
    return corners


def atlas_mesh(scene, atlas, width, height):
    owner = atlas.owner
    page = _page_corners(scene, width, height)[owner]
    a = page[:, 0]
    d1, d2 = page[:, 1] - a, page[:, 2] - a
    det = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    r = atlas.triangles - a[:, None]
    u = (r[..., 0] * d2[:, None, 1] - r[..., 1] * d2[:, None, 0]) / det[:, None]
    v = (d1[:, None, 0] * r[..., 1] - d1[:, None, 1] * r[..., 0]) / det[:, None]
    # (a corner outside the owner has barycentrics outside [0, 1]; the plane extends the surface there)
    lam = np.stack([1 - u - v, u, v], axis=-1)
    bake = _bake_corners(scene)[owner]
    normal = _face_normal(bake)
    # the back copy of a both_sides slot sits behind the surface, where the Translucent closure bakes the light from behind
    offset = np.where(atlas.role == 2, -1.0, 1.0) * _separation(bake)
    positions = np.einsum('nkb,nbd->nkd', lam, bake) + (offset[:, None] * normal)[:, None]
    # (a flipped winding would make Cycles shade the owner's back face)
    swap = np.einsum('nd,nd->n', np.cross(positions[:, 1] - positions[:, 0], positions[:, 2] - positions[:, 0]), normal) <= 0
    order = np.where(swap[:, None], [0, 2, 1], [0, 1, 2])
    rows = np.arange(len(owner))[:, None]
    uv = (atlas.triangles - atlas.slots[:, None] + atlas.pixels[:, None]) / atlas.side
    normals = np.einsum('nkb,nbd->nkd', lam, scene['normals'].astype(np.float64)[scene['tris'][owner]])
    tex_uv = np.einsum('nkb,nbd->nkd', lam, flip_v(scene['tex_uv']).astype(np.float64)[scene['tris'][owner]])
    return dict(positions=positions[rows, order].reshape(-1, 3).astype(np.float32),
                tris=np.arange(3 * len(owner), dtype=np.int32).reshape(-1, 3),
                normals=normals[rows, order].reshape(-1, 3).astype(np.float32),
                tex_uv=tex_uv[rows, order].reshape(-1, 2).astype(np.float32),
                atlas_uv=uv[rows, order].reshape(-1, 2).astype(np.float32),
                material_tri=owner.astype(np.int32))


def scatter_atlas(atlas, values, buffer_shape):
    buffer = np.full((*buffer_shape[:2], 3), -1, np.float32)
    got = values[atlas.pixels[:, 1], atlas.pixels[:, 0]]
    ok = (got >= 0).all(axis=1)
    single = (atlas.role == 0) & ok
    buffer[atlas.slots[single, 1], atlas.slots[single, 0]] = got[single]
    front = np.flatnonzero(atlas.role == 1)
    both = ok[front] & ok[atlas.twin[front]]
    # (the DIFFUSE pass divides by the summed closure colour, so the two-closure page value is the halves' mean)
    mean = 0.5 * (got[front] + got[atlas.twin[front]])
    buffer[atlas.slots[front[both], 1], atlas.slots[front[both], 0]] = mean[both]
    return buffer, int(((atlas.role == 0) & ~ok).sum() + (~both).sum())


def atlas_objects(inputs, atlas, mesh, objects=None):
    """One object per source holding the slots its triangles own, linked where the source is linked."""
    roles = {0: {}, 1: dict(both_sides=False), 2: dict(both_sides=True, back_only=True)}
    # (a caller that passes its list still holds what was made when a later source fails)
    objects = [] if objects is None else objects
    for number, source in enumerate(inputs.sources):
        slots = np.flatnonzero(np.isin(atlas.owner, source.picked))
        if not len(slots):
            continue
        corners = (3 * slots[:, None] + np.arange(3)).ravel()
        props = [source.tri_props[i] for i in np.searchsorted(source.picked, atlas.owner[slots])]
        keys = [(prop, int(role)) for prop, role in zip(props, atlas.role[slots])]
        distinct = sorted(set(keys))
        materials = [receiver_material(f'atlas_{inputs.page}_{number}_{n}', prop, inputs.job, inputs.textures,
                                       inputs.page, inputs.image, None, inputs.glow_images, inputs.glow_transforms,
                                       **roles[role])
                     for n, (prop, role) in enumerate(distinct)]
        obj = make_object(f'atlas_{inputs.page}_{number}', mesh['positions'][corners],
                          {UV_TEXTURE: mesh['tex_uv'][corners], UV_LIGHTMAP: mesh['atlas_uv'][corners]},
                          np.arange(3 * len(slots), dtype=np.int32).reshape(-1, 3), materials,
                          [distinct.index(key) for key in keys], False, mesh['normals'][corners])
        obj.visible_camera = True
        obj.visible_diffuse = obj.visible_glossy = obj.visible_transmission = False
        obj.visible_volume_scatter = obj.visible_shadow = False
        obj['flagged'] = source.obj['flagged']
        # (read now: add_suns and add_lamps linked the source after it was built, and a copy has to be lit as it is)
        for collection in bpy.data.collections:
            if source.obj.name in collection.objects:
                collection.objects.link(obj)
        objects.append(obj)
    return objects


def remove_atlas_objects(objects):
    for obj in objects:
        mesh, materials = obj.data, list(obj.data.materials)
        bpy.data.objects.remove(obj)
        bpy.data.meshes.remove(mesh)
        for material in materials:
            bpy.data.materials.remove(material)


def warn_if_masks_differ(diffuse_written, emit_written, page):
    # the page's written mask comes from the diffuse pass alone; the emit pass should write the same texels
    differs = not np.array_equal(diffuse_written, emit_written)
    if differs:
        log(f'warning: page {page}: the emit pass wrote {int((emit_written != diffuse_written).sum())} '
            f'texels differently from the diffuse pass (a Blender behaviour change?)')
    return differs


def bake_page(page, receivers, image, h, w, job, mask_image=None, diffuse_scale=None, atlas=None):
    chosen = [obj for obj, p, _ in receivers if p == page]
    if not chosen:
        return (np.zeros((h, w, 3), np.float32), np.zeros((h, w, 3), np.float32),
                np.zeros((h, w), np.bool_))
    size_page_image(image, w, h)
    try:
        return _bake_page(page, chosen, image, h, w, job, mask_image, diffuse_scale, atlas)
    finally:
        release_page_image(image)


def page_tier(image, mask_image, mask, buffer, h, w, pixels):
    mask_image.generated_width, mask_image.generated_height = w, h
    pixels.fill(0)
    pixels[mask] = 1
    mask_image.pixels.foreach_set(pixels.ravel())
    clear_page(image, buffer)
    bpy.ops.object.bake(type='DIFFUSE', pass_filter={'DIRECT', 'INDIRECT'}, margin=0, use_clear=False,
                        target='IMAGE_TEXTURES')
    return read_page(image, buffer, 3 * h, 3 * w)


def atlas_tier(n, mask, inputs, owner, written, h, w):
    """The tier's sub-texels baked on atlas copies and scattered back to the page, or None to bake the tier on the page."""
    plan = plan_atlas(written, mask, owner, inputs.two_sided)
    if plan is None:
        log(f'atlas tier {n}: page bake (a written sub-texel has no owning triangle)')
        return None
    if not len(plan.slots):
        log(f'atlas tier {n}: page bake (no slots)')
        return None
    plan = place_atlas(inputs.scene, plan, w, h)
    if plan is None:
        log(f'atlas tier {n}: page bake (no atlas layout)')
        return None
    image, side = inputs.image, plan.side
    # (one array: it sets the sentinel and then takes the baked pixels, so a large atlas costs one buffer)
    values = np.full(side * side * 4, -1, np.float32)
    objects = []
    try:
        atlas_objects(inputs, plan, atlas_mesh(inputs.scene, plan, w, h), objects)
        image.generated_width = image.generated_height = side
        image.pixels.foreach_set(values)
        select_only(objects)
        bpy.ops.object.bake(type='DIFFUSE', pass_filter={'DIRECT', 'INDIRECT'}, margin=0, use_clear=False,
                            target='IMAGE_TEXTURES')
        image.pixels.foreach_get(values)
    finally:
        remove_atlas_objects(objects)
    values = values.reshape(side, side, 4)[:, :, :3]
    slots = len(plan.slots)
    raw, missing = scatter_atlas(plan, values, (3 * h, 3 * w))
    # (merge_extra weights a tier by the base bake's owned counts, so a tier with a slot missing is never merged)
    if missing:
        log(f'atlas tier {n}: page bake ({missing} of {slots} slots unwritten)')
        return None
    baked = int((values >= 0).all(axis=2).sum())
    log(f'atlas tier {n}: slots {slots}, side {side}, atlas pixels baked {baked} ({baked / slots:.2f})')
    return raw


def top_up_edge_texels(diffuse, corrected, owned, image, mask_image, buffer, h, w, job, diffuse_scale=None,
                       atlas=None, written=None):
    """Bake again, only where a texel owns few sub-texels, with other seeds, and merge by sample count."""
    cycles = bpy.context.scene.cycles
    base_spp = cycles.samples
    tiers = plan_extra_passes(owned, base_spp, job['texel_samples'])
    extra = []
    extra_corrected = []
    pixels = None
    seed = cycles.seed
    owner = selected = active = None
    # (the page's own image is not needed while the tiers bake on the atlas, so it is held at one texel)
    held = False
    if atlas is not None and tiers:
        selected, active = list(bpy.context.selected_objects), bpy.context.view_layer.objects.active
        owner = subtexel_owner(atlas.scene, atlas.page, w, h)
        release_page_image(image)
        held = True
    try:
        for n, (spp, mask) in enumerate(tiers):
            # (a repeated seed would repeat the noise instead of averaging it away)
            cycles.seed = seed + 1 + n
            cycles.samples = spp
            raw = atlas_tier(n, mask, atlas, owner, written, h, w) if held else None
            if raw is None:
                if held:
                    size_page_image(image, w, h)
                    select_only(selected, active)
                if pixels is None:
                    pixels = np.empty((h, w, 4), np.float32)
                raw = page_tier(image, mask_image, mask, buffer, h, w, pixels)
                if held:
                    release_page_image(image)
            extra.append(downsample_written(raw)[0])
            if diffuse_scale is not None:
                extra_corrected.append(downsample_written(raw, diffuse_scale)[0])
    finally:
        cycles.seed, cycles.samples = seed, base_spp
        reset_mask(mask_image)
        if held:
            size_page_image(image, w, h)
            release_page_image(atlas.image)
            select_only(selected, active)
    diffuse = merge_extra(diffuse, owned, base_spp, tiers, extra)
    corrected = (merge_extra(corrected, owned, base_spp, tiers, extra_corrected)
                 if diffuse_scale is not None else diffuse)
    return diffuse, corrected


def select_only(objects, active=None):
    for obj in bpy.context.scene.objects:
        obj.select_set(False)
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = active or objects[0]


def _bake_page(page, chosen, image, h, w, job, mask_image, diffuse_scale=None, atlas=None):
    select_only(chosen)
    # a covered edge can miss every sub-texel; zero cannot distinguish that from a baked shadow
    buffer = np.empty(3 * h * 3 * w * 4, np.float32)
    clear_page(image, buffer)
    bpy.ops.object.bake(type='DIFFUSE', pass_filter={'DIRECT', 'INDIRECT'}, margin=0, use_clear=False,
                        target='IMAGE_TEXTURES')
    raw = read_page(image, buffer, 3 * h, 3 * w)
    diffuse, written, owned = downsample_and_owned(raw)
    corrected = downsample_written(raw, diffuse_scale)[0] if diffuse_scale is not None else diffuse
    if mask_image is not None:
        # (the top-up clears the buffer this view reads)
        diffuse, corrected = top_up_edge_texels(diffuse, corrected, owned, image, mask_image, buffer, h, w, job,
                                               diffuse_scale, atlas, (raw >= 0).all(axis=2) if atlas is not None else None)
    if job.get('lamps_only'):
        return corrected[::-1].copy(), diffuse[::-1].copy(), written[::-1].copy()
    clear_page(image, buffer)
    bpy.ops.object.bake(type='EMIT', margin=0, use_clear=False, target='IMAGE_TEXTURES')
    # (the glow run's emitters are the flagged receivers, whose own texels the gather has already filled)
    emit, emit_written = downsample_written(read_page(image, buffer, 3 * h, 3 * w),
                                            diffuse_scale if job.get('glow') else None)
    warn_if_masks_differ(written, emit_written, page)
    # Blender's row 0 is the bottom; the diffuse part alone comes back too, for the vertex pass to add emission once
    return (corrected + emit)[::-1].copy(), diffuse[::-1].copy(), written[::-1].copy()


def sample_bilinear(image, u, v, repeat):
    """image (h,w,c), texel centres at half-integers, row 0 = v 0; u, v arrays; clamped or wrapped at the edges."""
    h, w = image.shape[:2]
    x, y = u * w - 0.5, v * h - 0.5
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    if repeat:
        xs, ys = (x0 % w, (x0 + 1) % w), (y0 % h, (y0 + 1) % h)
    else:
        xs = (np.clip(x0, 0, w - 1), np.clip(x0 + 1, 0, w - 1))
        ys = (np.clip(y0, 0, h - 1), np.clip(y0 + 1, 0, h - 1))
    top = image[ys[0], xs[0]] * (1 - fx) + image[ys[0], xs[1]] * fx
    bottom = image[ys[1], xs[0]] * (1 - fx) + image[ys[1], xs[1]] * fx
    return top * (1 - fy) + bottom * fy


# a hit's own sub-texel can belong to an ordinary neighbour or the padding, so a flagged hit reads the nearest
# flagged sample within this many sub-texels (one texel), and nothing beyond
FLAGGED_REACH = 3


def flagged_subtexels(shape, xy, rgb):
    """Each flagged sample's light at its own sub-texel, spread outward to the sub-texels FLAGGED_REACH steps away
    that no sample holds; vertex_gpu's copy (raster.py) must stay identical."""
    height, width = shape
    source = np.zeros((height, width, 3), np.float32)
    filled = np.zeros((height, width), bool)
    source[xy[:, 1], xy[:, 0]] = rgb
    filled[xy[:, 1], xy[:, 0]] = True
    for _ in range(FLAGGED_REACH):
        padded, padded_filled = np.pad(source, ((1, 1), (1, 1), (0, 0))), np.pad(filled, 1)
        grown, grown_filled = source.copy(), filled.copy()
        for dy, dx in ((0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)):
            near = padded_filled[1 + dy:1 + dy + height, 1 + dx:1 + dx + width] & ~grown_filled
            grown[near] = padded[1 + dy:1 + dy + height, 1 + dx:1 + dx + width][near]
            grown_filled |= near
        source, filled = grown, grown_filled
    return source


def radical_inverse(i):
    result, scale = 0.0, 0.5
    while i:
        result += scale * (i & 1)
        i >>= 1
        scale *= 0.5
    return result


# the lamp rays are a copy of vertex_gpu's (this script cannot import the package); the Blender parity tests hold
# the two passes equal
def _unit(direction):
    d = np.float64(direction)
    return d / np.linalg.norm(d)


@dataclass
class LampRays:
    point: np.ndarray
    light: np.ndarray
    origin: np.ndarray
    direction: np.ndarray
    limit: np.ndarray
    skip: np.ndarray
    factor: np.ndarray


def _lamp_factor(light, d, to_light, normals, flagged):
    cosine = np.where(flagged, 1.0, np.einsum('ij,ij->i', normals, to_light))
    if light['kind'] != 3:
        return cosine / (d * d)
    s = -(to_light @ _unit(light['axis']))
    cutoff, falloff = float(light['cos_cutoff']), float(light['cos_falloff'])
    # the cutoff test comes first, as in lights.ramp: a falloff wider than the cutoff leaves a step at the cutoff
    width = np.where(falloff > cutoff, falloff - cutoff, 1.0)
    ramp = np.where(s <= cutoff, 0.0, np.where(s >= falloff, 1.0, (s - cutoff) / width))
    return cosine * ramp / (d * d)


def iter_lamp_rays(points, normals, flagged, usable, lights, chunk=LAMP_RAY_CHUNK):
    """The shadow rays of every lamp to every point it lights, ordered by light then point, in blocks of at most chunk rows."""
    points, normals = np.asarray(points, np.float64), np.asarray(normals, np.float64)
    flagged, usable = np.asarray(flagged, bool), np.asarray(usable, bool)
    for k, light in enumerate(lights):
        position = np.float64(light['position'])
        for start in range(0, len(points), chunk):
            rows = np.arange(start, min(start + chunk, len(points)))
            offset = position - points[rows]
            d = np.linalg.norm(offset, axis=1)
            near = (d >= 1e-4) & (d < float(light['reach'])) & usable[rows]
            rows, offset, d = rows[near], offset[near], d[near]
            direction = offset / d[:, None]
            factor = _lamp_factor(light, d, direction, normals[rows], flagged[rows])
            lit = factor > 0
            rows, direction, d, factor = rows[lit], direction[lit], d[lit], factor[lit]
            if not len(rows):
                continue
            yield LampRays(rows.astype(np.int64), np.full(len(rows), k, np.int64),
                           points[rows] + RAY_EPSILON * direction, direction,
                           np.maximum(0.0, d - 2 * RAY_EPSILON), np.full(len(rows), int(light['owner']), np.int64),
                           factor)


class VertexPass:
    """Incident radiosity at the query points as tool builds it: cosine-weighted rays over the receivers and occluders
    (one triangle id space, receivers first), shaded from the baked pages, a shadow ray per reaching sun and lamp,
    and the class ambient as one flat term along the normal."""

    def __init__(self, scene, job, textures, pages):
        n_recv_vertices = len(scene['positions'])
        positions = np.concatenate([scene['positions'], scene['occ_positions']]).astype(np.float64)
        self.n_recv = len(scene['tris'])
        self.tris = np.concatenate([scene['tris'], scene['occ_tris'] + n_recv_vertices]).astype(np.int64)
        self.corners = positions[self.tris]
        self.twin = scene['tri_twin']
        self.tex_uv = np.concatenate([scene['tex_uv'], scene['occ_uv']]).astype(np.float64)
        # occluders have no page, so their lightmap UVs are never read
        self.lm_uv = np.concatenate([scene['lm_uv'], np.zeros((len(scene['occ_positions']), 2))]).astype(np.float64)
        n_occ = len(scene['occ_tris'])
        self.texture_of = np.concatenate([scene['tri_texture'], scene['occ_tri_texture']])
        self.cutout = np.concatenate([scene['tri_cutout'], scene['occ_tri_cutout']])
        self.casts = np.concatenate([scene['tri_casts_shadow'], scene['occ_tri_casts_shadow']])
        self.tint = np.concatenate([scene['tri_tint'], scene['occ_tri_tint']]).astype(np.float64)
        self.page_of = np.concatenate([scene['tri_page'], np.full(n_occ, -1)])
        self.class_of = np.concatenate([scene['tri_class'], np.full(n_occ, -1)])
        self.two_sided = None
        if job.get('ignore_normals_mode') == 'enhanced' and scene.get('tri_ignore_normals', np.empty(0)).any():
            self.two_sided = np.concatenate([scene['tri_ignore_normals'], np.zeros(n_occ, bool)])
        self.emission = np.concatenate([scene['tri_emission'], scene['occ_tri_emission']]).astype(np.float64)
        # a see-through emitter among the occluders is still seen by the rays that gather light (a shadow ray passes it)
        self.glows = np.concatenate([np.zeros(self.n_recv, np.bool_), self.emission[self.n_recv:].any(axis=1)])
        self.normals = np.cross(self.corners[:, 1] - self.corners[:, 0], self.corners[:, 2] - self.corners[:, 0])
        self.owner = np.concatenate([np.full(self.n_recv, -1),
                                     scene.get('occ_tri_owner', np.full(n_occ, -1))]).astype(np.int64)
        self.scene = scene
        self.job = job
        self.light_flat = None
        self.pages = [np.asarray(p, np.float64) for p in pages]
        # per page, the flagged samples' own light that the reflected stages read on flagged hits (None outside them)
        self.flagged_pages = None
        self.textures = [np.asarray(t, np.float32) / 255.0 for t in textures]
        classes = [int(k) for k in job['ambient']] + [int(scene['query_class'].max(initial=0)),
                                                      int(self.class_of.max(initial=0))]
        # indexed by class + 1: class -1 (indoor) has its own ambient and must not wrap onto the last outdoor slot
        self.ambient = np.zeros((max(classes) + 2, 3))
        for key, rgb in job['ambient'].items():
            self.ambient[int(key) + 1] = rgb
        self.bvh = BVHTree.FromPolygons(positions.tolist(), self.tris.tolist(), all_triangles=True)

    def ambient_of(self, cls):
        cls = np.asarray(cls)
        return self.ambient[np.clip(cls + 1, 0, None)] * (cls >= -1)[:, None]

    def barycentric(self, tri, loc):
        """Weights of loc in each triangle, clamped into it (ray_cast gives the hit point but no barycentrics)."""
        a, b, c = self.corners[tri, 0], self.corners[tri, 1], self.corners[tri, 2]
        v0, v1, v2 = b - a, c - a, loc - a
        d00, d01, d11 = (v0 * v0).sum(1), (v0 * v1).sum(1), (v1 * v1).sum(1)
        d20, d21 = (v2 * v0).sum(1), (v2 * v1).sum(1)
        denom = d00 * d11 - d01 * d01
        safe = np.where(denom == 0, 1.0, denom)
        v = (d11 * d20 - d01 * d21) / safe
        w = (d00 * d21 - d01 * d20) / safe
        weights = np.clip(np.stack([1 - v - w, v, w], axis=1), 0, None)
        total = weights.sum(1, keepdims=True)
        # a zero-area triangle has no meaningful weights, so its centre stands in
        return np.where(total > 0, weights / np.where(total > 0, total, 1), 1 / 3)

    def interpolate(self, uv, tri, loc):
        weights = self.barycentric(tri, loc)
        return (uv[self.tris[tri]] * weights[:, :, None]).sum(1)

    def hit_lm_uv(self, tri, loc):
        return self.interpolate(self.lm_uv, np.array([tri]), np.array([loc]))[0]

    def texture_rgba(self, tri, loc):
        """Texture at each hit, raw 8-bit values over 255; rows without a texture are NaN."""
        out = np.full((len(tri), 4), np.nan)
        uv = self.interpolate(self.tex_uv, tri, loc)
        for k in np.unique(self.texture_of[tri]):
            if k < 0:
                continue
            rows = self.texture_of[tri] == k
            sampled = sample_bilinear(self.textures[k], uv[rows, 0], uv[rows, 1], True).astype(np.float64)
            out[rows] = sampled
        return out

    def faces(self, index, heading):
        return float(np.dot(self.normals[index], heading)) < 0

    def cast(self, origin, direction, carried=None, limit=RAY_LENGTH, skip=-1):
        """First solid hit (triangle, point) or (-1, None); cut-out holes and non-casting triangles are passed,
        and a passed triangle's tint multiplies carried (a length 3 array) when one is given. A hit on a triangle
        of owner skip (skip >= 0) is passed without tint, and SKIP_LIMIT of them end the ray as a miss."""
        start = mathutils.Vector(origin)
        heading = mathutils.Vector(direction)
        remaining = limit
        passes = skips = 0
        while passes <= PASS_LIMIT and remaining > 0:
            location, _, index, distance = self.bvh.ray_cast(start, heading, remaining)
            if index is None:
                return -1, None
            if skip >= 0 and self.owner[index] == skip:
                # a lamp's own placement never blocks it; the cap ends a stack of owner triangles as a miss
                skips += 1
                if skips >= SKIP_LIMIT:
                    return -1, None
                start = location + heading * RAY_EPSILON
                remaining -= distance + RAY_EPSILON
                continue
            passes += 1
            twin = int(self.twin[index])
            if twin >= 0 and not self.faces(index, heading) and self.faces(twin, heading):
                # the hit point stays put; interpolation recomputes weights in the facing twin's corner order
                index = twin
            if self.casts[index] or (carried is None and self.glows[index] and self.faces(index, heading)):
                if not (self.cutout[index] and self.texture_of[index] >= 0
                        and self.texture_rgba(np.array([index]), np.array([location]))[0, 3] < 0.5):
                    return index, np.array(location)
            if carried is not None and not self.casts[index]:
                carried *= self.tint[index]
            start = location + heading * RAY_EPSILON
            remaining -= distance + RAY_EPSILON
        return -1, None

    def lamp_rays(self, points, normals, flagged, usable):
        """(point, light, direction, factor, carried tint, cast) of every lamp ray that is not blocked, in the order of
        iter_lamp_rays; cast is whether a shadow ray was traced (a limit of 0 is unoccluded without one)."""
        for lamp in iter_lamp_rays(points, normals, flagged, usable, self.job.get('lights') or [], LAMP_RAY_CHUNK):
            for i in range(len(lamp.point)):
                carried = np.ones(3)
                cast = bool(lamp.limit[i] > 0)
                if cast and self.cast(lamp.origin[i].tolist(), lamp.direction[i].tolist(), carried, lamp.limit[i],
                                      lamp.skip[i])[0] >= 0:
                    yield None, None, None, None, None, True
                    continue
                yield lamp.point[i], lamp.light[i], lamp.direction[i], lamp.factor[i], carried, cast

    def _shade(self, tri, loc, direction, cls, with_ambient=True, rgb=False, with_pages=True, with_emission=True):
        """Radiance arriving along each ray: tri (n,) with -1 for a miss, loc (n,3), direction (n,3), cls (n,) of the query."""
        n = len(tri)
        radiance = np.zeros((n, 3) if rgb else n)
        # a ray that misses sees the sky, which tool never gathers: the flat ambient term in run() stands for it
        hit = np.flatnonzero(tri >= 0)
        if not len(hit):
            return radiance
        t, p = tri[hit], loc[hit]
        rgba = self.texture_rgba(t, p)
        textured = ~np.isnan(rgba[:, 0])
        albedo = np.full((len(hit), 3), float(self.job['albedo_fallback']))
        albedo[textured] = rgba[textured, :3] * float(self.job['albedo_scale'])
        albedo[self.glows[t] & ~self.casts[t]] = 0
        recv = t < self.n_recv
        facing = (direction[hit] * self.normals[t]).sum(1) < 0
        lit = np.zeros((len(hit), 3) if rgb else len(hit))
        occluder = ~recv
        colour = (albedo[occluder] * (self.ambient_of(cls[hit][occluder]) if with_ambient else 0)
                  + (self.emission[t][occluder] * facing[occluder, None] if with_emission else 0))
        lit[occluder] = colour if rgb else colour @ LUMA
        front = recv & (facing | self.two_sided[t]) if self.two_sided is not None else recv & facing
        if front.any():
            ft = t[front]
            lm = self.interpolate(self.lm_uv, ft, p[front])
            page_value = np.zeros((len(ft), 3))
            for k in np.unique(self.page_of[ft]) if with_pages else ():
                rows = self.page_of[ft] == k
                page_value[rows] = sample_bilinear(self.pages[k], lm[rows, 0], lm[rows, 1], False)
                if self.flagged_pages is not None and self.two_sided is not None:
                    rows &= self.two_sided[ft]
                    source = self.flagged_pages[k]
                    if source is None:
                        page_value[rows] = 0
                    else:
                        h, w = source.shape[:2]
                        x = np.clip(np.floor(lm[rows, 0] * w).astype(np.int64), 0, w - 1)
                        y = np.clip(np.floor(lm[rows, 1] * h).astype(np.int64), 0, h - 1)
                        page_value[rows] = source[y, x]
            if self.two_sided is None:
                colour = albedo[front] * (page_value + (self.ambient_of(self.class_of[ft])
                                                       if with_ambient else 0)) + (self.emission[ft] if with_emission else 0)
            else:
                colour = albedo[front] * (page_value + (self.ambient_of(self.class_of[ft]) * facing[front, None]
                                                       if with_ambient else 0)) + (self.emission[ft] * facing[front, None]
                                                                                  if with_emission else 0)
            lit[front] = colour if rgb else colour @ LUMA
        radiance[hit] = lit
        return radiance

    def shade(self, tri, loc, direction, cls):
        return self._shade(tri, loc, direction, cls)

    def shade_rgb(self, tri, loc, direction, cls, with_ambient=True, with_pages=True, with_emission=True):
        return self._shade(tri, loc, direction, cls, with_ambient=with_ambient, rgb=True,
                           with_pages=with_pages, with_emission=with_emission)

    def radiance(self, tri, loc, direction, cls):
        return float(self.shade(np.array([tri]), np.array([loc]), np.array([direction]), np.array([cls]))[0])

    def hemisphere(self, count):
        """Cosine-weighted directions about +z from a Hammersley set, with the polar radius and azimuth they came from."""
        i = np.arange(count)
        u1 = (i + 0.5) / count
        phi = 2 * np.pi * np.array([radical_inverse(int(k)) for k in i])
        r = np.sqrt(u1)
        return np.stack([r * np.cos(phi), r * np.sin(phi), np.sqrt(1 - u1)], axis=1), phi, r

    def sphere(self, count):
        i = np.arange(count)
        u1 = (i + 0.5) / count
        phi = 2 * np.pi * np.array([radical_inverse(int(k)) for k in i])
        z = 1 - 2 * u1
        r = np.sqrt(1 - z * z)
        return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1), phi, r

    def run(self):
        scene, job = self.scene, self.job
        texel_rays = self.gather_texels()
        rays = int(job.get('cpu_vertex_rays', job['vertex_rays']))
        self.rays_used = rays
        queries = len(scene['query_pos'])
        local, phi, r = self.hemisphere(rays)
        height = local[:, 2]
        ignore_normals = scene.get('query_ignore_normals', np.zeros(queries, bool))
        flagged = ignore_normals.any()
        if flagged:
            sphere_local, _, sphere_r = self.sphere(rays)
        pos = scene['query_pos'].astype(np.float64)
        nrm = scene['query_normal'].astype(np.float64)
        length = np.linalg.norm(nrm, axis=1)
        # a zero or non-finite normal has no hemisphere to cast into: that query stays at zero
        usable = np.isfinite(length) & (length > 1e-12)
        nrm = np.where(usable[:, None], nrm / np.where(usable, length, 1.0)[:, None], [0.0, 0.0, 1.0])
        qcls = scene['query_class'].astype(np.int64)
        vector = np.zeros((queries, 3))
        irradiance = np.zeros(queries)
        cast_count = texel_rays
        sun_visible = sun_facing_queries(scene, job['suns'])
        # which queries each sun reaches, for the faces whose plane turns away from it (they take it from their corners)
        self.sun_lit = np.zeros((queries, len(job['suns'])), np.bool_)
        self.sun_throughput = (np.zeros((queries, len(job['suns']), 3), np.float32)
                               if flagged and job.get('ignore_normals_mode', 'tool') == 'tool' else None)
        for first in range(0, queries, QUERY_CHUNK):
            ids = np.arange(first, min(first + QUERY_CHUNK, queries))
            m = len(ids)
            n = nrm[ids]
            axis = np.where((np.abs(n[:, 0]) < 0.9)[:, None], [[1.0, 0, 0]], [[0, 1.0, 0]])
            tangent = np.cross(n, axis)
            tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
            bitangent = np.cross(n, tangent)
            spin = 2 * np.pi * ((ids * GOLDEN_FRACTION + job.get('seed', 0) * SEED_SPIN) % 1.0)
            angle = phi[None, :] + spin[:, None]
            radius = np.where(ignore_normals[ids, None], sphere_r, r) if flagged else r[None, :]
            h = np.where(ignore_normals[ids, None], sphere_local[:, 2], height) if flagged else height[None, :]
            dirs = (radius[:, :, None] * np.cos(angle)[:, :, None] * tangent[:, None, :]
                    + radius[:, :, None] * np.sin(angle)[:, :, None] * bitangent[:, None, :]
                    + h[:, :, None] * n[:, None, :])
            # (tool starts every ray along its own direction: a start along the normal is inside the solid of a
            # neighbouring face that overhangs it)
            tri = np.full((m, rays), -1, np.int64)
            loc = np.zeros((m, rays, 3))
            for a in np.flatnonzero(usable[ids]):
                for b in range(rays):
                    index, point = self.cast((pos[ids[a]] + RAY_EPSILON * dirs[a, b]).tolist(), dirs[a, b].tolist())
                    if index >= 0:
                        tri[a, b] = index
                        loc[a, b] = point
            cast_count += int(usable[ids].sum()) * rays
            lum = self.shade(tri.ravel(), loc.reshape(-1, 3), dirs.reshape(-1, 3),
                             np.repeat(qcls[ids], rays)).reshape(m, rays)
            lum[~usable[ids]] = 0
            solid_angle = np.where(ignore_normals[ids], 4 * np.pi, np.pi) if flagged else np.pi
            vector[ids] = (solid_angle[:, None] if flagged else solid_angle) / rays * (lum[:, :, None] * dirs).sum(1)
            irradiance[ids] = solid_angle / rays * lum.sum(1)
            # (the pass works in pi x tool units, the sun's strength carries pi and a gathered hit has no 1/pi, so
            # the ambient that tool adds as lum x normal takes pi too; without it the sun-to-ambient ratio is off)
            flat = np.pi * (self.ambient_of(qcls[ids]) @ LUMA) * usable[ids]
            vector[ids] += flat[:, None] * n
            irradiance[ids] += flat
            for sun_index, sun in enumerate(job['suns']):
                d = np.float64(sun['direction'])
                d /= np.linalg.norm(d)
                colour = np.float64(sun['color'])
                for a in range(m):
                    cosine = 1.0 if ignore_normals[ids[a]] else float(n[a] @ d)
                    if (not usable[ids[a]] or cosine <= 0 or int(qcls[ids[a]]) not in sun['classes']
                            or not (ignore_normals[ids[a]] or sun_visible[ids[a], sun_index])):
                        continue
                    cast_count += 1
                    carried = np.ones(3)
                    if self.cast((pos[ids[a]] + RAY_EPSILON * d).tolist(), d.tolist(), carried)[0] < 0:
                        strength = float(colour * carried @ LUMA) * float(sun['strength'])
                        vector[ids[a]] += strength * cosine * d
                        irradiance[ids[a]] += strength * cosine
                        self.sun_lit[ids[a], sun_index] = True
                        if self.sun_throughput is not None:
                            self.sun_throughput[ids[a], sun_index] = carried
        lights = job.get('lights') or []
        if self.sun_throughput is not None:
            self.light_flat = np.zeros((queries, 3))
        for point, light, direction, factor, carried, cast in self.lamp_rays(pos, nrm, ignore_normals, usable):
            cast_count += cast
            if point is None:
                continue
            strength = np.float64(lights[light]['strength']) * carried
            weight = float(strength @ LUMA) * factor
            vector[point] += weight * direction
            irradiance[point] += weight
            if self.light_flat is not None and ignore_normals[point]:
                self.light_flat[point] += strength * factor / np.pi
        edge_pos = scene.get('edge_pos', np.empty((0, 3), np.float32))
        self.edge_sun_lit = np.zeros((len(edge_pos), len(job['suns'])), np.bool_)
        for first in range(0, len(edge_pos), QUERY_CHUNK):
            end = min(first + QUERY_CHUNK, len(edge_pos))
            normals = scene['edge_normal'][first:end].astype(np.float64)
            length = np.linalg.norm(normals, axis=1)
            usable = np.isfinite(length) & (length > 1e-12)
            normals /= np.where(usable, length, 1)[:, None]
            for k, sun in enumerate(job['suns']):
                d = np.float64(sun['direction'])
                d /= np.linalg.norm(d)
                enabled = (usable & (normals @ d > 0) & scene['edge_active'][first:end, k]
                           & np.isin(scene['edge_class'][first:end], sun['classes']))
                for i in np.flatnonzero(enabled) + first:
                    cast_count += 1
                    self.edge_sun_lit[i, k] = self.cast(
                        (edge_pos[i].astype(np.float64) + RAY_EPSILON * d).tolist(), d.tolist(), np.ones(3))[0] < 0
        if self.light_flat is not None:
            self.light_flat = self.light_flat.astype(np.float32)
        return vector.astype(np.float32), irradiance.astype(np.float32), cast_count

    def gather_texels(self):
        scene, job = self.scene, self.job
        count = len(scene.get('texel_pos', ()))
        self.texel_rgb = np.zeros((count, 3), np.float32)
        if not count:
            return 0
        pos = scene['texel_pos'].astype(np.float64)
        length = np.linalg.norm(scene['texel_normal'], axis=1)
        usable = np.isfinite(length) & (length > 1e-12)
        casts = 0
        for sun in job['suns']:
            d = np.float64(sun['direction'])
            d /= np.linalg.norm(d)
            for a in np.flatnonzero(usable & np.isin(scene['texel_class'], sun['classes'])):
                carried = np.ones(3)
                casts += 1
                if self.cast((pos[a] + RAY_EPSILON * d).tolist(), d.tolist(), carried)[0] < 0:
                    self.texel_rgb[a] += np.float64(sun['color']) * sun['strength'] / np.pi * carried
        lights = job.get('lights') or []
        normals = scene['texel_normal'].astype(np.float64) / np.where(usable, length, 1)[:, None]
        for point, light, _, factor, carried, cast in self.lamp_rays(pos, normals, np.ones(count, bool), usable):
            casts += cast
            if point is not None:
                self.texel_rgb[point] += np.float64(lights[light]['strength']) * carried * factor / np.pi
        sun_rgb, base = self.texel_rgb, self.pages
        # original emission is direct reception like the sun, so it is in place before the first reflected stage
        # (gathered inside stage zero, a flagged sheet it lights could pass that light on only a stage later)
        casts += self.gather_texel_reflection(with_pages=False, with_emission=True)
        total = sun_rgb + self.texel_rgb
        # (a mixed texel's average is output only: a flagged hit reads the flagged samples' light, an ordinary hit
        # the ordinary samples', or light would cross between surfaces that merely share a texel)
        self.pages, self.flagged_pages = self.ordinary_sources(base), self.flagged_sources(total)
        # cycles keeps its ordinary bounce depth; this bounds only flagged reflected stages, followed by one
        # terminal glow hop, so flagged paths may exceed bounces by one (not a global per-depth budget)
        for _ in range(max(0, int(job['bounces']))):
            casts += self.gather_texel_reflection(with_emission=False)
            total += self.texel_rgb
            # only the new stage travels again; the ordinary pages are consumed once
            self.pages = [np.zeros_like(p) for p in base]
            self.flagged_pages = self.flagged_sources(self.texel_rgb)
        self.texel_rgb = total
        self.pages, self.flagged_pages = self.install_texels(base, total), None
        return casts

    def owned(self, page, w, h):
        """How many of each texel's sub-texels a receiver owns, and how many of those hold a flagged sample."""
        scene = self.scene
        owned = receiver_diffuse_scale(scene, page, w, h, coverage=True)[::-1] >= 0
        owned = owned.reshape(h, 3, w, 3).sum(axis=(1, 3))
        xy = scene['texel_xy'][scene['texel_page'] == page] // 3
        flagged = np.zeros((h, w), np.int64)
        np.add.at(flagged, (xy[:, 1], xy[:, 0]), 1)
        return owned, flagged

    def ordinary_sources(self, base):
        """The baked pages over their ordinary sub-texels alone (the flagged ones bake black)."""
        pages = []
        for page, value in enumerate(base):
            h, w = value.shape[:2]
            owned, flagged = self.owned(page, w, h)
            ordinary = owned - flagged
            scale = np.where(flagged > 0, owned / np.maximum(ordinary, 1) * (ordinary > 0), 1.0)
            pages.append(value * scale[:, :, None])
        return pages

    def flagged_sources(self, rgb):
        scene = self.scene
        sources = []
        for page, value in enumerate(self.pages):
            rows = scene['texel_page'] == page
            sources.append(flagged_subtexels((3 * value.shape[0], 3 * value.shape[1]), scene['texel_xy'][rows],
                                             rgb[rows]) if rows.any() else None)
        return sources

    def gather_texel_reflection(self, with_pages=True, with_emission=True):
        scene, job = self.scene, self.job
        count = len(scene.get('texel_pos', ()))
        self.texel_rgb = np.zeros((count, 3), np.float32)
        if not count:
            return 0
        rays = max(1, round(job.get('cpu_samples', job['samples']) / 9))
        counts = np.full(count, rays, np.int64)
        target = job.get('cpu_texel_samples', job['texel_samples'])
        if target > 0:
            for page, (w, h) in enumerate(job['pages']):
                owned = receiver_diffuse_scale(scene, page, w, h, coverage=True)[::-1] >= 0
                owned = owned.reshape(h, 3, w, 3).sum(axis=(1, 3))
                chosen = scene['texel_page'] == page
                xy = scene['texel_xy'][chosen] // 3
                counts[chosen] = np.maximum(rays, -(-target // np.maximum(owned[xy[:, 1], xy[:, 0]], 1)))
        casts = 0
        pos = scene['texel_pos'].astype(np.float64)
        normals = scene['texel_normal'].astype(np.float64)
        length = np.linalg.norm(normals, axis=1)
        usable = np.isfinite(length) & (length > 1e-12)
        normals /= np.where(usable, length, 1)[:, None]
        batches = []
        for rays in np.unique(counts):
            rows = np.flatnonzero(counts == rays)
            batches.extend((int(rays), rows[first:first + QUERY_CHUNK]) for first in range(0, len(rows), QUERY_CHUNK))
        for rays, ids in batches:
            local, phi, r = self.sphere(rays)
            n = normals[ids]
            axis = np.where((np.abs(n[:, 0]) < .9)[:, None], [[1., 0, 0]], [[0, 1., 0]])
            tangent = np.cross(n, axis)
            tangent /= np.where(usable[ids], np.linalg.norm(tangent, axis=1), 1)[:, None]
            bitangent = np.cross(n, tangent)
            spin = 2 * np.pi * ((ids * GOLDEN_FRACTION + job.get('seed', 0) * SEED_SPIN) % 1)
            angle = phi[None, :] + spin[:, None]
            dirs = (r[None, :, None] * np.cos(angle)[:, :, None] * tangent[:, None, :]
                    + r[None, :, None] * np.sin(angle)[:, :, None] * bitangent[:, None, :]
                    + local[None, :, 2, None] * n[:, None, :])
            tri = np.full((len(ids), rays), -1, np.int64)
            loc = np.zeros((len(ids), rays, 3))
            for a in np.flatnonzero(usable[ids]):
                for b in range(rays):
                    index, point = self.cast((pos[ids[a]] + RAY_EPSILON * dirs[a, b]).tolist(), dirs[a, b].tolist())
                    if index >= 0:
                        tri[a, b], loc[a, b] = index, point
            casts += int(usable[ids].sum()) * rays
            rgb = self.shade_rgb(tri.ravel(), loc.reshape(-1, 3), dirs.reshape(-1, 3),
                                 np.repeat(scene['texel_class'][ids], rays), with_ambient=False,
                                 with_pages=with_pages, with_emission=with_emission)
            self.texel_rgb[ids] = 4 / rays * rgb.reshape(len(ids), rays, 3).sum(axis=1)
        return casts

    def install_texels(self, base, rgb):
        scene = self.scene
        # the gathered value replaces only flagged samples; the baked page is already zero there
        pages = [p.copy() for p in base]
        for page, value in enumerate(pages):
            h, w = value.shape[:2]
            owned = receiver_diffuse_scale(scene, page, w, h, coverage=True)[::-1] >= 0
            owned = owned.reshape(h, 3, w, 3).sum(axis=(1, 3))
            rows = scene['texel_page'] == page
            xy = scene['texel_xy'][rows] // 3
            total = np.zeros_like(value)
            np.add.at(total, (xy[:, 1], xy[:, 0]), rgb[rows])
            value += total / np.maximum(owned, 1)[:, :, None]
        return pages


def main():
    job_path = sys.argv[sys.argv.index('--') + 1]
    root = os.path.dirname(os.path.abspath(job_path))
    with open(job_path) as handle:
        job = json.load(handle)
    started = time.perf_counter()
    out = {
        'seconds_start': np.float64(time.time() - float(os.environ['LIGHTMAPPER_LAUNCH_TIME']))
        if 'LIGHTMAPPER_LAUNCH_TIME' in os.environ else np.float64(0),
        'seconds_scene': np.float64(0), 'seconds_pages': np.float64(0),
        'seconds_vertex': np.float64(0),
    }
    scene_started = time.perf_counter()
    with np.load(os.path.join(root, job['scene'])) as data:
        scene_data = {key: data[key] for key in data.files}
    out['vertex_rays_cast'] = np.int64(0)
    if job.get('vertex_only'):
        with np.load(os.path.join(root, job['diffuse'])) as data:
            diffuse_pages = [data[f'diffuse_{i}'] for i in range(len(job['pages']))]
    else:
        scene, device = setup_scene(job)
        out['samples_used'] = np.int64(job['samples'])
        out['texel_samples_used'] = np.int64(job['texel_samples'])
        log(f'device {device}')
        textures, page_images, mask_images = make_images(os.path.join(root, 'textures.npz'), job['pages'],
                                                        job['texel_samples'] > 0)
        glow_images, glow_transforms = [], []
        # (a lamps-only run adds to pages that already hold the emission)
        if job.get('glow') or job.get('lamps_only'):
            scene_data['tri_emission'] = np.zeros_like(scene_data['tri_emission'])
            scene_data['occ_tri_emission'] = np.zeros_like(scene_data['occ_tri_emission'])
        if job.get('glow'):
            glow_images, glow_transforms = load_glow_images(os.path.join(root, job['glow_pages']), len(job['pages']),
                                                            job.get('test_glow_uncropped'))
        sources = []
        atlas_image = make_atlas_image() if mask_images else None
        receivers, occluders = build_objects(scene_data, job, textures, page_images, mask_images, glow_images,
                                             glow_transforms, sources)
        if not job.get('lamps_only'):
            add_suns(job, receivers, occluders)
        add_lamps(job, receivers, occluders, job.get('lamp_batch', 0))
        out['seconds_scene'] = np.float64(time.perf_counter() - scene_started)
        diffuse_pages = []
        for i, (w, h) in enumerate(job['pages']):
            began = time.perf_counter()
            out[f'page_{i}'], diffuse, out[f'written_{i}'] = bake_page(i, receivers, page_images[i], h, w, job,
                                                                   mask_images[i] if mask_images else None,
                                                                   receiver_diffuse_scale(
                                                                        scene_data, i, w, h, enhanced=job.get('ignore_normals_mode') == 'enhanced'),
                                                                   atlas_inputs(scene_data, job, i, sources, textures,
                                                                                atlas_image, glow_images,
                                                                                glow_transforms) if atlas_image is not None else None)
            diffuse_pages.append(diffuse)
            out[f'diffuse_{i}'] = diffuse
            out['seconds_pages'] = np.float64(out['seconds_pages'] + time.perf_counter() - began)
            log(f'page {i} ({w}x{h}) baked in {time.perf_counter() - began:.2f} s')
        out['device'] = np.array(device)
    if job.get('vertex_only') or job.get('vertex_pass', 'cpu') == 'cpu':
        with np.load(os.path.join(root, 'textures.npz')) as data:
            raw_textures = [data[f'tex_{k}'] for k in range(len(data.files))]
        began = time.perf_counter()
        vertex_pass = VertexPass(scene_data, job, raw_textures, diffuse_pages)
        out['vertex_vector'], out['vertex_irradiance'], rays = vertex_pass.run()
        out['vertex_sun_lit'] = vertex_pass.sun_lit
        if vertex_pass.sun_throughput is not None:
            out['vertex_sun_throughput'] = vertex_pass.sun_throughput
        out['texel_rgb'] = vertex_pass.texel_rgb
        out['edge_sun_lit'] = vertex_pass.edge_sun_lit
        if vertex_pass.light_flat is not None:
            out['vertex_light_flat'] = vertex_pass.light_flat
        out['vertex_rays_cast'] = np.int64(rays)
        out['vertex_rays_used'] = np.int64(vertex_pass.rays_used)
        elapsed = time.perf_counter() - began
        out['seconds_vertex'] = np.float64(elapsed)
        log(f'vertex pass: {len(out["vertex_vector"])} queries, {rays} rays in {elapsed:.2f} s ({rays / max(elapsed, 1e-9):.0f} rays/s)')
    np.savez(os.path.join(root, job['out']), **out)
    log(f'done in {time.perf_counter() - started:.2f} s')


# a test imports this file to drive the vertex pass alone
if __name__ == '__main__':
    main()
