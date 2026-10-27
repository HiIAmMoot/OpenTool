"""Runs inside Blender only (blender -b --python bake.py -- job.json): builds the job's scene and bakes the lightmap pages.

This script imports Blender's Python API and needs a GPL-compatible license. It exchanges data with the
pipeline through job.json, scene.npz, textures.npz and out.npz. The page bakes are Cycles diffuse + emission
passes into one float image per page; rows come back in our convention (row 0 = v 0). The per-vertex pass
runs here only on the CPU path or a vertex-only relaunch: Blender's BVH tree casts the rays and numpy
shades the hits in batches.
"""

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
PASS_LIMIT = 8
LUMA = np.float64([0.2126, 0.7152, 0.0722])
GOLDEN_FRACTION = 0.618034
SEED_STRIDE = 1000
SEED_SPIN = 0.7548776662466927
# queries per batch: bounds the hit arrays held before they are shaded
QUERY_CHUNK = 256


def log(message):
    print(f'[bake] {message}', flush=True)


def pick_device(kinds: list[str], required: bool, hiprt: bool) -> str:
    """Enable every device of the first working GPU type; the name of what was used, 'CPU' when none."""
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


def setup_scene(job):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    cycles = scene.cycles
    cycles.samples = max(1, round(job['samples'] / 9))
    cycles.seed = job.get('seed', 0) * SEED_STRIDE
    log(f'cycles seed {cycles.seed}')
    cycles.max_bounces = cycles.diffuse_bounces = job['bounces']
    cycles.glossy_bounces = cycles.transmission_bounces = cycles.volume_bounces = 0
    cycles.use_denoising = False
    world = bpy.data.worlds.new('world')
    world.use_nodes = True
    background = world.node_tree.nodes['Background']
    background.inputs['Color'].default_value = (0, 0, 0, 1)
    background.inputs['Strength'].default_value = 0
    scene.world = world
    wanted = job.get('device', 'OPTIX')
    kinds = list(DEVICE_TYPES[DEVICE_TYPES.index(wanted):]) if wanted in DEVICE_TYPES else []
    return scene, pick_device(job.get('cycles_devices', kinds), job.get('cycles_required', False),
                              job.get('hiprt', False))


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
        # the 3x float buffer is only allocated while its page bakes (about 151 MB per 1024 page)
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


def size_page_image(image, w, h):
    image.generated_width, image.generated_height = 3 * w, 3 * h


def release_page_image(image):
    image.generated_width = image.generated_height = 1


def reset_mask(image):
    # one white texel reads as all ones everywhere (the node extends it), so a mask costs memory only while a top-up
    # pass needs a real one; a resized generated image comes back black, so the texel is written again
    image.generated_width = image.generated_height = 1
    image.pixels.foreach_set(np.ones(4, np.float32))


def make_material(name, texture, cutout, emission, job, page_image, tint=(), emit_only=False, mask_image=None):
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
    shader = diffuse.outputs['BSDF']
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
    return facing


def build_objects(scene_data, job, textures, page_images, mask_images=()):
    """Receivers per page, class, shadow and sun visibility; occluders per material and shadow."""
    positions, tris = scene_data['positions'], scene_data['tris']
    receivers = []
    # a see-through triangle still has to meet shadow rays when its tint dims them
    tinted = ~scene_data['tri_casts_shadow'] & (scene_data['tri_tint'] < 1).any(axis=1)
    keys = np.column_stack([scene_data['tri_page'], scene_data['tri_class'],
                            scene_data['tri_casts_shadow'] | tinted,
                            receiver_sun_bits(scene_data, job['suns'])])
    distinct_keys, labels = np.unique(keys, axis=0, return_inverse=True)
    order = np.argsort(labels, kind='stable')
    groups = np.split(order, np.flatnonzero(np.diff(labels[order])) + 1) if len(order) else []
    for key, picked in zip(distinct_keys, groups):
        page, cls, casts = int(key[0]), int(key[1]), bool(key[2])
        used, local = np.unique(tris[picked].ravel(), return_inverse=True)
        local = local.reshape(-1, 3)
        tri_props = [(int(scene_data['tri_texture'][t]), bool(scene_data['tri_cutout'][t]),
                      tuple(float(c) for c in scene_data['tri_emission'][t]),
                      tuple(float(c) for c in scene_data['tri_tint'][t]) if tinted[t] else ()) for t in picked]
        distinct = sorted(set(tri_props))
        materials = [make_material(f'recv_{page}_{cls}_{n}', textures[tex] if tex >= 0 else None, cut, emit,
                                   job, page_images[page], tint,
                                   mask_image=mask_images[page] if mask_images else None)
                     for n, (tex, cut, emit, tint) in enumerate(distinct)]
        obj = make_object(f'recv_{page}_{cls}_{int(casts)}', positions[used],
                          {UV_TEXTURE: flip_v(scene_data['tex_uv'][used]),
                           UV_LIGHTMAP: flip_v(scene_data['lm_uv'][used])},
                          local, materials, [distinct.index(p) for p in tri_props], casts,
                          scene_data['normals'][used])
        obj['sun_facing'] = key[3:].tolist()
        receivers.append((obj, page, cls))

    occluders = []
    occ_tris = scene_data['occ_tris']
    # a see-through occluder meets shadow rays too when its tint dims them
    otinted = ~scene_data['occ_tri_casts_shadow'] & (scene_data['occ_tri_tint'] < 1).any(axis=1)
    okeys = np.concatenate([
        np.stack([scene_data['occ_tri_texture'], scene_data['occ_tri_cutout'],
                  scene_data['occ_tri_casts_shadow'] | otinted], axis=1).astype(np.float64),
        scene_data['occ_tri_emission'].astype(np.float64),
        emit_only_of(scene_data)[:, None].astype(np.float64),
        np.where(otinted[:, None], scene_data['occ_tri_tint'], 1).astype(np.float64)], axis=1)
    for key in np.unique(okeys, axis=0) if len(occ_tris) else []:
        tex, cut, casts = int(key[0]), bool(key[1]), bool(key[2])
        emit, tint = tuple(float(c) for c in key[3:6]), tuple(float(c) for c in key[7:10])
        picked = np.flatnonzero((okeys == key).all(axis=1))
        used, local = np.unique(occ_tris[picked].ravel(), return_inverse=True)
        material = make_material(f'occ_{tex}_{int(cut)}', textures[tex] if tex >= 0 else None, cut,
                                  emit, job, None, tint if tint != (1.0, 1.0, 1.0) else (),
                                  emit_only=bool(key[6]))
        occluders.append(make_object(f'occ_{tex}_{int(cut)}_{int(casts)}',
                                     scene_data['occ_positions'][used],
                                     {UV_TEXTURE: flip_v(scene_data['occ_uv'][used])},
                                     local.reshape(-1, 3), [material], np.zeros(len(picked), np.int32), casts))
    return receivers, occluders


def emit_only_of(scene_data):
    # decided by write_job before the source weights zero the emission, so a pane stays transparent without it
    return np.asarray(scene_data['occ_tri_emit_only'], np.bool_)


# Cycles keeps at most 64 light-linking sets and past that warns on every link and bakes a hundred
# times slower, so more distinct receiver sets than this is refused rather than linked wrongly
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


def read_page(image, h, w):
    buffer = np.empty(h * w * 4, np.float32)
    image.pixels.foreach_get(buffer)
    return buffer.reshape(h, w, 4)[:, :, :3].copy()


def downsample_written(rgb):
    """Box-average 3x3 sub-texels, excluding the negative unwritten sentinel, not black samples."""
    h, w = rgb.shape[:2]
    written = (rgb >= 0).all(axis=2)
    count = written.reshape(h // 3, 3, w // 3, 3).sum(axis=(1, 3))
    total = np.where(written[:, :, None], rgb, 0).reshape(h // 3, 3, w // 3, 3, 3).sum(axis=(1, 3))
    return (total / np.maximum(count, 1)[:, :, None]).astype(np.float32), count > 0


# an edge texel owns few of its 9 sub-texels, so with the same samples per sub-texel it is a noisier
# estimate than an interior one; a texel is topped up unless it already has this share of the target
ENOUGH_SHARE = 0.95
# a tier tops its texels up for its smallest owned count, so the largest in it overshoots by about this
TIER_SPAN = 1.5


def owned_subsamples(rgb):
    """How many of each texel's 3x3 sub-texels the bake wrote."""
    h, w = rgb.shape[:2]
    return (rgb >= 0).all(axis=2).reshape(h // 3, 3, w // 3, 3).sum(axis=(1, 3))


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
    total = base * (owned * base_spp)[:, :, None]
    for (spp, mask), image in zip(tiers, extra):
        # (a texel outside the mask may hold inf or NaN, and a zero weight would not clear it)
        total = total + np.where(mask[:, :, None], image, 0) * (owned * spp)[:, :, None]
    weight = effective_samples(owned, base_spp, tiers)
    return (total / np.maximum(weight, 1)[:, :, None]).astype(np.float32)


def warn_if_masks_differ(diffuse_written, emit_written, page):
    # the page's written mask comes from the diffuse pass alone; the emit pass should write the same texels
    differs = not np.array_equal(diffuse_written, emit_written)
    if differs:
        log(f'warning: page {page}: the emit pass wrote {int((emit_written != diffuse_written).sum())} '
            f'texels differently from the diffuse pass (a Blender behaviour change?)')
    return differs


def bake_page(page, receivers, image, h, w, job, mask_image=None):
    chosen = [obj for obj, p, _ in receivers if p == page]
    if not chosen:
        return (np.zeros((h, w, 3), np.float32), np.zeros((h, w, 3), np.float32),
                np.zeros((h, w), np.bool_))
    size_page_image(image, w, h)
    try:
        return _bake_page(page, chosen, image, h, w, job, mask_image)
    finally:
        release_page_image(image)


def top_up_edge_texels(diffuse, owned, image, mask_image, sentinel, h, w, job):
    """Bake again, only where a texel owns few sub-texels, with other seeds, and merge by sample count."""
    cycles = bpy.context.scene.cycles
    base_spp = cycles.samples
    tiers = plan_extra_passes(owned, base_spp, job['texel_samples'])
    extra = []
    seed = cycles.seed
    try:
        for n, (spp, mask) in enumerate(tiers):
            # (a repeated seed would repeat the noise instead of averaging it away)
            cycles.seed = seed + 1 + n
            cycles.samples = spp
            mask_image.generated_width, mask_image.generated_height = w, h
            pixels = np.zeros((h, w, 4), np.float32)
            pixels[mask] = 1
            mask_image.pixels.foreach_set(pixels.ravel())
            image.pixels.foreach_set(sentinel)
            bpy.ops.object.bake(type='DIFFUSE', pass_filter={'DIRECT', 'INDIRECT'}, margin=0, use_clear=False,
                                target='IMAGE_TEXTURES')
            extra.append(downsample_written(read_page(image, 3 * h, 3 * w))[0])
    finally:
        cycles.seed, cycles.samples = seed, base_spp
        reset_mask(mask_image)
    return merge_extra(diffuse, owned, base_spp, tiers, extra)


def _bake_page(page, chosen, image, h, w, job, mask_image):
    view_layer = bpy.context.view_layer
    for obj in bpy.context.scene.objects:
        obj.select_set(False)
    for obj in chosen:
        obj.select_set(True)
    view_layer.objects.active = chosen[0]
    # a covered edge can miss every sub-texel; zero cannot distinguish that from a baked shadow
    sentinel = np.full(3 * h * 3 * w * 4, -1, np.float32)
    image.pixels.foreach_set(sentinel)
    bpy.ops.object.bake(type='DIFFUSE', pass_filter={'DIRECT', 'INDIRECT'}, margin=0, use_clear=False,
                        target='IMAGE_TEXTURES')
    raw = read_page(image, 3 * h, 3 * w)
    diffuse, written = downsample_written(raw)
    if mask_image is not None:
        diffuse = top_up_edge_texels(diffuse, owned_subsamples(raw), image, mask_image, sentinel, h, w, job)
    image.pixels.foreach_set(sentinel)
    bpy.ops.object.bake(type='EMIT', margin=0, use_clear=False, target='IMAGE_TEXTURES')
    emit, emit_written = downsample_written(read_page(image, 3 * h, 3 * w))
    warn_if_masks_differ(written, emit_written, page)
    # Blender's row 0 is the bottom; the diffuse part alone comes back too, for the vertex pass to add emission once
    return (diffuse + emit)[::-1].copy(), diffuse[::-1].copy(), written[::-1].copy()


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


def radical_inverse(i):
    result, scale = 0.0, 0.5
    while i:
        result += scale * (i & 1)
        i >>= 1
        scale *= 0.5
    return result


class VertexPass:
    """Incident radiosity at the query points as tool builds it: cosine-weighted rays over the receivers and occluders
    (one triangle id space, receivers first), shaded from the baked pages, a shadow ray per reaching sun, and the
    class ambient as one flat term along the normal."""

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
        self.emission = np.concatenate([scene['tri_emission'], scene['occ_tri_emission']]).astype(np.float64)
        # a see-through emitter among the occluders is still seen by the rays that gather light (a shadow ray passes it)
        self.glows = np.concatenate([np.zeros(self.n_recv, np.bool_), self.emission[self.n_recv:].any(axis=1)])
        self.normals = np.cross(self.corners[:, 1] - self.corners[:, 0], self.corners[:, 2] - self.corners[:, 0])
        self.scene = scene
        self.job = job
        self.pages = [np.asarray(p, np.float64) for p in pages]
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

    def cast(self, origin, direction, carried=None):
        """First solid hit (triangle, point) or (-1, None); cut-out holes and non-casting triangles are passed,
        and a passed triangle's tint multiplies carried (a length 3 array) when one is given."""
        start = mathutils.Vector(origin)
        heading = mathutils.Vector(direction)
        remaining = RAY_LENGTH
        for _ in range(PASS_LIMIT + 1):
            location, _, index, distance = self.bvh.ray_cast(start, heading, remaining)
            if index is None:
                return -1, None
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

    def shade(self, tri, loc, direction, cls):
        """Radiance arriving along each ray: tri (n,) with -1 for a miss, loc (n,3), direction (n,3), cls (n,) of the query."""
        n = len(tri)
        radiance = np.zeros(n)
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
        lit = np.zeros(len(hit))
        occluder = ~recv
        lit[occluder] = (albedo[occluder] * self.ambient_of(cls[hit][occluder])
                         + self.emission[t][occluder] * facing[occluder, None]) @ LUMA
        front = recv & facing
        if front.any():
            ft = t[front]
            lm = self.interpolate(self.lm_uv, ft, p[front])
            page_value = np.zeros((len(ft), 3))
            for k in np.unique(self.page_of[ft]):
                rows = self.page_of[ft] == k
                page_value[rows] = sample_bilinear(self.pages[k], lm[rows, 0], lm[rows, 1], False)
            colour = albedo[front] * (page_value + self.ambient_of(self.class_of[ft])) + self.emission[ft]
            lit[front] = colour @ LUMA
        radiance[hit] = lit
        return radiance

    def radiance(self, tri, loc, direction, cls):
        return float(self.shade(np.array([tri]), np.array([loc]), np.array([direction]), np.array([cls]))[0])

    def hemisphere(self, count):
        """Cosine-weighted directions about +z from a Hammersley set, with the polar radius and azimuth they came from."""
        i = np.arange(count)
        u1 = (i + 0.5) / count
        phi = 2 * np.pi * np.array([radical_inverse(int(k)) for k in i])
        r = np.sqrt(u1)
        return np.stack([r * np.cos(phi), r * np.sin(phi), np.sqrt(1 - u1)], axis=1), phi, r

    def run(self):
        scene, job = self.scene, self.job
        rays = int(job['vertex_rays'])
        queries = len(scene['query_pos'])
        local, phi, r = self.hemisphere(rays)
        height = local[:, 2]
        pos = scene['query_pos'].astype(np.float64)
        nrm = scene['query_normal'].astype(np.float64)
        length = np.linalg.norm(nrm, axis=1)
        # a zero or non-finite normal has no hemisphere to cast into: that query stays at zero
        usable = np.isfinite(length) & (length > 1e-12)
        nrm = np.where(usable[:, None], nrm / np.where(usable, length, 1.0)[:, None], [0.0, 0.0, 1.0])
        qcls = scene['query_class'].astype(np.int64)
        vector = np.zeros((queries, 3))
        irradiance = np.zeros(queries)
        cast_count = 0
        sun_visible = sun_facing_queries(scene, job['suns'])
        # which queries each sun reaches, for the faces whose plane turns away from it (they take it from their corners)
        self.sun_lit = np.zeros((queries, len(job['suns'])), np.bool_)
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
            dirs = (r[None, :, None] * np.cos(angle)[:, :, None] * tangent[:, None, :]
                    + r[None, :, None] * np.sin(angle)[:, :, None] * bitangent[:, None, :]
                    + height[None, :, None] * n[:, None, :])
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
            vector[ids] = np.pi / rays * (lum[:, :, None] * dirs).sum(1)
            irradiance[ids] = np.pi / rays * lum.sum(1)
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
                    cosine = float(n[a] @ d)
                    if (not usable[ids[a]] or cosine <= 0 or int(qcls[ids[a]]) not in sun['classes']
                            or not sun_visible[ids[a], sun_index]):
                        continue
                    cast_count += 1
                    carried = np.ones(3)
                    if self.cast((pos[ids[a]] + RAY_EPSILON * d).tolist(), d.tolist(), carried)[0] < 0:
                        strength = float(colour * carried @ LUMA) * float(sun['strength'])
                        vector[ids[a]] += strength * cosine * d
                        irradiance[ids[a]] += strength * cosine
                        self.sun_lit[ids[a], sun_index] = True
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
        return vector.astype(np.float32), irradiance.astype(np.float32), cast_count


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
        log(f'device {device}')
        textures, page_images, mask_images = make_images(os.path.join(root, 'textures.npz'), job['pages'],
                                                        job['texel_samples'] > 0)
        receivers, occluders = build_objects(scene_data, job, textures, page_images, mask_images)
        add_suns(job, receivers, occluders)
        out['seconds_scene'] = np.float64(time.perf_counter() - scene_started)
        diffuse_pages = []
        for i, (w, h) in enumerate(job['pages']):
            began = time.perf_counter()
            out[f'page_{i}'], diffuse, out[f'written_{i}'] = bake_page(i, receivers, page_images[i], h, w, job,
                                                                   mask_images[i] if mask_images else None)
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
        out['edge_sun_lit'] = vertex_pass.edge_sun_lit
        out['vertex_rays_cast'] = np.int64(rays)
        elapsed = time.perf_counter() - began
        out['seconds_vertex'] = np.float64(elapsed)
        log(f'vertex pass: {len(out["vertex_vector"])} queries, {rays} rays in {elapsed:.2f} s ({rays / max(elapsed, 1e-9):.0f} rays/s)')
    np.savez(os.path.join(root, job['out']), **out)
    log(f'done in {time.perf_counter() - started:.2f} s')


# a test imports this file to drive the vertex pass alone
if __name__ == '__main__':
    main()
