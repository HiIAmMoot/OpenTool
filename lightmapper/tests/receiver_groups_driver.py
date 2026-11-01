"""Runs inside Blender: builds the receiver objects of a job and records how each one was grouped and shaded."""

import importlib.util
import json
import os
import sys

import numpy as np


def main():
    root = sys.argv[sys.argv.index('--') + 1]
    spec = importlib.util.spec_from_file_location('lm_bake', os.path.join(os.environ['LIGHTMAPPER_BAKE_DIR'], 'bake.py'))
    bake = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bake)
    import bpy
    with open(os.path.join(root, 'job.json')) as handle:
        job = json.load(handle)
    bake.setup_scene(job)
    with np.load(os.path.join(root, 'scene.npz')) as data:
        scene_data = {key: data[key] for key in data.files}
    textures, images, masks = bake.make_images(os.path.join(root, 'textures.npz'), job['pages'], False)
    glow_images, glow_transforms = [], []
    if job.get('glow'):
        glow_images, glow_transforms = bake.load_glow_images(os.path.join(root, job['glow_pages']), len(job['pages']))
    receivers, _ = bake.build_objects(scene_data, job, textures, images, masks, glow_images, glow_transforms)
    groups = []
    for obj, page, cls in receivers:
        slots = []
        for material in obj.data.materials:
            nodes = material.node_tree.nodes
            diffuse = next(n for n in nodes if n.bl_idname == 'ShaderNodeBsdfDiffuse')
            surface = next(n for n in nodes if n.bl_idname == 'ShaderNodeOutputMaterial').inputs['Surface']
            slots.append({
                'pure_emission': surface.links[0].from_node.bl_idname == 'ShaderNodeEmission',
                'translucent': any(n.bl_idname == 'ShaderNodeBsdfTranslucent' for n in nodes),
                'zero_albedo': not diffuse.inputs['Color'].is_linked
                               and tuple(diffuse.inputs['Color'].default_value[:3]) == (0, 0, 0),
                'glow_image': next((n.image.name for n in nodes if n.bl_idname == 'ShaderNodeTexImage'
                                    and n.image is not None and n.image.name.startswith('glow_')), None)})
        groups.append({'page': page, 'cls': cls, 'triangles': len(obj.data.polygons), 'flagged': bool(obj['flagged']),
                       'sun_facing': [bool(v) for v in obj['sun_facing']], 'slots': slots,
                       'slot_of_triangle': [int(p.material_index) for p in obj.data.polygons]})
    with open(os.path.join(root, 'groups.json'), 'w') as handle:
        json.dump(groups, handle)


main()
