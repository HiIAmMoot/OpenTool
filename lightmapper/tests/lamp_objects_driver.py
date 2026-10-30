"""Runs inside Blender: builds the objects and lamps of a job and records their names, owners and collections."""

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
    with open(os.path.join(root, 'job.json')) as handle:
        job = json.load(handle)
    bake.setup_scene(job)
    with np.load(os.path.join(root, 'scene.npz')) as data:
        scene_data = {key: data[key] for key in data.files}
    textures, images, _ = bake.make_images(os.path.join(root, 'textures.npz'), job['pages'])
    receivers, occluders = bake.build_objects(scene_data, job, textures, images)
    bake.add_suns(job, receivers, occluders)
    out = {'error': None}
    try:
        bake.add_lamps(job, receivers, occluders, 0)
    except RuntimeError as error:
        out['error'] = str(error)
    bpy = bake.bpy
    out['objects'] = [{'name': o.name, 'owner': o.get('owner'),
                       'collections': [c.name for c in bpy.data.collections if o.name in c.objects]}
                      for o in bpy.data.objects if o.type == 'MESH']
    out['lamps'] = [{'name': o.name, 'receivers': o.light_linking.receiver_collection.name,
                     'blockers': o.light_linking.blocker_collection.name}
                    for o in bpy.data.objects if o.type == 'LIGHT' and o.name.startswith('lamp_')
                    and o.light_linking.receiver_collection]
    with open(os.path.join(root, 'lamp_objects.json'), 'w') as handle:
        json.dump(out, handle)


main()
