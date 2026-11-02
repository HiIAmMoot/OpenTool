"""Runs inside Blender: runs setup_scene on the job and records the Cycles sample clamps and adaptive sampling."""

import importlib.util
import json
import os
import sys

import bpy
import numpy as np


class Proxy:
    def __init__(self, real, over):
        self.real, self.over = real, over

    def __getattr__(self, name):
        return self.over[name] if name in self.over else getattr(self.real, name)


def main():
    root = sys.argv[sys.argv.index('--') + 1]
    spec = importlib.util.spec_from_file_location('lm_bake', os.path.join(os.environ['LIGHTMAPPER_BAKE_DIR'], 'bake.py'))
    bake = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bake)
    with open(os.path.join(root, 'job.json')) as handle:
        job = json.load(handle)
    scene, _ = bake.setup_scene(job)
    seen = []

    def spy(*args, **kwargs):
        seen.append({'tile_size': int(scene.cycles.tile_size), 'use_auto_tile': bool(scene.cycles.use_auto_tile)})
        return bpy.ops.object.bake(*args, **kwargs)

    bake.bpy = Proxy(bpy, {'ops': Proxy(bpy.ops, {'object': Proxy(bpy.ops.object, {'bake': spy})})})
    with np.load(os.path.join(root, 'scene.npz')) as data:
        scene_data = {key: data[key] for key in data.files}
    textures, images, masks = bake.make_images(os.path.join(root, 'textures.npz'), job['pages'])
    receivers, occluders = bake.build_objects(scene_data, job, textures, images)
    bake.add_suns(job, receivers, occluders)
    w, h = job['pages'][0]
    bake.bake_page(0, receivers, images[0], h, w, job)
    clamps = {'sample_clamp_direct': float(scene.cycles.sample_clamp_direct),
              'sample_clamp_indirect': float(scene.cycles.sample_clamp_indirect),
              'use_adaptive_sampling': bool(scene.cycles.use_adaptive_sampling),
              'first_bake': seen[0]}
    with open(os.path.join(root, 'clamps.json'), 'w') as handle:
        json.dump(clamps, handle)


main()
