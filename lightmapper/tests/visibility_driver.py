"""Runs inside Blender: builds the receiver objects of a job and records their ray visibility."""

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
    receivers, _ = bake.build_objects(scene_data, job, textures, images)
    out = {'objects': [
        {'min_x': float(min(v.co.x for v in obj.data.vertices)),
         **{name: bool(getattr(obj, name)) for name in (
             'visible_diffuse', 'visible_glossy', 'visible_transmission', 'visible_shadow', 'visible_camera')}}
        for obj, _, _ in receivers]}
    with open(os.path.join(root, 'visibility.json'), 'w') as handle:
        json.dump(out, handle)


main()
