"""Runs inside Blender: bakes each page in turn and records every page image's size at each read."""

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
    seen = []
    real_read = bake.read_page

    def spy(image, h, w):
        seen.append([int(i.size[0] * i.size[1]) for i in images])
        return real_read(image, h, w)

    bake.read_page = spy
    for k, (w, h) in enumerate(job['pages']):
        bake.bake_page(k, receivers, images[k], h, w, job)
    with open(os.path.join(root, 'sizes.json'), 'w') as handle:
        json.dump({'during': seen, 'after': [int(i.size[0] * i.size[1]) for i in images]}, handle)


main()
