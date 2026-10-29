"""Runs inside Blender: bakes each page with top-up masks and records every mask's size at each read."""

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
    textures, images, masks = bake.make_images(os.path.join(root, 'textures.npz'), job['pages'], True)
    before = [[int(m.size[0]), int(m.size[1])] for m in masks]
    receivers, occluders = bake.build_objects(scene_data, job, textures, images, masks)
    bake.add_suns(job, receivers, occluders)
    seen = []
    real_read = bake.read_page

    def spy(image, *rest):
        page = next(k for k, i in enumerate(images) if i == image)
        seen.append({'page': page, 'masks': [[int(m.size[0]), int(m.size[1])] for m in masks]})
        return real_read(image, *rest)

    bake.read_page = spy
    for k, (w, h) in enumerate(job['pages']):
        bake.bake_page(k, receivers, images[k], h, w, job, masks[k])
    after = []
    for m in masks:
        pixels = np.empty(m.size[0] * m.size[1] * 4, np.float32)
        m.pixels.foreach_get(pixels)
        after.append({'size': [int(m.size[0]), int(m.size[1])], 'min': float(pixels.min())})
    with open(os.path.join(root, 'mask_sizes.json'), 'w') as handle:
        json.dump({'before': before, 'during': seen, 'after': after}, handle)


main()
