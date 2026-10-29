"""Runs inside Blender: bakes one page with a fake bake operator and records what the page read back before each bake."""

import importlib.util
import json
import os
import sys
import types

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
    receivers, occluders = bake.build_objects(scene_data, job, textures, images, masks)
    bake.add_suns(job, receivers, occluders)
    image = images[0]
    out = {}
    calls = []
    reads = []

    def fake_bake(**_):
        n = len(calls)
        pixels = np.empty(image.size[0] * image.size[1] * 4, np.float32)
        image.pixels.foreach_get(pixels)
        pixels = pixels.reshape(image.size[1], image.size[0], 4)
        written = np.zeros(pixels.shape[:2], np.bool_)
        if n == 0:
            # two of texel (0, 0)'s nine sub-texels, so the page plans a top-up tier
            written[0, 0:2] = True
        else:
            written[0, 3 + n] = True
        pixels[written] = n + 1
        image.pixels.foreach_set(pixels.ravel())
        out[f'set_{n}'] = written
        calls.append(n)

    class Proxy:
        ops = types.SimpleNamespace(object=types.SimpleNamespace(bake=fake_bake))

        def __getattr__(self, name):
            return getattr(real_bpy, name)

    real_bpy = bake.bpy
    bake.bpy = Proxy()
    real_read = bake.read_page

    def spy(image, *rest):
        rgb = real_read(image, *rest)
        out[f'read_{len(reads)}'] = rgb.copy()
        reads.append(rgb)
        return rgb

    bake.read_page = spy
    w, h = job['pages'][0]
    bake.bake_page(0, receivers, image, h, w, job, masks[0])
    # a later read overwrites an earlier one only if they are views of the one buffer
    out['shared'] = np.array(all(np.shares_memory(reads[0], r) for r in reads))
    out['count'] = np.array(len(reads))
    np.savez(os.path.join(root, 'reads.npz'), **out)


main()
