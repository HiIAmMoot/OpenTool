"""Runs inside Blender to check the bake image contract independently of bake_page."""

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
    scene, _ = bake.setup_scene(job)
    with np.load(os.path.join(root, 'scene.npz')) as data:
        scene_data = {key: data[key] for key in data.files}
    textures, images, _ = bake.make_images(os.path.join(root, 'textures.npz'), job['pages'])
    receivers, occluders = bake.build_objects(scene_data, job, textures, images)
    bake.add_suns(job, receivers, occluders)
    for obj, _, _ in receivers:
        obj.select_set(True)
    bake.bpy.context.view_layer.objects.active = receivers[0][0]
    image = images[0]
    bake.size_page_image(image, *job['pages'][0])
    w, h = image.size
    out = {'samples': np.array(scene.cycles.samples), 'size': np.array([w, h])}
    for name, value in [('DIFFUSE', -1.0), ('EMIT', -2.0)]:
        image.pixels.foreach_set(np.full(h * w * 4, value, np.float32))
        options = {'pass_filter': {'DIRECT', 'INDIRECT'}} if name == 'DIFFUSE' else {}
        bake.bpy.ops.object.bake(type=name, margin=0, use_clear=False, target='IMAGE_TEXTURES', **options)
        out[name] = bake.read_page(image, np.empty(h * w * 4, np.float32), h, w)[::-1].copy()
    np.savez(os.path.join(root, 'probe.npz'), **out)


main()
