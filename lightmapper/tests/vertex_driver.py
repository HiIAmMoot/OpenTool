"""Runs inside Blender (blender -b --python vertex_driver.py -- dir): drives bake.py's vertex pass on the arrays in dir
with the page images given, so tests can pin the pass without a Cycles bake. dir holds scene.npz, job.json,
textures.npz, pages.npz and optionally probe.json (rays to trace one by one instead of the whole pass).
"""

import importlib.util
import json
import os
import sys

import numpy as np


def main():
    root = sys.argv[sys.argv.index('--') + 1]
    bake_path = os.path.join(os.environ['LIGHTMAPPER_BAKE_DIR'], 'bake.py')
    spec = importlib.util.spec_from_file_location('lm_bake', bake_path)
    bake = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bake)
    with open(os.path.join(root, 'job.json')) as handle:
        job = json.load(handle)
    with np.load(os.path.join(root, 'scene.npz')) as data:
        scene = {key: data[key] for key in data.files}
    with np.load(os.path.join(root, 'textures.npz')) as data:
        textures = [data[f'tex_{k}'] for k in range(len(data.files))]
    with np.load(os.path.join(root, 'pages.npz')) as data:
        pages = [data[f'page_{i}'] for i in range(len(job['pages']))]
    tracer = bake.VertexPass(scene, job, textures, pages)
    probe_path = os.path.join(root, 'probe.json')
    if os.path.exists(probe_path):
        with open(probe_path) as handle:
            probes = json.load(handle)
        out = {'uv': [], 'radiance': [], 'tri': []}
        for origin, direction, cls in probes:
            tri, loc = tracer.cast(np.float64(origin), np.float64(direction))
            out['tri'].append(tri)
            out['uv'].append(tracer.hit_lm_uv(tri, loc) if tri >= 0 else (np.nan, np.nan))
            out['radiance'].append(tracer.radiance(tri, loc, np.float64(direction), cls) if tri >= 0 else np.nan)
        np.savez(os.path.join(root, 'probe.npz'), **{k: np.array(v) for k, v in out.items()})
        return
    vector, irradiance, rays = tracer.run()
    np.savez(os.path.join(root, 'vp.npz'), vector=vector, irradiance=irradiance, rays=rays,
             edge_sun_lit=tracer.edge_sun_lit)


main()
