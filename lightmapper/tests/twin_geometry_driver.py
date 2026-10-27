"""Inspect real Blender receiver meshes and the unchanged vertex-pass geometry."""

import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np


root = Path(sys.argv[sys.argv.index('--') + 1])
spec = importlib.util.spec_from_file_location('lm_bake', Path(os.environ['LIGHTMAPPER_BAKE_DIR']) / 'bake.py')
bake = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bake)
job = json.loads((root / 'job.json').read_text())
bake.setup_scene(job)
with np.load(root / 'scene.npz') as data:
    scene = {key: data[key] for key in data.files}
before = {key: value.copy() for key, value in scene.items()}
textures, images, _ = bake.make_images(root / 'textures.npz', job['pages'])
receivers, occluders = bake.build_objects(scene, job, textures, images)
for key, value in before.items():
    np.testing.assert_array_equal(scene[key], value, err_msg=f'vertex-pass input changed: {key}')
tracer = bake.VertexPass(scene, job, [], [np.zeros((8, 16, 3), np.float32)])
np.testing.assert_array_equal(tracer.corners[:len(scene['tris'])], before['positions'][before['tris']])
np.testing.assert_array_equal(tracer.corners[len(scene['tris']):], before['occ_positions'][before['occ_tris']])
assert len(receivers) == 1
mesh = receivers[0][0].data
positions = np.empty((len(mesh.vertices), 3), np.float32)
mesh.vertices.foreach_get('co', positions.ravel())
indices = np.empty(len(mesh.loops), np.int32)
mesh.loops.foreach_get('vertex_index', indices)
out = dict(positions=positions, tris=indices.reshape(-1, 3),
           separation=np.array(getattr(bake, 'TWIN_SEPARATION', 0)))
for name in (bake.UV_TEXTURE, bake.UV_LIGHTMAP):
    uv = np.empty((len(mesh.loops), 2), np.float32)
    mesh.uv_layers[name].data.foreach_get('uv', uv.ravel())
    out[name] = uv.reshape(-1, 3, 2)
normals = np.empty((len(mesh.loops), 3), np.float32)
mesh.corner_normals.foreach_get('vector', normals.ravel())
out['normals'] = normals.reshape(-1, 3, 3)
original_occluders = set(map(tuple, before['occ_positions']))
for obj in occluders:
    assert all(tuple(v.co) in original_occluders for v in obj.data.vertices)
np.savez(root / 'geometry.npz', **out)
