"""Runs inside Blender: runs setup_scene on the job and records the Cycles sample clamps."""

import importlib.util
import json
import os
import sys


def main():
    root = sys.argv[sys.argv.index('--') + 1]
    spec = importlib.util.spec_from_file_location('lm_bake', os.path.join(os.environ['LIGHTMAPPER_BAKE_DIR'], 'bake.py'))
    bake = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bake)
    with open(os.path.join(root, 'job.json')) as handle:
        job = json.load(handle)
    scene, _ = bake.setup_scene(job)
    clamps = {'sample_clamp_direct': float(scene.cycles.sample_clamp_direct),
              'sample_clamp_indirect': float(scene.cycles.sample_clamp_indirect)}
    with open(os.path.join(root, 'clamps.json'), 'w') as handle:
        json.dump(clamps, handle)


main()
