"""Stands in for blender.exe in the job exchange tests: reads job.json and scene.npz after "--" and writes an out.npz derived from them."""

import json
import os
import sys
import time
from pathlib import Path

# stamped before numpy loads, which is slow under load and would swamp the 1.5 s the tests expect
launch_seconds = time.time() - float(os.environ.get('LIGHTMAPPER_LAUNCH_TIME', time.time()))
import numpy as np


def page(i, w, h):
    # different in every row, column and channel, so a flip or transpose shows
    y, x, c = np.mgrid[0:h, 0:w, 0:3]
    return (i * 1000 + y * 100 + x + c * 0.25).astype(np.float32)


def main():
    job_path = Path(sys.argv[sys.argv.index('--') + 1])
    job = json.loads(job_path.read_text())
    mode = os.environ.get('FAKE_BLENDER_MODE', 'ok')
    print('fake blender running')
    stage_start = launch_seconds
    if mode == 'stage_timings':
        time.sleep(1.5)
        if 'LIGHTMAPPER_LAUNCH_TIME' in os.environ:
            stage_start += 1.5
    if mode == 'nooutput':
        return
    with open(job_path.with_name('args.json'), 'w') as handle:
        json.dump(sys.argv[1:], handle)
    with np.load(job_path.with_name(job['scene'])) as scene:
        query_pos = scene['query_pos']
        query_class = scene['query_class']
        edges = len(scene['edge_pos'])
    arrays = {'vertex_rays_cast': np.int64(0)}
    if 'LIGHTMAPPER_LAUNCH_TIME' in os.environ:
        arrays.update(seconds_start=np.float64(stage_start),
                      seconds_scene=np.float64(2), seconds_pages=np.float64(3), seconds_vertex=np.float64(0))
    if not job.get('vertex_only'):
        cpu = not job['cycles_devices']
        arrays['samples_used'] = np.int64(job.get('cpu_samples', job['samples']) if cpu else job['samples'])
        arrays['texel_samples_used'] = np.int64(job.get('cpu_texel_samples', job['texel_samples'])
                                              if cpu else job['texel_samples'])
        for i, (w, h) in enumerate(job['pages']):
            arrays[f'page_{i}'] = page(i, w, h)
            arrays[f'diffuse_{i}'] = page(i, w, h)
            arrays[f'written_{i}'] = np.ones((h, w), bool)
            arrays[f'written_{i}'][0, i] = False
    else:
        with np.load(job_path.with_name(job['diffuse'])) as data:
            assert len(data.files) == len(job['pages'])
    if job.get('vertex_only') or job.get('vertex_pass', 'cpu') == 'cpu':
        arrays['vertex_vector'] = query_pos[:, [1, 2, 0]] * 2
        arrays['vertex_irradiance'] = (query_pos.sum(axis=1) + query_class).astype(np.float32)
        arrays['vertex_sun_lit'] = np.zeros((len(query_pos), len(job['suns'])), bool)
        arrays['edge_sun_lit'] = np.zeros((edges, len(job['suns'])), bool)
        rays = job.get('cpu_vertex_rays', job['vertex_rays'])
        arrays['vertex_rays_used'] = np.int64(rays)
        arrays['vertex_rays_cast'] = np.int64(len(query_pos) * rays)
    arrays['device'] = np.array(('+'.join(job['cycles_devices']) or 'CPU') + '-FAKE')
    np.savez(job_path.with_name(job['out']), **arrays)
    if mode == 'fail':
        # a result is on disk, so only the exit code can tell the run failed
        print('fake blender failing on purpose')
        sys.exit(1)


main()
