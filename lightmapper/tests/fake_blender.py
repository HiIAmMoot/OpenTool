"""Stands in for blender.exe in the job exchange tests: reads job.json and scene.npz after "--" and writes an out.npz derived from them."""

import json
import os
import sys
from pathlib import Path

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
    if mode == 'nooutput':
        return
    with open(job_path.with_name('args.json'), 'w') as handle:
        json.dump(sys.argv[1:], handle)
    with np.load(job_path.with_name(job['scene'])) as scene:
        query_pos = scene['query_pos']
        query_class = scene['query_class']
    arrays = {f'page_{i}': page(i, w, h) for i, (w, h) in enumerate(job['pages'])}
    for i, (w, h) in enumerate(job['pages']):
        arrays[f'written_{i}'] = np.ones((h, w), bool)
        arrays[f'written_{i}'][0, i] = False
    arrays['vertex_vector'] = query_pos[:, [1, 2, 0]] * 2
    arrays['vertex_irradiance'] = (query_pos.sum(axis=1) + query_class).astype(np.float32)
    arrays['device'] = np.array(('+'.join(job['cycles_devices']) or 'CPU') + '-FAKE')
    np.savez(job_path.with_name(job['out']), **arrays)
    if mode == 'fail':
        # a result is on disk, so only the exit code can tell the run failed
        print('fake blender failing on purpose')
        sys.exit(1)


main()
