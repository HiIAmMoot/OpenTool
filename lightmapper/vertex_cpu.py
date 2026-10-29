"""Compiled CPU vertex child and frozen-build cache probe."""

import json
import os
from pathlib import Path
from time import perf_counter

import numba
from numba import njit, prange
import numpy as np


@njit(parallel=True, cache=True)
def _probe_kernel(values):
    result = np.empty_like(values)
    for i in prange(len(values)):
        result[i] = values[i] * values[i]
    return result


def probe_main(out: Path) -> int:
    values = np.arange(1_000_000, dtype=np.float64)
    started = perf_counter()
    _probe_kernel(values)
    seconds = perf_counter() - started
    stats = _probe_kernel.stats
    out.write_text(json.dumps({
        'threading_layer': numba.threading_layer(),
        'threads': numba.get_num_threads(),
        'cache_hits': int(sum(stats.cache_hits.values())),
        'cache_misses': int(sum(stats.cache_misses.values())),
        'seconds': seconds,
        'cache_dir': os.environ.get('NUMBA_CACHE_DIR'),
    }))
    return 0


def child_main(argv: list[str]) -> int:
    if argv[:1] == ['--probe']:
        return probe_main(Path(argv[1]))
    raise ValueError('expected --probe and an output path')
