"""Lazy command entry so the worker can preload its runtime before numpy."""

import os
import sys

from . import frozen
from .frozen import CPU_VERTEX_FLAG, DENOISE_FLAG, PROBE_FLAG, VERTEX_FLAG


def cli_entry(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == [CPU_VERTEX_FLAG]:
        os.environ.setdefault('NUMBA_CACHE_DIR', str(frozen.numba_cache_dir()))
        from . import vertex_cpu
        return vertex_cpu.child_main(argv[1:])
    if argv[:1] == [DENOISE_FLAG]:
        from . import denoise_worker
        if len(argv) > 1:
            return denoise_worker.main(kind=argv[1])
        return denoise_worker.main()
    if argv[:1] == [PROBE_FLAG]:
        from . import gpu
        return gpu.probe_main()
    if argv[:1] == [VERTEX_FLAG]:
        from . import vertex_gpu
        return vertex_gpu.child_main(argv[1:])
    from . import cli
    return cli.main(argv)
