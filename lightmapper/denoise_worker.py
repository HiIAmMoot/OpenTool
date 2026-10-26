"""OIDN's isolated worker process."""

import ctypes
import os
import sys
from pathlib import Path

from . import frozen

WORKER_FLAG = '--denoise-worker'
RUNTIME_DLLS = ('vcruntime140', 'vcruntime140_1', 'msvcp140', 'msvcp140_1', 'msvcp140_2')


def runtime_dll_dir() -> Path:
    if frozen.is_frozen():
        return frozen.bundle_dir()
    return Path(os.environ['SystemRoot']) / 'System32'


def worker_command() -> list[str]:
    if frozen.is_frozen():
        return [str(frozen.cli_executable()), WORKER_FLAG]
    return [sys.executable, str(frozen.bundle_dir() / 'lightmap.py'), WORKER_FLAG]


def main(stdin=None, stdout=None) -> int:
    stdin = sys.stdin.buffer if stdin is None else stdin
    stdout = sys.stdout.buffer if stdout is None else stdout
    if os.name == 'nt':
        for name in RUNTIME_DLLS:
            path = runtime_dll_dir() / f'{name}.dll'
            if path.is_file():
                ctypes.WinDLL(str(path))
    import io
    import numpy as np
    import pyoidn
    img = np.load(io.BytesIO(stdin.read()))
    np.save(stdout, pyoidn.denoise(img, filter_type='RTLightmap'))
    return 0
