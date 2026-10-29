"""Facts that differ between source runs and a PyInstaller bundle."""

import os
import subprocess
import sys
from pathlib import Path


# a windowed app has no console, so windows would open one for every console child
NO_WINDOW = getattr(subprocess, 'CREATE_NO_WINDOW', 0) if sys.platform == 'win32' else 0
DENOISE_FLAG = '--denoise-worker'
PROBE_FLAG = '--gpu-probe'
VERTEX_FLAG = '--vertex-pass'
CPU_VERTEX_FLAG = '--cpu-vertex-pass'


def numba_cache_dir() -> Path:
    return Path(os.environ.get('LOCALAPPDATA') or Path.home()) / 'OpenTool' / 'numba-cache'


def is_frozen() -> bool:
    return bool(getattr(sys, 'frozen', False))


def bundle_dir() -> Path:
    if is_frozen():
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parents[1]


def cli_executable() -> Path:
    return Path(sys.executable).with_name('OpenTool-cli.exe')


def child_command(flag: str, *args: str) -> list[str]:
    if is_frozen():
        return [str(cli_executable()), flag, *args]
    return [sys.executable, str(bundle_dir() / 'lightmap.py'), flag, *args]
