"""Facts that differ between source runs and a PyInstaller bundle."""

import subprocess
import sys
from pathlib import Path


# A windowed app has no console, so Windows would open one for every console child.
NO_WINDOW = getattr(subprocess, 'CREATE_NO_WINDOW', 0) if sys.platform == 'win32' else 0


def is_frozen() -> bool:
    return bool(getattr(sys, 'frozen', False))


def bundle_dir() -> Path:
    if is_frozen():
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parents[1]


def cli_executable() -> Path:
    return Path(sys.executable).with_name('OpenTool-cli.exe')
