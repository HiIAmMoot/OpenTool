from dataclasses import dataclass
import os
from pathlib import Path
import re
import subprocess
import sys

from .frozen import NO_WINDOW


MIN_VERSION = (5, 2)
_UNINSTALL = r'SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall'


@dataclass(frozen=True)
class Blender:
    path: Path
    version: tuple[int, int]


class BlenderNotFound(Exception):
    pass


def blender_version(exe: Path, timeout: float = 60) -> tuple[int, int] | None:
    try:
        result = subprocess.run([str(exe), '--version'], capture_output=True, stdin=subprocess.DEVNULL,
                                creationflags=NO_WINDOW, timeout=timeout, text=True)
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    match = re.search(r'^Blender (\d+)\.(\d+)', result.stdout, re.MULTILINE)
    return (int(match.group(1)), int(match.group(2))) if match else None


def registry_installs() -> list[Path]:
    if sys.platform != 'win32':
        return []
    try:
        import winreg
        paths = []
        locations = ((winreg.HKEY_LOCAL_MACHINE, _UNINSTALL),
                     (winreg.HKEY_LOCAL_MACHINE, r'SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall'),
                     (winreg.HKEY_CURRENT_USER, _UNINSTALL))
        for hive, key_path in locations:
            with winreg.OpenKey(hive, key_path) as key:
                for index in range(winreg.QueryInfoKey(key)[0]):
                    try:
                        with winreg.OpenKey(key, winreg.EnumKey(key, index)) as install:
                            display, _ = winreg.QueryValueEx(install, 'DisplayName')
                            location, _ = winreg.QueryValueEx(install, 'InstallLocation')
                        if str(display).lower().startswith('blender') and location:
                            paths.append(Path(location) / 'blender.exe')
                    except OSError:
                        continue
        return paths
    except OSError:
        return []


def program_files_installs() -> list[Path]:
    root = os.environ.get('ProgramFiles')
    if not root:
        return []
    return [path for path in Path(root).glob('Blender Foundation/Blender */blender.exe') if path.is_file()]


def find_blender(remembered: str | None = None) -> Blender:
    old_versions = []
    for candidate in (os.environ.get('LIGHTMAPPER_BLENDER'), remembered):
        if candidate:
            path = Path(candidate)
            if path.is_file():
                version = blender_version(path)
                if version and version >= MIN_VERSION:
                    return Blender(path, version)
                if version:
                    old_versions.append(version)

    candidates = []
    seen = set()
    for path in registry_installs() + program_files_installs():
        try:
            identity = str(path.resolve()).casefold()
        except OSError:
            identity = str(path).casefold()
        if identity not in seen:
            seen.add(identity)
            candidates.append(path)
    found = []
    for path in candidates:
        version = blender_version(path) if path.is_file() else None
        if version is not None:
            if version >= MIN_VERSION:
                found.append(Blender(path, version))
            else:
                old_versions.append(version)
    if found:
        return max(found, key=lambda item: item.version)
    if old_versions:
        major, minor = max(old_versions)
        raise BlenderNotFound(f'Found Blender {major}.{minor}, but OpenTool needs Blender 5.2 or newer. '
                              'Install it from blender.org, or pick blender.exe yourself.')
    raise BlenderNotFound('Blender 5.2 or newer was not found. Install it from blender.org, or pick blender.exe yourself.')
