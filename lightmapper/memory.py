import ctypes
from ctypes import wintypes
from dataclasses import dataclass, replace
import logging
from math import ceil
import os
import shutil

import numpy as np


# peak process-tree commit per stage, fit to the kept bakes of 2026-11-01 (lightmapper/tests/data/memory_peaks.csv) with
# every row between 1.0 and 1.21 times its measured peak (vertex up to 1.35, finish up to 1.55). The rows come from a GPU
# (OptiX) machine; no CPU bake was measured, so the CPU device uses the same constants.
BLENDER_BASE: int = 1_700_000_000
BLENDER_PER_SUB_TEXEL: int = 270
BLENDER_PER_PAGE_TEXEL: int = 99
BLENDER_PER_VERTEX: int = 1205
# the whole-page tile's own term, outside MARGIN: it is the measured difference between a tiled and an untiled bake,
# so the margin on the other terms would count its error twice
PER_TILE_SUB_TEXEL: int = 102
VERTEX_BASE: int = 1_055_000_000
VERTEX_PER_PAGE_TEXEL: int = 273
VERTEX_PER_GLOW_SUB_TEXEL: int = 20
GLOW_PER_GLOW_SUB_TEXEL: int = 20
GLOW_PER_TILE_SUB_TEXEL: int = 110
FINISH_BASE: int = 300_000_000
FINISH_PER_SUB_TEXEL: int = 31
# Cycles' tile limit (tile.h:120): a page whose sub-texel side is bigger cannot be one tile
MAX_TILE = 8192
MARGIN = 1.15
GB = 10**9
STAGES = ('blender', 'vertex', 'glow', 'finish')
READ = object()
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SceneCounts:
    triangles: int
    vertices: int

    @classmethod
    def from_scene(cls, scene) -> 'SceneCounts':
        return cls(len(scene.tris) + len(scene.occ_tris),
                   len(scene.positions) + len(scene.occ_positions))

    @classmethod
    def from_npz(cls, path) -> 'SceneCounts':
        with np.load(path, allow_pickle=False) as scene:
            return cls(len(scene['tris']) + len(scene['occ_tris']),
                       len(scene['positions']) + len(scene['occ_positions']))


@dataclass(frozen=True)
class StageInputs:
    counts: SceneCounts
    pages: list[tuple[int, int]]
    device: str = 'gpu'
    # 9 * w * h over the pages holding a flagged triangle, only in the Enhanced ignore-normals mode
    glow_sub_texels: int = 0
    whole_page_tiles: bool = False


def _largest_sub_texels(pages) -> int:
    return max((9 * w * h for w, h in pages), default=0)


def _stages(inputs: StageInputs, largest: int, texels: int) -> dict[str, int]:
    tiles = inputs.whole_page_tiles
    glow = inputs.glow_sub_texels
    blender = (BLENDER_BASE + BLENDER_PER_SUB_TEXEL * largest + BLENDER_PER_PAGE_TEXEL * texels
               + BLENDER_PER_VERTEX * inputs.counts.vertices)
    return {
        'blender': ceil(MARGIN * blender) + (PER_TILE_SUB_TEXEL * largest if tiles else 0),
        'vertex': ceil(MARGIN * (VERTEX_BASE + VERTEX_PER_PAGE_TEXEL * texels + VERTEX_PER_GLOW_SUB_TEXEL * glow)),
        'glow': 0 if glow == 0 else (ceil(MARGIN * (blender + GLOW_PER_GLOW_SUB_TEXEL * glow))
                                     + (GLOW_PER_TILE_SUB_TEXEL * largest if tiles else 0)),
        'finish': ceil(MARGIN * (FINISH_BASE + FINISH_PER_SUB_TEXEL * largest)),
    }


def stage_bytes(inputs: StageInputs) -> dict[str, int]:
    return _stages(inputs, _largest_sub_texels(inputs.pages), sum(w * h for w, h in inputs.pages))


def peak_bytes(inputs: StageInputs) -> tuple[str, int]:
    stages = stage_bytes(inputs)
    stage = max(STAGES, key=stages.__getitem__)
    return stage, stages[stage]


def flagged_sub_texels(tri_page, flagged, pages) -> int:
    if flagged is None or not len(flagged):
        return 0
    held = {int(page) for page in np.asarray(tri_page)[np.asarray(flagged, bool)]}
    return sum(9 * pages[page][0] * pages[page][1] for page in held)


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ('dwLength', wintypes.DWORD),
        ('dwMemoryLoad', wintypes.DWORD),
        ('ullTotalPhys', ctypes.c_ulonglong),
        ('ullAvailPhys', ctypes.c_ulonglong),
        ('ullTotalPageFile', ctypes.c_ulonglong),
        ('ullAvailPageFile', ctypes.c_ulonglong),
        ('ullTotalVirtual', ctypes.c_ulonglong),
        ('ullAvailVirtual', ctypes.c_ulonglong),
        ('ullAvailExtendedVirtual', ctypes.c_ulonglong),
    ]


@dataclass(frozen=True)
class Machine:
    ram: int
    commit_limit: int
    commit_ceiling: int | None


@dataclass(frozen=True)
class PagingFile:
    drive: str
    initial_mb: int | None
    maximum_mb: int | None


def parse_paging_files(value: list[str]) -> list[PagingFile]:
    files = []
    for entry in value:
        parts = entry.rsplit(None, 2)
        sizes = (0, 0)
        if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
            path, sizes = parts[0], (int(parts[1]), int(parts[2]))
        else:
            path = entry.strip()
        drive = path[:1].upper()
        if drive == '?':
            drive = os.environ.get('SystemDrive', 'C:')[:1].upper()
        managed = sizes == (0, 0)
        files.append(PagingFile(drive, None if managed else sizes[0], None if managed else sizes[1]))
    return files


# (a system-managed file grows on demand, so what it can reach is bounded by its disk and by Windows' own cap of about
# three times the RAM; a fixed file never grows past its configured maximum)
def commit_ceiling(ram: int, commit_limit: int, files: list[PagingFile], free_on) -> int:
    if not files:
        return ram
    reach = 0
    for file in files:
        if file.maximum_mb is None:
            reach += min(3 * ram, free_on(file.drive))
        else:
            reach += file.maximum_mb * 2**20
    return max(commit_limit, ram + reach)


def _windows_free_physical_bytes() -> int:
    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel32.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(MEMORYSTATUSEX)]
    kernel32.GlobalMemoryStatusEx.restype = wintypes.BOOL
    status = MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(status)
    if not kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise ctypes.WinError(ctypes.get_last_error())
    return status.ullAvailPhys


def free_physical_bytes() -> int | None:
    try:
        if os.name == 'nt':
            return _windows_free_physical_bytes()
        return os.sysconf('SC_AVPHYS_PAGES') * os.sysconf('SC_PAGE_SIZE')
    except Exception:
        return None


def _memory_status() -> tuple[int, int]:
    if os.name != 'nt':
        ram = os.sysconf('SC_PHYS_PAGES') * os.sysconf('SC_PAGE_SIZE')
        return ram, ram
    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel32.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(MEMORYSTATUSEX)]
    kernel32.GlobalMemoryStatusEx.restype = wintypes.BOOL
    status = MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(status)
    if not kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise ctypes.WinError(ctypes.get_last_error())
    return status.ullTotalPhys, status.ullTotalPageFile


def _paging_files() -> list[str] | None:
    if os.name != 'nt':
        return None
    import winreg
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                        r'SYSTEM\CurrentControlSet\Control\Session Manager\Memory Management') as key:
        value, _ = winreg.QueryValueEx(key, 'PagingFiles')
    return [value] if isinstance(value, str) else list(value)


def _free_on(drive: str) -> int:
    return shutil.disk_usage(drive + ':\\').free


def read_machine() -> Machine | None:
    try:
        ram, limit = _memory_status()
    except Exception:
        return None
    try:
        names = _paging_files()
        ceiling = None if names is None else commit_ceiling(ram, limit, parse_paging_files(names), _free_on)
    except Exception:
        # (the page file's reach only decides the second warning level, so the first still works without it)
        ceiling = None
    return Machine(ram, limit, ceiling)


def _tiles_fit(inputs: StageInputs, limit: int | None, stage: str) -> bool:
    if limit is None or not inputs.pages:
        return False
    if 3 * max(max(page) for page in inputs.pages) > MAX_TILE:
        return False
    return stage_bytes(replace(inputs, whole_page_tiles=True))[stage] <= limit


def whole_page_tiles(inputs: StageInputs, machine: Machine | None) -> bool:
    return _tiles_fit(inputs, None if machine is None else machine.ram, 'blender')


# (the glow run starts after the page stage's arrays are gone, so what is free then, not what is installed, decides)
def glow_whole_page_tiles(inputs: StageInputs, free: int | None) -> bool:
    return _tiles_fit(inputs, free, 'glow')


def format_gb(n: int) -> str:
    return f'{n / GB:.1f}' if n < 10 * GB else f'{n / GB:.0f}'


def _stage_label(stage: str) -> str:
    from .progress import STAGE_LABELS
    label = STAGE_LABELS[stage]
    return label[:1].lower() + label[1:]


def _limit(machine: Machine | None) -> int | None:
    if machine is None:
        return None
    return machine.commit_limit if machine.commit_ceiling is None else machine.commit_ceiling


def page_file_message(inputs: StageInputs, machine: Machine) -> str:
    stage, need = peak_bytes(inputs)
    return (f'This bake needs about {format_gb(need)} GB at its peak ({_stage_label(stage)}), more than this '
            f"computer's {format_gb(machine.ram)} GB of memory. It will lean on the page file and slow down.")


def out_of_memory_message(inputs: StageInputs, machine: Machine | None) -> str:
    stage, need = peak_bytes(inputs)
    limit = _limit(machine)
    held = '' if limit is None else f' ({format_gb(limit)} GB)'
    text = (f'This bake needs about {format_gb(need)} GB at its peak ({_stage_label(stage)}), more than this '
            f"computer's memory and page file can hold{held}. It will likely run out of memory."
            ' Try a lower texel density, or close other programs.')
    if stage == 'glow':
        text += ' Or set Ignore-normals lighting to Traditional.'
    return text


@dataclass
class MemoryCheck:
    estimate: int
    stage: str = ''
    level: str = 'fits'
    message: str = ''
    machine: Machine | None = None

    @property
    def fits(self) -> bool:
        return self.level == 'fits'


def check_bake_memory(inputs: StageInputs, machine=READ) -> MemoryCheck:
    stage, need = peak_bytes(inputs)
    if machine is READ:
        machine = read_machine()
    if machine is None:
        logger.warning("this computer's memory is unknown, so the bake runs without the memory check")
        return MemoryCheck(need, stage)
    if need > _limit(machine):
        return MemoryCheck(need, stage, 'out_of_memory', out_of_memory_message(inputs, machine), machine)
    if need > machine.ram:
        return MemoryCheck(need, stage, 'page_file', page_file_message(inputs, machine), machine)
    return MemoryCheck(need, stage, machine=machine)


def summary(inputs: StageInputs, check: MemoryCheck) -> dict:
    machine = check.machine
    ceiling = None if machine is None else machine.commit_ceiling
    return {'peak_gb': round(check.estimate / GB, 2), 'stage': check.stage, 'level': check.level,
            'stages_gb': {stage: round(size / GB, 2) for stage, size in stage_bytes(inputs).items()},
            'ram_gb': None if machine is None else round(machine.ram / GB, 2),
            'ceiling_gb': None if ceiling is None else round(ceiling / GB, 2)}
