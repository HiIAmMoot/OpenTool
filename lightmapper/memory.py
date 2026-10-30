import ctypes
from ctypes import wintypes
from dataclasses import dataclass
import logging
from math import ceil
import os

import numpy as np


# b40_a and bloodgulch, 512/1024/2048, 2026-10-29; triangles fit to 0 because
# triangle and vertex counts rise together, so the vertex term carries both.
PER_SUB_TEXEL: int = 271
PER_PAGE_TEXEL: int = 63
PER_TRIANGLE: int = 0
PER_VERTEX: int = 1771
MARGIN = 1.15
GB = 10**9
PAGE_SIDES = (256, 512, 1024, 2048, 4096)
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


def page_term(w, h) -> int:
    return 9 * w * h * PER_SUB_TEXEL


def resident_term(pages) -> int:
    return PER_PAGE_TEXEL * sum(w * h for w, h in pages)


def scene_term(counts) -> int:
    return PER_TRIANGLE * counts.triangles + PER_VERTEX * counts.vertices


def bake_bytes(counts, pages) -> int:
    w, h = max(pages, key=lambda page: page[0] * page[1], default=(0, 0))
    return ceil(MARGIN * (page_term(w, h) + resident_term(pages) + scene_term(counts)))


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


def format_gb(n: int) -> str:
    return f'{n / GB:.1f}' if n < 10 * GB else f'{n / GB:.0f}'


def suggest_page_max(counts, pages, free: int | None) -> int | None:
    if not pages:
        return None
    largest = max(pages, key=lambda page: page[0] * page[1])
    fixed = resident_term(pages) + scene_term(counts)
    for side in reversed(PAGE_SIDES):
        if side < max(largest) and (free is None or MARGIN * (page_term(side, side) + fixed) <= free):
            return side
    return None


def out_of_memory_message(counts, pages, page_max: int, free: int | None) -> str:
    text = f'This bake needs about {format_gb(bake_bytes(counts, pages))} GB of free memory'
    if pages:
        w, h = max(pages, key=lambda page: page[0] * page[1])
        text += f' (largest page {w} x {h} at Max resolution {page_max})'
    if free is not None:
        text += f'; this computer has {format_gb(free)} GB free'
    side = suggest_page_max(counts, pages, free)
    if side is None:
        return text + '. Try a lower texel density, or close other programs.'
    return (text + f'. Try Max resolution {side}, or close other programs. '
            'If the level then needs too many pages or still does not fit, lower the texel density a little.')


@dataclass
class MemoryCheck:
    estimate: int
    free: int | None
    fits: bool = True
    message: str = ''


def check_bake_memory(counts, pages, page_max, free=READ) -> MemoryCheck:
    estimate = bake_bytes(counts, pages)
    if free is READ:
        free = free_physical_bytes()
    if free is None:
        logger.warning('free memory is unknown, so the bake runs without the memory check')
    elif estimate > free:
        return MemoryCheck(estimate, free, False, out_of_memory_message(counts, pages, page_max, free))
    return MemoryCheck(estimate, free)
