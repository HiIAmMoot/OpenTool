"""Finish a baked page in float, then tone and quantise it once.

Nothing here rounds to 8 or 16 bits before `quantise`: the denoiser runs on
float32 and the ambient is added after it, so only the final r5g6b5 encode loses
precision.
"""

import io
import logging
import subprocess
import sys
import tempfile

import numpy as np

from .formats.vertices import encode_r5g6b5
from .raster import dilate
from .seams import stitch_seams

log = logging.getLogger(__name__)


# a fresh interpreter loads the system's C++ runtime before anything else: OIDN
# needs the new msvcp140/vcruntime140, and once numpy or numba have loaded the
# 2019 copy that sits beside Anaconda's python.exe the device creation in this
# process dies with an access violation
_WORKER = """
import ctypes, os, sys
if os.name == 'nt':
    for n in ('vcruntime140', 'vcruntime140_1', 'msvcp140', 'msvcp140_1', 'msvcp140_2'):
        path = os.path.join(os.environ['SystemRoot'], 'System32', n + '.dll')
        if os.path.exists(path):
            ctypes.WinDLL(path)
import io
import numpy as np, pyoidn
img = np.load(io.BytesIO(sys.stdin.buffer.read()))
np.save(sys.stdout.buffer, pyoidn.denoise(img, filter_type='RTLightmap'))
"""


# the network misreads a page's border: a constant page comes back up to 13% off
# in its outer texels, and 0.1% off once edge texels are repeated around it
_BORDER = 16


def _denoise(page: np.ndarray, timeout: float) -> np.ndarray:
    data = io.BytesIO()
    padded = np.pad(page, ((_BORDER, _BORDER), (_BORDER, _BORDER), (0, 0)), mode='edge')
    np.save(data, padded)
    redistributable = ('; a missing or old Microsoft Visual C++ redistributable '
                       '(vcruntime140/msvcp140 14.4x) is the usual cause')
    try:
        with tempfile.TemporaryDirectory() as cwd:
            done = subprocess.run([sys.executable, '-c', _WORKER], input=data.getvalue(),
                                  capture_output=True, cwd=cwd, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f'OIDN denoise timed out after {timeout:g} s') from None
    if done.returncode:
        raise RuntimeError(f'OIDN denoise failed ({done.returncode}): '
                           f'{done.stderr.decode(errors="replace")[-400:]}{redistributable}')
    try:
        result = np.load(io.BytesIO(done.stdout))
    except ValueError as error:
        raise RuntimeError(f'OIDN denoise returned unreadable data: {error}{redistributable}') from None
    if result.shape != padded.shape or not np.isfinite(result).all():
        raise RuntimeError(f'OIDN denoise returned a bad result (shape {result.shape}, '
                           f'expected {padded.shape}, finite: {bool(np.isfinite(result).all())})')
    return result[_BORDER:-_BORDER, _BORDER:-_BORDER]


def expose(rgb: np.ndarray) -> np.ndarray:
    # tool divides a colour by its largest channel when that exceeds 1, so an
    # over-bright texel keeps its hue instead of each channel clipping alone
    rgb = np.maximum(rgb, 0)
    peak = rgb.max(axis=-1, keepdims=True)
    return np.clip(rgb / np.maximum(peak, 1), 0, 1)


def tone(e: np.ndarray, gamma: float) -> np.ndarray:
    return expose(np.maximum(e, 0) ** gamma)


def finish_float(raw, mask, cls, ambient, s, written=None) -> np.ndarray:
    raw = np.ascontiguousarray(raw, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.bool_)
    cls = np.asarray(cls, dtype=np.int16)
    written = mask if written is None else np.asarray(written, dtype=np.bool_)
    # a written texel the coverage test missed (a chart edge the 3x3 bake reached) has no class of
    # its own: it and the padding beyond it take the covered neighbour's, as padding always did
    # (two rings more than the padding: written texels reach one ring past the covered ones)
    seeded = dilate(raw, mask, cls, 2 * s.padding + 2)[2]
    page, dilated, classes = dilate(raw, written, np.where(written, seeded, cls), 2 * s.padding)
    missing = mask & ~written
    log.info('page %dx%d: %d covered texels unwritten; %d stayed unfilled after dilation',
             raw.shape[1], raw.shape[0], int(missing.sum()), int((missing & ~dilated).sum()))
    # covered texels keep their own ambient class even when a neighbour supplied their radiance
    classes[mask] = cls[mask]
    # (an all-black page comes back from the denoiser with texels up to 0.03)
    if s.denoise and page.any():
        page = _denoise(np.ascontiguousarray(page, dtype=np.float32), s.timeout_s)
        # (the denoiser blends chart texels into the padding and past it, so
        # the padding is rebuilt from the charts' own denoised texels)
        page[~dilated] = 0
        page, dilated, classes = dilate(
            np.ascontiguousarray(page, dtype=np.float32), (mask | written) & dilated, classes, 2 * s.padding)
    classes[mask] = cls[mask]
    dilated |= mask
    page = np.array(page, dtype=np.float32)
    for klass, colour in ambient.items():
        page[dilated & (classes == klass)] += np.asarray(colour, dtype=np.float32)
    page[~dilated] = 0
    clipped = float(np.mean(page[mask].max(axis=-1) > 1)) if mask.any() else 0.0
    log.info('page %dx%d: %.2f%% of covered texels clipped at 1',
             page.shape[1], page.shape[0], clipped * 100)
    return page


def quantise(page_f, s) -> np.ndarray:
    return encode_r5g6b5(tone(page_f, s.gamma), s.dither)


def finish_pages(pages, raw, masks, classes, ambient, s, written=None, joined_groups=()):
    written = [None] * len(pages) if written is None else written
    finished = [finish_float(r, m, c, ambient, s, written=w)
                for r, m, c, w in zip(raw, masks, classes, written)]
    # OIDN can recreate a step, and ambient can differ across pages, so enforce
    # continuity after both in float, before the single tone/quantisation pass
    if s.stitch_seams:
        stitch_seams(pages, finished, joined_groups)
    return finished
