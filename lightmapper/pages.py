"""Finish a baked page in float, then tone and quantise it once.

Nothing here rounds to 8 or 16 bits before `quantise`: the denoiser runs on
float32 and the ambient is added after it, so only the final r5g6b5 encode loses
precision.
"""

import io
import logging
import os
import struct
import subprocess
import sys
import tempfile
from time import perf_counter
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout

import numpy as np

from .formats.vertices import encode_r5g6b5
from . import frozen
from . import denoise_worker
from .raster import dilate
from .seams import stitch_seams

log = logging.getLogger(__name__)


# the network misreads a page's border: a constant page comes back up to 13% off
# in its outer texels, and 0.1% off once edge texels are repeated around it
_BORDER = 16
_REDISTRIBUTABLE_HINT = ('; a missing or old Microsoft Visual C++ redistributable '
                        '(vcruntime140/msvcp140 14.4x) is the usual cause')

_WORKER = """
import ctypes, os, sys
if os.name == 'nt':
    for n in ('vcruntime140', 'vcruntime140_1', 'msvcp140', 'msvcp140_1', 'msvcp140_2'):
        path = os.path.join(os.environ['SystemRoot'], 'System32', n + '.dll')
        if os.path.exists(path):
            ctypes.WinDLL(path)
import io, struct
import numpy as np, pyoidn
def receive():
    size = struct.unpack('<Q', sys.stdin.buffer.read(8))[0]
    return np.load(io.BytesIO(sys.stdin.buffer.read(size)))
def send(value):
    out = io.BytesIO(); np.save(out, value); data = out.getvalue()
    sys.stdout.buffer.write(struct.pack('<Q', len(data)) + data); sys.stdout.buffer.flush()
kind = sys.argv[1]
try:
    if kind == 'CPU':
        device = None
        filt = src = dst = None
        buffer_size = 0
    else:
        types = {'CUDA': pyoidn.OIDN_DEVICE_TYPE_CUDA, 'HIP': pyoidn.OIDN_DEVICE_TYPE_HIP,
                 'NOPE': 255}
        device = pyoidn.Device(types[kind]); device.commit()
        error = device.get_error()
        if error: raise RuntimeError(str(error))
        filt = pyoidn.Filter(device, 'RTLightmap')
        filt.set_quality(pyoidn.OIDN_QUALITY_HIGH)
        src = dst = None
        buffer_size = 0
    send(np.array([1], dtype=np.uint8))
except Exception as error:
    send(np.array([0], dtype=np.uint8))
    print(error, file=sys.stderr, flush=True)
    sys.exit(2)
while True:
    try: img = receive()
    except Exception: break
    if kind == 'CPU':
        result = pyoidn.denoise(img, filter_type='RTLightmap')
    else:
        nbytes = img.nbytes
        if nbytes != buffer_size:
            if src is not None: src.release(); dst.release()
            src = pyoidn.Buffer(device, nbytes, pyoidn.OIDN_STORAGE_DEVICE)
            dst = pyoidn.Buffer(device, nbytes, pyoidn.OIDN_STORAGE_DEVICE)
            buffer_size = nbytes
        src.write(0, nbytes, img)
        filt.set_image(pyoidn.OIDN_IMAGE_COLOR, src, pyoidn.OIDN_FORMAT_FLOAT3, img.shape[1], img.shape[0])
        filt.set_image(pyoidn.OIDN_IMAGE_OUTPUT, dst, pyoidn.OIDN_FORMAT_FLOAT3, img.shape[1], img.shape[0])
        filt.commit(); filt.execute(); device.wait()
        result = np.empty_like(img); dst.read(0, nbytes, result)
    send(result)
if filt is not None: filt.release()
if src is not None: src.release(); dst.release()
if device is not None: device.release()
"""


def _denoise(page: np.ndarray, timeout: float, worker=None) -> np.ndarray:
    if worker is not None:
        return worker.denoise(page)
    data = io.BytesIO()
    padded = np.pad(page, ((_BORDER, _BORDER), (_BORDER, _BORDER), (0, 0)), mode='edge')
    np.save(data, padded)
    try:
        with tempfile.TemporaryDirectory() as cwd:
            done = subprocess.run(denoise_worker.worker_command(), input=data.getvalue(),
                                  capture_output=True, cwd=cwd, timeout=timeout,
                                  creationflags=frozen.NO_WINDOW)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f'OIDN denoise timed out after {timeout:g} s') from None
    if done.returncode:
        raise RuntimeError(f'OIDN denoise failed ({done.returncode}): '
                           f'{done.stderr.decode(errors="replace")[-400:]}{_REDISTRIBUTABLE_HINT}')
    try:
        result = np.load(io.BytesIO(done.stdout))
    except ValueError as error:
        raise RuntimeError(f'OIDN denoise returned unreadable data: {error}{_REDISTRIBUTABLE_HINT}') from None
    if result.shape != padded.shape or not np.isfinite(result).all():
        raise RuntimeError(f'OIDN denoise returned a bad result (shape {result.shape}, '
                           f'expected {padded.shape}, finite: {bool(np.isfinite(result).all())})')
    return result[_BORDER:-_BORDER, _BORDER:-_BORDER]


class DenoiseTimeoutError(RuntimeError):
    pass


class DenoiseWorker:
    def __init__(self, device, timeout, process):
        self.device, self.timeout, self.process = device, timeout, process
        self.warnings = []
        self.cwd = None
        self.strict = True
        # start-up plus every page's denoise, for the bake's profile
        self.seconds = 0.0
        self._timing = False

    @classmethod
    def start(cls, device, timeout, strict):
        began = perf_counter()
        try:
            worker = cls._open(device, timeout)
        except (RuntimeError, OSError) as error:
            if strict:
                raise RuntimeError(f'OIDN {device} failed: {error}') from error
            worker = cls._open('CPU', timeout)
            worker._warn_fallback(f'OIDN {device} failed ({error}); denoising on the CPU')
        worker.strict = strict
        worker.seconds += perf_counter() - began
        return worker

    @classmethod
    def _open(cls, device, timeout):
        cwd = tempfile.TemporaryDirectory()
        process = subprocess.Popen([sys.executable, '-c', _WORKER, device], stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd.name)
        worker = cls(device, timeout, process)
        worker.cwd = cwd

        def read_ready():
            head = process.stdout.read(8)
            if len(head) != 8:
                raise RuntimeError(f'worker exited with code {process.poll()}')
            size = struct.unpack('<Q', head)[0]
            payload = process.stdout.read(size)
            if len(payload) != size:
                raise RuntimeError(f'worker returned a truncated ready frame ({len(payload)} of {size} bytes)')
            ready = np.load(io.BytesIO(payload))
            if not ready[0]:
                raise RuntimeError(process.stderr.read().decode(errors='replace')[-400:])

        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                ready = pool.submit(read_ready)
                try:
                    ready.result(timeout=timeout)
                except FutureTimeout:
                    # Terminate the producer before executor shutdown waits for
                    # the blocked pipe reader to finish.
                    process.kill()
                    process.wait()
                    cwd.cleanup()
                    raise DenoiseTimeoutError(
                        f'OIDN {device} startup timed out after {timeout:g} s') from None
        except FutureTimeout:
            raise
        except DenoiseTimeoutError:
            raise
        except Exception as error:
            if process.poll() is None:
                process.kill()
                process.wait()
            stderr = process.stderr.read().decode(errors='replace')[-400:]
            cwd.cleanup()
            if stderr:
                raise RuntimeError(f'{error}: {stderr}') from error
            raise
        return worker

    def _warn_fallback(self, warning):
        self.warnings.append(warning)
        log.warning('%s', warning)

    def _stop(self):
        if self.process.poll() is None:
            try:
                self.process.stdin.close()
                self.process.wait(timeout=min(self.timeout, 1))
            except (subprocess.TimeoutExpired, OSError):
                self.process.kill()
                self.process.wait()
        stderr = getattr(self.process, 'stderr', None)
        tail = stderr.read().decode(errors='replace')[-400:] if stderr is not None else ''
        if self.cwd is not None:
            self.cwd.cleanup()
            self.cwd = None
        return tail

    def _page_failure(self, page, error, process_failure):
        device = self.device
        if isinstance(error, DenoiseTimeoutError):
            self._stop()
            raise error
        stderr = self._stop()
        if device != 'CPU' and not self.strict:
            replacement = self._open('CPU', self.timeout)
            self.process, self.cwd, self.device = replacement.process, replacement.cwd, 'CPU'
            self._warn_fallback(f'OIDN {device} failed ({error}); denoising on the CPU')
            return self.denoise(page)
        detail = f'OIDN {device} denoise failed: {error}'
        if stderr:
            detail += f'; {stderr}'
        if process_failure:
            detail += _REDISTRIBUTABLE_HINT
        raise RuntimeError(detail) from error

    def _exchange(self, payload):
        self.process.stdin.write(struct.pack('<Q', len(payload)) + payload)
        self.process.stdin.flush()
        head = self.process.stdout.read(8)
        if len(head) != 8:
            raise RuntimeError('worker exited')
        size = struct.unpack('<Q', head)[0]
        result = self.process.stdout.read(size)
        if len(result) != size:
            raise RuntimeError(f'worker returned a truncated page ({len(result)} of {size} bytes)')
        return np.load(io.BytesIO(result))

    def denoise(self, page):
        # _page_failure retries through this method, so only the outermost call may add its time
        if self._timing:
            return self._denoise(page)
        self._timing = True
        began = perf_counter()
        try:
            return self._denoise(page)
        finally:
            self._timing = False
            self.seconds += perf_counter() - began

    def _denoise(self, page):
        padded = np.pad(page, ((_BORDER, _BORDER), (_BORDER, _BORDER), (0, 0)), mode='edge')
        data = io.BytesIO(); np.save(data, padded); payload = data.getvalue()
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(self._exchange, payload)
            try:
                result = future.result(timeout=self.timeout)
            except FutureTimeout:
                self.process.kill()
                self.process.wait()
                raise DenoiseTimeoutError(f'OIDN denoise timed out after {self.timeout:g} s') from None
            except Exception as error:
                return self._page_failure(page, error, process_failure=True)
        if result.shape != padded.shape or not np.isfinite(result).all():
            error = RuntimeError(f'OIDN denoise returned a bad result (shape {result.shape}, expected {padded.shape}, finite: {bool(np.isfinite(result).all())})')
            return self._page_failure(page, error, process_failure=False)
        return result[_BORDER:-_BORDER, _BORDER:-_BORDER]

    def close(self):
        self._stop()

    def __enter__(self): return self
    def __exit__(self, *args): self.close()


def expose(rgb: np.ndarray) -> np.ndarray:
    # tool divides a colour by its largest channel when that exceeds 1, so an
    # over-bright texel keeps its hue instead of each channel clipping alone
    rgb = np.maximum(rgb, 0)
    peak = rgb.max(axis=-1, keepdims=True)
    return np.clip(rgb / np.maximum(peak, 1), 0, 1)


def tone(e: np.ndarray, gamma: float) -> np.ndarray:
    return expose(np.maximum(e, 0) ** gamma)


def finish_float(raw, mask, cls, ambient, s, written=None, worker=None) -> np.ndarray:
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
        page = _denoise(np.ascontiguousarray(page, dtype=np.float32), s.timeout_s, worker)
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


def finish_pages(pages, raw, masks, classes, ambient, s, written=None, joined_groups=(), worker=None):
    written = [None] * len(pages) if written is None else written
    finished = [finish_float(r, m, c, ambient, s, written=w, worker=worker)
                for r, m, c, w in zip(raw, masks, classes, written)]
    # OIDN can recreate a step, and ambient can differ across pages, so enforce
    # continuity after both in float, before the single tone/quantisation pass
    if s.stitch_seams:
        stitch_seams(pages, finished, joined_groups)
    return finished
