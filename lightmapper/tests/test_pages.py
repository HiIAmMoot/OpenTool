"""Pin the float finishing order: denoise before ambient, one quantise at the end."""

import io
import logging
import shutil
import struct
import sys
import subprocess
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from lightmapper import denoise_worker, frozen, pages
from lightmapper.formats.vertices import encode_r5g6b5
from lightmapper.pages import expose, finish_float, quantise, tone
from lightmapper.raster import coverage, dilate
from lightmapper.settings import Settings


def full_page(n=48):
    mask = np.ones((n, n), dtype=bool)
    return mask, np.zeros((n, n), dtype=np.int16)


def noisy_page(seed=1, size=32):
    return np.random.default_rng(seed).random((size, size, 3), dtype=np.float32)


def fake_worker(monkeypatch, code):
    monkeypatch.setattr(pages.denoise_worker, 'worker_command',
                        lambda kind=None: [sys.executable, '-c', code, *([kind] if kind else [])])


@pytest.mark.parametrize('strict', [False, True])
def test_native_gpu_error_retries_the_same_page_on_cpu_unless_required(monkeypatch, strict):
    real_command = denoise_worker.worker_command
    root = str(Path(__file__).resolve().parents[2])
    code = (f'import sys; sys.path.insert(0, {root!r}); '
            'from lightmapper.tests import fake_oidn; '
            'sys.modules["pyoidn"] = fake_oidn; '
            'from lightmapper.denoise_worker import serve; '
            'sys.exit(serve("CUDA", sys.stdin.buffer, sys.stdout.buffer))')
    monkeypatch.setattr(denoise_worker, 'worker_command',
                        lambda kind=None: [sys.executable, '-c', code] if kind == 'CUDA' else real_command(kind))
    first = np.full((40, 40, 3), .25, np.float32)
    second = np.full_like(first, .75)
    with pages.DenoiseWorker.start('CUDA', 60, strict=strict) as worker:
        np.testing.assert_array_equal(worker.denoise(first), first)
        if strict:
            with pytest.raises(RuntimeError, match='native execute failure'):
                worker.denoise(second)
        else:
            result = worker.denoise(second)
            assert worker.device == 'CPU'
            np.testing.assert_array_equal(result, pages._denoise(second, 60))
            assert 'denoising on the CPU' in worker.warnings[0]


def test_worker_starts_without_a_window(monkeypatch):
    seen = {}
    real = pages.subprocess.Popen

    def spy(args, **kwargs):
        seen.update(kwargs, args=args)
        return real(args, **kwargs)

    monkeypatch.setattr(pages.subprocess, 'Popen', spy)
    with pages.DenoiseWorker.start('CPU', 60, strict=True):
        pass
    assert seen['creationflags'] == frozen.NO_WINDOW and seen['args'] == denoise_worker.worker_command('CPU')


def test_worker_without_a_timeout_denoises_and_closes():
    page = np.full((40, 40, 3), .5, np.float32)
    worker = pages.DenoiseWorker.start('CPU', None, strict=True)
    try:
        with worker:
            assert worker.denoise(page).shape == page.shape
            assert worker.process.poll() is None
        assert worker.process.poll() is not None
    finally:
        if worker.process.poll() is None:
            worker.process.kill()
            worker.process.wait()
        if worker.cwd is not None:
            worker.cwd.cleanup()


def test_worker_start_without_a_timeout_is_still_bounded(monkeypatch):
    monkeypatch.setattr(pages, 'START_TIMEOUT_S', 0.5, raising=False)
    monkeypatch.setattr(pages.denoise_worker, 'worker_command', lambda kind=None: [sys.executable, '-c', 'import time; time.sleep(60)'])
    worker = None
    try:
        with pytest.raises(pages.DenoiseTimeoutError, match='startup timed out after 0.5 s'):
            worker = pages.DenoiseWorker.start('CUDA', None, strict=True)
    finally:
        if worker is not None:
            if worker.process.poll() is None:
                worker.process.kill()
                worker.process.wait()
            worker.cwd.cleanup()


def test_one_worker_serves_every_page(monkeypatch):
    calls = []
    real_popen = pages.subprocess.Popen

    def count(*args, **kwargs):
        calls.append(args)
        return real_popen(*args, **kwargs)

    monkeypatch.setattr(pages.subprocess, 'Popen', count)
    s = Settings(denoise=True, stitch_seams=False)
    mask, cls = full_page(32)
    with pages.DenoiseWorker.start('CPU', 60, strict=True) as worker:
        pages.finish_pages([SimpleNamespace(width=32, height=32)] * 3, [noisy_page(i, 32) for i in range(3)],
                           [mask] * 3, [cls] * 3, {}, s, worker=worker)
    assert len(calls) == 1


def test_worker_cpu_result_equals_the_one_off_denoise():
    page = noisy_page(seed=1)
    with pages.DenoiseWorker.start('CPU', 60, strict=True) as worker:
        assert np.array_equal(worker.denoise(page), pages._denoise(page, 60))


def test_a_worker_writing_far_more_than_a_pipe_buffer_to_stderr_still_finishes_its_page(monkeypatch):
    fake_worker(monkeypatch, """
import io, struct, sys
import numpy as np
sys.stderr.write('x' * 1_000_000 + '\\n'); sys.stderr.flush()
def send(value):
    out = io.BytesIO(); np.save(out, value); data = out.getvalue()
    sys.stdout.buffer.write(struct.pack('<Q', len(data)) + data); sys.stdout.buffer.flush()
send(np.array([1], dtype=np.uint8))
while True:
    head = sys.stdin.buffer.read(8)
    if len(head) < 8: break
    size = struct.unpack('<Q', head)[0]
    send(np.load(io.BytesIO(sys.stdin.buffer.read(size))))
""")
    page = noisy_page()
    with pages.DenoiseWorker.start('CPU', 30, strict=True) as worker:
        np.testing.assert_array_equal(worker.denoise(page), page)


def test_a_failing_worker_start_reports_the_stderr_tail_once(monkeypatch):
    fake_worker(monkeypatch, """
import io, struct, sys
import numpy as np
out = io.BytesIO(); np.save(out, np.array([0], dtype=np.uint8)); data = out.getvalue()
sys.stdout.buffer.write(struct.pack('<Q', len(data)) + data); sys.stdout.buffer.flush()
print('no such device', file=sys.stderr, flush=True)
sys.exit(2)
""")
    with pytest.raises(RuntimeError) as error:
        pages.DenoiseWorker.start('CUDA', 30, strict=True)
    assert str(error.value).count('no such device') == 1


def test_gpu_failure_falls_back_with_a_warning_unless_strict(caplog):
    worker = pages.DenoiseWorker.start('NOPE', 60, strict=False)
    try:
        assert worker.device == 'CPU' and 'denoising on the CPU' in worker.warnings[0]
        assert 'invalid device type' in worker.warnings[0]
        assert '\n' not in worker.warnings[0] and '\r' not in worker.warnings[0]
        assert worker.warnings[0] in caplog.text
    finally:
        worker.close()
    with pytest.raises(RuntimeError, match='NOPE'):
        pages.DenoiseWorker.start('NOPE', 60, strict=True)


def test_worker_startup_timeout_kills_and_reaps_child(monkeypatch):
    import threading
    import time

    processes = []
    timeout = 0.1

    class Input:
        def close(self): pass
        def write(self, data): pass
        def flush(self): pass

    def capture(*args, **kwargs):
        class Output:
            def __init__(self, killed): self.killed = killed
            def read(self, size=-1):
                self.killed.wait(0.8)
                return b''

        class Process:
            def __init__(self):
                self.killed = threading.Event()
                self.stdout, self.stderr, self.stdin = Output(self.killed), Output(self.killed), Input()

            def kill(self): self.killed.set()
            def wait(self, timeout=None): return 1
            def poll(self): return 1 if self.killed.is_set() else None

        process = Process()
        processes.append(process)
        return process

    fake_worker(monkeypatch, 'import time; time.sleep(60)')
    monkeypatch.setattr(pages.subprocess, 'Popen', capture)
    started = time.monotonic()
    with pytest.raises(RuntimeError, match='startup timed out after 0.1 s'):
        pages.DenoiseWorker.start('CUDA', timeout, strict=True)
    elapsed = time.monotonic() - started
    assert elapsed < timeout + 0.3, f'startup took {elapsed:.3f}s for a {timeout:.3f}s timeout'
    assert processes[0].poll() is not None


@pytest.mark.parametrize('bad_output', ['nan', 'shape'])
def test_non_strict_gpu_bad_result_restarts_on_cpu_and_redoes_page(monkeypatch, caplog, bad_output):
    class Process:
        class Output:
            def __init__(self, data): self.data = data
            def read(self, size=-1):
                result, self.data = self.data[:size], self.data[size:]
                return result
        class Input:
            def close(self): pass
            def write(self, data): pass
            def flush(self): pass

        def __init__(self, device):
            array = np.full((2 + 2 * pages._BORDER, 2 + 2 * pages._BORDER, 3),
                            np.nan if device == 'CUDA' else 1, np.float32)
            payload = npy(array)
            self.stdout = self.Output(struct.pack('<Q', len(payload)) + payload)
            self.stdin = self.Input()
            self.returncode = None

        def poll(self): return self.returncode
        def wait(self, timeout=None):
            self.returncode = 0
            return 0

    opened = []

    def open_worker(cls, device, timeout):
        worker = cls(device, timeout, Process(device))
        opened.append(worker)
        worker.cwd = None
        return worker

    monkeypatch.setattr(pages.DenoiseWorker, '_open', classmethod(open_worker))
    def exchange(self, payload):
        if self.device == 'CPU':
            return np.load(io.BytesIO(payload))
        shape = (2 + 2 * pages._BORDER, 2 + 2 * pages._BORDER, 3)
        return np.full(shape if bad_output == 'nan' else shape[1:],
                       np.nan if bad_output == 'nan' else 1, np.float32)

    monkeypatch.setattr(pages.DenoiseWorker, '_exchange', exchange, raising=False)
    worker = pages.DenoiseWorker.start('CUDA', 10, strict=False)
    try:
        result = worker.denoise(np.ones((2, 2, 3), np.float32))
        assert worker.device == 'CPU'
        assert result.shape == (2, 2, 3) and np.isfinite(result).all()
        assert len(opened) == 2
        assert 'denoising on the CPU' in worker.warnings[0]
        assert worker.warnings[0] in caplog.text
    finally:
        worker.close()


def test_non_strict_gpu_timeout_does_not_fall_back(monkeypatch):
    class Process:
        class Input:
            def close(self): pass
            def write(self, data): pass
            def flush(self): pass
        class Output:
            def read(self, size=-1): return b''
        stdin = Input()
        stdout = Output()
        killed = False

        def poll(self): return 1 if self.killed else None
        def wait(self, timeout=None): self.killed = True; return 1
        def kill(self): self.killed = True

    opened = []

    def open_worker(cls, device, timeout):
        opened.append(device)
        return cls(device, timeout, Process())

    def timeout(self, payload):
        import time
        time.sleep(0.2)
        return np.load(io.BytesIO(payload))

    monkeypatch.setattr(pages.DenoiseWorker, '_open', classmethod(open_worker))
    monkeypatch.setattr(pages.DenoiseWorker, '_exchange', timeout, raising=False)
    worker = pages.DenoiseWorker.start('CUDA', 0.05, strict=False)
    with pytest.raises(RuntimeError, match='timed out after 0.05 s'):
        worker.denoise(np.ones((2, 2, 3), np.float32))
    assert worker.device == 'CUDA' and opened == ['CUDA']


def test_strict_page_failure_names_device_and_child_error(monkeypatch):
    class Process:
        class Output:
            def read(self, size=-1): return b''
        class Input:
            def close(self): pass
            def write(self, data): pass
            def flush(self): pass
        stdin, stdout = Input(), Output()

        def poll(self): return 1
        def wait(self, timeout=None): return 1

    def open_worker(cls, device, timeout):
        worker = cls(device, timeout, Process())
        worker.cwd = tempfile.TemporaryDirectory()
        (Path(worker.cwd.name) / 'stderr.txt').write_bytes(b'filter exploded\r\n')
        return worker

    def fail(self, payload):
        raise RuntimeError('worker exited')

    monkeypatch.setattr(pages.DenoiseWorker, '_open', classmethod(open_worker))
    monkeypatch.setattr(pages.DenoiseWorker, '_exchange', fail, raising=False)
    worker = pages.DenoiseWorker.start('CUDA', 10, strict=True)
    with pytest.raises(RuntimeError, match=r'OIDN CUDA.*filter exploded.*Visual C\+\+ redistributable'):
        worker.denoise(np.ones((2, 2, 3), np.float32))
    worker.close()


def test_denoise_seconds_count_a_fallback_page_once_and_include_the_start(monkeypatch):
    class Process:
        class Input:
            def close(self): pass
            def write(self, data): pass
            def flush(self): pass
        class Output:
            def read(self, size=-1): return b''
        stdin, stdout = Input(), Output()
        def poll(self): return 0
        def wait(self, timeout=None): return 0

    def open_worker(cls, device, timeout):
        time.sleep(0.2)
        worker = cls(device, timeout, Process())
        worker.cwd = None
        return worker

    def exchange(self, payload):
        if self.device == 'CPU':
            time.sleep(0.4)
            return np.load(io.BytesIO(payload))
        raise RuntimeError('worker exited')

    monkeypatch.setattr(pages.DenoiseWorker, '_open', classmethod(open_worker))
    monkeypatch.setattr(pages.DenoiseWorker, '_exchange', exchange, raising=False)
    worker = pages.DenoiseWorker.start('CUDA', 10, strict=False)
    startup = worker.seconds
    assert startup >= 0.2
    began = time.perf_counter()
    worker.denoise(np.ones((2, 2, 3), np.float32))
    wall = time.perf_counter() - began
    worker.close()
    # the CPU retry runs through denoise again; adding it to the outer call would double its 0.4 s
    assert worker.seconds - startup == pytest.approx(wall, abs=0.15)
    assert worker.device == 'CPU'


@pytest.mark.gpu_nvidia
def test_gpu_denoise_matches_cpu():
    page = noisy_page(seed=2, size=256)
    with pages.DenoiseWorker.start('CUDA', 60, strict=True) as gpu, pages.DenoiseWorker.start('CPU', 60, strict=True) as cpu:
        assert gpu.device == 'CUDA'
        a, b = gpu.denoise(page), cpu.denoise(page)
    rel = np.abs(a - b) / np.maximum(b, 1e-3)
    assert rel.mean() <= 5e-3 and np.percentile(rel, 99) <= 2e-2


@pytest.mark.parametrize('value', [0.05, 0.5, 4.0])
def test_constant_page_stays_constant(value):
    mask, cls = full_page()
    raw = np.full((48, 48, 3), value, dtype=np.float32)
    out = finish_float(raw, mask, cls, {}, Settings(denoise=True))
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out, value, rtol=0.02)


def test_ambient_adds_per_class_exactly():
    mask = np.zeros((32, 32), dtype=bool)
    cls = np.zeros((32, 32), dtype=np.int16)
    mask[:, :16] = True
    cls[:, :8] = 1
    cls[:, 8:16] = 2
    raw = np.where(mask[..., None], 0.5, 0.0).astype(np.float32) * np.ones(3, np.float32)
    s = Settings(denoise=True)
    base = finish_float(raw, mask, cls, {}, s)
    amb = {1: (0.1, 0.2, 0.3), 2: (0.01, 0.02, 0.03)}
    out = finish_float(raw, mask, cls, amb, s)
    d = out - base
    np.testing.assert_allclose(d[:, :8], np.broadcast_to([0.1, 0.2, 0.3], d[:, :8].shape), atol=1e-6)
    np.testing.assert_allclose(d[:, 8:16], np.broadcast_to([0.01, 0.02, 0.03], d[:, 8:16].shape), atol=1e-6)
    np.testing.assert_array_equal(d[:, 26:], 0)


def test_all_black_page():
    mask, cls = full_page(32)
    out = finish_float(np.zeros((32, 32, 3), np.float32), mask, cls, {}, Settings(denoise=True))
    assert not np.isnan(out).any()
    np.testing.assert_array_equal(out, 0)


def test_float_until_quantise_and_single_rounding():
    mask, cls = full_page(32)
    s = Settings(denoise=True)
    rng = np.random.default_rng(1)
    raw = rng.uniform(0, 2, (32, 32, 3)).astype(np.float32)
    out = finish_float(raw, mask, cls, {0: (0.1, 0.1, 0.1)}, s)
    assert out.dtype == np.float32
    px = quantise(out, s)
    assert px.dtype == np.uint16
    np.testing.assert_array_equal(px, encode_r5g6b5(tone(out, s.gamma), s.dither))


def test_tone():
    np.testing.assert_allclose(tone(np.array([-1.0, 0.25, 0.81]), 0.5), [0, 0.5, 0.9])


def test_expose_keeps_the_hue_of_an_over_bright_colour():
    np.testing.assert_allclose(expose(np.array([[1.34, 1.0, 1.0]])), [[1, 1 / 1.34, 1 / 1.34]])
    np.testing.assert_allclose(expose(np.array([1.34, 1.0, 1.0])), [1, 0.746, 0.746], atol=1e-3)


def test_expose_leaves_a_colour_within_range_alone():
    inside = np.array([[1.0, 0.5, 0.0], [0.2, 0.1, 0.9]])
    np.testing.assert_array_equal(expose(inside), inside)
    np.testing.assert_array_equal(expose(np.array([[-0.5, 0.5, 0.2]])), [[0, 0.5, 0.2]])


def test_tone_exposes_after_the_gamma():
    # gamma 2 turns (1.2, 0.6, 0.6) into (1.44, 0.36, 0.36), which exposure then rescales
    np.testing.assert_allclose(tone(np.array([1.2, 0.6, 0.6]), 2.0), [1, 0.25, 0.25])


def test_denoise_unbiased():
    mask, cls = full_page(64)
    value = 0.5
    rng = np.random.default_rng(7)
    raw = (value * (1 + 0.3 * rng.standard_normal((64, 64, 3)))).astype(np.float32)
    out = finish_float(raw, mask, cls, {}, Settings(denoise=True))
    assert not np.isnan(out).any()
    assert abs(out.mean() - value) / value < 0.02
    assert out.std() <= raw.std() / 2


def test_padding_comes_from_charts_not_denoiser():
    uv = np.array([[0.375, 0.375], [0.625, 0.375], [0.375, 0.625], [0.625, 0.625]])
    mask, cls = coverage(uv, np.array([[0, 1, 2], [1, 3, 2]]), np.array([3, 3]), 64, 64)
    raw = np.where(mask[..., None], 1.0, 0.0).astype(np.float32) * np.ones(3, np.float32)
    s = Settings(denoise=True)
    out = finish_float(raw, mask, cls, {3: (0.5, 0.5, 0.5)}, s)
    dilated = dilate(mask.astype(np.float32)[..., None], mask, cls, 2 * s.padding)[1]
    padding = dilated & ~mask
    assert padding.any()
    # ambient reaches the padding through the class map; far texels stay 0
    np.testing.assert_allclose(out[mask], 1.5, atol=0.02)
    np.testing.assert_allclose(out[padding], 1.5, atol=0.02)
    assert out[0, 0].max() == 0


def fake_run(monkeypatch, **kw):
    def run(*args, **kwargs):
        run.kwargs = kwargs
        if 'raise_' in kw:
            raise kw['raise_']
        return subprocess.CompletedProcess(args, kw.get('code', 0), kw['stdout'](kwargs) if 'stdout' in kw else b'', kw.get('stderr', b''))
    monkeypatch.setattr(pages.subprocess, 'run', run)
    return run


def test_denoise_timeout_becomes_runtime_error(monkeypatch):
    run = fake_run(monkeypatch, raise_=subprocess.TimeoutExpired('x', 3))
    with pytest.raises(RuntimeError, match='timed out after 3 s'):
        pages._denoise(np.ones((4, 4, 3), np.float32), 3)
    assert run.kwargs['timeout'] == 3


def test_finish_float_passes_the_settings_timeout(monkeypatch):
    run = fake_run(monkeypatch, raise_=subprocess.TimeoutExpired('x', 9))
    mask, cls = full_page(8)
    with pytest.raises(RuntimeError, match='timed out'):
        finish_float(np.ones((8, 8, 3), np.float32), mask, cls, {}, Settings(denoise=True, timeout_s=9))
    assert run.kwargs['timeout'] == 9


def npy(array):
    buffer = io.BytesIO()
    np.save(buffer, array)
    return buffer.getvalue()


def test_denoise_rejects_wrong_shape(monkeypatch):
    fake_run(monkeypatch, stdout=lambda kw: npy(np.ones((4, 4, 3), np.float32)))
    with pytest.raises(RuntimeError, match='bad result'):
        pages._denoise(np.ones((4, 4, 3), np.float32), 5)


def test_denoise_unreadable_output_keeps_redistributable_hint(monkeypatch):
    fake_run(monkeypatch, stdout=lambda kw: b'unreadable output')
    with pytest.raises(RuntimeError, match=r'unreadable data.*Visual C\+\+ redistributable'):
        pages._denoise(np.ones((4, 4, 3), np.float32), 5)


def test_denoise_rejects_nan(monkeypatch):
    size = 4 + 2 * pages._BORDER
    result = np.ones((size, size, 3), np.float32)
    result[3, 3, 1] = np.nan
    fake_run(monkeypatch, stdout=lambda kw: npy(result))
    with pytest.raises(RuntimeError, match='bad result'):
        pages._denoise(np.ones((4, 4, 3), np.float32), 5)


def test_denoise_failure_names_the_redistributable(monkeypatch):
    fake_run(monkeypatch, code=3221225477, stderr=b'boom')
    with pytest.raises(RuntimeError, match=r'boom.*Visual C\+\+ redistributable'):
        pages._denoise(np.ones((4, 4, 3), np.float32), 5)


def test_intermediates_stay_float32(monkeypatch):
    seen = []
    real_dilate, real_denoise = pages.dilate, pages._denoise
    monkeypatch.setattr(pages, 'dilate', lambda *a: seen.append(('dilate', a[0].dtype)) or real_dilate(*a))
    monkeypatch.setattr(pages, '_denoise', lambda page, timeout, worker=None:
                        seen.append(('denoise', page.dtype)) or real_denoise(page, timeout, worker))
    mask, cls = full_page(16)
    raw = np.full((16, 16, 3), 0.5, np.float64)
    out = finish_float(raw, mask, cls, {0: (0.1, 0.1, 0.1)}, Settings(denoise=True))
    assert [name for name, _ in seen] == ['dilate', 'dilate', 'denoise', 'dilate']
    assert {dtype for _, dtype in seen} == {np.dtype(np.float32)}
    assert out.dtype == np.float32


def test_covered_unwritten_texel_fills_from_written_neighbour(caplog):
    import inspect
    assert 'written' in inspect.signature(finish_float).parameters, 'written mask is missing'
    raw = np.zeros((1, 9, 3), np.float32)
    raw[0, 1] = [0.2, 0.4, 0.6]
    mask = np.zeros((1, 9), bool)
    mask[0, [1, 2, 8]] = True
    written = np.zeros_like(mask)
    written[0, 1] = True
    cls = np.zeros_like(mask, dtype=np.int16)
    cls[0, 2] = 1
    with caplog.at_level('INFO', logger='lightmapper.pages'):
        out = finish_float(raw, mask, cls, {0: (0.1, 0.1, 0.1), 1: (0.3, 0.2, 0.1)},
                           Settings(denoise=False, padding=1), written=written)
    np.testing.assert_allclose(out[0, 2], [0.5, 0.6, 0.7])
    np.testing.assert_allclose(out[0, 8], [0.1, 0.1, 0.1])
    np.testing.assert_array_equal(out[0, 6], 0)
    assert '2 covered texels unwritten; 1 stayed unfilled after dilation' in caplog.text


def test_written_black_is_not_filled_over():
    import inspect
    assert 'written' in inspect.signature(finish_float).parameters, 'written mask is missing'
    raw = np.float32([[[1, 1, 1], [0, 0, 0], [0, 0, 0]]])
    out = finish_float(raw, np.ones((1, 3), bool), np.zeros((1, 3), np.int16), {},
                       Settings(denoise=False), written=np.bool_([[True, True, False]]))
    np.testing.assert_array_equal(out, [[[1, 1, 1], [0, 0, 0], [0, 0, 0]]])


@pytest.mark.parametrize('denoise', [False, True])
def test_written_uncovered_texels_and_padding_beyond_take_the_covered_class(denoise):
    # a covered column of class -1, then a texel the bake wrote without covering its centre
    # (class 0 by default), then the padding past it: all three get the class -1 ambient
    mask = np.zeros((16, 16), dtype=bool)
    cls = np.zeros((16, 16), dtype=np.int16)
    mask[4:12, 2:6] = True
    cls[mask] = -1
    written = mask.copy()
    written[4:12, 6] = True
    raw = np.where(written[..., None], 0.2, 0.0).astype(np.float32) * np.ones(3, np.float32)
    s = Settings(denoise=denoise, padding=2)
    base = finish_float(raw, mask, cls, {}, s, written=written)
    out = finish_float(raw, mask, cls, {-1: (0.1, 0.1, 0.1), 0: (0.5, 0.5, 0.5)}, s, written=written)
    added = (out - base)[5:11]
    np.testing.assert_allclose(added[:, 2:6], 0.1, atol=1e-6)
    np.testing.assert_allclose(added[:, 6], 0.1, atol=1e-6)
    np.testing.assert_allclose(added[:, 7:11], 0.1, atol=1e-6)


def test_a_temp_folder_that_cannot_be_deleted_does_not_fail_close(caplog):
    worker = pages.DenoiseWorker.start('CPU', 60, strict=True)
    folder = worker.cwd.name
    # (an open handle keeps stderr.txt undeletable on Windows)
    leaked = open(Path(folder) / pages.STDERR_FILE, 'rb')
    try:
        with caplog.at_level(logging.INFO, logger='lightmapper.pages'):
            worker.close()
    finally:
        leaked.close()
        shutil.rmtree(folder, ignore_errors=True)
    assert [(r.levelno, folder in r.getMessage()) for r in caplog.records if 'left temp folder' in r.getMessage()]         == [(logging.INFO, True)]


def test_one_off_denoise_survives_an_undeletable_temp_folder(monkeypatch):
    real = tempfile.TemporaryDirectory

    class Stuck(real):
        def cleanup(self):
            raise PermissionError(32, 'in use')

    monkeypatch.setattr(pages.tempfile, 'TemporaryDirectory', Stuck)
    page = np.full((8, 8, 3), 0.5, np.float32)
    assert pages._denoise(page, 60).shape[:2] == (8, 8)
