"""OIDN's isolated worker process."""

import ctypes
import os
import sys
from pathlib import Path

from . import frozen

WORKER_FLAG = frozen.DENOISE_FLAG
RUNTIME_DLLS = ('vcruntime140', 'vcruntime140_1', 'msvcp140', 'msvcp140_1', 'msvcp140_2')


def runtime_dll_dir() -> Path:
    if frozen.is_frozen():
        return frozen.bundle_dir()
    return Path(os.environ['SystemRoot']) / 'System32'


def worker_command(kind: str | None = None) -> list[str]:
    return frozen.child_command(WORKER_FLAG, *([kind] if kind is not None else []))


def main(stdin=None, stdout=None, kind: str | None = None) -> int:
    stdin = sys.stdin.buffer if stdin is None else stdin
    stdout = sys.stdout.buffer if stdout is None else stdout
    if os.name == 'nt':
        # oidn needs the 14.4x msvcp140/vcruntime140 runtime. anaconda keeps a 2019 copy beside python.exe;
        # once numpy or numba load it, oidn device creation dies with an access violation. source runs use the
        # newer system32 copy; frozen runs ship it beside the exe, so load it before either import.
        for name in RUNTIME_DLLS:
            path = runtime_dll_dir() / f'{name}.dll'
            if path.is_file():
                ctypes.WinDLL(str(path))
    if kind is not None:
        return serve(kind, stdin, stdout)
    import io
    import numpy as np
    import pyoidn
    img = np.load(io.BytesIO(stdin.read()))
    np.save(stdout, pyoidn.denoise(img, filter_type='RTLightmap'))
    return 0


def serve(kind: str, stdin, stdout) -> int:
    import io
    import struct
    import numpy as np
    import pyoidn

    def check_error():
        error = device.get_error()
        if error:
            raise RuntimeError(str(error))

    def receive():
        size = struct.unpack('<Q', stdin.read(8))[0]
        return np.load(io.BytesIO(stdin.read(size)))

    def send(value):
        out = io.BytesIO()
        np.save(out, value)
        data = out.getvalue()
        stdout.write(struct.pack('<Q', len(data)) + data)
        stdout.flush()

    try:
        if kind == 'CPU':
            device = None
            filt = src = dst = None
            buffer_size = 0
        else:
            types = {'CUDA': pyoidn.OIDN_DEVICE_TYPE_CUDA, 'HIP': pyoidn.OIDN_DEVICE_TYPE_HIP,
                     'NOPE': 255}
            device = pyoidn.Device(types[kind])
            device.commit()
            check_error()
            filt = pyoidn.Filter(device, 'RTLightmap')
            filt.set_quality(pyoidn.OIDN_QUALITY_HIGH)
            check_error()
            src = dst = None
            buffer_size = 0
        send(np.array([1], dtype=np.uint8))
    except Exception as error:
        send(np.array([0], dtype=np.uint8))
        print(error, file=sys.stderr, flush=True)
        return 2
    while True:
        try:
            img = receive()
        except Exception:
            break
        if kind == 'CPU':
            result = pyoidn.denoise(img, filter_type='RTLightmap')
        else:
            try:
                nbytes = img.nbytes
                if nbytes != buffer_size:
                    if src is not None:
                        src.release()
                        dst.release()
                    src = pyoidn.Buffer(device, nbytes, pyoidn.OIDN_STORAGE_DEVICE)
                    dst = pyoidn.Buffer(device, nbytes, pyoidn.OIDN_STORAGE_DEVICE)
                    check_error()
                    buffer_size = nbytes
                src.write(0, nbytes, img)
                check_error()
                filt.set_image(pyoidn.OIDN_IMAGE_COLOR, src, pyoidn.OIDN_FORMAT_FLOAT3, img.shape[1], img.shape[0])
                filt.set_image(pyoidn.OIDN_IMAGE_OUTPUT, dst, pyoidn.OIDN_FORMAT_FLOAT3, img.shape[1], img.shape[0])
                check_error()
                filt.commit()
                check_error()
                filt.execute()
                check_error()
                device.wait()
                check_error()
                result = np.empty_like(img)
                dst.read(0, nbytes, result)
                check_error()
            except Exception as error:
                print(error, file=sys.stderr, flush=True)
                return 2
        send(result)
    if filt is not None:
        filt.release()
    if src is not None:
        src.release()
        dst.release()
    if device is not None:
        device.release()
    return 0
