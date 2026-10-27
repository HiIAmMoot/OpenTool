"""Low-level OIDN double with native errors that do not raise."""

import numpy as np

OIDN_DEVICE_TYPE_CUDA = 2
OIDN_DEVICE_TYPE_HIP = 3
OIDN_QUALITY_HIGH = 1
OIDN_STORAGE_DEVICE = 1
OIDN_IMAGE_COLOR = 'color'
OIDN_IMAGE_OUTPUT = 'output'
OIDN_FORMAT_FLOAT3 = 3
failure = 'execute'


class Device:
    def __init__(self, kind):
        self.error = None
        self.page = 0

    def commit(self): pass
    def release(self): pass

    def fail(self, operation):
        if operation == failure and (operation == 'setup' or self.page == 2):
            self.error = f'native {operation} failure'
            return True
        return False

    def get_error(self):
        error, self.error = self.error, None
        return error

    def wait(self):
        self.fail('wait')


class Buffer:
    def __init__(self, device, size, storage):
        self.device = device
        self.data = np.zeros(size // 4, np.float32)

    def release(self): pass

    def write(self, offset, size, image):
        self.device.page += 1
        if not self.device.fail('write'):
            self.data[:] = image.ravel()

    def read(self, offset, size, image):
        self.device.fail('read')
        image[:] = self.data.reshape(image.shape)


class Filter:
    def __init__(self, device, kind):
        self.device = device
        self.images = {}

    def set_quality(self, quality):
        self.device.fail('setup')

    def set_image(self, name, buffer, fmt, width, height):
        self.images[name] = buffer

    def commit(self):
        self.device.fail('commit')

    def execute(self):
        if not self.device.fail('execute'):
            self.images['output'].data[:] = self.images['color'].data

    def release(self): pass
