import re
import threading
from pathlib import Path
from typing import Callable


STAGES = ('gather', 'charts', 'scene', 'blender', 'finish', 'write', 'validate')
STAGE_LABELS: dict[str, str] = {
    'gather': 'Reading the level',
    'charts': 'Laying out lightmap pages',
    'scene': 'Building the Blender scene',
    'blender': 'Baking in Blender',
    'finish': 'Denoising and finishing pages',
    'write': 'Writing tags',
    'validate': 'Checking the result',
}
_SPANS = {
    'gather': (0.00, 0.05),
    'charts': (0.05, 0.15),
    'scene': (0.15, 0.20),
    'blender': (0.20, 0.90),
    'finish': (0.90, 0.95),
    'write': (0.95, 0.98),
    'validate': (0.98, 1.00),
}


def parse_line(line: str) -> tuple | None:
    match = re.match(r'^\[bake\] device (\S+)', line)
    if match:
        return 'device', match.group(1)
    match = re.match(r'^\[bake\] page (\d+) \(', line)
    if match:
        return 'page', int(match.group(1))
    if line.startswith('[bake] vertex pass:'):
        return ('vertex',)
    return None


def fraction(stage: str, pages_done: int = 0, pages_total: int = 0) -> float:
    if stage == 'done':
        return 1.0
    if stage == 'vertex':
        return 0.90
    start, end = _SPANS[stage]
    if stage == 'blender':
        if pages_total <= 0:
            return start
        return 0.20 + 0.60 * min(max(pages_done, 0), pages_total) / pages_total
    return start


class BlenderLogFollower(threading.Thread):
    def __init__(self, log: Path, on_line: Callable[[str], None], interval: float = 0.25):
        super().__init__(daemon=True)
        self.log = Path(log)
        self.on_line = on_line
        self.interval = interval
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def run(self):
        offset = 0
        pending = b''
        while not self._stop_event.wait(self.interval):
            offset, pending = self._read(offset, pending)
        self._read(offset, pending)

    def _read(self, offset: int, pending: bytes) -> tuple[int, bytes]:
        try:
            if self.log.stat().st_size < offset:
                offset = 0
                pending = b''
            with self.log.open('rb') as handle:
                handle.seek(offset)
                data = handle.read()
                offset = handle.tell()
        except OSError:
            return offset, pending
        lines = (pending + data).split(b'\n')
        for line in lines[:-1]:
            self.on_line(line.rstrip(b'\r').decode('utf-8', errors='replace'))
        return offset, lines[-1]
