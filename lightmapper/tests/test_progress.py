import logging
import time

import pytest

from test_pipeline import tiny, api


def wait_until(predicate):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.01)
    raise AssertionError('condition was not met before timeout')


def test_pipeline_reports_each_stage_in_order(tiny):
    tags, settings, work = tiny
    seen = []
    api().bake(tags, 'level', 'room', settings, work=work, progress=seen.append)
    assert seen == ['gather', 'charts', 'scene', 'blender', 'vertex', 'finish', 'write']


def test_pipeline_logs_stage_starts(tiny, caplog):
    caplog.set_level(logging.INFO)
    tags, settings, work = tiny
    api().bake(tags, 'level', 'room', settings, work=work)
    assert 'stage: blender' in caplog.text


@pytest.mark.parametrize('line, event', [
    ('[bake] device HIP', ('device', 'HIP')),
    ('[bake] page 3 (512x256) baked in 1.20 s', ('page', 3)),
    ('[bake] vertex pass: 10 queries, 40 rays in 0.1 s (400 rays/s)', ('vertex',)),
    ('[bake] warning: page 3: the emit pass wrote 5 texels the diffuse pass did not', None),
    ('Blender 5.2.2 (hash ...)', None)])
def test_parse_line(line, event):
    from lightmapper.progress import parse_line
    assert parse_line(line) == event


def test_fraction_spans():
    from lightmapper.progress import fraction, STAGE_LABELS
    assert fraction('vertex') == 0.80 and STAGE_LABELS['vertex'] == 'Lighting the vertices'
    assert fraction('gather') == 0 and fraction('blender') == .2
    assert fraction('blender', 2, 4) == pytest.approx(.5) and fraction('blender', 0, 0) == .2
    assert fraction('finish') == .9 and fraction('validate') == .98 and fraction('done') == 1


def test_fraction_never_goes_back():
    from lightmapper.progress import STAGES, fraction
    points = [fraction(s) for s in STAGES]
    assert points == sorted(points)


def test_follower_delivers_complete_lines_only(tmp_path):
    from lightmapper.progress import BlenderLogFollower
    log = tmp_path / 'blender.log'
    got = []
    follower = BlenderLogFollower(log, got.append, interval=.01)
    follower.start()
    log.write_bytes(b'[bake] device OPTIX\n[bake] page 0 (8x8) ba')
    wait_until(lambda: got == ['[bake] device OPTIX'])
    with open(log, 'ab') as handle:
        handle.write(b'ked in 1 s\n')
    follower.stop()
    follower.join(2)
    assert got == ['[bake] device OPTIX', '[bake] page 0 (8x8) baked in 1 s']


def test_follower_restarts_after_log_truncation(tmp_path):
    from lightmapper.progress import BlenderLogFollower
    log = tmp_path / 'blender.log'
    got = []
    follower = BlenderLogFollower(log, got.append, interval=.01)
    follower.start()
    log.write_bytes(b'one\ntwo\nthree\n')
    wait_until(lambda: got == ['one', 'two', 'three'])
    log.write_bytes(b'new run\n')
    try:
        wait_until(lambda: got == ['one', 'two', 'three', 'new run'])
    finally:
        follower.stop()
        follower.join(2)
    assert got == ['one', 'two', 'three', 'new run']
