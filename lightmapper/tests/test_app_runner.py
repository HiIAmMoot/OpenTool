import logging
import shutil

import pytest

from lightmapper import pipeline
from lightmapper.tagio.tagfile import new_element, read_tag, resolve_reference_path, write_tag
from test_pipeline import child, tiny
import opentool.runner as runner
from opentool.runner import BakeRequest, run_bake


def run(tiny, **kw):
    tags, settings, work = tiny; events = []
    out = run_bake(BakeRequest(tags, 'level', 'room', settings, kw.pop('new_names', True), work), events.append)
    return out, events


def test_runner_bakes_into_new_names(tiny):
    out, events = run(tiny)
    tags = tiny[0]
    assert out.status == 'done' and (tags / 'room_lit.scenario_structure_bsp').is_file()
    assert (tags / 'level_lit.scenario').is_file() and out.output_folder == tags
    assert events[-1] == ('done', out)
    assert out.message == 'Done. The lit level is level_lit.'


@pytest.fixture
def two_bsps(tiny):
    tags, settings, work = tiny
    shutil.copy(tags / 'room.scenario_structure_bsp', tags / 'room2.scenario_structure_bsp')
    scenario = read_tag(tags / 'level.scenario')
    second = child(scenario.root, 'structure_bsps'); second.set_reference('structure_bsp', 'sbsp', 'room2')
    scenario.root.set_block('structure_bsps', [*scenario.root.block('structure_bsps'), second])
    write_tag(scenario, tags / 'level.scenario')
    return tiny


def test_second_bsp_keeps_the_first_bsps_lit_reference(two_bsps):
    tags, settings, work = two_bsps
    run_bake(BakeRequest(tags, 'level', 'room', settings, True, work / 'a'), lambda e: None)
    run_bake(BakeRequest(tags, 'level', 'room2', settings, True, work / 'b'), lambda e: None)
    refs = [e.reference('structure_bsp')[1] for e in read_tag(tags / 'level_lit.scenario').root.block('structure_bsps')]
    assert refs == ['room_lit', 'room2_lit']
    assert [e.reference('structure_bsp')[1] for e in read_tag(tags / 'level.scenario').root.block('structure_bsps')] == ['room', 'room2']


def test_rebaking_the_same_bsp_keeps_one_lit_name(two_bsps):
    tags, settings, work = two_bsps
    for w in ('a', 'b'):
        assert run_bake(BakeRequest(tags, 'level', 'room', settings, True, work / w), lambda e: None).status == 'done'
    refs = [e.reference('structure_bsp')[1] for e in read_tag(tags / 'level_lit.scenario').root.block('structure_bsps')]
    assert refs == ['room_lit', 'room2'] and not (tags / 'room_lit_lit.scenario_structure_bsp').exists()


def test_rebake_reads_the_original_bsp(two_bsps, monkeypatch):
    tags, settings, work = two_bsps
    run_bake(BakeRequest(tags, 'level', 'room', settings, True, work / 'a'), lambda e: None)
    seen = []
    real = pipeline.bake
    monkeypatch.setattr(runner.pipeline, 'bake', lambda *a, **k: seen.append(a[1:3]) or real(*a, **k))
    run_bake(BakeRequest(tags, 'level', 'room', settings, True, work / 'b'), lambda e: None)
    assert seen == [('level', 'room')]


def test_reused_work_drops_an_old_blender_log(tiny):
    tags, settings, work = tiny
    work.mkdir(parents=True, exist_ok=True); (work / 'blender.log').write_text('[bake] device CUDA\n')
    _, events = run(tiny)
    assert ('device', 'CUDA') not in events


def test_runner_restores_the_root_log_level(tiny):
    root = logging.getLogger(); root.setLevel(logging.WARNING); run(tiny)
    assert root.level == logging.WARNING


def test_runner_overwrites_in_place_only_when_asked(tiny):
    out, _ = run(tiny, new_names=False)
    assert out.status == 'done'
    assert not (tiny[0] / 'room_lit.scenario_structure_bsp').exists()
    assert not (tiny[0] / 'level_lit.scenario').exists()


def test_runner_stages_end_with_validate(tiny):
    _, events = run(tiny)
    assert [e[1] for e in events if e[0] == 'stage'] == ['gather', 'charts', 'scene', 'blender', 'finish', 'write', 'validate']


def test_runner_forwards_log_lines_and_writes_a_log_file(tiny):
    out, events = run(tiny)
    assert any(e[0] == 'log' and 'stage: charts' in e[1] for e in events)
    assert 'stage: write' in (out.work / 'opentool.log').read_text()


def test_runner_reports_validation_problems(tiny, monkeypatch):
    monkeypatch.setattr(runner, 'validate_bsp', lambda *a, **k: ['last element must have bitmap -1'])
    out, _ = run(tiny)
    assert out.status == 'problems' and out.problems == ['last element must have bitmap -1']
    assert out.message == 'Finished with problems. The tags were written; the problems are listed in the log.'


def test_runner_output_folder_is_written_bsp_folder(tiny):
    tags, settings, work = tiny
    nested = tags / 'levels' / 'x'
    nested.mkdir(parents=True)
    shutil.move(tags / 'room.scenario_structure_bsp', nested / 'room.scenario_structure_bsp')
    scenario = read_tag(tags / 'level.scenario')
    scenario.root.block('structure_bsps')[0].set_reference('structure_bsp', 'sbsp', 'levels\\x\\room')
    write_tag(scenario, nested / 'level.scenario')
    (tags / 'level.scenario').unlink()
    out = run_bake(BakeRequest(tags, 'levels\\x\\level', 'room', settings, True, work), lambda e: None)
    assert out.output_folder == resolve_reference_path(tags, 'sbsp', 'levels\\x\\room_lit').parent


def test_runner_validates_tags_read_back_from_disk(tiny, monkeypatch):
    tags, settings, work = tiny
    seen_reads = []
    seen_validation = []
    read = runner.read_tag

    def track_read(path):
        tag = read(path)
        seen_reads.append((path, tag, path.read_bytes()))
        return tag

    def validate(bsp_tag, bitmap_tag):
        seen_validation.append((bsp_tag, bitmap_tag))
        return []

    monkeypatch.setattr(runner, 'read_tag', track_read)
    monkeypatch.setattr(runner, 'validate_bsp', validate)
    out, _ = run(tiny)
    bsp_path = resolve_reference_path(tags, 'sbsp', 'room_lit')
    written_bsp = read(bsp_path)
    bitmap_path = resolve_reference_path(tags, 'bitm', written_bsp.root.reference('lightmaps_bitmap')[1])
    assert len(seen_validation) == 1
    bsp_tag, bitmap_tag = seen_validation[0]
    assert any(path == bsp_path and tag is bsp_tag and data == bsp_path.read_bytes()
               for path, tag, data in seen_reads)
    assert any(path == bitmap_path and tag is bitmap_tag and data == bitmap_path.read_bytes()
               for path, tag, data in seen_reads)
    assert out.status == 'done'


def test_runner_setup_failure_still_emits_failed_outcome(tiny, monkeypatch):
    def denied(*a, **k):
        raise PermissionError(13, 'denied', 'C:\\tags\\opentool.log')

    monkeypatch.setattr(runner.logging, 'FileHandler', denied)
    events = []
    out = run_bake(BakeRequest(tiny[0], 'level', 'room', tiny[1], True, tiny[2]), events.append)
    assert out.status == 'failed'
    assert events[-1] == ('done', out)
    assert [event for event in events if event[0] == 'done'] == [('done', out)]


def test_runner_ignores_raising_event_callback(tiny):
    def broken_emit(event):
        raise RuntimeError('event sink closed')

    assert run_bake(BakeRequest(tiny[0], 'level', 'room', tiny[1], True, tiny[2]), broken_emit).status == 'done'


def test_broken_log_sink_does_not_recurse(tiny, monkeypatch):
    tags, settings, work = tiny
    calls = []

    def broken_emit(event):
        if event[0] == 'log' and ('one line' in event[1] or 'Could not report bake event' in event[1]):
            calls.append(event)
            raise RuntimeError('event sink closed')

    real_bake = pipeline.bake

    def log_once(*args, **kwargs):
        logging.getLogger('lightmapper.x').info('one line')
        return real_bake(*args, **kwargs)

    monkeypatch.setattr(runner.pipeline, 'bake', log_once)
    out = run_bake(BakeRequest(tags, 'level', 'room', settings, True, work), broken_emit)
    assert out.status == 'done'
    assert len(calls) <= 3
    assert (work / 'opentool.log').stat().st_size < 50_000


def test_runner_continues_when_follower_cannot_start(tiny, monkeypatch):
    def fail_start(self):
        raise RuntimeError('thread could not start')

    monkeypatch.setattr(runner.BlenderLogFollower, 'start', fail_start)
    assert run(tiny)[0].status == 'done'


def test_runner_reports_write_failure(tiny, monkeypatch):
    def denied(*a, **k): raise PermissionError(13, 'denied', 'C:\\tags\\room_lit.bitmap')
    monkeypatch.setattr(pipeline, '_write', denied)
    out, events = run(tiny)
    assert out.status == 'failed' and 'could not write' in out.message
    assert 'Traceback' in (out.work / 'opentool.log').read_text()


def test_runner_follows_blender_log(tiny, monkeypatch):
    # replace the tiny fixture's fake Blender with one that writes blender.log lines then returns
    tags, settings, work = tiny
    real = pipeline.run_blender

    def write_log(path, s):
        (path / 'blender.log').write_text('[bake] device OPTIX\n[bake] page 0 (0, 0)\n')
        return real(path, s)

    monkeypatch.setattr(pipeline, 'run_blender', write_log)
    _, events = run(tiny)
    assert ('device', 'OPTIX') in events and ('pages', 1, 1) in events


def test_runner_removes_its_log_handlers(tiny):
    before = list(logging.getLogger().handlers); run(tiny); assert logging.getLogger().handlers == before
