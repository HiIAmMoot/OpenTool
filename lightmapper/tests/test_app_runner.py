import json
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


def test_rebake_carries_current_scenario_root_edits(tiny):
    tags, settings, work = tiny
    assert run_bake(BakeRequest(tags, 'level', 'room', settings, True, work / 'a'), lambda e: None).status == 'done'
    scenario = read_tag(tags / 'level.scenario')
    scenario.root.set('local_north', 1.25)
    write_tag(scenario, tags / 'level.scenario')
    assert run_bake(BakeRequest(tags, 'level', 'room', settings, True, work / 'b'), lambda e: None).status == 'done'
    assert read_tag(tags / 'level_lit.scenario').root.get('local_north') == pytest.approx(1.25)


def test_rebake_after_bsp_reorder_keeps_lit_refs_in_current_order(two_bsps):
    tags, settings, work = two_bsps
    run_bake(BakeRequest(tags, 'level', 'room', settings, True, work / 'a'), lambda e: None)
    run_bake(BakeRequest(tags, 'level', 'room2', settings, True, work / 'b'), lambda e: None)
    scenario = read_tag(tags / 'level.scenario')
    entries = scenario.root.block('structure_bsps')
    scenario.root.set_block('structure_bsps', [entries[1], entries[0]])
    write_tag(scenario, tags / 'level.scenario')
    assert run_bake(BakeRequest(tags, 'level', 'room', settings, True, work / 'c'), lambda e: None).status == 'done'
    refs = [e.reference('structure_bsp')[1] for e in read_tag(tags / 'level_lit.scenario').root.block('structure_bsps')]
    assert refs == ['room2_lit', 'room_lit']


def test_lit_scenario_matches_the_baked_bsp_by_name_in_a_folder(tiny):
    tags, _, _ = tiny
    scenario = read_tag(tags / 'level.scenario')
    scenario.root.block('structure_bsps')[0].set_reference('structure_bsp', 'sbsp', 'levels\\x\\room')
    write_tag(scenario, tags / 'level.scenario')
    runner.write_lit_scenario(tags, 'level', 'room', 'levels\\x\\room_baked', 'level_lit')
    refs = [e.reference('structure_bsp')[1] for e in read_tag(tags / 'level_lit.scenario').root.block('structure_bsps')]
    assert refs == ['levels\\x\\room_baked']


def test_rebake_reads_the_original_bsp(two_bsps, monkeypatch):
    tags, settings, work = two_bsps
    run_bake(BakeRequest(tags, 'level', 'room', settings, True, work / 'a'), lambda e: None)
    seen = []
    real = pipeline.bake
    monkeypatch.setattr(runner.pipeline, 'bake', lambda *a, **k: seen.append(a[1:3]) or real(*a, **k))
    run_bake(BakeRequest(tags, 'level', 'room', settings, True, work / 'b'), lambda e: None)
    assert seen == [('level', 'room')]


def test_bake_request_charts_reach_the_pipeline(tiny, monkeypatch, tmp_path):
    tags, settings, work = tiny
    seen = []
    real = pipeline.bake
    monkeypatch.setattr(runner.pipeline, 'bake', lambda *a, **k: seen.append(k.get('charts')) or real(*a, **k))
    run_bake(BakeRequest(tags, 'level', 'room', settings, True, work, charts=tmp_path / 'c.npz'), lambda e: None)
    assert seen == [tmp_path / 'c.npz']


def test_chart_cache_path_is_one_file_per_level(tmp_path):
    first = runner.chart_cache_path(tmp_path, 'level', 'room')
    assert first == runner.chart_cache_path(tmp_path, 'level', 'room')
    assert first != runner.chart_cache_path(tmp_path, 'level', 'other')
    assert first.suffix == '.npz' and first.parent.name == 'charts'


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


def test_runner_can_disable_all_backups(tiny):
    tags, settings, work = tiny
    assert BakeRequest(tags, 'level', 'room', settings, False).backups is True
    out = run_bake(BakeRequest(tags, 'level', 'room', settings, False, work, backups=False), lambda e: None)
    assert out.status == 'done'
    assert not list(tags.rglob('backups/*'))
    runner.write_lit_scenario(tags, 'level', 'room', 'room_lit', 'level_lit', backups=False)
    runner.write_lit_scenario(tags, 'level', 'room', 'room_lit', 'level_lit', backups=False)
    assert not list(tags.rglob('backups/*'))


def _bake_twice_new_names(tiny, backups):
    tags, settings, work = tiny
    for i in range(2):
        out = run_bake(BakeRequest(tags, 'level', 'room', settings, True, work / str(i), backups=backups), lambda e: None)
        assert out.status == 'done'
    return sorted(p.name.split('_', 1)[1] for p in tags.rglob('backups/*'))


def test_runner_new_names_without_backups_keeps_none(tiny):
    assert _bake_twice_new_names(tiny, False) == []
    assert not list(tiny[0].rglob('backups'))


def test_runner_new_names_with_backups_covers_bsp_bitmap_and_lit_scenario(tiny):
    assert _bake_twice_new_names(tiny, True) == [
        'level_lit.scenario', 'room_lit.bitmap', 'room_lit.scenario_structure_bsp']


def test_runner_stages_end_with_validate(tiny):
    _, events = run(tiny)
    assert [e[1] for e in events if e[0] == 'stage'] == ['gather', 'charts', 'scene', 'blender', 'vertex', 'finish', 'write', 'validate']


def test_runner_forwards_log_lines_and_writes_a_log_file(tiny):
    out, events = run(tiny)
    assert any(e[0] == 'log' and 'stage: charts' in e[1] for e in events)
    assert 'stage: write' in (out.work / 'opentool.log').read_text()


def test_verbose_controls_info_forwarding_but_file_keeps_info(tiny, monkeypatch):
    tags, settings, work = tiny
    real_bake = pipeline.bake
    def with_info(*args, **kwargs):
        logging.getLogger('lightmapper.test').info('an info record')
        return real_bake(*args, **kwargs)
    monkeypatch.setattr(runner.pipeline, 'bake', with_info)
    quiet_events = []
    quiet = run_bake(BakeRequest(tags, 'level', 'room', settings, True, work / 'quiet', verbose=False), quiet_events.append)
    assert not any(event[0] == 'log' and 'an info record' in event[1] for event in quiet_events)
    assert 'an info record' in (quiet.work / 'opentool.log').read_text()
    loud_events = []
    run_bake(BakeRequest(tags, 'level', 'room', settings, True, work / 'loud', verbose=True), loud_events.append)
    assert any(event[0] == 'log' and 'an info record' in event[1] for event in loud_events)
    for events in (quiet_events, loud_events):
        assert sum(event[0] == 'log' and 'stage: charts' in event[1] for event in events) == 1


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


@pytest.mark.parametrize('verbose', [False, True])
def test_runner_follows_blender_log(tiny, monkeypatch, verbose):
    tags, settings, work = tiny
    real = pipeline.run_blender

    def write_log(path, s):
        (path / 'blender.log').write_text('[bake] device OPTIX\n[bake] page 0 (0, 0)\n')
        return real(path, s)

    class ImmediateFollower:
        def __init__(self, _path, callback):
            self.callback = callback
        def start(self):
            self.callback('[bake] device OPTIX')
            self.callback('[bake] page 0 (0, 0)')
        def stop(self):
            pass
        def join(self):
            pass
    monkeypatch.setattr(runner, 'BlenderLogFollower', ImmediateFollower)
    monkeypatch.setattr(pipeline, 'run_blender', write_log)
    events = []
    outcome = run_bake(BakeRequest(tags, 'level', 'room', settings, True, work, verbose=verbose), events.append)
    assert ('device', 'OPTIX') in events and ('pages', 1, 1) in events
    blender_lines = [event for event in events if event[0] == 'log' and '[bake]' in event[1]]
    assert bool(blender_lines) is verbose
    assert 'INFO: [bake] device OPTIX' in (outcome.work / 'opentool.log').read_text()


def test_runner_removes_its_log_handlers(tiny):
    before = list(logging.getLogger().handlers); run(tiny); assert logging.getLogger().handlers == before


def test_outcome_carries_devices_rays_and_warnings_from_the_summary(tiny):
    out, _ = run(tiny)
    summary = json.loads((out.work / 'summary.json').read_text())
    assert out.devices == summary['devices'] and out.devices
    assert out.rays == summary['rays'] and out.rays
    assert out.warnings == summary['warnings']


def test_failed_bake_has_no_devices(tiny, monkeypatch):
    monkeypatch.setattr(runner.pipeline, 'bake', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('boom')))
    out, _ = run(tiny)
    assert out.status == 'failed' and out.devices is None and out.rays is None and out.warnings == []


def test_unreadable_summary_leaves_devices_empty(tiny, monkeypatch):
    real = runner.pipeline.bake

    def bake_then_break(*a, **k):
        result = real(*a, **k)
        (k['work'] / 'summary.json').write_text('{not json')
        return result

    monkeypatch.setattr(runner.pipeline, 'bake', bake_then_break)
    out, _ = run(tiny)
    assert out.status == 'done' and out.devices is None and out.warnings == []


def test_summary_without_all_three_devices_leaves_devices_empty(tiny, monkeypatch):
    real = runner.pipeline.bake

    def bake_then_cut(*a, **k):
        result = real(*a, **k)
        path = k['work'] / 'summary.json'
        summary = json.loads(path.read_text())
        del summary['devices']['denoise']
        path.write_text(json.dumps(summary))
        return result

    monkeypatch.setattr(runner.pipeline, 'bake', bake_then_cut)
    out, _ = run(tiny)
    assert out.status == 'done' and out.devices is None and out.warnings == []


def test_lamp_batch_stages_reach_the_status_line_not_the_stage_list(tiny, monkeypatch):
    from test_pipeline import batch_runs, many_lamps
    many_lamps(monkeypatch, 70)
    batch_runs(monkeypatch)
    _, events = run(tiny)
    assert ('status', 'Baking lamp batch 1 of 1') in events
    assert [e[1] for e in events if e[0] == 'stage'] == ['gather', 'charts', 'scene', 'blender', 'vertex', 'finish', 'write', 'validate']
