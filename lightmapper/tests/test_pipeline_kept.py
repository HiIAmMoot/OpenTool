import json
from dataclasses import replace

import pytest

from lightmapper import cpu_vertex_job as vertex_cpu, kept
from lightmapper.errors import KeptBakeError
from lightmapper.tagio.tagfile import read_tag, resolve_reference_path, write_tag
from test_pipeline import api, put, run_enhanced_pipeline, tiny  # noqa: F401


def kept_work(tmp_path, tags, name='a'):
    return kept.level_folder(tmp_path / 'k', tags, 'level', 'room') / name


def tool(settings):
    return replace(settings, ignore_normals_mode='tool')


def test_kept_bake_holds_what_a_write_needs(tiny, tmp_path):
    tags, settings, _ = tiny
    work = kept_work(tmp_path, tags)
    before = (tags / 'room.scenario_structure_bsp').read_bytes()
    api().bake(tags, 'level', 'room', tool(settings), out_bsp='copy', work=work, keep=True)
    folder = kept.kept_dir(work)
    assert sorted(p.name for p in folder.iterdir()) == sorted([kept.SOURCE, kept.SCENARIO, kept.CHARTS, kept.RESULT])
    manifest = kept.read_manifest(work)
    assert manifest.settings['ignore_normals_mode'] == 'tool' and manifest.glow_device == 'off'
    assert manifest.glow_reason == 'Traditional ignore normals' and manifest.vertex_device == 'cpu'
    assert (folder / kept.SOURCE).read_bytes() == before
    assert kept.load_result(work).device == 'test CPU'


def test_kept_bake_survives_a_failed_write(tiny, tmp_path, monkeypatch):
    tags, settings, _ = tiny
    pipeline = api()
    work = kept_work(tmp_path, tags)

    def refuse(tags, group, path, tag, backups=True):
        if group == 'bitm':
            raise PermissionError('the bitmap is open elsewhere')

    monkeypatch.setattr(pipeline, '_write', refuse)
    with pytest.raises(PermissionError):
        pipeline.bake(tags, 'level', 'room', tool(settings), out_bsp='copy', work=work, keep=True)
    assert kept.read_manifest(work).tags == str(tags)


def test_kept_enhanced_bake_keeps_the_glow(tiny, tmp_path, monkeypatch):
    tags = tiny[0]
    work = kept_work(tmp_path, tags)
    run_enhanced_pipeline(tiny, monkeypatch, 'enhanced',
                          bake_kwargs={'work': work, 'dry_run': False, 'out_bsp': 'copy', 'keep': True})
    manifest = kept.read_manifest(work)
    assert manifest.glow_device == 'OPTIX' and manifest.glow_reason == ''
    assert len(kept.load_glow(work)) == 1


def test_a_new_kept_bake_replaces_the_last_one(tiny, tmp_path):
    tags, settings, _ = tiny
    first, second = kept_work(tmp_path, tags, 'a'), kept_work(tmp_path, tags, 'b')
    for work in (first, second):
        api().bake(tags, 'level', 'room', tool(settings), out_bsp='copy', work=work, keep=True)
    assert [p.name for p in second.parent.iterdir()] == ['b']
    assert kept.latest(tmp_path / 'k', tags, 'level', 'room') == second


def test_a_kept_bake_in_a_folder_of_the_users_own_prunes_nothing(tiny, tmp_path):
    tags, settings, _ = tiny
    neighbour = tmp_path / 'mine' / 'other'
    neighbour.mkdir(parents=True)
    work = tmp_path / 'mine' / 'bake'
    api().bake(tags, 'level', 'room', tool(settings), out_bsp='copy', work=work, keep=True)
    assert neighbour.is_dir() and kept.read_manifest(work)


@pytest.mark.parametrize('kwargs', [{'dry_run': True}, {'sources': 'sun'}, {'sources': 'emit'}])
def test_keep_refuses_dry_runs_and_partial_sources(tiny, tmp_path, kwargs):
    tags, settings, _ = tiny
    work = kept_work(tmp_path, tags)
    with pytest.raises(ValueError, match='a kept bake must be a full bake that writes tags'):
        api().bake(tags, 'level', 'room', settings, work=work, keep=True, **kwargs)
    assert not work.exists()


def test_kept_bake_drops_the_superseded_outputs(tiny, tmp_path):
    tags, settings, _ = tiny
    work = kept_work(tmp_path, tags)
    work.mkdir(parents=True)
    (work / 'out.npz').write_bytes(b'x')
    (work / 'lamps_1_out.npz').write_bytes(b'x')
    (work / 'blender.log').write_text('log')
    api().bake(tags, 'level', 'room', tool(settings), out_bsp='copy', work=work, keep=True)
    assert not (work / 'out.npz').exists() and not (work / 'lamps_1_out.npz').exists()
    assert (work / 'blender.log').read_text() == 'log'


def test_kept_source_is_the_bsp_from_before_an_in_place_bake(tiny, tmp_path):
    tags, settings, _ = tiny
    work = kept_work(tmp_path, tags)
    before = (tags / 'room.scenario_structure_bsp').read_bytes()
    api().bake(tags, 'level', 'room', tool(settings), work=work, keep=True)
    assert (tags / 'room.scenario_structure_bsp').read_bytes() != before
    assert (kept.kept_dir(work) / kept.SOURCE).read_bytes() == before


def test_a_failed_rebake_into_a_kept_folder_leaves_no_old_bake(tiny, tmp_path, monkeypatch):
    tags, settings, _ = tiny
    work = kept_work(tmp_path, tags)
    api().bake(tags, 'level', 'room', tool(settings), out_bsp='copy', work=work, keep=True)
    kept.read_manifest(work)

    def broken(work, pages, timeout):
        raise RuntimeError('out of memory')

    monkeypatch.setattr(vertex_cpu, 'run_in_child', broken)
    with pytest.raises(RuntimeError):
        api().bake(tags, 'level', 'room', tool(settings), out_bsp='copy', work=work, keep=True)
    with pytest.raises(KeptBakeError):
        kept.read_manifest(work)


def test_a_rebake_drops_the_glow_of_the_earlier_kept_bake(tiny, tmp_path, monkeypatch):
    tags = tiny[0]
    work = kept_work(tmp_path, tags)
    both = {'work': work, 'dry_run': False, 'out_bsp': 'copy', 'keep': True}
    run_enhanced_pipeline(tiny, monkeypatch, 'enhanced', bake_kwargs=both)
    assert (kept.kept_dir(work) / kept.GLOW).is_file()
    run_enhanced_pipeline(tiny, monkeypatch, 'tool', bake_kwargs=both)
    assert not (kept.kept_dir(work) / kept.GLOW).exists()
    assert kept.read_manifest(work).glow_device == 'off'


def test_keep_needs_a_work_folder(tiny):
    tags, settings, _ = tiny
    with pytest.raises(ValueError, match='a kept bake needs a work folder'):
        api().bake(tags, 'level', 'room', settings, out_bsp='copy', keep=True)


def fail(*args, **kwargs):
    pytest.fail('a write again must not bake')


def forbid_baking(monkeypatch):
    from lightmapper import vertex_gpu
    pipeline = api()
    for name in ('run_blender', 'run_blender_glow', 'run_blender_lamp_batch', 'run_blender_vertex_only'):
        monkeypatch.setattr(pipeline, name, fail)
    monkeypatch.setattr(vertex_cpu, 'run_in_child', fail)
    monkeypatch.setattr(vertex_gpu, 'run_in_child', fail)


def written(tags, name='copy'):
    return [resolve_reference_path(tags, group, name) for group in ('bitm', 'sbsp')]


def tag_bytes(tag, folder):
    write_tag(tag, folder / 'bytes', backup=False)
    return (folder / 'bytes').read_bytes()


def kept_tool_bake(tiny, tmp_path):
    tags, settings, _ = tiny
    work = kept_work(tmp_path, tags)
    settings = tool(settings)
    api().bake(tags, 'level', 'room', settings, out_bsp='copy', work=work, keep=True)
    return tags, settings, work


@pytest.mark.parametrize('mode', ['tool', 'enhanced'])
def test_write_again_writes_what_the_bake_wrote(tiny, tmp_path, monkeypatch, mode):
    tags, settings, _ = tiny
    work = kept_work(tmp_path, tags)
    run_enhanced_pipeline(tiny, monkeypatch, mode,
                          bake_kwargs={'work': work, 'dry_run': False, 'out_bsp': 'copy', 'keep': True})
    first = [path.read_bytes() for path in written(tags)]
    for path in written(tags):
        path.unlink()
    forbid_baking(monkeypatch)
    api().bake(tags, 'level', 'room', replace(settings, ignore_normals_mode=mode, stitch_seams=False, bounces=1),
               out_bsp='copy', work=work, reuse=True)
    assert [path.read_bytes() for path in written(tags)] == first


def test_write_again_ignores_the_tags_as_they_are_now(tiny, tmp_path, monkeypatch):
    tags, settings, work = kept_tool_bake(tiny, tmp_path)
    first = read_tag(resolve_reference_path(tags, 'sbsp', 'copy'))
    lit = read_tag(resolve_reference_path(tags, 'sbsp', 'copy'))
    first.root.set_reference('lightmaps_bitmap', 'bitm', 'copy2')
    put(tags, 'sbsp', 'room', lit)
    scenario = read_tag(resolve_reference_path(tags, 'scnr', 'level'))
    scenario.root.block('structure_bsps')[0].set_reference('structure_bsp', 'sbsp', 'elsewhere')
    put(tags, 'scnr', 'level', scenario)
    forbid_baking(monkeypatch)
    api().bake(tags, 'level', 'room', settings, out_bsp='copy2', work=work, reuse=True)
    again = read_tag(resolve_reference_path(tags, 'sbsp', 'copy2'))
    assert again.root.reference('lightmaps_bitmap') == ('bitm', 'copy2')
    assert tag_bytes(again, tmp_path) == tag_bytes(first, tmp_path)


def test_write_again_switches_32bit(tiny, tmp_path, monkeypatch):
    tags, settings, work = kept_tool_bake(tiny, tmp_path)
    forbid_baking(monkeypatch)
    api().bake(tags, 'level', 'room', replace(settings, lightmap_32bit=True), out_bsp='copy', work=work, reuse=True)
    bitmap = read_tag(resolve_reference_path(tags, 'bitm', 'copy'))
    assert [b.get('format') for b in bitmap.root.block('bitmaps')] == [10]


def test_write_again_refuses_other_settings(tiny, tmp_path, monkeypatch):
    tags, settings, work = kept_tool_bake(tiny, tmp_path)
    for path in written(tags):
        path.unlink()
    forbid_baking(monkeypatch)
    with pytest.raises(KeptBakeError, match='samples') as refused:
        api().bake(tags, 'level', 'room', replace(settings, samples=settings.samples + 1),
                   out_bsp='copy', work=work, reuse=True)
    assert "can't be written again with these" in str(refused.value)
    assert not any(path.exists() for path in written(tags))


def test_write_again_refuses_another_level(tiny, tmp_path, monkeypatch):
    tags, settings, work = kept_tool_bake(tiny, tmp_path)
    forbid_baking(monkeypatch)
    with pytest.raises(KeptBakeError, match='another level'):
        api().bake(tags, 'level', 'other', settings, out_bsp='copy', work=work, reuse=True)


def test_write_again_refuses_changed_shaders(tiny, tmp_path, monkeypatch):
    tags, settings, work = kept_tool_bake(tiny, tmp_path)
    for path in written(tags):
        path.unlink()
    shader = read_tag(tags / 'wall.shader_environment')
    shader.root.set('radiosity.flags', shader.root.get('radiosity.flags') ^ 8)
    put(tags, 'senv', 'wall', shader)
    forbid_baking(monkeypatch)
    with pytest.raises(KeptBakeError, match='shaders changed since this bake was kept'):
        api().bake(tags, 'level', 'room', settings, out_bsp='copy', work=work, reuse=True)
    assert not any(path.exists() for path in written(tags))


def test_write_again_refuses_a_folder_that_is_not_kept(tiny, tmp_path, monkeypatch):
    tags, settings, _ = tiny
    forbid_baking(monkeypatch)
    folder = tmp_path / 'empty'
    folder.mkdir()
    with pytest.raises(KeptBakeError, match='not a kept bake'):
        api().bake(tags, 'level', 'room', settings, out_bsp='copy', work=folder, reuse=True)


def test_write_again_refuses_a_damaged_result_before_writing(tiny, tmp_path, monkeypatch):
    tags, settings, work = kept_tool_bake(tiny, tmp_path)
    for path in written(tags):
        path.unlink()
    (kept.kept_dir(work) / kept.RESULT).write_bytes(b'garbage')
    assert kept.latest(tmp_path / 'k', tags, 'level', 'room') == work
    forbid_baking(monkeypatch)
    with pytest.raises(KeptBakeError):
        api().bake(tags, 'level', 'room', settings, out_bsp='copy', work=work, reuse=True)
    assert not any(path.exists() for path in written(tags))


def test_write_again_keeps_the_bake_summary(tiny, tmp_path, monkeypatch):
    tags, settings, work = kept_tool_bake(tiny, tmp_path)
    before = (work / 'summary.json').read_bytes()
    forbid_baking(monkeypatch)
    api().bake(tags, 'level', 'room', settings, out_bsp='copy', work=work, reuse=True)
    assert (work / 'summary.json').read_bytes() == before
    summary = json.loads((work / 'write_summary.json').read_text())
    assert summary['devices']['vertex_pass'] == 'cpu' and summary['devices']['glow_pass'] == 'off'


def test_write_again_reports_its_progress(tiny, tmp_path, monkeypatch):
    tags, settings, work = kept_tool_bake(tiny, tmp_path)
    forbid_baking(monkeypatch)
    stages = []
    api().bake(tags, 'level', 'room', settings, out_bsp='copy', work=work, reuse=True, progress=stages.append)
    assert stages == ['gather', 'charts', 'scene', 'finish', 'write']


def test_keep_and_reuse_exclude_each_other(tiny, tmp_path):
    tags, settings, work = kept_tool_bake(tiny, tmp_path)
    with pytest.raises(ValueError, match='keep'):
        api().bake(tags, 'level', 'room', settings, work=work, keep=True, reuse=True)


def test_a_kept_folder_that_cannot_be_cleared_is_named(tiny, tmp_path, monkeypatch):
    tags, settings, work = kept_tool_bake(tiny, tmp_path)
    monkeypatch.setattr(api().shutil, 'rmtree', lambda *args, **kwargs: None)
    with pytest.raises(KeptBakeError, match='could not be cleared') as refused:
        api().bake(tags, 'level', 'room', settings, out_bsp='copy', work=work, keep=True)
    assert str(kept.kept_dir(work)) in str(refused.value)


def test_a_kept_bake_may_reuse_the_charts_it_kept(tiny, tmp_path):
    tags, settings, work = kept_tool_bake(tiny, tmp_path)
    api().bake(tags, 'level', 'room', settings, out_bsp='copy', work=work, keep=True,
               charts=kept.kept_dir(work) / kept.CHARTS)
    assert (kept.kept_dir(work) / kept.CHARTS).is_file()
    assert kept.read_manifest(work).bsp == 'room'
