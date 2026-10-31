from dataclasses import replace

import pytest

from lightmapper import cpu_vertex_job as vertex_cpu, kept
from lightmapper.errors import KeptBakeError
from test_pipeline import api, run_enhanced_pipeline, tiny  # noqa: F401


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
