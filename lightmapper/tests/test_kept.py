import json
from dataclasses import asdict, fields

import numpy as np
import pytest

from lightmapper import kept
from lightmapper.blender_job import BakeResult
from lightmapper.errors import KeptBakeError
from lightmapper.settings import Settings


def arr(seed, shape, dtype=np.float32):
    return (np.random.default_rng(seed).random(shape) * 100).astype(dtype)


def full_result():
    return BakeResult(
        pages=[arr(1, (4, 4, 3)), arr(2, (4, 4, 3))],
        vertex_vector=arr(3, (5, 3)), vertex_irradiance=arr(4, (5, 3), np.float64),
        device='cuda', written=[arr(5, (2,), np.int32), arr(6, (3,), np.int32)],
        vertex_sun_lit=arr(7, (5, 2)) > 50, edge_sun_lit=arr(8, (5, 2)) > 50,
        diffuse_pages=[arr(9, (4, 4, 3))], vertex_rays_cast=1234,
        stages={'bake': 1.5, 'denoise': {'a': 2}}, samples_used=256, texel_samples_used=64,
        vertex_sun_throughput=arr(10, (5, 2)), texel_rgb=arr(11, (4, 3)), light_flat=arr(12, (6,)))


def same(a, b):
    assert a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b)


def test_result_round_trips(tmp_path):
    result = full_result()
    kept.kept_dir(tmp_path).mkdir(parents=True)
    kept.save_result(tmp_path, result)
    loaded = kept.load_result(tmp_path)
    for field in fields(BakeResult):
        want, got = getattr(result, field.name), getattr(loaded, field.name)
        if field.name == 'diffuse_pages':
            assert got is None
        elif field.name in ('pages', 'written'):
            assert len(want) == len(got)
            for w, g in zip(want, got):
                same(w, g)
        elif isinstance(want, np.ndarray):
            same(want, got)
        else:
            assert want == got, field.name


def test_result_round_trips_with_none_fields(tmp_path):
    result = BakeResult(pages=[arr(1, (2, 2, 3))], vertex_vector=None, vertex_irradiance=None, device='cpu', written=[])
    kept.kept_dir(tmp_path).mkdir(parents=True)
    kept.save_result(tmp_path, result)
    loaded = kept.load_result(tmp_path)
    same(result.pages[0], loaded.pages[0])
    assert loaded.written == [] and loaded.device == 'cpu'
    for field in fields(BakeResult):
        if field.name not in ('pages', 'written', 'device'):
            assert getattr(loaded, field.name) is None, field.name


def test_glow_round_trips(tmp_path):
    pages = [arr(1, (3, 3, 3)), arr(2, (3, 3, 3), np.float64)]
    kept.kept_dir(tmp_path).mkdir(parents=True)
    kept.save_glow(tmp_path, pages)
    loaded = kept.load_glow(tmp_path)
    assert len(loaded) == 2
    for w, g in zip(pages, loaded):
        same(w, g)


def test_level_folder_is_one_per_level(tmp_path):
    a = kept.level_folder(tmp_path, tmp_path / 'tags', 'sc', 'bsp')
    assert a == kept.level_folder(tmp_path, tmp_path / 'tags', 'sc', 'bsp')
    assert a.parent == tmp_path and a.name.startswith('bsp-')
    assert a != kept.level_folder(tmp_path, tmp_path / 'tags', 'sc', 'other')
    assert a != kept.level_folder(tmp_path, tmp_path / 'tags2', 'sc', 'bsp')


def make_kept(work, settings=None):
    kept.kept_dir(work).mkdir(parents=True)
    for name in ('source.scenario_structure_bsp', 'scenario.scenario', 'charts.npz', 'result.npz', 'scene.npz', 'job.json'):
        (kept.kept_dir(work) / name).write_bytes(b'x')
    kept.write_manifest(work, kept.Manifest(
        tags='t', scenario='s', bsp='b', input_path='i', has_template=False, vertex_device='cpu',
        glow_device='cpu', glow_reason='', settings=asdict(settings or Settings()), kept_at='now'))


def test_manifest_round_trips(tmp_path):
    make_kept(tmp_path)
    assert kept.read_manifest(tmp_path).bsp == 'b'
    assert json.loads((tmp_path / kept.MANIFEST).read_text())['version'] == kept.VERSION


@pytest.mark.parametrize('name', [kept.SCENE, kept.JOB])
def test_a_kept_bake_without_its_scene_or_job_is_refused(tmp_path, name):
    make_kept(tmp_path)
    (kept.kept_dir(tmp_path) / name).unlink()
    with pytest.raises(KeptBakeError):
        kept.read_manifest(tmp_path)


def test_latest_is_the_newest_readable_kept_bake(tmp_path):
    assert kept.latest(tmp_path / 'nope', tmp_path, 's', 'b') is None
    level = kept.level_folder(tmp_path, tmp_path, 's', 'b')
    assert kept.latest(tmp_path, tmp_path, 's', 'b') is None
    make_kept(level / '20260101-000000')
    make_kept(level / '20260102-000000')
    (level / '20260103-000000').mkdir()
    make_kept(level / '20260104-000000')
    (level / '20260104-000000' / kept.MANIFEST).write_text('[]')
    assert kept.latest(tmp_path, tmp_path, 's', 'b') == level / '20260102-000000'


def test_new_work_is_not_created(tmp_path):
    work = kept.new_work(tmp_path, tmp_path, 's', 'b')
    assert work.parent == kept.level_folder(tmp_path, tmp_path, 's', 'b') and not work.exists()


def test_prune_keeps_only_this_bake(tmp_path):
    level = tmp_path / 'level'
    for name in ('a', 'b', 'c'):
        (level / name).mkdir(parents=True)
        (level / name / 'f').write_text(name)
    kept.prune(level / 'b')
    assert [p.name for p in level.iterdir()] == ['b']


@pytest.mark.parametrize('damage', ['missing', 'garbled', 'version', 'result', 'list', 'null'])
def test_read_manifest_refuses(tmp_path, damage):
    make_kept(tmp_path)
    manifest = tmp_path / kept.MANIFEST
    if damage == 'missing':
        manifest.unlink()
    elif damage == 'garbled':
        manifest.write_text('{not json')
    elif damage in ('list', 'null'):
        manifest.write_text({'list': '[]', 'null': 'null'}[damage])
    elif damage == 'version':
        data = json.loads(manifest.read_text())
        data['version'] = 2
        manifest.write_text(json.dumps(data))
    else:
        (kept.kept_dir(tmp_path) / 'result.npz').unlink()
    with pytest.raises(KeptBakeError, match='is not a kept bake OpenTool can write again'):
        kept.read_manifest(tmp_path)


def test_setting_differences():
    base = Settings()
    kept_dict = json.loads(json.dumps(asdict(base)))
    assert kept.setting_differences(kept_dict, Settings(lightmap_32bit=True, blender='other', timeout_s=5)) == []
    assert kept.setting_differences(kept_dict, Settings(device='cpu' if base.device != 'cpu' else 'auto')) == []
    out = kept.setting_differences(kept_dict, Settings(samples=base.samples + 1, denoise=not base.denoise))
    assert sorted(out) == sorted([
        f'samples: {base.samples!r} when kept, {base.samples + 1!r} now',
        f'denoise: {base.denoise!r} when kept, {not base.denoise!r} now'])
    del kept_dict['samples']
    assert kept.setting_differences(kept_dict, base) == ['samples: not in the kept bake']


def test_drop_superseded_leaves_logs_and_kept_files(tmp_path):
    make_kept(tmp_path)
    for name in (*kept.SUPERSEDED, 'lamps_0_out.npz', 'lamps_1_out.npz', 'blender.log', 'job.json'):
        (tmp_path / name).write_bytes(b'x')
    kept.drop_superseded(tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ['blender.log', 'job.json', 'kept', kept.MANIFEST]
    assert (kept.kept_dir(tmp_path) / 'result.npz').exists()


def test_save_sources_copies_bytes(tmp_path):
    src = tmp_path / 'src'
    src.mkdir()
    for name in ('i', 't', 's', 'c'):
        (src / name).write_bytes(name.encode())
    work = tmp_path / 'work'
    kept.kept_dir(work).mkdir(parents=True)
    kept.save_sources(work, src / 'i', src / 't', src / 's', src / 'c')
    assert (kept.kept_dir(work) / 'source.scenario_structure_bsp').read_bytes() == b'i'
    assert (kept.kept_dir(work) / 'template.bitmap').read_bytes() == b't'
    assert (kept.kept_dir(work) / 'scenario.scenario').read_bytes() == b's'
    assert (kept.kept_dir(work) / 'charts.npz').read_bytes() == b'c'
    work2 = tmp_path / 'work2'
    kept.kept_dir(work2).mkdir(parents=True)
    kept.save_sources(work2, src / 'i', None, src / 's', src / 'c')
    assert not (kept.kept_dir(work2) / 'template.bitmap').exists()


def damaged_result(tmp_path, drop=None, raw=None):
    kept.kept_dir(tmp_path).mkdir(parents=True)
    kept.save_result(tmp_path, full_result())
    path = kept.kept_dir(tmp_path) / kept.RESULT
    if raw is not None:
        path.write_bytes(raw)
    elif drop:
        with np.load(path) as data:
            keep = {k: data[k] for k in data.files if k not in drop}
        np.savez(path, **keep)


@pytest.mark.parametrize('drop, raw', [
    (('pages_count',), None), (('device',), None), (('written_count',), None), (('pages_1',), None),
    (None, b'not a zip file'), (None, b'PKtruncated')])
def test_load_result_refuses_a_damaged_file(tmp_path, drop, raw):
    damaged_result(tmp_path, drop, raw)
    with pytest.raises(KeptBakeError, match='is not a kept bake OpenTool can write again'):
        kept.load_result(tmp_path)


def test_load_result_keeps_optional_fields_absent(tmp_path):
    damaged_result(tmp_path, ('vertex_vector', 'texel_rgb'))
    loaded = kept.load_result(tmp_path)
    assert loaded.vertex_vector is None and loaded.texel_rgb is None

def test_create_work_numbers_a_clash_inside_one_second(tmp_path, monkeypatch):
    import datetime as dt

    class Frozen(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 1, 2, 3, 4, 5)

    monkeypatch.setattr(kept, 'datetime', Frozen)
    works = [kept.create_work(tmp_path, tmp_path, 's', 'b') for _ in range(3)]
    assert [w.name for w in works] == ['20260102-030405', '20260102-030405-2', '20260102-030405-3']
    assert all(w.is_dir() and w.parent == works[0].parent for w in works)


def test_prune_logs_a_folder_it_cannot_remove(tmp_path, monkeypatch, caplog):
    level = tmp_path / 'level'
    for name in ('a', 'b', 'c'):
        (level / name).mkdir(parents=True)
    real = kept.shutil.rmtree

    def stuck(path, **handlers):
        if path.name == 'a':
            (handlers.get('onexc') or handlers['onerror'])(None, str(path / 'f'), PermissionError())
        else:
            real(path)

    monkeypatch.setattr(kept.shutil, 'rmtree', stuck)
    with caplog.at_level('WARNING', logger='lightmapper.kept'):
        kept.prune(level / 'b')
    assert sorted(p.name for p in level.iterdir()) == ['a', 'b']
    assert len(caplog.messages) == 1 and str(level / 'a') in caplog.messages[0]
