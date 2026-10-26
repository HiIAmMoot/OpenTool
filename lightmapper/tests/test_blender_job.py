import json
import os
import stat
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

from lightmapper.blender_job import BakeResult, BlenderError, Scene, run_blender, write_job
from lightmapper.gather import Sun
from lightmapper.settings import Settings

FAKE = Path(__file__).with_name('fake_blender.py')


def page(i, w, h):
    # the stub's page, spelled out per texel so the test does not share its code
    out = np.zeros((h, w, 3), np.float32)
    for y in range(h):
        for x in range(w):
            for c in range(3):
                out[y, x, c] = i * 1000 + y * 100 + x + c * 0.25
    return out


def make_scene(emission=2.0):
    return Scene(
        positions=np.float32([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
        normals=np.float32([[0, 0, 1]] * 3),
        lm_uv=np.float32([[0, 0], [1, 0], [0, 1]]),
        tex_uv=np.float32([[0, 0], [1, 0], [0, 1]]),
        tris=np.int32([[0, 1, 2]]),
        tri_page=np.int16([0]),
        tri_class=np.int16([1]),
        tri_casts_shadow=np.bool_([True]),
        tri_emission=np.float32([[emission, emission, emission]]),
        tri_texture=np.int32([0]),
        tri_cutout=np.bool_([False]),
        tri_tint=np.float32([[0.25, 0.5, 1.0]]),
        occ_positions=np.float32([[0, 0, 1], [1, 0, 1], [0, 1, 1]]),
        occ_uv=np.float32([[0, 0], [1, 0], [0, 1]]),
        occ_tris=np.int32([[0, 1, 2]]),
        occ_tri_texture=np.int32([-1]),
        occ_tri_cutout=np.bool_([False]),
        occ_tri_casts_shadow=np.bool_([True]),
        occ_tri_emission=np.float32([[1.0, 1.0, 1.0]]),
        occ_tri_tint=np.float32([[1.0, 1.0, 1.0]]),
        query_pos=np.float32([[0.2, 0.3, 0.5], [1, 2, 3], [-4, 5, 0.25]]),
        query_normal=np.float32([[0, 0, 1], [0, 1, 0], [1, 0, 0]]),
        query_class=np.int16([1, 2, -1]),
    )


def make_suns():
    return [
        Sun(np.float64([0, 0, -1]), 0.1, np.float64([1, 1, 0.5]), 3.0, (1, 2)),
        Sun(np.float64([1, 0, 0]), 0.2, np.float64([1, 1, 1]), 5.0, ()),
    ]


AMBIENT = {-1: np.float64([0.1, 0.2, 0.3]), 1: np.float64([0.4, 0.4, 0.4])}
TEXTURE = np.arange(2 * 3 * 4, dtype=np.uint8).reshape(2, 3, 4)


def job_for(tmp_path, sources='all', **overrides):
    s = Settings(w_sun=2.0, w_emit=3.0, w_amb=0.5, **overrides)
    path = write_job(tmp_path, make_scene(), [(16, 8)], make_suns(), AMBIENT, s, sources,
                     textures=[TEXTURE])
    return json.loads(path.read_text()), s


def test_write_job_files_load_back(tmp_path):
    job, s = job_for(tmp_path, samples=12, device='CUDA')
    assert job['version'] == 1 and job['scene'] == 'scene.npz' and job['out'] == 'out.npz'
    assert job['pages'] == [[16, 8]]
    assert (job['samples'], job['bounces'], job['vertex_rays'], job['device']) == (12, 1, 4096, 'nvidia')
    assert (job['albedo_scale'], job['albedo_fallback']) == (1.0, 0.5)
    scene = make_scene()
    with np.load(tmp_path / 'scene.npz') as data:
        expected_fields = [f.name for f in fields(Scene) if getattr(scene, f.name) is not None]
        assert sorted(data.files) == sorted(expected_fields + [
            'occ_tri_emit_only', 'edge_pos', 'edge_normal', 'edge_class', 'edge_active', 'edge_keys', 'edge_offsets'])
        for f in fields(Scene):
            expected = getattr(scene, f.name)
            if expected is None:
                continue
            if f.name in ('tri_emission', 'occ_tri_emission'):
                expected = expected * s.w_emit
            np.testing.assert_array_equal(data[f.name], expected, err_msg=f.name)
            assert data[f.name].dtype == getattr(scene, f.name).dtype, f.name
        assert data['edge_pos'].shape == data['edge_normal'].shape == (0, 3)
        assert data['edge_pos'].dtype == data['edge_normal'].dtype == np.float32
        assert data['edge_class'].shape == data['edge_keys'].shape == (0,)
        assert data['edge_active'].shape == (0, 1)
        assert data['edge_active'].dtype == np.bool_
        np.testing.assert_array_equal(data['edge_offsets'], [0])
    assert sorted(job) == sorted([
        'version', 'scene', 'out', 'pages', 'suns', 'ambient', 'samples', 'bounces',
        'albedo_scale', 'albedo_fallback', 'vertex_rays', 'device', 'texel_samples', 'seed',
        'cycles_devices', 'cycles_required', 'hiprt', 'vertex_pass'])
    with np.load(tmp_path / 'textures.npz') as data:
        assert list(data.files) == ['tex_0']
        np.testing.assert_array_equal(data['tex_0'], TEXTURE)
        assert data['tex_0'].dtype == np.uint8


def test_query_inverse_survives_job_exchange(tmp_path):
    scene = make_scene()
    scene.query_inverse = np.int32([2, 0, 1])
    write_job(tmp_path, scene, [(16, 8)], [], {}, Settings())
    with np.load(tmp_path / 'scene.npz') as data:
        np.testing.assert_array_equal(data['query_inverse'], [2, 0, 1])


def test_weights_applied_for_all(tmp_path):
    job, s = job_for(tmp_path)
    assert len(job['suns']) == 1
    sun = job['suns'][0]
    assert sun['strength'] == pytest.approx(6.0)
    assert sun['classes'] == [1, 2] and sun['angle'] == pytest.approx(0.1)
    assert sun['direction'] == [0, 0, -1] and sun['color'] == [1, 1, 0.5]
    assert job['ambient']['-1'] == pytest.approx([0.05, 0.1, 0.15])
    assert job['ambient']['1'] == pytest.approx([0.2, 0.2, 0.2])
    with np.load(tmp_path / 'scene.npz') as data:
        np.testing.assert_allclose(data['tri_emission'], 6.0)
        np.testing.assert_allclose(data['occ_tri_emission'], 3.0)


def test_sun_only_drops_emission_and_ambient(tmp_path):
    job, _ = job_for(tmp_path, sources='sun')
    assert len(job['suns']) == 1 and job['ambient'] == {}
    with np.load(tmp_path / 'scene.npz') as data:
        assert not data['tri_emission'].any()
        assert not data['occ_tri_emission'].any()


@pytest.mark.parametrize('sources, weight', [('sun', 1.), ('all', 0.)])
def test_emitter_only_role_survives_source_weighting(tmp_path, sources, weight):
    scene = make_scene()
    scene.occ_tri_casts_shadow[:] = False
    write_job(tmp_path, scene, [(16, 8)], [], {}, Settings(w_emit=weight), sources=sources)
    with np.load(tmp_path / 'scene.npz') as data:
        assert 'occ_tri_emit_only' in data.files
        np.testing.assert_array_equal(data['occ_tri_emit_only'], [True])
        np.testing.assert_array_equal(data['occ_tri_emission'], 0)


def test_shadow_casting_tinted_emitter_is_not_an_emitter_pane(tmp_path):
    scene = make_scene()
    scene.occ_tri_casts_shadow[:] = True
    scene.occ_tri_tint[:] = 0.5
    write_job(tmp_path, scene, [(16, 8)], [], {}, Settings())
    with np.load(tmp_path / 'scene.npz') as data:
        assert data['occ_tri_emission'].any()
        np.testing.assert_array_equal(data['occ_tri_emit_only'], [False])


def test_emit_only_drops_suns_and_ambient(tmp_path):
    job, _ = job_for(tmp_path, sources='emit')
    assert job['suns'] == [] and job['ambient'] == {}
    with np.load(tmp_path / 'scene.npz') as data:
        np.testing.assert_allclose(data['tri_emission'], 6.0)


def test_bad_sources_rejected(tmp_path):
    with pytest.raises(ValueError):
        job_for(tmp_path, sources='moon')


@pytest.fixture
def fake(tmp_path):
    # a launcher the command can run directly, as the real blender.exe is
    if os.name == 'nt':
        launcher = tmp_path / 'fake_blender.cmd'
        launcher.write_text(f'@"{sys.executable}" "{FAKE}" %*\r\n')
    else:
        launcher = tmp_path / 'fake_blender.sh'
        launcher.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{FAKE}" "$@"\n')
        launcher.chmod(launcher.stat().st_mode | stat.S_IEXEC)
    return str(launcher)


def test_run_blender_returns_arrays(tmp_path, fake):
    work = tmp_path / 'work'
    s = Settings(blender=fake)
    write_job(work, make_scene(), [(16, 8), (4, 4)], make_suns(), AMBIENT, s)
    result = run_blender(work, s)
    assert isinstance(result, BakeResult)
    assert [p.shape for p in result.pages] == [(8, 16, 3), (4, 4, 3)]
    for i, (w, h) in enumerate([(16, 8), (4, 4)]):
        np.testing.assert_array_equal(result.pages[i], page(i, w, h))
    assert result.pages[0][1, 0, 0] == 100 and result.pages[0][0, 1, 0] == 1
    qp = make_scene().query_pos
    np.testing.assert_array_equal(result.vertex_vector, qp[:, [1, 2, 0]] * 2)
    np.testing.assert_allclose(result.vertex_irradiance, [2.0, 8.0, 0.25])
    assert result.vertex_vector.dtype == np.float32 and result.vertex_irradiance.dtype == np.float32
    assert result.device == 'OPTIX+CUDA+HIP-FAKE'
    assert hasattr(result, 'written'), 'written masks were not returned'
    for i, (w, h) in enumerate([(16, 8), (4, 4)]):
        expected = np.ones((h, w), bool)
        expected[0, i] = False
        np.testing.assert_array_equal(result.written[i], expected)
        assert result.written[i].dtype == np.bool_
    assert 'fake blender running' in (work / 'blender.log').read_text()
    args = json.loads((work / 'args.json').read_text())
    assert args[:4] == ['-b', '--factory-startup', '--python-exit-code', '1']
    assert args[4] == '--python' and args[5].endswith('bake.py') and args[6:] == ['--', str(work / 'job.json')]


def test_run_blender_without_timeout(tmp_path, fake):
    work = tmp_path / 'work'
    s = Settings(blender=fake, timeout_s=None)
    write_job(work, make_scene(), [(16, 8)], make_suns(), AMBIENT, s)
    run_blender(work, s)


def test_blender_failure_raises(tmp_path, fake, monkeypatch):
    work = tmp_path / 'work'
    s = Settings(blender=fake)
    write_job(work, make_scene(), [(16, 8)], make_suns(), AMBIENT, s)
    monkeypatch.setenv('FAKE_BLENDER_MODE', 'fail')
    with pytest.raises(BlenderError, match='failing on purpose'):
        run_blender(work, s)


def test_missing_output_raises(tmp_path, fake, monkeypatch):
    work = tmp_path / 'work'
    s = Settings(blender=fake)
    write_job(work, make_scene(), [(16, 8)], make_suns(), AMBIENT, s)
    monkeypatch.setenv('FAKE_BLENDER_MODE', 'nooutput')
    with pytest.raises(BlenderError, match='out.npz'):
        run_blender(work, s)


def test_stale_output_not_taken_for_a_new_run(tmp_path, fake, monkeypatch):
    work = tmp_path / 'work'
    s = Settings(blender=fake)
    write_job(work, make_scene(), [(16, 8)], make_suns(), AMBIENT, s)
    run_blender(work, s)
    monkeypatch.setenv('FAKE_BLENDER_MODE', 'nooutput')
    with pytest.raises(BlenderError):
        run_blender(work, s)


def test_missing_executable_raises(tmp_path):
    work = tmp_path / 'work'
    s = Settings(blender=str(tmp_path / 'nope.exe'))
    write_job(work, make_scene(), [(16, 8)], make_suns(), AMBIENT, s)
    with pytest.raises(BlenderError):
        run_blender(work, s)


def test_job_carries_the_resolved_edge_texel_target(tmp_path):
    job, _ = job_for(tmp_path / 'follow', samples=64)
    assert job['texel_samples'] == 9 * round(64 / 9)
    job, _ = job_for(tmp_path / 'off', samples=64, texel_samples=0)
    assert job['texel_samples'] == 0


def test_job_carries_the_seed(tmp_path):
    write_job(tmp_path, make_scene(), [(8, 8)], [], {}, Settings(seed=7))
    assert json.loads((tmp_path / 'job.json').read_text())['seed'] == 7


def test_unprobed_auto_is_todays_job(tmp_path):
    write_job(tmp_path, make_scene(), [(8, 8)], [], {}, Settings())
    job = json.loads((tmp_path / 'job.json').read_text())
    assert (job['device'], job['cycles_devices'], job['cycles_required'], job['hiprt']) == (
        'auto', ['OPTIX', 'CUDA', 'HIP'], False, False)


def test_job_uses_the_supplied_choice(tmp_path):
    from lightmapper.gpu import Adapter, choose
    choice = choose('amd', [Adapter('Radeon', 0x1002, (2,), 'vulkan', True)])
    write_job(tmp_path, make_scene(), [(8, 8)], [], {}, Settings(device='amd'), choice=choice)
    job = json.loads((tmp_path / 'job.json').read_text())
    assert (job['device'], job['cycles_devices'], job['cycles_required'], job['hiprt']) == (
        'amd', ['HIP'], True, False)


@pytest.mark.blender
def test_vertex_only_relaunch_equals_the_in_blender_pass(tmp_path):
    from conftest import LIGHTMAPPER_BLENDER
    from lightmapper import blender_job
    s = Settings(samples=4, vertex_rays=16, blender=str(LIGHTMAPPER_BLENDER))
    write_job(tmp_path, make_scene(), [(8, 8)], make_suns(), AMBIENT, s, textures=[TEXTURE])
    first = run_blender(tmp_path, s)
    relaunch = blender_job.run_blender_vertex_only(tmp_path, s, first.diffuse_pages)
    np.testing.assert_array_equal(relaunch.vector, first.vertex_vector)
    np.testing.assert_array_equal(relaunch.irradiance, first.vertex_irradiance)
    np.testing.assert_array_equal(relaunch.sun_lit, first.vertex_sun_lit)
    np.testing.assert_array_equal(relaunch.edge_sun_lit, first.edge_sun_lit)
    assert relaunch.rays == first.vertex_rays_cast and relaunch.rays > 0
    log = (tmp_path / 'blender_vertex.log').read_text()
    assert 'page 0' not in log and 'device ' not in log


@pytest.mark.blender
def test_skip_returns_diffuse_pages_and_no_vector(tmp_path):
    s = Settings(samples=4, vertex_rays=16)
    write_job(tmp_path, make_scene(), [(8, 8)], make_suns(), AMBIENT, s, textures=[TEXTURE], vertex_pass='skip')
    result = run_blender(tmp_path, s)
    assert result.vertex_vector is None and result.vertex_irradiance is None
    assert len(result.diffuse_pages) == 1 and result.vertex_rays_cast == 0
    assert result.pages[0].mean() >= result.diffuse_pages[0].mean()
    assert 'vertex pass:' not in (tmp_path / 'blender.log').read_text()
