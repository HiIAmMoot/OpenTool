"""Check the weight and gamma fit and the vector-form choice on data with known answers."""

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from lightmapper import cli, pipeline
from lightmapper.blender_job import BakeResult
from lightmapper.calibrate import calibrate, choose_vector_mode, fit, predict
from lightmapper.diff import LUMA, Samples
from lightmapper.formats.bitmap import write_bitmap
from lightmapper.formats.vertices import compress, encode_r5g6b5, pack_uncompressed, unpack_uncompressed
from lightmapper.radiosity import finalize
from lightmapper.tagio.tagfile import read_tag, resolve_reference_path, write_tag
from test_diff import grid_pair
from test_pipeline import put, tiny  # noqa: F401  (tiny is the fixture)


def samples(rgb):
    n = len(rgb)
    z = np.zeros((n, 3))
    return Samples(z, np.asarray(rgb, float), np.ones(n), np.array(['m'] * n, object),
                   z, z, z, z, np.arange(n), {})


def fields(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    # independent per-sample fields, scaled so the model's sum stays under the clip
    return rng.random((n, 3)) * 0.8, rng.random((n, 3)) * 0.5, rng.random((n, 3)) * 0.2


def test_predict_exposes_an_over_bright_sum_keeping_its_hue():
    sun = np.array([[1.34, 1.0, 1.0]])
    zero = np.zeros((1, 3))
    np.testing.assert_allclose(predict(1.0, 0.0, 0.0, 1.0, sun, zero, zero), [[1, 1 / 1.34, 1 / 1.34]])


def test_fit_recovers_weights_and_gamma():
    sun, emit, ambient = fields()
    tool = predict(0.8, 1.3, 0.5, 0.7, sun, emit, ambient)
    result = fit(samples(sun), samples(emit), ambient, tool)
    for got, want in zip((result.w_sun, result.w_emit, result.w_amb, result.gamma), (0.8, 1.3, 0.5, 0.7)):
        assert got == pytest.approx(want, rel=0.02)
    assert result.pearson_r > 0.99 and result.median_abs_error < 0.01


def test_fit_leaves_out_dark_samples_and_lets_brighter_ours_cost_nothing_on_saturated_ones():
    sun, emit, ambient = fields()
    tool = predict(0.8, 1.3, 0.5, 0.7, sun, emit, ambient)
    # black tool texels that ours puts high would drag the fit; clipped ones that ours
    # puts well above the clip are no evidence against it
    saturated, dark = np.arange(0, 3000, 10), np.arange(5, 3000, 10)
    tool[saturated], sun[saturated], emit[saturated] = 1.0, 3.0, 3.0
    tool[dark], sun[dark] = 0.0, 3.0
    result = fit(samples(sun), samples(emit), ambient, tool)
    assert result.samples_used == (tool @ LUMA > 0.01).sum()
    for got, want in zip((result.w_sun, result.w_emit, result.w_amb, result.gamma), (0.8, 1.3, 0.5, 0.7)):
        assert got == pytest.approx(want, rel=0.02)


def test_fit_learns_a_weight_only_saturated_samples_bound_from_below():
    rng = np.random.default_rng(5)
    n = 1000
    # half the samples carry no sun, so they fix everything but the sun weight; the other
    # half are clipped in tool, with a sun field so dim that the weight must exceed 2.4
    emit, ambient = rng.random((2 * n, 3)) * 0.5, rng.random((2 * n, 3)) * 0.2
    sun = np.zeros((2 * n, 3))
    sun[n:], emit[n:], ambient[n:] = 0.4, 0, 0
    tool = predict(1.0, 1.3, 0.5, 1.0, sun, emit, ambient)
    tool[n:] = 1.0
    result = fit(samples(sun), samples(emit), ambient, tool)
    assert result.w_sun > 2.4
    assert result.samples_used == 2 * n
    assert result.median_abs_error < 0.01


def test_fit_refuses_when_every_usable_sample_has_a_clipped_channel():
    sun = np.array([[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]])
    tool = np.array([[1, 0.2, 0.3], [0.3, 1, 0.2], [0.2, 0.3, 1]])
    zero = np.zeros_like(sun)
    with pytest.raises(ValueError, match='no unclipped samples'):
        fit(samples(sun), samples(zero), zero, tool)


def test_clipped_channel_prediction_between_threshold_and_one_costs_nothing():
    sun, emit, ambient = fields(30)
    tool = predict(1, 1, 1, 1, sun, emit, ambient)
    sun = np.vstack([sun, np.full((30, 3), 0.99)])
    emit = np.vstack([emit, np.zeros((30, 3))])
    ambient = np.vstack([ambient, np.zeros((30, 3))])
    tool = np.vstack([tool, np.ones((30, 3))])
    result = fit(samples(sun), samples(emit), ambient, tool)
    np.testing.assert_allclose([result.w_sun, result.w_emit, result.w_amb, result.gamma],
                               [1, 1, 1, 1], rtol=0, atol=1e-6)


def test_vertical_flip_has_low_correlation():
    flipped, straight = grid_pair(True), grid_pair(False)
    assert len(flipped.surface) == 64
    for pairs, low in ((flipped, True), (straight, False)):
        zero = np.zeros_like(pairs.tool_rgb)
        result = fit(samples(pairs.ours_rgb), samples(pairs.ours_rgb), zero, pairs.tool_rgb)
        assert (result.pearson_r < 0.3) == low
    assert straight.tool_rgb.shape == (64, 3)


def vector_data(seed=3):
    rng = np.random.default_rng(seed)
    n = 4000
    direction = rng.normal(size=(n, 3))
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    E = rng.random(n) * 4 + 0.5
    # coherence-like sums are bounded by E; magnitudes of V and E are otherwise unrelated
    V = direction * (E * rng.random(n) * 0.9 + 0.05)[:, None]
    N = rng.normal(size=(n, 3)).astype(np.float32)
    return V, E, N


def test_vector_mode_picks_coherence():
    V, E, N = vector_data()
    tool = finalize(V, E, N, 'coherence', 1)
    choice = choose_vector_mode(tool, N, V, E, N)
    assert choice.mode == 'coherence' and choice.excluded == 0


def test_vector_mode_picks_scaled_with_its_scale():
    V, E, N = vector_data()
    V = V * rng_scale(len(V))
    tool = finalize(V, E, N, 'scaled', 0.37)
    choice = choose_vector_mode(tool, N, V, E, N)
    assert choice.mode == 'scaled'
    assert choice.scale == pytest.approx(0.37, rel=0.15)


def rng_scale(n):
    return np.random.default_rng(4).random(n)[:, None] * 3 + 0.1


def test_vector_mode_compares_vectors_not_lengths():
    V, E, N = vector_data()
    tool = -finalize(V, E, N, 'coherence', 1)
    choice = choose_vector_mode(tool, N, V, E, N)
    # same lengths, opposite directions: only a vector difference sees it
    assert choice.errors['coherence'] > 0.5


def test_vector_scale_follows_the_vector_difference():
    V, E, N = vector_data()
    signs = np.random.default_rng(9).choice([-1, 1], size=V.shape)
    tool = finalize(V * signs, E, N, 'scaled', 0.37)
    choice = choose_vector_mode(tool, N, V, E, N)
    # (lengths alone would pick 0.398 here; the directions disagree, so the vector error picks lower)
    assert choice.mode == 'scaled'
    assert choice.scale == pytest.approx(0.2512, rel=0.01)


@pytest.mark.parametrize('count', [3000, 4000])
def test_vector_mode_keeps_ambient_dominated_normals_as_measurements(count):
    V, E, N = vector_data()
    tool = finalize(V, E, N, 'coherence', 1)
    tool[:count] = N[:count]
    choice = choose_vector_mode(tool, N, V, E, N)
    assert choice.excluded == 0
    assert choice.errors['coherence'] > 0.1


# ---------- the command on tiny tags, with Blender faked


@pytest.fixture
def lit(tiny, monkeypatch):
    tags, settings, work = tiny
    room_path = resolve_reference_path(tags, 'sbsp', 'room')
    room = read_tag(room_path)
    material = room.root.block('lightmaps')[0].block('materials')[0]
    env, lm = unpack_uncompressed(material.data('uncompressed_vertices'), 6, 6)
    lm['texcoord'] = np.array([[4, 4], [20, 4], [4, 20], [36, 4], [52, 4], [36, 20]]) / 64
    material.set_data('uncompressed_vertices', pack_uncompressed(env, lm))
    material.set_data('compressed_vertices', compress(env, lm))
    room.root.set_reference('lightmaps_bitmap', 'bitm', 'room')
    write_tag(room, room_path)
    page = np.full((64, 64, 3), 0.4) + np.arange(64)[None, :, None] / 400
    put(tags, 'bitm', 'room', write_bitmap([encode_r5g6b5(page, False)]))

    def fake_blender(job_work, s):
        job = json.loads((job_work / 'job.json').read_text())
        with np.load(job_work / 'scene.npz') as scene:
            vectors = scene['query_normal'] * 0.5
        pages = [np.full((h, w, 3), 0.3, np.float32) for w, h in job['pages']]
        irradiance = np.ones(len(vectors), np.float32)
        np.savez(job_work / 'out.npz', vertex_vector=vectors, vertex_irradiance=irradiance,
                 device=np.array('test'))
        return BakeResult(pages, vectors, irradiance, 'test CPU',
                          [np.ones(p.shape[:2], bool) for p in pages])

    monkeypatch.setattr(pipeline, 'run_blender', fake_blender)
    return tags, replace(settings, texels_per_unit=2.0), work


@pytest.mark.parametrize('passes', [True, False])
def test_calibrate_writes_fit_and_report(lit, tmp_path, monkeypatch, passes):
    from lightmapper import calibrate as module, settings as settings_module
    tags, settings, work = lit
    before = {p: p.read_bytes() for p in tags.rglob('*') if p.is_file()}
    shipped_path = Path(settings_module.__file__).with_name('calibration.json')
    shipped = shipped_path.read_bytes()
    real = module.report

    def forced(*args):
        data = real(*args)
        data['acceptance'] = {'pearson_r': passes}
        return data

    monkeypatch.setattr(module, 'report', forced)
    data = calibrate(tags, 'level', 'room', work, settings)
    # the fit never replaces the package's calibration, accepted or not
    assert shipped_path.read_bytes() == shipped
    written = json.loads((work / 'calibration.json').read_text())
    assert set(written) == {'w_sun', 'w_emit', 'w_amb', 'gamma', 'radiosity_mode', 'radiosity_scale'}
    assert written['radiosity_mode'] in ('coherence', 'scaled') and 0.25 <= written['gamma'] <= 4
    assert data['samples'] == 2 and data['clipped_fraction'] == 0
    assert 'Acceptance' in (work / 'calibration.md').read_text()
    assert {p: p.read_bytes() for p in tags.rglob('*') if p.is_file()} == before


def test_calibrate_refuses_unpaired_sun_and_emit(lit, tmp_path, monkeypatch):
    from lightmapper import calibrate as module
    tags, settings, work = lit
    real, calls = module.paired_samples, []

    def uneven(*args):
        samples = real(*args)
        calls.append(1)
        if len(calls) == 2:
            samples.surface = samples.surface[:1]
        return samples

    monkeypatch.setattr(module, 'paired_samples', uneven)
    with pytest.raises(ValueError, match='same surfaces'):
        calibrate(tags, 'level', 'room', work, settings)


def test_calibrate_logs_a_missing_class_ambient(lit, tmp_path, caplog):
    tags, settings, work = lit
    calibrate(tags, 'level', 'room', work, settings)
    assert 'no ambient for class' in caplog.text


def test_diff_command_against_itself(lit, capsys, caplog):
    tags, _, work = lit
    assert cli.main(['diff', 'room', 'room', '--tags', str(tags), '--work', str(work)]) == 0
    assert 'Pearson r: 1.0000' in capsys.readouterr().out
    assert (work / 'diff.md').is_file()
    assert 'identity' in caplog.text
