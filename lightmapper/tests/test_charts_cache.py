"""The chart cache preserves numeric layout without pickle payloads."""

import json

import numpy as np
import pytest

from lightmapper.formats.bsp import OutMaterial, OutPage
from lightmapper.formats.vertices import ENV, LM


def test_cache_roundtrip(tmp_path):
    from lightmapper.charts import load_charts, save_charts

    path = tmp_path / 'charts.npz'
    env, lm = np.zeros(3, ENV), np.zeros(3, LM)
    env['position'] = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
    lm['texcoord'] = [[.1, .1], [.9, .1], [.1, .9]]
    pages = [OutPage(8, 16, [OutMaterial(7, env, lm, np.array([[2, 0, 1]]), np.array([19]))])]
    assert load_charts(path, 'key') is None
    save_charts(path, pages, 'key')
    with np.load(path, allow_pickle=False) as data:
        assert json.loads(str(data['meta'])) == {'version': 11, 'key': 'key', 'pages': [[8, 16, 1]]}
        assert data['p0_m0_group'].dtype == np.int32
        assert data['p0_m0_tris'].dtype == np.uint16
        assert data['p0_m0_surface_ids'].dtype == np.int32
    result = load_charts(path, 'key')
    assert (result[0].width, result[0].height) == (8, 16)
    material = result[0].materials[0]
    assert material.group == 7
    for name in ('env', 'lm', 'tris', 'surface_ids'):
        np.testing.assert_array_equal(getattr(material, name), getattr(pages[0].materials[0], name))
    assert load_charts(path, 'changed') is None
    np.savez(path, meta=json.dumps({'version': 99, 'key': 'key', 'pages': []}))
    assert load_charts(path, 'key') is None


def test_cache_from_before_density_per_detail_level_is_not_reused(tmp_path):
    from lightmapper.charts import load_charts

    path = tmp_path / 'charts.npz'
    np.savez(path, meta=json.dumps({'version': 1, 'key': 'key', 'pages': []}))
    assert load_charts(path, 'key') is None


def test_cache_key_changes_with_the_detail_level_scale(tmp_path):
    from lightmapper.charts import ChartSettings
    from lightmapper.pipeline import chart_cache_key

    bsp = tmp_path / 'room.bsp'
    bsp.write_bytes(b'tag bytes')
    same = chart_cache_key(bsp, ChartSettings(), [])
    assert chart_cache_key(bsp, ChartSettings(), []) == same
    assert chart_cache_key(bsp, ChartSettings(detail_level_scale=(3.0, 1.0, 0.6, 0.03)), []) != same
    assert chart_cache_key(bsp, ChartSettings(texels_per_unit=3.0), []) != same


def test_cache_from_before_no_shrink_scaling_is_not_reused(tmp_path):
    from lightmapper.charts import load_charts

    path = tmp_path / 'charts.npz'
    np.savez(path, meta=json.dumps({'version': 2, 'key': 'key', 'pages': []}))
    assert load_charts(path, 'key') is None


def test_cache_from_before_packing_search_is_not_reused(tmp_path):
    from lightmapper.charts import load_charts

    path = tmp_path / 'charts.npz'
    np.savez(path, meta=json.dumps({'version': 3, 'key': 'key', 'pages': []}))
    assert load_charts(path, 'key') is None


def test_cache_from_separate_atlases_is_not_reused(tmp_path):
    from lightmapper.charts import load_charts

    path = tmp_path / 'charts.npz'
    np.savez(path, meta=json.dumps({'version': 4, 'key': 'key', 'pages': []}))
    assert load_charts(path, 'key') is None


def test_cache_before_density_tolerance_guard_is_not_reused(tmp_path):
    from lightmapper.charts import load_charts

    path = tmp_path / 'charts.npz'
    np.savez(path, meta=json.dumps({'version': 5, 'key': 'key', 'pages': []}))
    assert load_charts(path, 'key') is None


def test_cache_before_material_budget_packing_is_not_reused(tmp_path):
    from lightmapper.charts import load_charts

    path = tmp_path / 'charts.npz'
    np.savez(path, meta=json.dumps({'version': 6, 'key': 'key', 'pages': []}))
    assert load_charts(path, 'key') is None


def test_cache_before_tight_spill_packing_is_not_reused(tmp_path):
    from lightmapper.charts import load_charts

    path = tmp_path / 'charts.npz'
    np.savez(path, meta=json.dumps({'version': 7, 'key': 'key', 'pages': []}))
    assert load_charts(path, 'key') is None


def test_cache_before_page_count_priority_is_not_reused(tmp_path):
    from lightmapper.charts import load_charts

    path = tmp_path / 'charts.npz'
    np.savez(path, meta=json.dumps({'version': 8, 'key': 'key', 'pages': []}))
    assert load_charts(path, 'key') is None


def test_cache_before_material_aggregation_is_not_reused(tmp_path):
    from lightmapper.charts import load_charts
    path = tmp_path / 'charts.npz'
    np.savez(path, meta=json.dumps({'version': 9, 'key': 'key', 'pages': []}))
    assert load_charts(path, 'key') is None


def test_cache_key_changes_with_respecting_normal_seams(tmp_path):
    from lightmapper.charts import ChartSettings
    from lightmapper.pipeline import chart_cache_key

    bsp = tmp_path / 'room.bsp'
    bsp.write_bytes(b'tag bytes')
    assert (chart_cache_key(bsp, ChartSettings(respect_normal_seams=False), [])
            != chart_cache_key(bsp, ChartSettings(), []))


def test_cache_key_needs_the_shader_radiosity(tmp_path):
    from lightmapper.charts import ChartSettings
    from lightmapper.pipeline import chart_cache_key

    bsp = tmp_path / 'room.bsp'
    bsp.write_bytes(b'tag bytes')
    with pytest.raises(TypeError):
        chart_cache_key(bsp, ChartSettings())


def test_cache_before_joined_simple_charts_is_not_reused(tmp_path):
    from lightmapper.charts import load_charts
    path = tmp_path / 'charts.npz'
    np.savez(path, meta=json.dumps({'version': 10, 'key': 'key', 'pages': []}))
    assert load_charts(path, 'key') is None


def test_interrupted_save_preserves_previous_cache(tmp_path, monkeypatch):
    from lightmapper import charts

    path = tmp_path / 'charts.npz'
    charts.save_charts(path, [OutPage(8, 16, [])], 'old')

    def interrupted(stream, **arrays):
        stream.write(b'partial zip')
        raise OSError('interrupted')

    monkeypatch.setattr(charts.np, 'savez', interrupted)
    with pytest.raises(OSError, match='interrupted'):
        charts.save_charts(path, [OutPage(32, 64, [])], 'new')
    result = charts.load_charts(path, 'old')
    assert [(p.width, p.height) for p in result] == [(8, 16)]
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize('damage', ['truncated', 'bad_json', 'missing_array'])
def test_corrupt_cache_is_a_miss(tmp_path, damage):
    from lightmapper import charts

    path = tmp_path / 'charts.npz'
    if damage == 'truncated':
        charts.save_charts(path, [OutPage(8, 16, [])], 'key')
        path.write_bytes(path.read_bytes()[:40])
    elif damage == 'bad_json':
        np.savez(path, meta='not json')
    else:
        np.savez(path, meta=json.dumps({'version': charts.CACHE_VERSION, 'key': 'key', 'pages': [[8, 16, 1]]}))
    assert charts.load_charts(path, 'key') is None
