import pytest

from lightmapper import memory, pipeline
from lightmapper.charts import CapError
from lightmapper.memory import SceneCounts
from opentool import runner


@pytest.fixture
def result(monkeypatch):
    monkeypatch.setattr(memory, 'PER_SUB_TEXEL', 1000)
    monkeypatch.setattr(memory, 'PER_PAGE_TEXEL', 100)
    monkeypatch.setattr(memory, 'PER_TRIANGLE', 0)
    monkeypatch.setattr(memory, 'PER_VERTEX', 0)
    monkeypatch.setattr(memory, 'MARGIN', 1)
    return pipeline.SceneEstimate(SceneCounts(2, 4), [(1024, 1024)])


@pytest.mark.parametrize('free, text', [
    (15_000_000_000, 'Needs about 9.5 GB; 15 GB free'),
    (None, 'Needs about 9.5 GB'),
    (9_542_041_600, 'Needs about 9.5 GB; 9.5 GB free'),
])
def test_describe_fits(result, free, text):
    from opentool.estimate import describe
    assert describe(result, 1024, free) == (text, 'fits')


def test_describe_warns(result):
    from opentool.estimate import describe
    assert describe(result, 1024, 1) == (
        'This bake needs about 9.5 GB of free memory (largest page 1024 x 1024 at Max resolution 1024); '
        'this computer has 0.0 GB free. Try a lower texel density, or close other programs. '
        'You can still bake, at your own risk.', 'warning')


@pytest.mark.parametrize('exc, text, state', [
    (CapError('too many pages'), 'This level needs more lightmap space than the stock format allows at these settings. '
     'Lower the texel density, or raise Max resolution under Advanced.', 'refused'),
    (RuntimeError('broken'), 'The bake stopped: broken. The full error is in the log.', 'error'),
])
def test_describe_error(exc, text, state):
    from opentool.estimate import describe
    assert describe(exc, 1024, None) == (text, state)


def test_run_estimate_uses_chart_settings_and_level_cache(monkeypatch, tmp_path):
    from opentool.estimate import EstimateInputs, run_estimate
    from lightmapper.settings import Settings
    seen = []
    expected = pipeline.SceneEstimate(SceneCounts(2, 4), [(8, 16)])

    def scene(*args):
        seen.append(args)
        return expected

    monkeypatch.setattr(pipeline, 'estimate_scene', scene)
    inputs = EstimateInputs(tmp_path, 'level', 'room', 8.0, 2048)
    assert run_estimate(inputs) is expected
    tags, scenario, bsp, settings, cache = seen[0]
    assert (tags, scenario, bsp) == (tmp_path, 'level', 'room')
    assert settings == Settings(texels_per_unit=8.0, page_max=2048)
    assert cache == runner.chart_cache_path(tmp_path, 'level', 'room')


def test_chart_cache_normalises_tags_path(tmp_path):
    assert runner.chart_cache_path(tmp_path / 'child' / '..', 'level', 'room') == runner.chart_cache_path(tmp_path, 'level', 'room')


def test_describe_counts_all_127_resident_pages(result):
    from opentool.estimate import describe
    result.pages.extend([(256, 256)] * 126)
    # 9 * 1024**2 * 1000 + (1024**2 + 126 * 256**2) * 100
    need = 10_367_795_200
    assert describe(result, 1024, need) == ('Needs about 10 GB; 10 GB free', 'fits')
    assert describe(result, 1024, need - 1)[1] == 'warning'
