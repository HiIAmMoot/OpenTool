import pytest

from lightmapper import memory, pipeline
from lightmapper.charts import CapError
from lightmapper.memory import SceneCounts
from opentool import runner


GB = memory.GB


@pytest.fixture
def result(monkeypatch):
    # only the Blender stage costs anything: 10 GB for any level
    for name in ('BLENDER_PER_SUB_TEXEL', 'BLENDER_PER_PAGE_TEXEL', 'BLENDER_PER_VERTEX', 'VERTEX_BASE',
                 'VERTEX_PER_PAGE_TEXEL', 'VERTEX_PER_GLOW_SUB_TEXEL', 'GLOW_PER_GLOW_SUB_TEXEL', 'FINISH_BASE',
                 'FINISH_PER_SUB_TEXEL', 'PER_TILE_SUB_TEXEL', 'GLOW_PER_TILE_SUB_TEXEL'):
        monkeypatch.setattr(memory, name, 0)
    monkeypatch.setattr(memory, 'BLENDER_BASE', 10 * GB)
    monkeypatch.setattr(memory, 'MARGIN', 1)
    return pipeline.SceneEstimate(SceneCounts(2, 4), [(8, 8)])


@pytest.mark.parametrize('machine, text', [
    (memory.Machine(16 * GB, 24 * GB, 40 * GB), 'Needs about 10 GB; this computer has 16 GB'),
    (None, 'Needs about 10 GB'),
    (memory.Machine(10 * GB, 10 * GB, None), 'Needs about 10 GB; this computer has 10 GB'),
])
def test_describe_fits(result, machine, text):
    from opentool.estimate import describe
    assert describe(result, machine) == (text, 'fits')


def test_describe_warns_about_the_page_file(result):
    from opentool.estimate import RISK_SUFFIX, describe
    machine = memory.Machine(8 * GB, 8 * GB, 40 * GB)
    assert describe(result, machine) == (
        "This bake needs about 10 GB at its peak (baking in Blender), more than this computer's 8.0 GB of memory. "
        'It will lean on the page file and slow down.' + RISK_SUFFIX, 'warning')


def test_describe_flags_a_risk_of_running_out_of_memory(result):
    from opentool.estimate import RISK_SUFFIX, describe
    machine = memory.Machine(8 * GB, 8 * GB, 9 * GB)
    inputs = pipeline.stage_inputs(result, 'tool', machine)
    assert describe(result, machine) == (
        memory.out_of_memory_message(inputs, machine) + RISK_SUFFIX, 'risk')
    assert 'run out of memory' in describe(result, machine)[0]


def test_describe_counts_the_glow_stage_only_in_enhanced_mode(result, monkeypatch):
    from opentool.estimate import describe
    monkeypatch.setattr(memory, 'GLOW_PER_GLOW_SUB_TEXEL', 1000)
    result.flagged_sub_texels = 9 * GB // 1000
    machine = memory.Machine(15 * GB, 15 * GB, 15 * GB)
    assert describe(result, machine, 'tool')[1] == 'fits'
    text, state = describe(result, machine, 'enhanced')
    assert state == 'risk' and text.startswith('This bake needs about 19 GB at its peak (baking light from')
    assert 'Traditional' in text


def test_describe_has_no_glow_without_bounces(result, monkeypatch):
    from opentool.estimate import describe
    monkeypatch.setattr(memory, 'GLOW_PER_GLOW_SUB_TEXEL', 1000)
    result.flagged_sub_texels = 9 * GB // 1000
    machine = memory.Machine(15 * GB, 15 * GB, 15 * GB)
    assert describe(result, machine, 'enhanced', 0)[1] == 'fits'
    assert describe(result, machine, 'enhanced', 1)[1] == 'risk'


def test_describe_decides_tiles_from_the_machine(result, monkeypatch):
    from opentool.estimate import describe
    monkeypatch.setattr(memory, 'PER_TILE_SUB_TEXEL', 10**9)
    # (one 8 x 8 page is 576 sub-texels: tiles would add 576 GB)
    assert describe(result, memory.Machine(16 * GB, 16 * GB, None))[0] == 'Needs about 10 GB; this computer has 16 GB'
    monkeypatch.setattr(memory, 'PER_TILE_SUB_TEXEL', 1)
    assert describe(result, memory.Machine(16 * GB, 16 * GB, None))[1] == 'fits'


@pytest.mark.parametrize('exc, text, state', [
    (CapError('too many pages'), 'This level needs more lightmap space than the stock format allows at these settings. '
     'Lower the texel density, or set Max resolution to Auto under Advanced.', 'refused'),
    (RuntimeError('broken'), 'The bake stopped: broken. The full error is in the log.', 'error'),
])
def test_describe_error(exc, text, state):
    from opentool.estimate import describe
    assert describe(exc, None) == (text, state)


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
    run_estimate(EstimateInputs(tmp_path, 'level', 'room', 8.0, 2048, 'enhanced', 'cpu', 0))
    assert seen[1][3] == Settings(texels_per_unit=8.0, page_max=2048, ignore_normals_mode='enhanced', device='cpu',
                                  bounces=0)
    assert cache == runner.chart_cache_path(tmp_path, 'level', 'room')


def test_chart_cache_normalises_tags_path(tmp_path):
    assert runner.chart_cache_path(tmp_path / 'child' / '..', 'level', 'room') == runner.chart_cache_path(tmp_path, 'level', 'room')


def test_describe_counts_all_127_resident_pages(result, monkeypatch):
    from opentool.estimate import describe
    monkeypatch.setattr(memory, 'BLENDER_BASE', 0)
    monkeypatch.setattr(memory, 'BLENDER_PER_PAGE_TEXEL', 100)
    result.pages[:] = [(1024, 1024)] + [(256, 256)] * 126
    need = (1024 ** 2 + 126 * 256 ** 2) * 100
    assert describe(result, memory.Machine(need, need, None))[1] == 'fits'
    assert describe(result, memory.Machine(need - 1, need - 1, None))[1] == 'risk'
    assert describe(result, memory.Machine(need - 1, need, need + 1))[1] == 'warning'


def test_the_stages_run_quick_then_rough_then_full(monkeypatch, tmp_path):
    from opentool import estimate
    seen = []
    for name in ('quick_estimate', 'estimate_scene', 'full_estimate'):
        monkeypatch.setattr(pipeline, name, lambda *args, name=name: seen.append(name) or name)
    inputs = estimate.EstimateInputs(tmp_path, 'level', 'room', 8.0, 'auto')
    assert [run(inputs) for _, run in estimate.STAGES] == ['quick_estimate', 'estimate_scene', 'full_estimate']
    labels = [label for label, _ in estimate.STAGES]
    assert labels[0].startswith('Quick estimate') and labels[1].startswith('Rough estimate') and labels[2] == ''


def test_rough_counts_the_square_when_it_lands_on_a_half_height_shape():
    # (a nested block can need w x w where the rectangles settled on w x w/2, so the largest page is counted square)
    sizes, flagged, scale = pipeline.rough_pages([(512, 1024), (512, 1024)], (1024, 512), 0, False)
    assert max(w * h for w, h in sizes) == 1024 * 1024 and flagged == 0 and scale == 1.0
    sizes, _, _ = pipeline.rough_pages([(1024, 1024)], (1024, 1024), 0, False)
    assert sizes == [(1024, 1024)]


def test_rough_counts_every_page_as_flagged_when_any_triangle_ignores_normals():
    sizes, flagged, _ = pipeline.rough_pages([(256, 256), (256, 128)], (256, 256), 0, True)
    assert flagged == 9 * (256 * 256 + 256 * 128)


@pytest.mark.parametrize('stepped, scale', [(0, 1.0), (11, pipeline.ROUGH_REDUCTION), (pipeline.ROUGH_REACH, pipeline.ROUGH_REDUCTION),
                                            (pipeline.ROUGH_REACH + 1, 1.0)])
def test_rough_scales_down_when_the_page_count_stepped_up_from_within_reach(stepped, scale):
    # (the rectangles stepped past a shape where they needed few enough pages for the nest to plausibly get in)
    assert pipeline.rough_pages([(4096, 4096)] * 3, (4096, 2048), stepped, False)[2] == scale


def test_the_scale_lowers_every_stage(result):
    full = memory.stage_bytes(pipeline.stage_inputs(result, 'tool', None))
    half = memory.stage_bytes(pipeline.stage_inputs(pipeline.SceneEstimate(result.counts, result.pages, scale=.5),
                                                    'tool', None))
    assert half['blender'] == full['blender'] // 2
