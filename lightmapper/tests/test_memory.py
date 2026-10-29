import logging
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from lightmapper import memory
from lightmapper.memory import SceneCounts
from lightmapper.blender_job import write_job
from lightmapper.settings import Settings
from lightmapper.tests.test_blender_job import make_scene


@pytest.fixture
def counts():
    return SceneCounts(10_000, 20_000)


@pytest.fixture
def pages():
    return [(4096, 4096)]


@pytest.fixture
def message_model(monkeypatch):
    for name, value in dict(PER_SUB_TEXEL=1000, PER_PAGE_TEXEL=0,
                            PER_TRIANGLE=0, PER_VERTEX=0, MARGIN=1.0).items():
        monkeypatch.setattr(memory, name, value)


def test_page_term_is_nine_sub_texels_per_texel():
    assert memory.page_term(1024, 1024) == 9 * 1024 * 1024 * memory.PER_SUB_TEXEL


def test_page_term_grows_with_the_square_of_the_side():
    assert memory.page_term(2048, 2048) == 4 * memory.page_term(1024, 1024)


def test_a_bigger_level_needs_more_memory_at_the_same_max_resolution():
    small = memory.bake_bytes(SceneCounts(10_000, 20_000), [(1024, 1024)] * 2)
    big = memory.bake_bytes(SceneCounts(400_000, 900_000), [(1024, 1024)] * 9)
    assert big > small


def test_the_largest_page_decides_the_page_term(monkeypatch, message_model):
    monkeypatch.setattr(memory, 'PER_PAGE_TEXEL', 2)
    monkeypatch.setattr(memory, 'PER_TRIANGLE', 3)
    monkeypatch.setattr(memory, 'PER_VERTEX', 5)
    monkeypatch.setattr(memory, 'MARGIN', 1.15)
    assert memory.resident_term([(256, 256), (2048, 1024), (512, 512)]) == 4_849_664
    assert memory.scene_term(SceneCounts(7, 11)) == 76
    assert memory.bake_bytes(SceneCounts(7, 11), [(256, 256), (2048, 1024), (512, 512)]) == 21_711_100_401


def test_largest_supported_page_set_uses_the_fitted_model():
    assert memory.bake_bytes(SceneCounts(400_000, 900_000), [(4096, 4096)] * 127) == 203_260_240_296


def test_refuses_when_the_estimate_exceeds_free_memory(counts, pages):
    need = memory.bake_bytes(counts, pages)
    assert memory.check_bake_memory(counts, pages, 4096, free=need) == memory.MemoryCheck(need, need)
    with pytest.raises(memory.BakeMemoryError) as caught:
        memory.check_bake_memory(counts, pages, 4096, free=need - 1)
    assert caught.value.estimate == need
    assert caught.value.free == need - 1


def test_message_names_the_need_the_free_memory_and_a_fitting_max_resolution(counts, pages, message_model):
    with pytest.raises(memory.BakeMemoryError) as caught:
        memory.check_bake_memory(counts, pages, 4096, free=9_800_000_000)
    error = caught.value
    assert str(error) == ('This bake needs about 151 GB of free memory (largest page 4096 x 4096 at Max '
                          'resolution 4096); this computer has 9.8 GB free. Try Max resolution 1024, or close '
                          'other programs. If the level then needs too many pages or still does not fit, lower '
                          'the texel density a little.')


def test_suggestion_counts_the_resident_pages(counts, monkeypatch, message_model):
    monkeypatch.setattr(memory, 'PER_PAGE_TEXEL', 100)
    assert memory.suggest_page_max(counts, [(2048, 2048)], 10_000_000_000) == 1024
    assert memory.suggest_page_max(counts, [(2048, 2048)] * 2, 10_000_000_000) == 512


def test_suggestion_counts_the_scene_and_margin(monkeypatch, message_model):
    monkeypatch.setattr(memory, 'PER_VERTEX', 100_000)
    monkeypatch.setattr(memory, 'MARGIN', 1.15)
    assert memory.suggest_page_max(SceneCounts(0, 20_000), [(4096, 4096)], 11_000_000_000) == 512


def test_suggests_texel_density_when_no_choice_fits(counts, pages, message_model):
    with pytest.raises(memory.BakeMemoryError) as caught:
        memory.check_bake_memory(counts, pages, 4096, free=500_000_000)
    error = caught.value
    assert str(error).endswith('this computer has 0.5 GB free. Try a lower texel density, or close other programs.')


def test_suggestion_is_worded_as_try(counts, pages, message_model):
    fitting = memory.out_of_memory_message(counts, pages, 4096, 9_800_000_000)
    no_fit = memory.out_of_memory_message(counts, pages, 4096, 500_000_000)
    assert 'Try' in fitting and 'lower the texel density' in fitting
    assert 'Try' in no_fit


def test_message_names_a_non_square_largest_page(counts, message_model):
    text = memory.out_of_memory_message(counts, [(4096, 2048)], 4096, None)
    assert 'largest page 4096 x 2048' in text
    assert 'Try Max resolution 2048' in text


def test_message_without_free_memory_suggests_the_next_lower_side(counts, message_model):
    text = memory.out_of_memory_message(counts, [(4096, 4096)], 4096, None)
    assert 'this computer has' not in text and 'Try Max resolution 2048' in text


def test_suggestion_uses_actual_page_side_and_accepts_exact_fit(counts, message_model):
    assert memory.suggest_page_max(counts, [(1024, 256)], 2_359_296_000) == 512
    assert memory.suggest_page_max(counts, [(256, 256)], None) is None


def test_format_gb():
    assert [memory.format_gb(n) for n in (9_840_000_000, 10_400_000_000, 25_462_081_056)] == ['9.8', '10', '25']


def test_unknown_free_memory_does_not_refuse(counts, pages, caplog):
    with caplog.at_level(logging.WARNING, logger='lightmapper.memory'):
        result = memory.check_bake_memory(counts, pages, 4096, free=None)
    assert result == memory.MemoryCheck(memory.bake_bytes(counts, pages), None)
    assert [(record.levelname, record.getMessage()) for record in caplog.records] == [
        ('WARNING', 'free memory is unknown, so the bake runs without the memory check')]


def test_free_memory_is_read_when_not_given(counts, pages, monkeypatch):
    monkeypatch.setattr(memory, 'free_physical_bytes', lambda: 1)
    with pytest.raises(memory.BakeMemoryError) as caught:
        memory.check_bake_memory(counts, pages, 4096)
    assert caught.value.free == 1


def test_free_memory_failure_reads_as_unknown(monkeypatch):
    def fail():
        raise OSError('memory status unavailable')

    monkeypatch.setattr(memory, 'os', SimpleNamespace(name='nt'))
    monkeypatch.setattr(memory, '_windows_free_physical_bytes', fail)
    assert memory.free_physical_bytes() is None


def test_free_physical_bytes_reads_this_machine():
    free = memory.free_physical_bytes()
    assert isinstance(free, int) and free > 0


def test_free_physical_bytes_uses_available_pages(monkeypatch):
    values = {'SC_AVPHYS_PAGES': 123, 'SC_PAGE_SIZE': 4096}
    monkeypatch.setattr(memory, 'os', SimpleNamespace(name='posix', sysconf=values.__getitem__))
    assert memory.free_physical_bytes() == 503_808


@pytest.mark.parametrize('failure', [AttributeError, OSError, ValueError])
def test_sysconf_failure_reads_as_unknown(monkeypatch, failure):
    def fail(name):
        raise failure(name)

    monkeypatch.setattr(memory, 'os', SimpleNamespace(name='posix', sysconf=fail))
    assert memory.free_physical_bytes() is None


def test_scene_counts_from_npz_match_the_scene(tmp_path):
    scene = make_scene()
    write_job(tmp_path, scene, [(16, 8)], [], {}, Settings(device='cpu'))
    assert SceneCounts.from_scene(scene) == SceneCounts(2, 6)
    assert SceneCounts.from_npz(tmp_path / 'scene.npz') == SceneCounts.from_scene(scene)


def test_scene_counts_are_frozen(counts):
    with pytest.raises(FrozenInstanceError):
        counts.vertices = 0


def test_bake_memory_error_carries_its_message_estimate_and_free():
    error = memory.BakeMemoryError('This bake needs ...', 1, 2)
    assert str(error) == 'This bake needs ...'
    assert error.estimate == 1
    assert error.free == 2


def test_bake_memory_error_is_a_bake_error():
    from lightmapper import pipeline
    from lightmapper.errors import BakeError

    assert issubclass(memory.BakeMemoryError, pipeline.BakeError)
    assert pipeline.BakeError is BakeError
