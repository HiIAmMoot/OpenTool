import csv
import logging
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from lightmapper import memory
from lightmapper.blender_job import write_job
from lightmapper.memory import Machine, PagingFile, SceneCounts, StageInputs
from lightmapper.settings import Settings
from lightmapper.tests.test_blender_job import make_scene

GB = memory.GB


@pytest.fixture
def small_model(monkeypatch):
    for name, value in dict(BLENDER_BASE=1000, BLENDER_PER_SUB_TEXEL=10, BLENDER_PER_PAGE_TEXEL=3, BLENDER_PER_VERTEX=5,
                            PER_TILE_SUB_TEXEL=2, VERTEX_BASE=500, VERTEX_PER_PAGE_TEXEL=7, VERTEX_PER_GLOW_SUB_TEXEL=11,
                            GLOW_PER_GLOW_SUB_TEXEL=13, GLOW_PER_TILE_SUB_TEXEL=17, FINISH_BASE=100,
                            FINISH_PER_SUB_TEXEL=19, MARGIN=1).items():
        monkeypatch.setattr(memory, name, value)


def inputs(**changes):
    # the largest page is 9 * 4 * 2 = 72 sub-texels, the pages hold 12 texels, the scene 10 vertices
    base = StageInputs(SceneCounts(0, 10), [(2, 2), (4, 2)], 'gpu', 36, False)
    return replace(base, **changes)


def only(monkeypatch, stage, **others):
    # a model where one stage is huge and the others negligible, so the peak and its message are easy to state
    for name in ('BLENDER_BASE', 'BLENDER_PER_SUB_TEXEL', 'BLENDER_PER_PAGE_TEXEL', 'BLENDER_PER_VERTEX',
                 'VERTEX_BASE', 'VERTEX_PER_PAGE_TEXEL', 'VERTEX_PER_GLOW_SUB_TEXEL', 'GLOW_PER_GLOW_SUB_TEXEL',
                 'FINISH_BASE', 'FINISH_PER_SUB_TEXEL', 'PER_TILE_SUB_TEXEL', 'GLOW_PER_TILE_SUB_TEXEL'):
        monkeypatch.setattr(memory, name, 0)
    monkeypatch.setattr(memory, 'MARGIN', 1)
    for name, value in others.items():
        monkeypatch.setattr(memory, name, value)


def test_stage_bytes_for_a_fixed_input(small_model):
    assert memory.stage_bytes(inputs()) == {'blender': 1806, 'vertex': 980, 'glow': 2274, 'finish': 1468}


def test_tiles_add_to_the_blender_and_glow_stages_only(small_model):
    tiled = memory.stage_bytes(inputs(whole_page_tiles=True))
    assert tiled == {'blender': 1806 + 2 * 72, 'vertex': 980, 'glow': 2274 + 17 * 72, 'finish': 1468}


def test_margin_scales_each_stage_but_not_the_tile_term(small_model, monkeypatch):
    monkeypatch.setattr(memory, 'MARGIN', 2)
    assert memory.stage_bytes(inputs(whole_page_tiles=True)) == {
        'blender': 2 * 1806 + 2 * 72, 'vertex': 2 * 980, 'glow': 2 * 2274 + 17 * 72, 'finish': 2 * 1468}


def test_glow_stage_is_zero_without_glow(small_model):
    stages = memory.stage_bytes(inputs(glow_sub_texels=0))
    assert stages['glow'] == 0
    assert stages['vertex'] == 500 + 7 * 12


def test_peak_is_the_largest_stage(small_model):
    assert memory.peak_bytes(inputs()) == ('glow', 2274)
    assert memory.peak_bytes(inputs(glow_sub_texels=0)) == ('blender', 1806)


def test_empty_pages_cost_only_the_base_and_the_vertices(small_model):
    stages = memory.stage_bytes(StageInputs(SceneCounts(0, 10), []))
    assert stages == {'blender': 1050, 'vertex': 500, 'glow': 0, 'finish': 100}


def test_a_bigger_level_needs_more_memory_at_the_same_max_resolution():
    small = memory.peak_bytes(StageInputs(SceneCounts(10_000, 20_000), [(1024, 1024)] * 2))[1]
    big = memory.peak_bytes(StageInputs(SceneCounts(400_000, 900_000), [(1024, 1024)] * 9))[1]
    assert big > small


def test_the_largest_page_decides_the_sub_texel_terms(small_model):
    blender = memory.stage_bytes(StageInputs(SceneCounts(0, 0), [(256, 256), (2048, 1024), (512, 512)]))['blender']
    texels = 256 * 256 + 2048 * 1024 + 512 * 512
    assert blender == 1000 + 10 * 9 * 2048 * 1024 + 3 * texels


def machine(ram, limit=None, ceiling=None):
    return Machine(ram, ram if limit is None else limit, ceiling)


def test_two_levels(small_model):
    peak = memory.peak_bytes(inputs())[1]
    fits = memory.check_bake_memory(inputs(), 2048, machine(peak, ceiling=peak * 2))
    assert (fits.level, fits.fits, fits.message) == ('fits', True, '')
    assert (fits.stage, fits.estimate) == ('glow', peak)
    page_file = memory.check_bake_memory(inputs(), 2048, machine(peak - 1, ceiling=peak * 2))
    assert (page_file.level, page_file.fits) == ('page_file', False)
    oom = memory.check_bake_memory(inputs(), 2048, machine(peak // 2, ceiling=peak - 1))
    assert (oom.level, oom.fits) == ('out_of_memory', False)
    edge = memory.check_bake_memory(inputs(), 2048, machine(peak // 2, ceiling=peak))
    assert edge.level == 'page_file'


def test_an_unknown_ceiling_falls_back_to_the_commit_limit(small_model):
    peak = memory.peak_bytes(inputs())[1]
    assert memory.check_bake_memory(inputs(), 2048, machine(peak // 2, limit=peak)).level == 'page_file'
    assert memory.check_bake_memory(inputs(), 2048, machine(peak // 2, limit=peak - 1)).level == 'out_of_memory'


def test_unknown_machine_does_not_warn(caplog):
    with caplog.at_level(logging.WARNING, logger='lightmapper.memory'):
        result = memory.check_bake_memory(StageInputs(SceneCounts(1, 3), [(8, 8)]), 1024, None)
    assert (result.level, result.message, result.machine) == ('fits', '', None)
    assert [(r.levelname, r.getMessage()) for r in caplog.records] == [
        ('WARNING', "this computer's memory is unknown, so the bake runs without the memory check")]


def test_read_machine_failing_gives_none(monkeypatch):
    def fail():
        raise OSError('memory status unavailable')

    monkeypatch.setattr(memory, '_memory_status', fail)
    assert memory.read_machine() is None


def test_read_machine_reads_this_computer():
    found = memory.read_machine()
    assert found.ram > 0 and found.commit_limit >= found.ram


def test_a_page_file_that_cannot_be_read_leaves_the_ceiling_unknown(monkeypatch):
    def fail():
        raise OSError('registry unavailable')

    monkeypatch.setattr(memory, '_memory_status', lambda: (16 * GB, 24 * GB))
    monkeypatch.setattr(memory, '_paging_files', fail)
    assert memory.read_machine() == Machine(16 * GB, 24 * GB, None)


SYSTEM_DRIVE = 'C'


@pytest.fixture(autouse=True)
def system_drive(monkeypatch):
    monkeypatch.setenv('SystemDrive', SYSTEM_DRIVE + ':')


@pytest.mark.parametrize('value, want', [
    (['?:\\pagefile.sys'], [PagingFile(SYSTEM_DRIVE, None, None)]),
    (['C:\\pagefile.sys 0 0'], [PagingFile('C', None, None)]),
    (['C:\\pagefile.sys 16384 32768'], [PagingFile('C', 16384, 32768)]),
    ([], []),
    (['C:\\pagefile.sys 0 0', 'D:\\pagefile.sys 4096 8192'],
     [PagingFile('C', None, None), PagingFile('D', 4096, 8192)]),
])
def test_parse_paging_files(value, want):
    assert memory.parse_paging_files(value) == want


def read_with(monkeypatch, files, ram=16 * GB, limit=24 * GB, free=1000 * GB):
    monkeypatch.setattr(memory, '_memory_status', lambda: (ram, limit))
    monkeypatch.setattr(memory, '_paging_files', lambda: files)
    monkeypatch.setattr(memory, '_free_on', lambda drive: free)
    return memory.read_machine()


def test_fixed_page_file_has_no_room_to_grow(monkeypatch):
    # (the current page file is 8 GB and the fixed size says 8 GB)
    found = read_with(monkeypatch, ['C:\\pagefile.sys 8192 8192'], limit=16 * GB + 8192 * 2**20)
    assert found.commit_ceiling == found.commit_limit


def test_fixed_page_file_ceiling_is_ram_plus_its_maximum(monkeypatch):
    found = read_with(monkeypatch, ['C:\\pagefile.sys 4096 32768'], limit=16 * GB + 4096 * 2**20)
    assert found.commit_ceiling == 16 * GB + 32768 * 2**20 > found.commit_limit


def test_system_managed_page_file_grows_to_three_times_ram_or_the_free_disk(monkeypatch):
    room = read_with(monkeypatch, ['?:\\pagefile.sys'], ram=16 * GB, limit=24 * GB, free=1000 * GB)
    assert room.commit_ceiling == 16 * GB + 3 * 16 * GB
    tight = read_with(monkeypatch, ['?:\\pagefile.sys'], ram=16 * GB, limit=24 * GB, free=20 * GB)
    assert tight.commit_ceiling == 16 * GB + 20 * GB
    full = read_with(monkeypatch, ['?:\\pagefile.sys'], ram=16 * GB, limit=24 * GB, free=1 * GB)
    assert full.commit_ceiling == 24 * GB
    # a peak between the limit and the ceiling is the page-file level, not out of memory
    inputs_ = StageInputs(SceneCounts(0, 0), [(2048, 2048)] * 2)
    monkeypatch.setattr(memory, 'stage_bytes', lambda _inputs: {'blender': 40 * GB, 'vertex': 0, 'glow': 0, 'finish': 0})
    assert memory.check_bake_memory(inputs_, 2048, room).level == 'page_file'
    assert memory.check_bake_memory(inputs_, 2048, tight).level == 'out_of_memory'


def test_two_page_files_add_their_reach(monkeypatch):
    found = read_with(monkeypatch, ['C:\\pagefile.sys 0 0', 'D:\\pagefile.sys 4096 8192'], ram=16 * GB,
                      limit=24 * GB, free=1000 * GB)
    assert found.commit_ceiling == 16 * GB + 3 * 16 * GB + 8192 * 2**20


def test_no_page_file_means_ram_is_the_ceiling(monkeypatch):
    assert read_with(monkeypatch, [], limit=15 * GB).commit_ceiling == 16 * GB


def test_no_page_file_list_on_another_system_leaves_the_ceiling_unknown(monkeypatch):
    assert read_with(monkeypatch, None).commit_ceiling is None


def test_whole_page_tiles_use_the_blender_stage_against_ram(small_model):
    pages = [(2048, 2048)]
    base = StageInputs(SceneCounts(0, 0), pages)
    need = memory.stage_bytes(replace(base, whole_page_tiles=True))['blender']
    assert memory.whole_page_tiles(base, machine(need))
    assert not memory.whole_page_tiles(base, machine(need - 1))
    assert not memory.whole_page_tiles(StageInputs(SceneCounts(0, 0), [(4096, 4096)]), machine(10**15))
    assert not memory.whole_page_tiles(base, None)
    assert not memory.whole_page_tiles(StageInputs(SceneCounts(0, 0), []), machine(10**15))


def test_messages_exact(monkeypatch):
    only(monkeypatch, 'blender', BLENDER_BASE=20 * GB)
    pages = StageInputs(SceneCounts(0, 0), [(1024, 1024)])
    page_file = memory.check_bake_memory(pages, 1024, Machine(16 * GB, 16 * GB, 32 * GB))
    assert page_file.message == (
        "This bake needs about 20 GB at its peak (baking in Blender), more than this computer's 16 GB of memory. "
        'It will lean on the page file and slow down.')
    oom = memory.check_bake_memory(pages, 1024, Machine(8 * GB, 8 * GB, 10 * GB))
    assert oom.message == (
        "This bake needs about 20 GB at its peak (baking in Blender), more than this computer's memory and page "
        'file can hold (10 GB). It will likely run out of memory. Try a lower texel density, or close other programs.')
    assert memory.out_of_memory_message(pages, 1024, Machine(8 * GB, 8 * GB, 10 * GB)) == oom.message


def test_message_suggests_a_fitting_max_resolution(monkeypatch):
    only(monkeypatch, 'blender', BLENDER_PER_SUB_TEXEL=1000)
    pages = StageInputs(SceneCounts(0, 0), [(4096, 4096)])
    text = memory.out_of_memory_message(pages, 4096, Machine(10 * GB, 10 * GB, 12 * GB))
    assert text == (
        "This bake needs about 151 GB at its peak (baking in Blender), more than this computer's memory and page "
        'file can hold (12 GB). It will likely run out of memory. Try Max resolution 1024, or close other programs. '
        'If the level then needs too many pages or still does not fit, lower the texel density a little.')


def test_message_without_a_machine_leaves_out_the_limit(monkeypatch):
    only(monkeypatch, 'blender', BLENDER_PER_SUB_TEXEL=1000)
    text = memory.out_of_memory_message(StageInputs(SceneCounts(0, 0), [(4096, 2048)]), 4096, None)
    assert 'page file can hold. It will' in text and 'Try Max resolution 2048' in text


def test_the_stage_label_is_lower_cased_mid_sentence(monkeypatch):
    only(monkeypatch, 'vertex', VERTEX_BASE=20 * GB)
    text = memory.out_of_memory_message(StageInputs(SceneCounts(0, 0), [(256, 256)]), 256, Machine(8 * GB, 8 * GB, 9 * GB))
    assert '(lighting the vertices)' in text


def test_glow_peak_names_traditional(monkeypatch):
    only(monkeypatch, 'glow', GLOW_PER_GLOW_SUB_TEXEL=100)
    glow = StageInputs(SceneCounts(0, 0), [(512, 512)], glow_sub_texels=10**9)
    text = memory.out_of_memory_message(glow, 512, Machine(8 * GB, 8 * GB, 9 * GB))
    assert '(baking light from ignore-normals surfaces)' in text
    assert text.endswith(' Or set "ignore_normals_mode" to "tool" in the --settings file.')
    other = memory.out_of_memory_message(replace(glow, glow_sub_texels=0, counts=SceneCounts(0, 10**9)), 512,
                                         Machine(8 * GB, 8 * GB, 9 * GB))
    assert 'ignore_normals_mode' not in other


def test_suggestion_keeps_the_glow_total(monkeypatch):
    # (the flagged pages repack into more, smaller pages, so their total stays about the same)
    only(monkeypatch, 'glow', GLOW_PER_GLOW_SUB_TEXEL=1000)
    glow = StageInputs(SceneCounts(0, 0), [(2048, 2048)], glow_sub_texels=10**6)
    assert memory.suggest_page_max(glow, 10**9) == 1024
    assert memory.suggest_page_max(glow, 10**9 - 1) is None


def test_suggestion_for_a_level_of_several_full_size_pages():
    level = StageInputs(SceneCounts(38951, 106283), [(2048, 2048)] * 2, glow_sub_texels=75_497_472)
    assert memory.peak_bytes(level)[1] > 12 * GB
    assert memory.suggest_page_max(level, 12 * GB) == 1024
    smaller = replace(level, pages=[(1024, 1024)] * 2)
    assert memory.peak_bytes(smaller)[1] < 12 * GB


def test_suggestion_keeps_the_total_page_texels(monkeypatch):
    only(monkeypatch, 'vertex', VERTEX_PER_PAGE_TEXEL=1000)
    level = StageInputs(SceneCounts(0, 0), [(2048, 2048)] * 4)
    # (the vertex stage follows the page texels, which a smaller Max resolution does not shrink)
    assert memory.suggest_page_max(level, 4 * 2048 * 2048 * 1000 - 1) is None
    assert memory.suggest_page_max(level, 4 * 2048 * 2048 * 1000) == 1024


def test_suggestion_without_pages_or_a_fitting_side():
    assert memory.suggest_page_max(StageInputs(SceneCounts(0, 0), []), None) is None
    assert memory.suggest_page_max(StageInputs(SceneCounts(0, 0), [(256, 256)]), None) is None


def test_format_gb():
    assert [memory.format_gb(n) for n in (9_840_000_000, 10_400_000_000, 25_462_081_056)] == ['9.8', '10', '25']


def test_summary_ceiling_is_null_when_unknown(small_model):
    check = memory.check_bake_memory(inputs(), 2048, Machine(2000, 2500, None))
    assert memory.summary(inputs(), check)['ceiling_gb'] is None


def test_glow_tiles_against_the_free_memory(small_model):
    base = StageInputs(SceneCounts(0, 0), [(2048, 2048)], glow_sub_texels=100)
    need = memory.stage_bytes(replace(base, whole_page_tiles=True))['glow']
    assert memory.glow_whole_page_tiles(base, need)
    assert not memory.glow_whole_page_tiles(base, need - 1)
    assert not memory.glow_whole_page_tiles(base, None)
    assert not memory.glow_whole_page_tiles(replace(base, pages=[(4096, 4096)]), 10**15)


def test_free_physical_bytes_reads_this_computer():
    free = memory.free_physical_bytes()
    assert isinstance(free, int) and free > 0


def test_summary(small_model):
    check = memory.check_bake_memory(inputs(), 2048, Machine(2000, 2500, 3000))
    assert memory.summary(inputs(), check) == {
        'peak_gb': 0.0, 'stage': 'glow', 'level': 'page_file',
        'stages_gb': {'blender': 0.0, 'vertex': 0.0, 'glow': 0.0, 'finish': 0.0}, 'ram_gb': 0.0, 'ceiling_gb': 0.0}
    assert memory.summary(inputs(), memory.check_bake_memory(inputs(), 2048, None))['ram_gb'] is None


def test_flagged_sub_texels_count_the_pages_with_a_flagged_triangle():
    pages = [(2, 2), (4, 2), (8, 8)]
    assert memory.flagged_sub_texels([0, 0, 2, 2], [True, False, False, True], pages) == 9 * 4 + 9 * 8 * 8
    assert memory.flagged_sub_texels([0, 1], [True, True], pages) == 9 * 4 + 9 * 8
    assert memory.flagged_sub_texels([0, 1], [False, False], pages) == 0
    assert memory.flagged_sub_texels([], None, pages) == 0


def test_scene_counts_from_npz_match_the_scene(tmp_path):
    scene = make_scene()
    write_job(tmp_path, scene, [(16, 8)], [], {}, Settings(device='cpu'))
    assert SceneCounts.from_scene(scene) == SceneCounts(2, 6)
    assert SceneCounts.from_npz(tmp_path / 'scene.npz') == SceneCounts.from_scene(scene)


def test_scene_counts_are_frozen():
    with pytest.raises(FrozenInstanceError):
        SceneCounts(1, 2).vertices = 0


DATA = Path(__file__).parent / 'data' / 'memory_peaks.csv'
# (finish is never the peak stage, so its looseness never moves a warning; the vertex fit is held tight on the rows
# that decide warnings, which leaves its two small rows looser)
UPPER = {'vertex': 1.35, 'finish': 1.6}


def measured_rows():
    with open(DATA, newline='') as file:
        return [pytest.param(row, id=f"{row['case']}-{row['stage']}-{row['adaptive']}-{row['whole_page_tiles']}")
                for row in csv.DictReader(file)]


@pytest.mark.parametrize('row', measured_rows())
def test_estimate_covers_the_measured_peaks(row):
    pages = [tuple(int(side) for side in page.split('x')) for page in row['pages'].split()]
    stage_inputs = StageInputs(SceneCounts(int(row['triangles']), int(row['vertices'])), pages, 'gpu',
                               int(row['glow_sub_texels']), row['whole_page_tiles'] == '1')
    estimate = memory.stage_bytes(stage_inputs)[row['stage']]
    measured = int(row['tree_bytes'])
    assert measured <= estimate <= UPPER.get(row['stage'], 1.3) * measured


def test_the_measured_rows_include_a_glow_level_and_every_stage():
    rows = [row.values[0] for row in measured_rows()]
    assert {row['stage'] for row in rows} == {'blender', 'vertex', 'glow', 'finish'}
    assert any(int(row['glow_sub_texels']) > 0 and row['stage'] == 'blender' for row in rows)
