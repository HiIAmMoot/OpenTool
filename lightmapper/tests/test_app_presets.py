from dataclasses import replace

import pytest

from lightmapper import memory
from lightmapper.settings import Settings
from opentool import devices, presets
from opentool.presets import ADVANCED, CPU_PRESETS, CUSTOM, DEVICES, GPU_NOTE, LABELS, PAGE_MAX_CHOICES, PAGE_MAX_HINT, page_max_hint, PRESETS, PRESET_HINTS, PRESET_NAMES, build_settings, parse_advanced


def test_page_max_choices_are_the_memory_page_sides():
    assert PAGE_MAX_CHOICES == memory.PAGE_SIDES == (256, 512, 1024, 2048, 4096)


def test_gpu_preset_values():
    assert PRESETS == {
        'Draft': dict(texels_per_unit=1.0, samples=256, vertex_rays=256, sky_grid=2, bounces=1, page_max=512, ignore_normals_mode='tool'),
        'Normal': dict(texels_per_unit=2.0, samples=1024, vertex_rays=1024, sky_grid=4, bounces=1, page_max=512, ignore_normals_mode='tool'),
        'High': dict(texels_per_unit=4.0, samples=2048, vertex_rays=4096, sky_grid=4, bounces=1, page_max=1024, ignore_normals_mode='tool'),
        'Ultra': dict(texels_per_unit=8.0, samples=2048, vertex_rays=16384, sky_grid=4, bounces=1, page_max=2048, ignore_normals_mode='enhanced')}
    assert PRESET_NAMES == ('Draft', 'Normal', 'High', 'Ultra')


def test_cpu_preset_values():
    assert CPU_PRESETS == {'Draft': dict(samples=64), 'Normal': dict(samples=256), 'High': dict(samples=512),
                           'Ultra': dict(samples=512)}


@pytest.mark.parametrize('name, mode', [('Draft', 'tool'), ('Normal', 'tool'), ('High', 'tool'), ('Ultra', 'enhanced')])
def test_presets_carry_ignore_normals(name, mode):
    assert build_settings(name, {}, 'b').ignore_normals_mode == mode


def test_ultra_hint_and_gpu_note():
    assert PRESET_HINTS['Ultra'] == 'Sharpest. Slow and memory-hungry; for a final bake.'
    assert GPU_NOTE == 'Ray counts are for a GPU. On the CPU, the presets use fewer rays per texel.'
    assert devices.CPU_HINT == 'Bakes on the CPU, with fewer rays per texel for the presets.'


def test_custom_bakes_traditional_by_default():
    values = {field: PRESETS['Ultra'][field] for field in ADVANCED}
    assert build_settings(CUSTOM, values, 'b').ignore_normals_mode == 'tool'


def test_normal_is_the_defaults_with_gpu_counts():
    s = build_settings('Normal', {}, 'b.exe')
    assert replace(s, blender=Settings().blender, timeout_s=7200) == replace(
        Settings(), page_max=512, samples=1024, vertex_rays=1024, cpu_samples=256, cpu_vertex_rays=None)


def test_named_preset_with_the_apps_filled_fields_keeps_its_cpu_counts():
    s = build_settings('High', dict(PRESETS['High']), 'b')
    assert (s.samples, s.vertex_rays, s.cpu_samples, s.cpu_vertex_rays) == (2048, 4096, 512, None)


def test_an_overridden_count_is_used_on_every_device():
    s = build_settings('Normal', {'samples': 300}, 'b')
    assert (s.samples, s.cpu_samples, s.vertex_rays, s.cpu_vertex_rays) == (300, None, 1024, None)


def test_custom_keeps_its_values_on_every_device():
    s = build_settings(CUSTOM, {field: PRESETS['Draft'][field] for field in ADVANCED}, 'b')
    assert (s.cpu_samples, s.cpu_vertex_rays) == (None, None)


def test_device_reaches_the_settings():
    assert build_settings('Normal', {}, 'b').device == 'auto'
    assert build_settings('Normal', {}, 'b', device='cpu').device == 'cpu'


def test_app_settings_have_no_timeout():
    assert build_settings('Normal', {}, 'b').timeout_s is None


@pytest.mark.parametrize('preset, samples, vertex_rays, cpu_samples', [
    ('Draft', 256, 256, 64),
    ('Normal', 1024, 1024, 256),
    ('High', 2048, 4096, 512),
    ('Ultra', 2048, 16384, 512),
])
def test_named_preset_counts(preset, samples, vertex_rays, cpu_samples):
    s = build_settings(preset, {}, 'b')
    assert (s.samples, s.vertex_rays, s.cpu_samples, s.cpu_vertex_rays) == (samples, vertex_rays, cpu_samples, None)
    assert s.resolved_cpu_vertex_rays() == vertex_rays


def test_an_overridden_vertex_count_is_used_on_every_device():
    s = build_settings('High', {'vertex_rays': 300}, 'b')
    assert (s.samples, s.cpu_samples, s.vertex_rays, s.cpu_vertex_rays) == (2048, 512, 300, None)


def test_exact_preset_interface_values():
    assert DEVICES == (('Auto', 'auto'), ('NVIDIA', 'nvidia'), ('AMD', 'amd'), ('CPU only', 'cpu'))
    assert GPU_NOTE == 'Ray counts are for a GPU. On the CPU, the presets use fewer rays per texel.'
    assert ADVANCED == ('texels_per_unit', 'samples', 'vertex_rays', 'sky_grid', 'bounces', 'page_max')
    assert PAGE_MAX_CHOICES == (256, 512, 1024, 2048, 4096)
    assert LABELS == dict(texels_per_unit='Texel density', samples='Rays per texel', vertex_rays='Rays per vertex', sky_grid='Suns per sky light', bounces='Bounces', page_max='Max resolution')
    assert CUSTOM == 'Custom' and PRESET_HINTS['Custom'] == 'Your own values from Advanced, on any device.'


def test_custom_requires_all_values():
    with pytest.raises(ValueError):
        build_settings(CUSTOM, {'texels_per_unit': 3.0}, 'b')


def test_advanced_overrides_the_preset():
    assert build_settings('High', {'texels_per_unit': 3.0}, 'b').texels_per_unit == 3.0


def test_parse_advanced_blank_means_preset():
    assert parse_advanced({'texels_per_unit': ' ', 'bounces': ''}) == {}


def test_parse_advanced_values():
    assert parse_advanced({'texels_per_unit': '3', 'bounces': '2', 'page_max': '512'}) == {'texels_per_unit': 3.0, 'bounces': 2, 'page_max': 512}


@pytest.mark.parametrize('field, text', [('texels_per_unit', 'abc'), ('texels_per_unit', '0'), ('bounces', '9'), ('bounces', '1.5'), ('page_max', '300')])
def test_parse_advanced_rejects(field, text):
    with pytest.raises(ValueError):
        parse_advanced({field: text})


@pytest.mark.parametrize('field,text', [('samples','0'),('samples','4097'),('vertex_rays','0'),('vertex_rays','65537'),('sky_grid','0'),('sky_grid','9')])
def test_parse_advanced_rejects_new_ranges(field, text):
    with pytest.raises(ValueError, match=LABELS[field]):
        parse_advanced({field: text})


@pytest.mark.parametrize('value,shown', [('256', False), ('512', False), ('1024', False), ('2048', True), ('4096', True), ('', False), ('x', False)])
def test_page_max_hint(value, shown):
    assert page_max_hint(value) == (PAGE_MAX_HINT if shown else '')


def test_density_choices():
    assert presets.DENSITY_CHOICES == (1, 2, 4, 8, 12, 16, 24, 32)


def test_density_up_to_32_is_valid():
    assert presets.parse_advanced({'texels_per_unit': '32'}) == {'texels_per_unit': 32.0}
    with pytest.raises(ValueError, match='0.25 to 32'):
        presets.parse_advanced({'texels_per_unit': '33'})


def test_density_values_add_a_custom_entry():
    assert presets.density_values('8') == ['1', '2', '4', '8', '12', '16', '24', '32']
    assert presets.density_values('3') == ['1', '2', '4', '8', '12', '16', '24', '32', '3']


def test_row_hint():
    assert presets.row_hint('page_max', '2048', False) == presets.PAGE_MAX_HINT
    assert presets.row_hint('page_max', '512', True) == ''
    assert presets.row_hint('ignore_normals_mode', 'Enhanced', False) == presets.ENHANCED_HINT
    assert presets.row_hint('ignore_normals_mode', 'Traditional', False) == ''
    assert presets.row_hint('samples', '1024', True) == presets.ADAPTIVE_HINT
    assert presets.row_hint('samples', '1024', False) == ''
    for name in ('texels_per_unit', 'vertex_rays', 'sky_grid', 'bounces', 'adaptive_sampling', 'lightmap_32bit'):
        assert presets.row_hint(name, 'Enhanced', True) == ''
    assert presets.ENHANCED_HINT == 'the bake can take up to twice as long, and needs more memory'
    assert presets.ADAPTIVE_HINT == 'the most rays per texel while adaptive sampling is on'
