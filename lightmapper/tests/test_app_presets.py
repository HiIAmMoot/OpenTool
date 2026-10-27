from dataclasses import replace

import pytest

from lightmapper.settings import Settings
from opentool.presets import ADVANCED, CUSTOM, LABELS, PAGE_MAX_CHOICES, PRESETS, PRESET_HINTS, build_settings, parse_advanced


def test_normal_is_todays_defaults():
    s = build_settings('Normal', {}, 'b.exe')
    assert replace(s, blender=Settings().blender, timeout_s=7200) == replace(Settings(), page_max=512, vertex_rays=1024)


def test_app_settings_have_no_timeout():
    assert build_settings('Normal', {}, 'b').timeout_s is None


def test_draft_and_high_values():
    d, h = build_settings('Draft', {}, 'b'), build_settings('High', {}, 'b')
    assert {key: getattr(d, key) for key in ADVANCED} == dict(texels_per_unit=1.0, samples=64, vertex_rays=256, sky_grid=2, bounces=1, page_max=512)
    assert {key: getattr(h, key) for key in ADVANCED} == dict(texels_per_unit=4.0, samples=512, vertex_rays=4096, sky_grid=4, bounces=1, page_max=1024)


def test_exact_preset_interface_values():
    assert PRESETS == {'Draft': dict(texels_per_unit=1.0, samples=64, vertex_rays=256, sky_grid=2, bounces=1, page_max=512), 'Normal': dict(texels_per_unit=2.0, samples=256, vertex_rays=1024, sky_grid=4, bounces=1, page_max=512), 'High': dict(texels_per_unit=4.0, samples=512, vertex_rays=4096, sky_grid=4, bounces=1, page_max=1024)}
    assert ADVANCED == ('texels_per_unit', 'samples', 'vertex_rays', 'sky_grid', 'bounces', 'page_max')
    assert PAGE_MAX_CHOICES == (256, 512, 1024, 2048, 4096)
    assert LABELS == dict(texels_per_unit='Texel density', samples='Rays per texel', vertex_rays='Rays per vertex', sky_grid='Suns per sky light', bounces='Bounces', page_max='Max resolution')
    assert CUSTOM == 'Custom' and PRESET_HINTS['Custom'] == 'Your own values from Advanced.'


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
