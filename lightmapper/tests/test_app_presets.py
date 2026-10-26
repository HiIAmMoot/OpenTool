from dataclasses import replace

import pytest

from lightmapper.settings import Settings
from opentool.presets import build_settings, parse_advanced


def test_normal_is_todays_defaults():
    s = build_settings('Normal', {}, 'b.exe')
    assert replace(s, blender=Settings().blender, timeout_s=7200) == Settings()


def test_app_settings_have_no_timeout():
    assert build_settings('Normal', {}, 'b').timeout_s is None


def test_draft_and_high_values():
    d, h = build_settings('Draft', {}, 'b'), build_settings('High', {}, 'b')
    assert (d.samples, d.vertex_rays, d.sky_grid, d.texels_per_unit) == (64, 1024, 2, 2.0)
    assert (h.texels_per_unit, h.samples) == (4.0, 256)


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
