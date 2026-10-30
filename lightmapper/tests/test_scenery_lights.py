"""Check the light record against what tool reads from a light tag."""

import math

import numpy as np
import pytest

from conftest import HCEEK_TAGS as TAGS
from lightmapper.formats.lights import (LightDefinition, bakes, light_kind, ramp, read_light, reach,
                                        scene_light)
from lightmapper.tagio.tagfile import new_tag, write_tag


@pytest.mark.ek
def test_light_layout_reads_the_radiosity_fields():
    light = read_light(TAGS, r'levels\b40\scenery\b40_ctorch\ctorch', {})
    assert (light.dynamic, light.falloff, light.cutoff) == (False, 0.0, 0.0)
    assert light.intensity == np.float32(0.38)
    np.testing.assert_allclose(light.colour, (0.160784, 0.839216, 0.772549), atol=1e-6)


def test_light_round_trips_through_a_written_tag(tmp_path):
    tag = new_tag('ligh')
    tag.root.set('flags', 1)
    tag.root.set('radiosity.intensity', 2.5)
    tag.root.set('radiosity.color', (1, .5, 0))
    tag.root.set('shape.falloff_angle', .3)
    tag.root.set('shape.cutoff_angle', .6)
    (tmp_path / 'a').mkdir()
    write_tag(tag, tmp_path / 'a' / 'lamp.light')
    light = read_light(tmp_path, r'a\lamp', {})
    assert light.dynamic is True and light.intensity == 2.5
    assert light.colour == (1.0, .5, 0.0)
    assert light.falloff == float(np.float32(.3)) and light.cutoff == float(np.float32(.6))


def test_a_missing_light_gives_none_once(tmp_path, caplog):
    cache = {}
    assert read_light(tmp_path, r'a\none', cache) is None
    assert read_light(tmp_path, r'a\none', cache) is None
    assert len(caplog.records) == 1


@pytest.mark.parametrize('falloff, cutoff, kind', [
    (0.0, 0.0, 2), (4.0, 3.2, 2), (1.0, 3.2, 3), (0.3, 0.6, 3), (0.3, 0.0, 2), (0.3, -1.0, 2)])
def test_light_kind_follows_tools_rule(falloff, cutoff, kind):
    assert light_kind(falloff, cutoff) == kind


def test_reach_is_tools_float32_distance():
    assert reach(np.float32(0.38)) == np.float32(np.sqrt(np.float64(np.float32(0.38) / np.float32(1 / 255))))
    assert abs(float(reach(1.0)) - 15.9687) < 1e-3


@pytest.mark.parametrize('dynamic, intensity, colour, want', [
    (False, 1.0, (1, 1, 1), True), (True, 1.0, (1, 1, 1), False), (False, 0.0, (1, 1, 1), False),
    (False, -1.0, (1, 1, 1), False), (False, 1.0, (0, 0, 0), False)])
def test_bakes_only_static_lit_lights(dynamic, intensity, colour, want):
    assert bakes(LightDefinition(dynamic, intensity, colour, 0.0, 0.0)) is want


@pytest.mark.parametrize('s, want', [
    (0.8, 0.0), (0.9, (0.9 - math.cos(.6)) / (math.cos(.3) - math.cos(.6))), (0.99, 1.0)])
def test_ramp_is_linear_in_the_cosine(s, want):
    assert ramp(s, math.cos(.6), math.cos(.3)) == pytest.approx(want)


def test_ramp_is_a_step_when_the_falloff_is_wider():
    assert ramp(0.9, math.cos(.6), math.cos(.7)) == 1.0 and ramp(0.8, math.cos(.6), math.cos(.7)) == 0.0


def test_scene_light_carries_tools_record():
    light = scene_light(LightDefinition(False, 2.0, (1, .5, .25), .3, .6), np.zeros(3), np.float64([0, 0, -1]), 7, 't')
    assert light.kind == 3 and light.owner == 7 and np.allclose(light.rgb, (2, 1, .5))
    assert light.cos_cutoff == np.float32(np.cos(np.float64(np.float32(.6))))
