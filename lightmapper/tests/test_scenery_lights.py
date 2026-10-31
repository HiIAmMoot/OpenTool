"""Check the light record against what tool reads from a light tag."""

import math

import numpy as np
import pytest

from conftest import HCEEK_TAGS as TAGS
from lightmapper.formats.lights import (LightDefinition, bakes, light_kind, ramp, read_light, reach,
                                        scene_light)
from lightmapper.formats.scenery import gather_lights, gather_occluders, quaternion_matrix
from lightmapper.tagio.tagfile import new_tag, read_tag, write_tag
from test_scenery import add, write_model


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


S = math.sqrt(.5)
IDENTITY = (0, 0, 0, 1)
TURN_Z = (0, 0, S, S)
ONE_TRIANGLE = [[(0, [(0, 0, 0), (1, 0, 0), (0, 1, 0)], [0, 1, 2])]]


def write_light(tags, name, **fields):
    tag = new_tag('ligh')
    values = {'radiosity.intensity': 1.0, 'radiosity.color': (1, 1, 1)}
    values.update({k.replace('__', '.'): v for k, v in fields.items()})
    for key, value in values.items():
        tag.root.set(key, value)
    write_tag(tag, tags / (name + '.light'))


def lamp_scene(tmp_path, markers=(), nodes=((-1, (0, 0, 0), IDENTITY),), attachments=(('lamp', 'light'),),
               placements=(dict(placement__position=(0, 0, 0)),), origin_offset=(0, 0, 0), lods=ONE_TRIANGLE):
    tags = tmp_path / 'tags'
    tags.mkdir()
    write_model(tags, 'm', lods, nodes=list(nodes), markers=list(markers))
    write_light(tags, 'lamp', radiosity__intensity=2.0)
    scenery = new_tag('scen')
    scenery.root.set_reference('model', 'mode', 'm')
    scenery.root.set('origin_offset', origin_offset)
    for path, marker in attachments:
        attachment = add(scenery.root, 'attachments', marker=marker.encode())
        attachment.set_reference('type', 'ligh', path)
    write_tag(scenery, tags / 's.scenery')
    scn = new_tag('scnr')
    add(scn.root, 'scenery_palette').set_reference('name', 'scen', 's')
    for kw in placements:
        add(scn.root, 'scenery', **kw)
    return scn, tags


def test_halo_quaternion_turns_forward_the_engines_way():
    s = c = math.sqrt(.5)
    np.testing.assert_allclose(quaternion_matrix((0, 0, s, c))[:, 0], (0, -1, 0), atol=1e-12)
    np.testing.assert_allclose(quaternion_matrix((0, s, 0, c))[:, 0], (0, 0, 1), atol=1e-12)


def test_a_quaternion_that_is_not_unit_length_turns_like_its_unit_one():
    s = c = math.sqrt(.5)
    np.testing.assert_allclose(quaternion_matrix((0, 0, 3 * s, 3 * c)), quaternion_matrix((0, 0, s, c)), atol=1e-12)


def test_light_is_placed_at_its_marker_with_the_origin_offset_at_yaw_90(tmp_path):
    scn, tags = lamp_scene(tmp_path, markers=[(0, '', 'light', 0, (1, 0, 0), IDENTITY)], origin_offset=(0, 0, 2),
                           placements=[dict(placement__position=(10, 20, 30), placement__rotation=(math.pi / 2, 0, 0))])
    lights, counts = gather_lights(scn, tags, 0)
    np.testing.assert_allclose(lights[0].position, (10, 21, 32), atol=1e-9)
    assert counts['attached'] == counts['placed'] == 1


def test_marker_composes_its_node_chain(tmp_path):
    scn, tags = lamp_scene(tmp_path, nodes=[(-1, (0, 0, 1), IDENTITY), (0, (1, 0, 0), TURN_Z)],
                           markers=[(0, '', 'light', 1, (1, 0, 0), IDENTITY)])
    lights, _ = gather_lights(scn, tags, 0)
    np.testing.assert_allclose(lights[0].position, (1, -1, 1), atol=1e-6)
    np.testing.assert_allclose(lights[0].axis, (0, -1, 0), atol=1e-6)


def test_missing_marker_falls_back_to_node_0_with_one_warning(tmp_path, caplog):
    scn, tags = lamp_scene(tmp_path, nodes=[(-1, (0, 0, 1), IDENTITY)], attachments=[('lamp', 'nope')],
                           placements=[dict(placement__position=(5, 0, 0)), dict(placement__position=(9, 0, 0))])
    lights, counts = gather_lights(scn, tags, 0)
    assert [tuple(l.position) for l in lights] == [(5, 0, 1), (9, 0, 1)]
    assert counts['missing_marker'] == 2 and caplog.text.count('nope') == 1


def test_a_marker_missing_in_two_cases_warns_once(tmp_path, caplog):
    scn, tags = lamp_scene(tmp_path, attachments=[('lamp', 'Nope'), ('lamp', 'nope')],
                           placements=[dict(placement__position=(5, 0, 0))])
    _, counts = gather_lights(scn, tags, 0)
    assert counts['missing_marker'] == 2 and caplog.text.lower().count('nope') == 1


def test_empty_marker_name_is_node_0(tmp_path, caplog):
    scn, tags = lamp_scene(tmp_path, nodes=[(-1, (0, 0, 1), IDENTITY)], attachments=[('lamp', '')],
                           placements=[dict(placement__position=(5, 0, 0))])
    lights, counts = gather_lights(scn, tags, 0)
    assert tuple(lights[0].position) == (5, 0, 1)
    assert counts['missing_marker'] == 0 and not caplog.records


def test_marker_name_matches_without_case(tmp_path):
    scn, tags = lamp_scene(tmp_path, markers=[(0, '', 'LiGhT', 0, (1, 2, 3), IDENTITY)],
                           attachments=[('lamp', 'lIGHT')])
    lights, counts = gather_lights(scn, tags, 0)
    np.testing.assert_allclose(lights[0].position, (1, 2, 3))
    assert counts['missing_marker'] == 0


def test_marker_comes_from_the_chosen_permutation(tmp_path):
    scn, tags = lamp_scene(tmp_path, markers=[(0, 'p', 'light', 0, (1, 0, 0), IDENTITY),
                                              (0, 'p 2', 'light', 0, (0, 5, 0), IDENTITY)],
                           placements=[dict(placement__desired_permutation=2), dict()])
    lights, _ = gather_lights(scn, tags, 0)
    np.testing.assert_allclose([l.position for l in lights], [(0, 5, 0), (1, 0, 0)])


def test_first_marker_instance_in_region_order_wins(tmp_path):
    scn, tags = lamp_scene(tmp_path, markers=[(0, '', 'light', 0, (1, 0, 0), IDENTITY),
                                              (1, '', 'light', 0, (0, 5, 0), IDENTITY)])
    lights, _ = gather_lights(scn, tags, 0)
    np.testing.assert_allclose(lights[0].position, (1, 0, 0))


def test_dynamic_and_dark_lights_are_counted_not_placed(tmp_path):
    scn, tags = lamp_scene(tmp_path, attachments=[('lamp', ''), ('dyn', ''), ('dark', ''), ('black', ''), ('gone', '')])
    write_light(tags, 'dyn', flags=1)
    write_light(tags, 'dark', radiosity__intensity=0.0)
    write_light(tags, 'black', radiosity__color=(0, 0, 0))
    lights, counts = gather_lights(scn, tags, 0)
    assert len(lights) == 1
    assert counts == dict(attached=5, dynamic=1, dark=2, missing_light=1, missing_model=0, missing_marker=0, placed=1)


def test_missing_model_is_counted(tmp_path):
    scn, tags = lamp_scene(tmp_path)
    (tags / 'm.gbxmodel').unlink()
    lights, counts = gather_lights(scn, tags, 0)
    assert not lights and counts['missing_model'] == 1


def test_non_light_attachments_are_ignored(tmp_path):
    scn, tags = lamp_scene(tmp_path, attachments=[])
    scenery = read_tag(tags / 's.scenery')
    for group, path in (('effe', 'puff'), ('ligh', 'lamp')):
        attachment = add(scenery.root, 'attachments')
        attachment.set_reference('type', group, path)
    write_tag(scenery, tags / 's.scenery')
    lights, counts = gather_lights(scn, tags, 0)
    assert len(lights) == 1 and counts['attached'] == 1


def test_lights_follow_the_placement_filters(tmp_path):
    scn, tags = lamp_scene(tmp_path, placements=[
        dict(placement__not_placed=1), dict(bsp_indices=2), dict(bsp_indices=1), dict(type=5), dict()])
    lights, _ = gather_lights(scn, tags, 0)
    assert [l.owner for l in lights] == [2, 4]


def test_scenery_without_triangles_still_carries_its_light(tmp_path):
    scn, tags = lamp_scene(tmp_path, lods=[[]], markers=[(0, '', 'light', 0, (1, 0, 0), IDENTITY)])
    lights, _ = gather_lights(scn, tags, 0)
    np.testing.assert_allclose(lights[0].position, (1, 0, 0))
    assert len(gather_occluders(scn, tags, 0).tris) == 0


def test_occluder_triangles_carry_their_placement_index(tmp_path):
    scn, tags = lamp_scene(tmp_path, lods=[[(0, [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)], [0, 1, 2, 3])]],
                           placements=[dict(placement__not_placed=1), dict(), dict()])
    occ = gather_occluders(scn, tags, 0)
    assert occ.tri_owner.dtype == np.int32 and occ.tri_owner.tolist() == [1, 1, 2, 2]


@pytest.mark.ek
def test_ctorch_light_sits_on_its_marker(tmp_path):
    scn = new_tag('scnr')
    add(scn.root, 'scenery_palette').set_reference('name', 'scen', r'levels\b40\scenery\b40_ctorch\ctorch')
    add(scn.root, 'scenery', placement__position=(0, 0, 0))
    lights, _ = gather_lights(scn, TAGS, 0)
    np.testing.assert_allclose(lights[0].position, (0.040749, 0.004020, 0.702933), atol=1e-5)
    assert lights[0].kind == 2 and abs(float(lights[0].reach) - 9.8438) < 1e-3
