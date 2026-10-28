"""Check lighting against independent tag fields and exact surface-class votes."""

import importlib
import importlib.util
import math

import numpy as np
import pytest

from conftest import HCEEK_TAGS as TAGS
from lightmapper.formats.bsp import BspGeometry, SourceGroup, NO_CLUSTER
from lightmapper.formats.vertices import ENV
from lightmapper.tagio.tagfile import new_element, new_tag, resolve_reference_path, write_tag


def api():
    assert importlib.util.find_spec('lightmapper.gather') is not None, 'lighting API is missing'
    return importlib.import_module('lightmapper.gather')


def child(parent, key):
    return new_element(next(f.layout for f in parent.layout.fields if f.key == key))


@pytest.mark.parametrize('yaw,pitch,want', [
    (0, math.pi / 2, [0, 0, 1]), (math.pi / 2, 0, [0, 1, 0]),
    (math.pi, 0, [-1, 0, 0]), (0, -math.pi / 2, [0, 0, -1]),
])
def test_sky_direction(yaw, pitch, want):
    np.testing.assert_allclose(api().sky_direction(yaw, pitch), want, atol=1e-15)


def geometry(triangles, classes):
    groups = []
    for i, points in enumerate(triangles):
        env = np.zeros(3, dtype=ENV)
        env['position'] = points
        groups.append(SourceGroup(('senv', str(i), 0, 0, -1), None, env,
                                  np.array([[0, 1, 2]]), np.array([i])))
    clusters = [i if c != NO_CLUSTER else NO_CLUSTER for i, c in enumerate(classes)]
    skies = [c if c != NO_CLUSTER else -1 for c in classes]
    return BspGeometry(groups, None, np.array(clusters, dtype=np.int16), np.array(skies, dtype=np.int16))


def test_no_cluster_takes_neighbours_class(caplog):
    a, b, c = (0, 0, 0), (1, 0, 0), (0, 1, 0)
    g = geometry([[a, b, c], [a, b, c], [a, b, c], [a, b, c],
                  [(10, 0, 0), (11, 0, 0), (10, 1, 0)]], [1, 1, -1, NO_CLUSTER, NO_CLUSTER])
    with caplog.at_level('INFO', logger='lightmapper.gather'):
        result = api().surface_classes(g, 2)
    np.testing.assert_array_equal(result, [1, 1, -1, 1, 1])
    assert 'neighbours=1' in caplog.text and 'fallback=1' in caplog.text


def test_neighbours_count_surfaces_not_shared_vertices():
    a, b, c = (0, 0, 0), (1, 0, 0), (0, 1, 0)
    g = geometry([[a, b, c], [a, (2, 0, 0), (3, 0, 0)],
                  [b, (4, 0, 0), (5, 0, 0)], [a, b, c]], [1, -1, -1, NO_CLUSTER])
    np.testing.assert_array_equal(api().surface_classes(g, 2), [1, -1, -1, -1])


def test_neighbour_tie_uses_lowest_class_and_exact_positions():
    a = (0, 0, 0)
    g = geometry([[a, (1, 0, 0), (0, 1, 0)], [a, (2, 0, 0), (0, 2, 0)],
                  [a, (3, 0, 0), (0, 3, 0)],
                  [(0.000001, 0, 0), (4, 0, 0), (0, 4, 0)]], [-1, 1, NO_CLUSTER, NO_CLUSTER])
    np.testing.assert_array_equal(api().surface_classes(g, 2), [-1, 1, -1, 1])


def test_no_outdoor_fallback_and_no_geometry():
    g = geometry([[(0, 0, 0), (1, 0, 0), (0, 1, 0)]], [NO_CLUSTER])
    np.testing.assert_array_equal(api().surface_classes(g, 0), [-1])
    g.groups = []
    np.testing.assert_array_equal(api().surface_classes(g, 0), [-1])
    g.surface_cluster = np.empty(0, dtype=np.int16)
    assert api().surface_classes(g, 0).shape == (0,)


@pytest.mark.parametrize('sky', [-2, 2])
def test_invalid_cluster_sky_refuses_even_unused_cluster(sky):
    g = BspGeometry([], None, np.empty(0, dtype=np.int16), np.array([sky], dtype=np.int16))
    with pytest.raises(ValueError, match=rf'cluster 0.*(?:index|sky) {sky}\b'):
        api().surface_classes(g, 2)


def test_high_valence_at_largest_named_bsp_size():
    n = 140168
    env = np.zeros(3, dtype=ENV)
    env['position'] = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
    group = SourceGroup(('senv', 'grid', 0, 0, -1), None, env,
                        np.tile([0, 1, 2], (n, 1)), np.arange(n))
    clusters = np.full(n, NO_CLUSTER, dtype=np.int16)
    clusters[:n // 2] = 0
    g = BspGeometry([group], None, clusters, np.array([1], dtype=np.int16))
    np.testing.assert_array_equal(api().surface_classes(g, 2), np.ones(n, dtype=np.int16))


def test_no_skies():
    result = api().gather_lighting(new_tag('scnr'), TAGS)
    assert result.suns == [] and result.ambient == {}


@pytest.mark.parametrize('flags,want', [(0, []), (1, [(0,), (1,)]),
                                         (2, [(-1,)]), (3, [(0, -1), (1,)])])
def test_interior_light_classes(tmp_path, flags, want):
    scenario = new_tag('scnr')
    refs = []
    for i in range(2):
        sky = new_tag('sky ')
        sky.root.set('indoor_ambient_radiosity.power', i + 2.0)
        sky.root.set('indoor_ambient_radiosity.color', (0.5, 0.25, 1.0))
        light = child(sky.root, 'lights')
        light.set('radiosity.flags', flags)
        light.set('radiosity.power', 4.0)
        light.set('radiosity.direction', (0.0, 0.0))
        sky.root.set_block('lights', [light])
        write_tag(sky, tmp_path / f'sky{i}.sky')
        ref = child(scenario.root, 'skies')
        ref.set_reference('sky', 'sky ', f'sky{i}')
        refs.append(ref)
    scenario.root.set_block('skies', refs)
    result = api().gather_lighting(scenario, tmp_path, sky_grid=1)
    assert [s.classes for s in result.suns] == want
    np.testing.assert_array_equal(result.ambient[-1], [1, 0.5, 2])


@pytest.mark.parametrize('group,extension', [('senv', 'shader_environment'), ('sgla', 'shader_transparent_glass')])
def test_radiosity_fields_and_simple_parameterization(tmp_path, group, extension):
    shader = new_tag(group)
    for key, value in [('flags', 5), ('detail_level', 2), ('power', 3.0),
                       ('color_of_emitted_light', (0.25, 0.5, 1.0)), ('tint_color', (1.0, 0.5, 0.25))]:
        shader.root.set('radiosity.' + key, value)
    write_tag(shader, tmp_path / ('shader.' + extension))
    result = api().read_radiosity(tmp_path, group, 'shader')
    assert result.flags & api().SIMPLE_PARAMETERIZATION
    assert (result.flags, result.detail_level, result.power) == (5, 2, 3.0)
    np.testing.assert_array_equal(result.emitted, [0.25, 0.5, 1])
    np.testing.assert_array_equal(result.tint, [1, 0.5, 0.25])
    np.testing.assert_array_equal(api().emission(result), [0.75, 1.5, 3])


@pytest.mark.parametrize('group', ['mod2', '', None, 'sky '])
def test_unsupported_shader_refuses_before_resolving(tmp_path, group):
    with pytest.raises(ValueError):
        api().read_radiosity(tmp_path, group, 'missing')


@pytest.mark.parametrize('power', [0.0, -1.0])
def test_emission_zero_power(power):
    module = api()
    np.testing.assert_array_equal(module.emission(module.ShaderRadiosity(
        0, 0, power, np.ones(3), np.ones(3))), np.zeros(3))


def one_light_scenario(tmp_path, yaw, pitch, diameter, power, flags=1):
    scenario = new_tag('scnr')
    sky = new_tag('sky ')
    light = child(sky.root, 'lights')
    light.set('radiosity.flags', flags)
    light.set('radiosity.power', power)
    light.set('radiosity.color', (1.0, 0.5, 0.25))
    light.set('radiosity.direction', (yaw, pitch))
    light.set('radiosity.diameter', diameter)
    sky.root.set_block('lights', [light])
    write_tag(sky, tmp_path / 'sky0.sky')
    ref = child(scenario.root, 'skies')
    ref.set_reference('sky', 'sky ', 'sky0')
    scenario.root.set_block('skies', [ref])
    return scenario


def test_sky_light_becomes_tools_grid_of_suns(tmp_path):
    yaw, pitch, d, power, n = 0.7, 0.4, 0.1, 8.0, 4
    result = api().gather_lighting(one_light_scenario(tmp_path, yaw, pitch, d, power), tmp_path, sky_grid=n)
    assert len(result.suns) == n * n
    # sample (i, j) is the i-th yaw step and the j-th pitch step, in that order
    for i in range(n):
        for j in range(n):
            sun = result.suns[i * n + j]
            y = yaw + d * (2 * i / (n - 1) - 1)
            p = pitch + d * (2 * j / (n - 1) - 1)
            np.testing.assert_allclose(sun.direction,
                [math.cos(y) * math.cos(p), math.sin(y) * math.cos(p), math.sin(p)])
            assert sun.power == pytest.approx(power / n ** 2)
            np.testing.assert_array_equal(sun.color, [1.0, 0.5, 0.25])
            assert sun.classes == (0,)
            assert sun.angle < 0.01


def test_sky_grid_of_one_is_the_lights_own_direction(tmp_path):
    result = api().gather_lighting(one_light_scenario(tmp_path, 0.7, 0.4, 0.1, 8.0), tmp_path, sky_grid=1)
    assert len(result.suns) == 1
    np.testing.assert_allclose(result.suns[0].direction, api().sky_direction(0.7, 0.4))
    assert result.suns[0].power == 8.0


def test_sky_grid_defaults_to_four(tmp_path):
    result = api().gather_lighting(one_light_scenario(tmp_path, 0.7, 0.4, 0.1, 8.0), tmp_path)
    assert len(result.suns) == 16


def radiosity_shader(group_tag, tmp_path, extension, flags=5):
    shader = new_tag(group_tag)
    for key, value in [('flags', flags), ('detail_level', 2), ('power', 3.0),
                       ('color_of_emitted_light', (0.25, 0.5, 1.0)), ('tint_color', (1.0, 0.5, 0.25))]:
        shader.root.set('radiosity.' + key, value)
    file = tmp_path / ('shader.' + extension)
    write_tag(shader, file)
    return file


@pytest.mark.parametrize('group,extension', [
    ('soso', 'shader_model'), ('schi', 'shader_transparent_chicago'),
    ('scex', 'shader_transparent_chicago_extended')])
def test_radiosity_of_the_other_layout_groups(tmp_path, group, extension):
    radiosity_shader(group, tmp_path, extension)
    result = api().read_radiosity(tmp_path, group, 'shader')
    assert (result.flags, result.detail_level, result.power) == (5, 2, 3.0)
    np.testing.assert_array_equal(result.tint, [1, 0.5, 0.25])


@pytest.mark.parametrize('group,extension', [
    ('swat', 'shader_transparent_water'), ('smet', 'shader_transparent_meter'),
    ('spla', 'shader_transparent_plasma'), ('sotr', 'shader_transparent_generic')])
def test_radiosity_of_groups_without_a_layout_is_read_at_the_base_offsets(tmp_path, group, extension):
    # the base fields sit first in every shader group, so a glass tag's bytes stand in for the group's own
    file = radiosity_shader('sgla', tmp_path, extension)
    renamed = bytearray(file.read_bytes())
    renamed[36:40] = group.encode()
    file.write_bytes(bytes(renamed))
    result = api().read_radiosity(tmp_path, group, 'shader')
    assert (result.flags, result.detail_level, result.power) == (5, 2, 3.0)
    np.testing.assert_array_equal(result.emitted, [0.25, 0.5, 1])
    np.testing.assert_array_equal(result.tint, [1, 0.5, 0.25])


def test_a_base_read_of_another_group_is_refused(tmp_path):
    file = radiosity_shader('sgla', tmp_path, 'shader_transparent_water')
    with pytest.raises(ValueError, match="swat.*sgla|sgla.*swat"):
        api()._read_base_radiosity(file, 'swat')


@pytest.mark.parametrize('size', [0, 35, 39, 64])
def test_truncated_base_shader_reports_too_short_before_fourcc(tmp_path, size):
    file = tmp_path / 'shader.shader_transparent_water'
    file.write_bytes(bytes(size))
    with pytest.raises(ValueError, match='too short'):
        api()._read_base_radiosity(file, 'swat')


@pytest.mark.parametrize('flags', [2, 5])
def test_ignore_normals_flag_is_read_without_a_warning(tmp_path, caplog, flags):
    radiosity_shader('senv', tmp_path, 'shader_environment', flags=flags)
    with caplog.at_level('WARNING'):
        result = api().read_radiosity(tmp_path, 'senv', 'shader')
    assert result.flags == flags
    assert not caplog.records


@pytest.mark.parametrize('group', ['swat', 'smet', 'spla', 'sotr'])
@pytest.mark.parametrize('damage', ['missing', 'short', 'fourcc'])
def test_unreadable_base_only_shader_is_dark_with_a_warning(tmp_path, caplog, group, damage):
    file = resolve_reference_path(tmp_path, group, 'broken')
    if damage == 'short':
        file.write_bytes(b'bad')
    elif damage == 'fourcc':
        file.write_bytes(b'\0' * 36 + b'sgla' + b'\0' * 100)
    with caplog.at_level('WARNING'):
        result = api().read_radiosity(tmp_path, group, 'broken')
    np.testing.assert_array_equal(api().emission(result), [0, 0, 0])
    assert len(caplog.records) == 1
    assert group in caplog.text and 'broken' in caplog.text and 'emission' in caplog.text


def test_negative_sky_power_is_clamped_and_logged(tmp_path, caplog):
    scenario = one_light_scenario(tmp_path, 0, 0, 0, -2)
    with caplog.at_level('WARNING'):
        lighting = api().gather_lighting(scenario, tmp_path, 1)
    assert all(s.power >= 0 for s in lighting.suns)
    assert 'negative' in caplog.text and 'power' in caplog.text
