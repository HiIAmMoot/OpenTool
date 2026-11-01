"""Render-only surfaces (no collision under them) cast no shadow unless asked; water is never lit or a blocker."""

import json
import struct

import numpy as np
import pytest

from lightmapper.settings import Settings, load_settings
from lightmapper.tagio.tagfile import read_tag, write_tag
from test_bsp import api as bsp_api, synthetic
from test_pipeline import add_unlit_quad, api, tiny


def test_a_surface_is_render_only_when_no_leaf_reference_has_a_node():
    bsp = synthetic()
    refs = bsp.root.block('leaf_surfaces')
    # references in order: surface 0, 1, 1, 2, 2, 3 (surface 4 has none)
    for ref, node in zip(refs, [-1, -1, 7, -1, -1, 0]):
        ref.set('node', node)
    geometry = bsp_api().read_geometry(bsp)
    np.testing.assert_array_equal(geometry.render_only, [True, False, True, False, True])


def unlink_first_triangle(tags):
    bsp = read_tag(tags / 'room.scenario_structure_bsp')
    bsp.root.block('leaf_surfaces')[0].set('node', -1)
    write_tag(bsp, tags / 'room.scenario_structure_bsp')


def shadow_of_each_triangle(work):
    with np.load(work / 'scene.npz') as scene:
        left = scene['positions'][scene['tris']][:, :, 0].max(axis=1) < 5
        return scene['tri_casts_shadow'][left][0], scene['tri_casts_shadow'][~left][0], scene['tri_tint'][left][0]


@pytest.mark.parametrize('enabled, expected', [(False, False), (True, True)])
def test_setting_decides_whether_a_render_only_triangle_is_an_occluder(tiny, enabled, expected):
    tags, settings, work = tiny
    unlink_first_triangle(tags)
    settings.render_only_shadows = enabled
    api().bake(tags, 'level', 'room', settings, work=work, dry_run=True)
    render_only, collidable, tint = shadow_of_each_triangle(work)
    assert bool(render_only) is expected
    assert collidable
    np.testing.assert_array_equal(tint, [1, 1, 1])


def test_a_render_only_triangle_is_still_a_lit_receiver_with_the_setting_off(tiny):
    tags, settings, work = tiny
    unlink_first_triangle(tags)
    api().bake(tags, 'level', 'room', settings, work=work, dry_run=True)
    with np.load(work / 'scene.npz') as scene:
        assert len(scene['tris']) == 2
        left = scene['positions'][scene['tris']][:, :, 0].max(axis=1) < 5
        uv = scene['lm_uv'][scene['tris'][left][0]].astype(np.float64)
        area = abs(np.cross(uv[1] - uv[0], uv[2] - uv[0])) / 2
        assert area > 0
        assert scene['tri_page'][left][0] == 0


def put_water(tags, power):
    header = bytearray(64)
    header[36:40] = b'swat'
    target = tags / 'water.shader_transparent_water'
    target.write_bytes(bytes(header) + struct.pack('>Hhf3f3f', 0, 0, power, 1., .5, .25, .5, .5, .5))


@pytest.mark.parametrize('enabled', [False, True])
def test_powered_water_is_an_emit_only_source_that_blocks_nothing(tiny, enabled):
    tags, settings, work = tiny
    put_water(tags, 10.)
    add_unlit_quad(tags, 'water', group='swat')
    settings.render_only_shadows = enabled
    api().bake(tags, 'level', 'room', settings, work=work, dry_run=True)
    with np.load(work / 'scene.npz') as scene:
        assert len(scene['occ_tris']) == 2
        assert scene['occ_tri_emission'].any()
        assert not scene['occ_tri_casts_shadow'].any()
        np.testing.assert_array_equal(scene['occ_tri_tint'], np.ones_like(scene['occ_tri_tint']))


@pytest.mark.parametrize('enabled', [False, True])
def test_water_on_a_lit_page_is_not_an_occluder(tiny, enabled):
    tags, settings, work = tiny
    put_water(tags, 0.)
    bsp = read_tag(tags / 'room.scenario_structure_bsp')
    bsp.root.block('lightmaps')[0].block('materials')[0].set_reference('shader', 'swat', 'water')
    write_tag(bsp, tags / 'room.scenario_structure_bsp')
    settings.render_only_shadows = enabled
    api().bake(tags, 'level', 'room', settings, work=work, dry_run=True)
    with np.load(work / 'scene.npz') as scene:
        assert not scene['tri_casts_shadow'].any()
        np.testing.assert_array_equal(scene['tri_tint'], np.ones_like(scene['tri_tint']))


def test_setting_defaults_off_and_round_trips(tmp_path):
    assert Settings().render_only_shadows is False
    path = tmp_path / 'settings.json'
    path.write_text('{"render_only_shadows": true}')
    settings = load_settings(path)
    assert settings.render_only_shadows is True
    assert json.loads(settings.to_json())['render_only_shadows'] is True


@pytest.mark.parametrize('argv, text', [([], '{"render_only_shadows": true, "denoise": false}'),
                                        (['--render-only-shadows'], '{"denoise": false}')])
def test_cli_turns_it_on_and_records_it(tiny, tmp_path, argv, text):
    from lightmapper.cli import main

    tags, _, work = tiny
    unlink_first_triangle(tags)
    path = tmp_path / 'settings.json'
    path.write_text(text)
    assert main(['bake', 'level', 'room', '--tags', str(tags), '--work', str(work),
                 '--settings', str(path), '--dry-run', *argv]) == 0
    assert shadow_of_each_triangle(work)[0]
    assert json.loads((work / 'summary.json').read_text())['settings']['render_only_shadows'] is True


def test_the_settings_docs_have_its_row():
    from pathlib import Path

    docs = (Path(__file__).resolve().parents[2] / 'docs' / 'lightmapper-settings.md').read_text(encoding='utf-8')
    assert '| `render_only_shadows` | false |' in docs


def test_render_only_triangle_is_a_pass_through_receiver_with_the_setting_off(tiny):
    tags, settings, work = tiny
    unlink_first_triangle(tags)
    api().bake(tags, 'level', 'room', settings, work=work, dry_run=True)
    with np.load(work / 'scene.npz') as scene:
        left = scene['positions'][scene['tris']][:, :, 0].max(axis=1) < 5
        assert scene['tri_pass_through'][left].all()
        assert not scene['tri_pass_through'][~left].any()
    settings.render_only_shadows = True
    api().bake(tags, 'level', 'room', settings, work=work, dry_run=True)
    with np.load(work / 'scene.npz') as scene:
        assert not scene['tri_pass_through'].any()


def test_an_emissive_render_only_triangle_is_not_pass_through(tiny):
    tags, settings, work = tiny
    unlink_first_triangle(tags)
    shader = read_tag(tags / 'wall.shader_environment')
    shader.root.set('radiosity.power', 5.)
    shader.root.set('radiosity.color_of_emitted_light', (1., .5, .25))
    write_tag(shader, tags / 'wall.shader_environment')
    api().bake(tags, 'level', 'room', settings, work=work, dry_run=True)
    with np.load(work / 'scene.npz') as scene:
        assert scene['tri_emission'].any()
        assert not scene['tri_pass_through'].any()


@pytest.mark.parametrize('power, emits', [(0., False), (10., True)])
def test_lit_page_water_blocks_nothing_and_emits_only_when_powered(tiny, caplog, power, emits):
    tags, settings, work = tiny
    put_water(tags, power)
    bsp = read_tag(tags / 'room.scenario_structure_bsp')
    bsp.root.block('lightmaps')[0].block('materials')[0].set_reference('shader', 'swat', 'water')
    write_tag(bsp, tags / 'room.scenario_structure_bsp')
    with caplog.at_level('WARNING'):
        api().bake(tags, 'level', 'room', settings, work=work, dry_run=True)
    with np.load(work / 'scene.npz') as scene:
        assert scene['tri_emission'].any() == emits
        assert scene['tri_pass_through'].all() != emits
        assert not scene['tri_casts_shadow'].any()
    assert [r.getMessage() for r in caplog.records if 'invisible receiver' in r.getMessage()] == [
        'water: a water shader on a lit page is kept as an invisible receiver only for layout']


@pytest.mark.parametrize('enabled', [False, True])
def test_water_on_scenery_emits_but_blocks_nothing(tiny, enabled):
    from test_bsp import child
    from test_scenery import write_model, write_scenery

    tags, settings, work = tiny
    put_water(tags, 4.)
    write_model(tags, 'model', [[(0, [(0, 0, 0), (4, 0, 0), (0, 4, 0), (4, 4, 0)], [0, 1, 2, 3])]],
                shaders=(('swat', 'water'),), scale=(1., 1.))
    write_scenery(tags, 'prop', 'model')
    scenario = read_tag(tags / 'level.scenario')
    palette = child(scenario.root, 'scenery_palette')
    palette.set_reference('name', 'scen', 'prop')
    scenario.root.set_block('scenery_palette', [palette])
    placement = child(scenario.root, 'scenery')
    placement.set('bsp_indices', 1)
    placement.set('placement.position', (0., 0., 2.))
    scenario.root.set_block('scenery', [placement])
    write_tag(scenario, tags / 'level.scenario')
    settings.render_only_shadows = enabled
    api().bake(tags, 'level', 'room', settings, work=work, dry_run=True)
    with np.load(work / 'scene.npz') as scene:
        assert len(scene['occ_tris']) == 2
        assert not scene['occ_tri_casts_shadow'].any()
        assert scene['occ_tri_emission'].any()


@pytest.mark.parametrize('enabled', [False, True])
def test_a_render_only_tinted_glass_triangle_has_no_tint_unless_it_shadows(tiny, enabled):
    from lightmapper.formats.bsp import OutMaterial, OutPage, read_geometry
    from lightmapper.formats.vertices import LM
    from lightmapper.gather import ShaderRadiosity
    from lightmapper.tagio.tagfile import new_tag
    from test_pipeline import put

    tags, _, _ = tiny
    put(tags, 'sgla', 'glass', new_tag('sgla'))
    geometry = read_geometry(read_tag(tags / 'room.scenario_structure_bsp'))
    geometry.render_only[:] = [True, False]
    group = geometry.groups[0]
    group.key = ('sgla', 'glass') + tuple(group.key[2:])
    material = OutMaterial(0, group.env, np.zeros(len(group.env), LM), group.tris, group.surface_ids)
    tint = np.array([.46, .6, .66])
    scene, _, _ = api()._scene(tags, read_tag(tags / 'level.scenario'), 0, geometry, [OutPage(8, 8, [material])],
                               [ShaderRadiosity(0, 0, 0., np.zeros(3), tint)], np.array([-1, -1], np.int16),
                               True, enabled)
    render_only = int(np.argmin(scene.positions[scene.tris][:, :, 0].max(axis=1)))
    np.testing.assert_allclose(scene.tri_tint[render_only], tint if enabled else np.ones(3), rtol=1e-6)
    np.testing.assert_allclose(scene.tri_tint[1 - render_only], tint, rtol=1e-6)


def test_a_bsp_without_leaf_surfaces_has_no_render_only_surfaces(caplog):
    bsp = synthetic()
    bsp.root.set_block('leaf_surfaces', [])
    with caplog.at_level('WARNING'):
        geometry = bsp_api().read_geometry(bsp)
    assert not geometry.render_only.any() and len(geometry.render_only) == 5
    assert any('leaf_surfaces' in r.getMessage() for r in caplog.records)
