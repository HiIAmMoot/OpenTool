"""Scenery emission is optional without disabling the BSP's unlit emitters."""

from dataclasses import replace
import json

import numpy as np
import pytest

from lightmapper.settings import Settings, load_settings
from test_pipeline import tiny, api, add_unlit_quad, put_scenery_over_the_room, put_sotr, rewind_room


def test_scenery_emission_defaults_on_and_round_trips(tmp_path):
    assert getattr(Settings(), 'scenery_emission', None) is True
    path = tmp_path / 'settings.json'
    path.write_text('{"scenery_emission": false}')
    settings = load_settings(path)
    assert settings.scenery_emission is False
    assert json.loads(settings.to_json())['scenery_emission'] is False


def test_setting_filters_only_scenery_emission_in_the_scene(tiny):
    tags, settings, work = tiny
    put_scenery_over_the_room(tags, 4.)
    put_sotr(tags, 'bsp_glow', 10.)
    add_unlit_quad(tags, 'bsp_glow', z=1.)
    settings.scenery_emission = False
    api().bake(tags, 'level', 'room', settings, work=work, dry_run=True)
    with np.load(work / 'scene.npz') as scene:
        np.testing.assert_array_equal(scene['occ_tri_emission'][:2], np.zeros((2, 3)))
        np.testing.assert_array_equal(scene['occ_tri_emission'][2:], [[10., 5., 2.5]] * 2)
        np.testing.assert_array_equal(scene['occ_tri_casts_shadow'], [True, True, False, False])


def test_cli_settings_file_disables_scenery_and_records_it(tiny, tmp_path):
    from lightmapper.cli import main

    tags, _, work = tiny
    put_scenery_over_the_room(tags, 4.)
    path = tmp_path / 'settings.json'
    path.write_text('{"scenery_emission": false, "texels_per_unit": 0.25, "denoise": false}')
    assert main(['bake', 'level', 'room', '--tags', str(tags), '--work', str(work),
                 '--settings', str(path), '--dry-run']) == 0
    with np.load(work / 'scene.npz') as scene:
        assert not scene['occ_tri_emission'].any()
    assert json.loads((work / 'summary.json').read_text())['settings']['scenery_emission'] is False


@pytest.mark.blender
def test_scenery_setting_controls_light_but_keeps_unlit_bsp_light(tiny, monkeypatch, blender_exe):
    from lightmapper import blender_job

    tags, settings, work = tiny
    monkeypatch.setattr(api(), 'run_blender', blender_job.run_blender)
    settings = replace(settings, samples=16, vertex_rays=8, blender=str(blender_exe))
    rewind_room(tags)
    put_scenery_over_the_room(tags, 10.)

    def light(enabled, name):
        settings.scenery_emission = enabled
        out = api().bake(tags, 'level', 'room', settings, work=work / name, sources='emit', dry_run=True)
        return float(sum(page.sum() for page in out.float_pages))

    on = light(True, 'on')
    off = light(False, 'off')
    put_sotr(tags, 'bsp_glow', 10.)
    add_unlit_quad(tags, 'bsp_glow', z=1.)
    bsp_only = light(False, 'bsp_only')
    assert on > 0
    assert off == 0
    assert bsp_only > 0
