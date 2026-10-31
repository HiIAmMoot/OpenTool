import json
from pathlib import Path

import pytest

from lightmapper.settings import Settings, load_settings


def test_defaults_come_from_calibration_json():
    s = Settings()
    assert (s.w_sun, s.w_emit, s.w_amb, s.gamma, s.radiosity_scale) == (1, 1, 1, 1, 1)
    assert s.radiosity_mode == 'coherence'
    assert (s.samples, s.bounces, s.vertex_rays, s.device, s.timeout_s) == (256, 1, 4096, 'auto', 7200)
    assert s.page_max == 1024 and s.denoise and s.dither


def test_seed_defaults_to_zero_and_stays_in_range():
    assert Settings().seed == 0
    for bad in (-1, 1000):
        with pytest.raises(ValueError, match='seed'):
            Settings(seed=bad)


@pytest.mark.parametrize('value,expected', [
    ('auto', 'auto'), ('NVIDIA', 'nvidia'), ('amd', 'amd'), ('cpu', 'cpu'),
    ('OPTIX', 'nvidia'), ('CUDA', 'nvidia'), ('HIP', 'amd'), ('CPU', 'cpu')])
def test_legacy_device_values_map_to_vendors(value, expected):
    assert Settings(device=value).device == expected


def test_unknown_device_names_the_allowed_values():
    with pytest.raises(ValueError, match='auto, nvidia, amd, cpu'):
        Settings(device='intel')


def test_shipped_calibration_is_tools_analytic_model(monkeypatch):
    from math import pi
    from lightmapper import settings
    monkeypatch.setattr(settings, 'CALIBRATION_PATH', Path(settings.__file__).with_name('calibration.json'))
    s = Settings()
    assert (s.w_sun, s.w_emit, s.w_amb, s.gamma) == (pi, 1.0, 1.0, 1.0)
    assert (s.radiosity_mode, s.radiosity_scale) == ('coherence', 1.0)


def test_none_gives_defaults():
    assert load_settings(None) == Settings()


def test_settings_file_accepts_null_timeout(tmp_path):
    p = tmp_path / 's.json'
    p.write_text('{"timeout_s": null}')
    assert load_settings(p).timeout_s is None


def test_blender_default_follows_environment_at_construction(monkeypatch):
    monkeypatch.setenv('LIGHTMAPPER_BLENDER', r'C:\Apps\Blender\blender.exe')
    assert load_settings(None).blender == r'C:\Apps\Blender\blender.exe'
    monkeypatch.setenv('LIGHTMAPPER_BLENDER', r'C:\Other\blender.exe')
    assert Settings().blender == r'C:\Other\blender.exe'


def test_blender_default_without_environment(monkeypatch):
    monkeypatch.delenv('LIGHTMAPPER_BLENDER', raising=False)
    assert Settings().blender == r'C:\Program Files\Blender Foundation\Blender 5.2\blender.exe'


def test_blender_setting_overrides_environment(tmp_path, monkeypatch):
    monkeypatch.setenv('LIGHTMAPPER_BLENDER', r'C:\Other\blender.exe')
    path = tmp_path / 'settings.json'
    path.write_text(json.dumps({'blender': r'C:\Apps\Blender\blender.exe'}))
    assert load_settings(path).blender == r'C:\Apps\Blender\blender.exe'


def test_round_trip(tmp_path):
    original = Settings(samples=8, w_sun=1.5, radiosity_mode='scaled', page_max=256, denoise=False)
    path = tmp_path / 's.json'
    path.write_text(original.to_json())
    assert load_settings(path) == original


def test_partial_file_keeps_other_defaults(tmp_path):
    path = tmp_path / 's.json'
    path.write_text(json.dumps({'samples': 4}))
    s = load_settings(path)
    assert s.samples == 4 and s.bounces == 1


def test_unknown_key_is_an_error(tmp_path):
    path = tmp_path / 's.json'
    path.write_text(json.dumps({'sampels': 4}))
    with pytest.raises(ValueError, match='sampels'):
        load_settings(path)


def test_bad_radiosity_mode_is_an_error():
    with pytest.raises(ValueError):
        Settings(radiosity_mode='bogus')


def test_detail_level_scale_survives_a_settings_file(tmp_path):
    assert Settings().detail_level_scale == (2.667, 1.0, 0.6, 0.03)
    original = Settings(detail_level_scale=(2.0, 1.0, 0.5, 0.1))
    path = tmp_path / 's.json'
    path.write_text(original.to_json())
    assert load_settings(path) == original


@pytest.mark.parametrize('bad', [0, -1])
def test_sky_grid_must_be_positive(bad):
    with pytest.raises(ValueError, match='sky_grid'):
        Settings(sky_grid=bad)


def test_texel_samples_follows_samples_unless_set():
    from lightmapper.settings import Settings

    assert Settings().texel_samples is None
    assert Settings(samples=64).resolved_texel_samples() == 9 * round(64 / 9)
    assert Settings(samples=1024).resolved_texel_samples() == 9 * round(1024 / 9)
    assert Settings(samples=4).resolved_texel_samples() == 9
    assert Settings(samples=1024, texel_samples=100).resolved_texel_samples() == 100
    assert Settings(samples=1024, texel_samples=0).resolved_texel_samples() == 0


def test_cpu_counts_default_to_the_gpu_counts():
    s = Settings(samples=1024, vertex_rays=4096)
    assert (s.resolved_cpu_samples(), s.resolved_cpu_vertex_rays(), s.resolved_cpu_texel_samples()) == (1024, 4096, 1026)


def test_cpu_counts_resolve_their_own_texel_samples():
    s = Settings(samples=1024, cpu_samples=256)
    assert s.resolved_cpu_texel_samples() == 252 and s.resolved_texel_samples() == 1026


@pytest.mark.parametrize('target', [0, 100])
def test_explicit_texel_target_applies_to_both_devices(target):
    s = Settings(samples=1024, cpu_samples=256, texel_samples=target)
    assert s.resolved_cpu_texel_samples() == target and s.resolved_texel_samples() == target


@pytest.mark.parametrize('field', ['cpu_samples', 'cpu_vertex_rays'])
def test_cpu_counts_must_be_positive(field):
    with pytest.raises(ValueError, match=field):
        Settings(**{field: 0})


@pytest.mark.parametrize('field', ['cpu_samples', 'cpu_vertex_rays'])
def test_cpu_counts_reject_negative_values(field):
    with pytest.raises(ValueError, match=field):
        Settings(**{field: -1})


def test_quantise_follows_the_32bit_setting():
    import numpy as np
    from lightmapper.pages import quantise

    page = np.full((2, 2, 3), 0.5, np.float32)
    assert quantise(page, Settings(dither=False)).dtype == np.uint16
    assert quantise(page, Settings(dither=False, lightmap_32bit=True)).dtype == np.uint32


def test_ignore_normals_mode_defaults_to_enhanced_and_refuses_others():
    assert Settings().ignore_normals_mode == 'enhanced'
    assert Settings(ignore_normals_mode='enhanced').ignore_normals_mode == 'enhanced'
    with pytest.raises(ValueError, match='tool.*enhanced'):
        Settings(ignore_normals_mode='wrong')


def test_cpu_vertex_backend_defaults_to_numba():
    assert Settings().cpu_vertex_backend == 'numba'


def test_cpu_vertex_backend_is_checked():
    assert Settings(cpu_vertex_backend='blender').cpu_vertex_backend == 'blender'
    with pytest.raises(ValueError, match=r"\('numba', 'blender'\)"):
        Settings(cpu_vertex_backend='wrong')


def test_cpu_vertex_backend_is_a_settings_file_key(tmp_path):
    path = tmp_path / 's.json'
    path.write_text('{"cpu_vertex_backend": "blender"}')
    assert load_settings(path).cpu_vertex_backend == 'blender'


def test_cpu_vertex_backend_is_not_on_the_cli():
    from lightmapper.cli import _parser
    import argparse

    commands = next(a for a in _parser()._actions if isinstance(a, argparse._SubParsersAction))
    assert not any('vertex_backend' in option for action in commands.choices['bake']._actions
                   for option in action.option_strings)
    root = Path(__file__).resolve().parents[2]
    for name in ('presets.py', 'usersettings.py'):
        assert 'cpu_vertex_backend' not in (root / 'opentool' / name).read_text()
