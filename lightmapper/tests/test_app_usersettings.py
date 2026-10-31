import json

import pytest

from opentool.usersettings import UserSettings, load, save, settings_path
import opentool.usersettings as usersettings


def test_device_round_trips_and_defaults_to_auto(tmp_path):
    assert UserSettings().device == 'auto'
    us = UserSettings(device='amd')
    save(us, tmp_path / 's.json')
    assert load(tmp_path / 's.json') == us


def test_v0_1_0_settings_file_loads_with_auto_device(tmp_path):
    old = {'tags': 'T', 'scenario': 's', 'bsp': 'b', 'blender': 'B', 'preset': 'High', 'new_names': True,
           'backups': False, 'advanced': {'samples': '512'}, 'advanced_open': True, 'verbose': True}
    path = tmp_path / 's.json'; path.write_text(json.dumps(old))
    assert load(path) == UserSettings(**old, device='auto')


@pytest.mark.parametrize('value', ['OPTIX', 'gpu', 3, None])
def test_unknown_device_falls_back_to_auto(tmp_path, value):
    path = tmp_path / 's.json'
    path.write_text(json.dumps({'device': value}))
    assert load(path).device == 'auto'


def test_settings_file_without_ignore_normals_mode_loads_enhanced(tmp_path):
    path = tmp_path / 's.json'
    path.write_text(json.dumps({'preset': 'High'}))
    assert load(path).ignore_normals_mode == 'enhanced'


@pytest.mark.parametrize('value', ['bogus', 'Tool', 3, None])
def test_unknown_ignore_normals_mode_falls_back_to_enhanced(tmp_path, value):
    path = tmp_path / 's.json'
    path.write_text(json.dumps({'ignore_normals_mode': value}))
    assert load(path).ignore_normals_mode == 'enhanced'


def test_known_ignore_normals_mode_is_loaded(tmp_path):
    path = tmp_path / 's.json'
    path.write_text(json.dumps({'ignore_normals_mode': 'tool'}))
    assert load(path).ignore_normals_mode == 'tool'


@pytest.mark.parametrize('device', ['auto', 'nvidia', 'amd', 'cpu'])
def test_known_device_is_loaded(tmp_path, device):
    path = tmp_path / 's.json'
    path.write_text(json.dumps({'device': device}))
    assert load(path).device == device


def test_round_trip(tmp_path):
    us = UserSettings(tags='D:\\t', scenario='levels\\a\\a', bsp='a', preset='High', new_names=False, advanced={'bounces': '2'}, ignore_normals_mode='tool')
    save(us, tmp_path / 's.json')
    assert load(tmp_path / 's.json') == us


def test_custom_verbose_round_trip_and_defaults(tmp_path):
    advanced = {key: str(value) for key, value in dict(texels_per_unit=3, samples=128, vertex_rays=512, sky_grid=3, bounces=2, page_max=512).items()}
    us = UserSettings(preset='Custom', advanced=advanced, verbose=True)
    save(us, tmp_path / 's.json')
    assert load(tmp_path / 's.json') == us
    assert UserSettings().new_names is False and UserSettings().verbose is False
    assert UserSettings().backups is True


def test_backup_setting_round_trips_and_defaults_true(tmp_path):
    assert UserSettings().backups is True
    settings = UserSettings(backups=False)
    save(settings, tmp_path / 's.json')
    assert load(tmp_path / 's.json').backups is False


def test_old_settings_file_without_backups_loads_true(tmp_path):
    (tmp_path / 's.json').write_text('{"preset": "High", "new_names": true}')
    loaded = load(tmp_path / 's.json')
    assert loaded.preset == 'High' and loaded.backups is True


def test_default_path_is_under_appdata(monkeypatch, tmp_path):
    monkeypatch.setenv('APPDATA', str(tmp_path))
    assert settings_path() == tmp_path / 'OpenTool' / 'settings.json'


def test_missing_file_gives_defaults(tmp_path):
    assert load(tmp_path / 'none.json') == UserSettings()


def test_corrupt_settings_file_gives_defaults(tmp_path):
    p = tmp_path / 's.json'
    p.write_text('{oops')
    assert load(p) == UserSettings()


def test_unknown_keys_and_wrong_types_are_dropped(tmp_path):
    p = tmp_path / 's.json'
    p.write_text('{"tags": 5, "future": 1, "bsp": "x"}')
    assert load(p) == UserSettings(bsp='x')


def test_unknown_preset_falls_back_to_normal(tmp_path):
    p = tmp_path / 's.json'
    p.write_text('{"preset": "Ultra"}')
    assert load(p).preset == 'Normal'


def test_save_creates_the_folder(tmp_path):
    save(UserSettings(), tmp_path / 'OpenTool' / 's.json')
    assert (tmp_path / 'OpenTool' / 's.json').is_file()


def test_oversized_integer_gives_defaults(tmp_path):
    p = tmp_path / 's.json'
    p.write_text('{"n": ' + '9' * 5000 + '}')
    assert load(p) == UserSettings()


def test_deeply_nested_json_gives_defaults(tmp_path):
    p = tmp_path / 's.json'
    p.write_text('[' * 100000 + '0' + ']' * 100000)
    assert load(p) == UserSettings()


def test_advanced_settings_keep_only_string_known_fields(tmp_path):
    p = tmp_path / 's.json'
    p.write_text('{"advanced": {"bounces": 2, "page_max": "512", "x": "1"}, "bounces": null}')
    assert load(p).advanced == {'page_max': '512'}


def test_save_removes_temp_file_when_replace_fails(tmp_path, monkeypatch):
    target = tmp_path / 's.json'

    def fail_replace(source, destination):
        raise OSError('replace failed')

    monkeypatch.setattr(usersettings.os, 'replace', fail_replace)
    with pytest.raises(OSError, match='replace failed'):
        save(UserSettings(), target)
    assert list(tmp_path.iterdir()) == []
