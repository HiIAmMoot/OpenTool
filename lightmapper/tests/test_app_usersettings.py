import pytest

from opentool.usersettings import UserSettings, load, save, settings_path
import opentool.usersettings as usersettings


def test_round_trip(tmp_path):
    us = UserSettings(tags='D:\\t', scenario='levels\\a\\a', bsp='a', preset='High', new_names=False, advanced={'bounces': '2'})
    save(us, tmp_path / 's.json')
    assert load(tmp_path / 's.json') == us


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
