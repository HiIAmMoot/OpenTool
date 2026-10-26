import pytest

from conftest import HCEEK_TAGS
from test_pipeline import tiny
from opentool.form import bsp_tag_path, list_bsps, new_names, scenario_tag_path, tags_folder_problem


def test_new_names_add_lit():
    assert new_names('levels\\a\\bg', 'levels\\a\\bg') == ('levels\\a\\bg_lit', 'levels\\a\\bg_lit')


def test_new_names_do_not_stack():
    assert new_names('levels\\a\\bg_lit', 'levels\\a\\bg_lit') == ('levels\\a\\bg_lit', 'levels\\a\\bg_lit')


def test_scenario_tag_path(tmp_path):
    f = tmp_path / 'levels' / 'x' / 'x.scenario'
    f.parent.mkdir(parents=True)
    f.write_bytes(b'')
    assert scenario_tag_path(tmp_path, f) == 'levels\\x\\x'


def test_scenario_outside_tags_is_refused(tmp_path):
    with pytest.raises(ValueError, match='inside the tags folder'):
        scenario_tag_path(tmp_path / 'tags', tmp_path / 'x.scenario')


def test_tags_folder_problem(tmp_path):
    assert tags_folder_problem(tmp_path / 'none') == 'That folder does not exist.'
    assert 'no globals folder' in tags_folder_problem(tmp_path)
    (tmp_path / 'globals').mkdir()
    assert tags_folder_problem(tmp_path) is None


def test_list_bsps(tiny):
    tags, *_ = tiny
    assert list_bsps(tags, 'level') == ['room']
    assert bsp_tag_path(tags, 'level', 'room') == 'room'


def test_list_bsps_resolves_scenario_reference(tmp_path, monkeypatch):
    import opentool.form as form

    expected = tmp_path / 'resolved.scenario'
    seen = []

    class Scenario:
        class Root:
            @staticmethod
            def block(name):
                assert name == 'structure_bsps'
                return []

        root = Root()

    def resolve(tags, group, scenario):
        seen.append((tags, group, scenario))
        return expected

    monkeypatch.setattr(form, 'resolve_reference_path', resolve, raising=False)
    monkeypatch.setattr(form, 'read_tag', lambda path: Scenario())
    assert form.list_bsps(tmp_path, 'levels\\test\\level') == []
    assert seen == [(tmp_path, 'scnr', 'levels\\test\\level')]


@pytest.mark.ek
def test_bloodgulch_has_one_bsp():
    assert list_bsps(HCEEK_TAGS, 'levels\\test\\bloodgulch\\bloodgulch') == ['bloodgulch']
