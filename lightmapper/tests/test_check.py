from pathlib import Path

import pytest

from lightmapper import check, cli
from lightmapper.resources import REQUIRED, missing


REPO = Path(__file__).resolve().parents[2]


def test_required_files_are_all_present_in_the_repo():
    assert missing() == []


def test_required_covers_every_layout_and_the_blender_script():
    layouts = {f'lightmapper/tagio/layouts/{path.name}'
               for path in (REPO / 'lightmapper/tagio/layouts').glob('*.xml')}
    assert layouts <= set(REQUIRED)
    assert {'lightmapper/calibration.json', 'lightmapper/blender/bake.py'} <= set(REQUIRED)


def test_missing_names_the_absent_file(tmp_path):
    assert 'lightmapper/calibration.json' in missing(tmp_path)


def test_check_command_reports_and_fails_on_a_missing_part(monkeypatch, capsys):
    monkeypatch.setattr(check, 'run_checks', lambda: [('files', True, 'all there'), ('blender', False, 'not found')])
    assert cli.main(['check']) == 1
    out = capsys.readouterr().out
    assert 'ok   files: all there' in out and 'FAIL blender: not found' in out


def test_check_needs_no_tags(monkeypatch):
    monkeypatch.delenv('HCEEK_TAGS', raising=False)
    monkeypatch.setattr(check, 'run_checks', lambda: [('files', True, '')])
    assert cli.main(['check']) == 0


@pytest.mark.blender
def test_real_checks_pass_on_this_machine():
    assert all(ok for _, ok, _ in check.run_checks())


def test_runtime_check_passes_when_not_frozen(monkeypatch):
    monkeypatch.delattr('sys.frozen', raising=False)
    assert check.check_runtime() == ('runtime', True, 'not frozen')


def test_runtime_problems_compare_paths_against_the_bundle(tmp_path):
    inside = str(tmp_path / '_internal' / 'msvcp140.dll')
    paths = {'msvcp140.dll': inside, 'vcruntime140_1.dll': 'C:/Windows/SYSTEM32/VCRUNTIME140_1.dll'}
    problems = check.runtime_problems(str(tmp_path / '_internal'), paths)
    assert len(problems) == 1 and 'vcruntime140_1.dll' in problems[0]
    assert check.runtime_problems(str(tmp_path / '_internal'), {'msvcp140.dll': None}) == ['msvcp140.dll is not loaded']
    # a sibling folder sharing the bundle's name as a prefix is outside it
    sibling = str(tmp_path / '_internal2' / 'msvcp140.dll')
    assert check.runtime_problems(str(tmp_path / '_internal'), {'msvcp140.dll': sibling})
