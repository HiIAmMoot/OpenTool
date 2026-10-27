from pathlib import Path

import pytest

from lightmapper import check, cli
from lightmapper.resources import REQUIRED, missing


REPO = Path(__file__).resolve().parents[2]


def dict_of(results):
    return {name: (ok, detail) for name, ok, detail in results}


def test_gpu_check_passes_when_the_probe_answers_without_a_gpu(monkeypatch):
    monkeypatch.setattr(check.gpu, 'run_probe', lambda: ([], 'vulkan: no driver'))
    assert dict_of(check.run_checks(('blender',)))['gpu'] == (True, 'no GPU with ray queries (vulkan: no driver)')


def test_gpu_check_names_the_ray_query_gpus(monkeypatch):
    from lightmapper.gpu import Adapter, NVIDIA
    rtx = Adapter('NVIDIA GeForce RTX 4060 Laptop GPU', NVIDIA, (1,), 'vulkan', True)
    no_rays = Adapter('No ray queries', NVIDIA, (2,), 'vulkan', False)
    monkeypatch.setattr(check.gpu, 'run_probe', lambda: ([rtx, no_rays], None))
    results = check.run_checks(('blender',))
    assert dict_of(results)['gpu'] == (True, 'NVIDIA GeForce RTX 4060 Laptop GPU (vulkan)')
    names = [name for name, _, _ in results]
    assert names == ['files', 'xatlas', 'numba', 'runtime', 'denoiser', 'gpu', 'blender']


def test_gpu_check_fails_when_slangpy_cannot_import(monkeypatch):
    monkeypatch.setattr(check.gpu, 'run_probe', lambda: ([], "slangpy: No module named 'slangpy.x'"))
    assert dict_of(check.run_checks(('blender',)))['gpu'] == (False, "slangpy: No module named 'slangpy.x'")


def test_gpu_check_names_the_adapters_when_none_has_ray_queries(monkeypatch):
    from lightmapper.gpu import Adapter, NVIDIA
    no_rays = Adapter('No ray queries', NVIDIA, (2,), 'vulkan', False)
    monkeypatch.setattr(check.gpu, 'run_probe', lambda: ([no_rays], None))
    assert dict_of(check.run_checks(('blender',)))['gpu'] == (
        True, 'no GPU with ray queries (adapters without them: No ray queries (vulkan))')


def test_gpu_check_fails_when_the_probe_child_dies(monkeypatch):
    def dies():
        raise check.gpu.ProbeError('GPU probe process died (exit 3)')
    monkeypatch.setattr(check.gpu, 'run_probe', dies)
    assert dict_of(check.run_checks(('blender',)))['gpu'] == (False, 'GPU probe process died (exit 3)')


def test_gpu_check_lists_nvidia_before_amd(monkeypatch):
    from lightmapper.gpu import Adapter, AMD, NVIDIA
    amd = Adapter('AMD Radeon(TM) Graphics', AMD, (1,), 'vulkan', True)
    nvidia = Adapter('NVIDIA GeForce RTX 4060 Laptop GPU', NVIDIA, (2,), 'vulkan', True)
    monkeypatch.setattr(check.gpu, 'run_probe', lambda: ([amd, nvidia], None))
    assert dict_of(check.run_checks(('blender',)))['gpu'] == (
        True, 'NVIDIA GeForce RTX 4060 Laptop GPU (vulkan), AMD Radeon(TM) Graphics (vulkan)')


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
    monkeypatch.setattr(check, 'run_checks', lambda skip=(): [('files', True, 'all there'), ('blender', False, 'not found')])
    assert cli.main(['check']) == 1
    out = capsys.readouterr().out
    assert 'ok   files: all there' in out and 'FAIL blender: not found' in out


def test_check_needs_no_tags(monkeypatch):
    monkeypatch.delenv('HCEEK_TAGS', raising=False)
    monkeypatch.setattr(check, 'run_checks', lambda skip=(): [('files', True, '')])
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


def test_skip_blender_reports_it_unchecked_and_passes(monkeypatch):
    def no_blender(*args):
        raise AssertionError('find_blender must not run when blender is skipped')
    def broken(*args, **kwargs):
        raise RuntimeError('not under test')
    monkeypatch.setattr(check.blender_find, 'find_blender', no_blender)
    # (the other rows are not what this test is about; they only have to not run for real)
    monkeypatch.setattr(check.xatlas_c, 'library', broken)
    monkeypatch.setattr(check.raster, 'dilate', broken)
    monkeypatch.setattr(check.pages, '_denoise', broken)
    results = check.run_checks(skip=('blender',))
    assert [row for row in results if row[0] == 'blender'] == [('blender', True, check.SKIPPED)]


def test_plain_check_skips_nothing(monkeypatch):
    seen = []
    monkeypatch.setattr(check, 'run_checks', lambda skip=(): seen.append(skip) or [('files', True, '')])
    assert cli.main(['check']) == 0
    assert seen == [()]


def test_check_skip_flag_prints_the_skip_line_and_exits_zero(monkeypatch, capsys):
    seen = []
    def fake(skip=()):
        seen.append(skip)
        return [('files', True, 'all there'), ('blender', True, check.SKIPPED)]
    monkeypatch.setattr(check, 'run_checks', fake)
    assert cli.main(['check', '--skip', 'blender']) == 0
    assert seen == [('blender',)]
    assert 'skip blender: not checked in this build' in capsys.readouterr().out


def test_check_rejects_an_unknown_skip_name():
    with pytest.raises(SystemExit):
        cli.main(['check', '--skip', 'files'])
