import json
import subprocess
import sys
from pathlib import Path

import pytest
from lightmapper import cli, kept
from test_pipeline import tiny  # noqa: F401


REPO = Path(__file__).resolve().parents[2]
SCRIPT = '''
import logging, sys, types
from lightmapper import cli
log = logging.getLogger('lightmapper.fake')
def fake_bake(*a, **k):
    log.info('an info line'); log.warning('a warning line')
    return types.SimpleNamespace(work='W')
cli.bake = fake_bake
raise SystemExit(cli.main(sys.argv[1:]))
'''


def run_cli(tmp_path, *extra):
    return subprocess.run([sys.executable, '-c', SCRIPT, 'bake', '--tags', str(tmp_path), 'level', 'room', *extra],
                          capture_output=True, text=True, cwd=REPO, timeout=60)


def test_default_hides_info_but_shows_warnings(tmp_path):
    done = run_cli(tmp_path)
    assert done.returncode == 0, done.stderr
    assert 'an info line' not in done.stderr and 'WARNING: a warning line' in done.stderr
    assert 'Work: W' in done.stdout


@pytest.mark.parametrize('flag', ['-v', '--verbose'])
def test_verbose_shows_info(tmp_path, flag):
    done = run_cli(tmp_path, flag)
    assert 'INFO: an info line' in done.stderr and 'WARNING: a warning line' in done.stderr


def test_every_command_takes_verbose():
    for argv in (['validate', 'x', '-v'], ['copy', 's', 'b', '--out-bsp', 'o', '--out-scenario', 'o', '-v'],
                 ['diff', 'a', 'b', '--work', 'w', '-v'], ['calibrate', 's', 'b', '--work', 'w', '-v'], ['check', '-v']):
        args = cli._parser().parse_args(argv)
        assert args.verbose


def test_no_backups_flag_for_bake_and_copy():
    assert cli._parser().parse_args(['bake', 'level', 'room']).no_backups is False
    assert cli._parser().parse_args(
        ['copy', 'level', 'room', '--out-bsp', 'out', '--out-scenario', 'out']
    ).no_backups is False
    for argv in (['bake', 'level', 'room', '--no-backups'],
                 ['copy', 'level', 'room', '--out-bsp', 'out', '--out-scenario', 'out', '--no-backups']):
        assert cli._parser().parse_args(argv).no_backups is True


def test_cli_forwards_no_backups(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(cli, 'bake', lambda *args, **kwargs: calls.append(kwargs) or type('O', (), {'work': 'W'})())
    cli.main(['bake', '--tags', str(tmp_path), 'level', 'room', '--no-backups'])
    assert calls[0]['backups'] is False


def test_cli_forwards_no_backups_to_copy(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(cli, 'copy_tags', lambda *args, **kwargs: calls.append(kwargs))
    cli.main(['copy', '--tags', str(tmp_path), 'level', 'room', '--out-bsp', 'out',
              '--out-scenario', 'out', '--no-backups'])
    assert calls == [{'backups': False}]


def test_cli_32bit_flag_sets_the_setting(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(cli, 'bake', lambda *args, **kwargs: calls.append(args[3]) or type('O', (), {'work': 'W'})())
    cli.main(['bake', '--tags', str(tmp_path), 'level', 'room'])
    cli.main(['bake', '--tags', str(tmp_path), 'level', 'room', '--32bit'])
    assert [s.lightmap_32bit for s in calls] == [False, True]


def test_cli_keep_uses_the_kept_root(monkeypatch, tmp_path):
    from lightmapper import kept
    calls = []
    monkeypatch.setattr(kept, 'kept_root', lambda: tmp_path / 'k')
    monkeypatch.setattr(cli, 'bake', lambda *args, **kwargs: calls.append((args, kwargs)) or type('O', (), {'work': 'W'})())
    cli.main(['bake', '--tags', str(tmp_path), 'level', 'room', '--keep'])
    args, kwargs = calls[0]
    work = args[6]
    assert kwargs['keep'] is True and work.is_dir()
    assert work.parent == kept.level_folder(tmp_path / 'k', tmp_path, 'level', 'room')


def test_cli_keep_with_an_explicit_work_folder_uses_it(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(cli, 'bake', lambda *args, **kwargs: calls.append((args, kwargs)) or type('O', (), {'work': 'W'})())
    cli.main(['bake', '--tags', str(tmp_path), 'level', 'room', '--keep', '--work', str(tmp_path / 'w')])
    assert calls[0][0][6] == tmp_path / 'w' and calls[0][1]['keep'] is True


def test_cli_without_keep_keeps_nothing(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(cli, 'bake', lambda *args, **kwargs: calls.append(kwargs) or type('O', (), {'work': 'W'})())
    cli.main(['bake', '--tags', str(tmp_path), 'level', 'room'])
    assert calls[0]['keep'] is False


@pytest.mark.parametrize('extra', [['--dry-run'], ['--sources', 'sun'], ['--sources', 'emit']])
def test_cli_keep_refuses_dry_runs_and_partial_sources(tmp_path, extra):
    with pytest.raises(SystemExit) as stopped:
        cli.main(['bake', '--tags', str(tmp_path), 'level', 'room', '--keep', *extra])
    assert stopped.value.code == 2



def kept_manifest(tmp_path, **settings):
    from dataclasses import asdict
    from lightmapper.settings import Settings
    work = tmp_path / 'kept'
    kept.kept_dir(work).mkdir(parents=True)
    for name in (kept.SOURCE, kept.SCENARIO, kept.CHARTS, kept.RESULT, kept.SCENE, kept.JOB):
        (kept.kept_dir(work) / name).write_bytes(b'x')
    base = {**asdict(Settings()), **settings, 'dropped_in_a_later_release': 1}
    kept.write_manifest(work, kept.Manifest(
        tags=str(tmp_path / 'tags'), scenario='level', bsp='room', input_path='room', has_template=False,
        vertex_device='cpu', glow_device='off', glow_reason='', settings=base, kept_at='now',
        output_path='room', output_scenario=''))
    return work


def test_cli_write_uses_the_manifest(monkeypatch, tmp_path):
    work = kept_manifest(tmp_path, lightmap_32bit=True)
    calls = []
    monkeypatch.setattr(cli, 'bake', lambda *args, **kwargs: calls.append((args, kwargs)) or type('O', (), {'work': work})())
    monkeypatch.delenv('HCEEK_TAGS', raising=False)
    assert cli.main(['write', str(work), '--out-bsp', 'copy']) == 0
    args, kwargs = calls[0]
    assert args[:3] == (tmp_path / 'tags', 'level', 'room')
    assert args[3].lightmap_32bit is True and args[4] == 'copy'
    assert kwargs['reuse'] is True and args[6] == work


def test_cli_write_prints_the_summary_report(monkeypatch, tmp_path, capsys):
    work = kept_manifest(tmp_path)
    (work / 'write_summary.json').write_text('{"devices": {"denoise": "off"}, "warnings": ["a warning"]}')
    monkeypatch.setattr(cli, 'bake', lambda *args, **kwargs: type('O', (), {'work': work})())
    cli.main(['write', str(work)])
    out = capsys.readouterr().out
    assert 'a warning' in out and f'Work: {work}' in out


def test_cli_write_16bit_overrides(monkeypatch, tmp_path):
    work = kept_manifest(tmp_path, lightmap_32bit=True)
    calls = []
    monkeypatch.setattr(cli, 'bake', lambda *args, **kwargs: calls.append(args[3]) or type('O', (), {'work': work})())
    cli.main(['write', str(work), '--16bit'])
    cli.main(['write', str(work), '--32bit'])
    assert [s.lightmap_32bit for s in calls] == [False, True]


def test_cli_write_refusal_exits_1(monkeypatch, tmp_path, capsys):
    from lightmapper.errors import KeptBakeError
    work = kept_manifest(tmp_path)

    def refuse(*args, **kwargs):
        raise KeptBakeError('This kept bake is of another level.')

    monkeypatch.setattr(cli, 'bake', refuse)
    assert cli.main(['write', str(work)]) == 1
    assert 'This kept bake is of another level.' in capsys.readouterr().err


def test_cli_write_refuses_a_folder_that_is_not_kept(tmp_path, capsys):
    assert cli.main(['write', str(tmp_path)]) == 1
    assert 'not a kept bake' in capsys.readouterr().err


def test_bake_prints_the_report_before_the_work_folder(monkeypatch, tmp_path, capsys):
    work = tmp_path / 'w'
    work.mkdir()
    (work / 'summary.json').write_text('{"seconds": {"total": 45.2}}')
    monkeypatch.setattr(cli, 'bake', lambda *args, **kwargs: type('O', (), {'work': work})())
    cli.main(['bake', '--tags', str(tmp_path), 'level', 'room'])
    out = capsys.readouterr().out
    assert 'Total time: 45 s' in out and out.index('Total time:') < out.index('Work:')


def test_cli_write_prints_the_report_and_keeps_its_warnings(monkeypatch, tmp_path, capsys):
    work = kept_manifest(tmp_path)
    (work / 'write_summary.json').write_text(json.dumps({
        'seconds': {'total': 3.0}, 'written': {'bitmap': 1048576, 'bsp': 2097152}, 'warnings': ['a warning'],
        'devices': {'page_bake': 'CPU', 'vertex_pass': 'cpu', 'denoise': 'off', 'glow_pass': 'off',
                    'glow_reason': 'bounces set to 0'}}))
    monkeypatch.setattr(cli, 'bake', lambda *args, **kwargs: type('O', (), {'work': work})())
    cli.main(['write', str(work)])
    out = capsys.readouterr().out
    assert 'BSP written: 2.0 MB' in out and 'warning: a warning' in out
    assert out.count('denoise') == 1 and 'glow pass skipped (bounces set to 0)' in out
    assert out.index('Total time:') < out.index('Work:')


def kept_bake_of_a_tiny_level(tiny, tmp_path, **kwargs):
    from dataclasses import replace
    from test_pipeline import api
    tags, settings, _ = tiny
    work = kept.level_folder(tmp_path / 'k', tags, 'level', 'room') / 'a'
    api().bake(tags, 'level', 'room', replace(settings, ignore_normals_mode='tool'), work=work, keep=True, **kwargs)
    return tags, work


def test_cli_write_writes_the_tags_the_bake_wrote(tiny, tmp_path, monkeypatch):
    from lightmapper.tagio.tagfile import read_tag
    tags, work = kept_bake_of_a_tiny_level(tiny, tmp_path, out_bsp='copy')
    room = (tags / 'room.scenario_structure_bsp').read_bytes()
    for name in ('copy.scenario_structure_bsp', 'copy.bitmap'):
        (tags / name).unlink()
    monkeypatch.delenv('HCEEK_TAGS', raising=False)
    assert cli.main(['write', str(work), '--32bit']) == 0
    assert (tags / 'room.scenario_structure_bsp').read_bytes() == room and not (tags / 'room.bitmap').exists()
    assert [b.get('format') for b in read_tag(tags / 'copy.bitmap').root.block('bitmaps')] == [10]


def test_cli_write_rewrites_the_scenario_the_bake_wrote(tiny, tmp_path, monkeypatch):
    from lightmapper.tagio.tagfile import read_tag
    tags, work = kept_bake_of_a_tiny_level(tiny, tmp_path, out_bsp='copy', out_scenario='lit')
    (tags / 'lit.scenario').unlink()
    monkeypatch.delenv('HCEEK_TAGS', raising=False)
    assert cli.main(['write', str(work)]) == 0
    entries = read_tag(tags / 'lit.scenario').root.block('structure_bsps')
    assert [e.reference('structure_bsp')[1] for e in entries] == ['copy']


def test_cli_keep_in_an_unwritable_kept_root_says_so(monkeypatch, tmp_path, capsys):
    blocker = tmp_path / 'k'
    blocker.write_text('a file')
    monkeypatch.setattr(kept, 'kept_root', lambda: blocker)
    monkeypatch.setattr(cli, 'bake', lambda *a, **k: pytest.fail('a bake without a kept folder'))
    assert cli.main(['bake', '--tags', str(tmp_path), 'level', 'room', '--keep']) == 1
    assert capsys.readouterr().err.strip() == (
        f'OpenTool could not make a folder for kept bakes in {blocker}. '
        'Leave out --keep, or move OpenTool to a folder you can write to.')
