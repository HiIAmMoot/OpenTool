import subprocess
import sys
from pathlib import Path

import pytest
from lightmapper import cli, kept


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
        vertex_device='cpu', glow_device='off', glow_reason='', settings=base, kept_at='now'))
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
