import subprocess
import sys
from pathlib import Path

import pytest
from lightmapper import cli


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
