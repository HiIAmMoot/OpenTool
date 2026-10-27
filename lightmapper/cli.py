"""Command-line entry points for baking, isolated tag copies, validation, differential and calibration."""

import argparse
from dataclasses import replace
import logging
import os
from pathlib import Path

import numpy as np

from .calibrate import calibrate
from .blender_find import find_blender
from .diff import load_pages, paired_samples, bitmap_pages, render_markdown, report
from .pipeline import bake, copy_tags
from . import settings as settings_module
from .settings import load_settings
from .tagio.tagfile import read_tag, resolve_reference_path
from .validate import validate_bsp


logger = logging.getLogger(__name__)


def _blender_fallback(s):
    if (not os.environ.get('LIGHTMAPPER_BLENDER') and s.blender == settings_module.DEFAULT_BLENDER
            and not Path(s.blender).is_file()):
        return replace(s, blender=str(find_blender().path))
    return s


def _parser():
    parser = argparse.ArgumentParser(description='Bake stock H1A lightmaps with Blender Cycles')
    commands = parser.add_subparsers(dest='command', required=True)
    check = commands.add_parser('check')
    check.add_argument('-v', '--verbose', action='store_true')
    for name in ('bake', 'copy', 'validate', 'diff', 'calibrate'):
        command = commands.add_parser(name)
        command.add_argument('-v', '--verbose', action='store_true')
        command.add_argument('--tags', type=Path, default=os.environ.get('HCEEK_TAGS') or None)
        if name == 'validate':
            command.add_argument('bsp')
            command.add_argument('--strict', action='store_true')
            continue
        if name == 'diff':
            command.add_argument('tool_bsp')
            command.add_argument('our_bsp')
            command.add_argument('--work', type=Path, required=True)
            continue
        command.add_argument('scenario')
        command.add_argument('bsp')
        if name == 'calibrate':
            command.add_argument('--work', type=Path, required=True)
            command.add_argument('--settings', type=Path)
            continue
        command.add_argument('--out-bsp', required=name == 'copy')
        command.add_argument('--out-scenario', required=name == 'copy')
        if name in ('bake', 'copy'):
            command.add_argument('--no-backups', action='store_true')
        if name == 'bake':
            command.add_argument('--settings', type=Path)
            command.add_argument('--work', type=Path)
            command.add_argument('--charts', type=Path)
            command.add_argument('--sources', choices=('all', 'sun', 'emit'), default='all')
            command.add_argument('--dry-run', action='store_true')
    return parser


def main(argv=None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == 'check':
        from .check import run_checks
        results = run_checks()
        for name, ok, detail in results:
            print(f'{"ok  " if ok else "FAIL"} {name}: {detail}')
        return int(not all(ok for _, ok, _ in results))
    if args.tags is None:
        parser.error('pass --tags with the editing-kit tags folder, or set HCEEK_TAGS')
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format='%(levelname)s: %(message)s')
    if args.command == 'copy':
        copy_tags(args.tags, args.scenario, args.bsp, args.out_bsp, args.out_scenario,
                  backups=not args.no_backups)
        return 0
    if args.command == 'bake':
        result = bake(args.tags, args.scenario, args.bsp, _blender_fallback(load_settings(args.settings)),
                      args.out_bsp, args.out_scenario, args.work, args.sources, args.charts, args.dry_run,
                      backups=not args.no_backups)
        print(f'Work: {result.work}')
        return 0
    if args.command == 'calibrate':
        data = calibrate(args.tags, args.scenario, args.bsp, args.work, _blender_fallback(load_settings(args.settings)))
        print((args.work / 'calibration.md').read_text())
        if not all(data['acceptance'].values()):
            print(f'acceptance failed: the fit is in {args.work / "calibration.json"}')
            return 1
        return 0
    if args.command == 'diff':
        return _diff(args)
    path = Path(args.bsp)
    if not path.is_file():
        path = resolve_reference_path(args.tags, 'sbsp', args.bsp)
    bsp = read_tag(path)
    group, reference = bsp.root.reference('lightmaps_bitmap')
    bitmap_path = resolve_reference_path(args.tags, group, reference) if group == 'bitm' and reference else None
    bitmap = read_tag(bitmap_path) if bitmap_path is not None and bitmap_path.is_file() else None
    problems = validate_bsp(bsp, bitmap, strict_compression=args.strict)
    for problem in problems:
        print(problem)
    return int(bool(problems))


def _diff(args) -> int:
    tool_bsp, our_bsp = (read_tag(_bsp_path(args.tags, name)) for name in (args.tool_bsp, args.our_bsp))
    # (the bake's work folder holds the surface mapping; a copy of tool's tags has none and keeps the order)
    count = len(our_bsp.root.block('surfaces'))
    if len(tool_bsp.root.block('surfaces')) != count:
        raise SystemExit('the two BSPs have different surface counts; pair them with a bake work folder instead')
    new_to_input = _read_mapping(args.work, count)
    tool_bitmap, our_bitmap = load_pages(args.tags, tool_bsp), load_pages(args.tags, our_bsp)
    samples = paired_samples(tool_bsp, tool_bitmap, bitmap_pages(our_bitmap), our_bsp, new_to_input)
    data = report(samples)
    args.work.mkdir(parents=True, exist_ok=True)
    (args.work / 'diff.md').write_text(render_markdown(data))
    print((args.work / 'diff.md').read_text())
    return 0


def _bsp_path(tags: Path, name: str) -> Path:
    path = Path(name)
    return path if path.is_file() else resolve_reference_path(tags, 'sbsp', name)


def _read_mapping(work: Path, count: int):
    saved = work / 'new_to_input.npy'
    if saved.is_file():
        return np.load(saved)
    logger.warning('no new_to_input.npy in %s: pairing surfaces by identity', work)
    return np.arange(count)
