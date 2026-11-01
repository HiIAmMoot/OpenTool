"""Command-line entry points for baking, isolated tag copies, validation, differential and calibration."""

import argparse
from dataclasses import fields, replace
import json
import logging
import os
from pathlib import Path
import sys

import numpy as np

from . import kept, report as bake_report
from .calibrate import calibrate
from .blender_find import find_blender
from .errors import KeptBakeError
from .diff import load_pages, paired_samples, bitmap_pages, render_markdown, report
from .pipeline import bake, copy_tags
from . import settings as settings_module
from .settings import Settings, load_settings
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
    check.add_argument('--skip', action='append', choices=('blender',), default=[])
    write = commands.add_parser('write')
    write.add_argument('-v', '--verbose', action='store_true')
    write.add_argument('work', type=Path)
    # (no HCEEK_TAGS default here: the kept bake knows its own tags, and the variable could name others)
    write.add_argument('--tags', type=Path)
    write.add_argument('--settings', type=Path)
    write.add_argument('--out-bsp')
    write.add_argument('--out-scenario')
    write.add_argument('--no-backups', action='store_true')
    wide = write.add_mutually_exclusive_group()
    wide.add_argument('--32bit', dest='lightmap_32bit', action='store_const', const=True)
    wide.add_argument('--16bit', dest='lightmap_32bit', action='store_const', const=False)
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
            command.add_argument('--keep', action='store_true')
            command.add_argument('--32bit', dest='lightmap_32bit', action='store_true')
    return parser


def main(argv=None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == 'bake' and args.keep and (args.dry_run or args.sources != 'all'):
        parser.error('--keep needs a full bake that writes tags: drop --dry-run and --sources')
    if args.command == 'check':
        from .check import SKIPPED, run_checks
        results = run_checks(tuple(args.skip))
        for name, ok, detail in results:
            label = 'skip' if detail == SKIPPED else 'ok  ' if ok else 'FAIL'
            print(f'{label} {name}: {detail}')
        return int(not all(ok for _, ok, _ in results))
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format='%(levelname)s: %(message)s')
    if args.command == 'write':
        return _write(args)
    if args.tags is None:
        parser.error('pass --tags with the editing-kit tags folder, or set HCEEK_TAGS')
    if args.command == 'copy':
        copy_tags(args.tags, args.scenario, args.bsp, args.out_bsp, args.out_scenario,
                  backups=not args.no_backups)
        return 0
    if args.command == 'bake':
        settings = _blender_fallback(load_settings(args.settings))
        if args.lightmap_32bit:
            settings.lightmap_32bit = True
        work = args.work
        if args.keep and work is None:
            try:
                work = kept.create_work(kept.kept_root(), args.tags, args.scenario, args.bsp)
            except OSError:
                print(f'OpenTool could not make a folder for kept bakes in {kept.kept_root()}. '
                      'Leave out --keep, or move OpenTool to a folder you can write to.', file=sys.stderr)
                return 1
        result = bake(args.tags, args.scenario, args.bsp, settings,
                      args.out_bsp, args.out_scenario, work, args.sources, args.charts, args.dry_run,
                      backups=not args.no_backups, keep=args.keep)
        summary = bake_report.read_summary(Path(result.work) / 'summary.json')
        if summary is not None:
            print_report(summary)
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


def print_report(summary: dict) -> None:
    for line in bake_report.bake_report(summary):
        print(line)


def _write(args) -> int:
    try:
        manifest = kept.read_manifest(args.work)
        if args.settings is not None:
            settings = load_settings(args.settings)
        else:
            known = {f.name for f in fields(Settings)}
            settings = Settings(**{k: v for k, v in manifest.settings.items() if k in known})
        # (without --32bit or --16bit the kept value stands: a 32-bit bake is not silently written 16-bit)
        if args.lightmap_32bit is not None:
            settings = replace(settings, lightmap_32bit=args.lightmap_32bit)
        result = bake(args.tags or Path(manifest.tags), manifest.scenario, manifest.bsp, settings,
                      args.out_bsp, args.out_scenario or manifest.output_scenario or None, args.work, reuse=True,
                      backups=not args.no_backups)
    except KeptBakeError as error:
        print(error, file=sys.stderr)
        return 1
    summary_file = args.work / 'write_summary.json'
    if summary_file.is_file():
        summary = json.loads(summary_file.read_text())
        print_report(summary)
        for warning in summary['warnings']:
            print(f'warning: {warning}')
    print(f'Work: {result.work}')
    return 0


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
