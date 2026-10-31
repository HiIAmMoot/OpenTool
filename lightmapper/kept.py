"""Kept bakes: a bake's intermediate results, stored in the app folder so the tags can be written again later.

One folder per level (tags folder, scenario and BSP) holds the last kept bake; a kept bake is a work folder whose
kept/ subfolder carries the sources and results the finish needs, described by kept.json.
"""

import hashlib
import json
import logging
import shutil
import sys
import zipfile
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path

import numpy as np

from . import frozen
from .blender_job import BakeResult
from .errors import KeptBakeError
from .settings import Settings


logger = logging.getLogger(__name__)

KEPT_FOLDER = 'kept bakes'
MANIFEST = 'kept.json'
VERSION = 1
# these cannot change what the pages hold: the Blender path and its time limit do not reach the output, and 32-bit
# only changes how the tags are written
ALLOWED_CHANGES = frozenset({'lightmap_32bit', 'blender', 'timeout_s'})
SUPERSEDED = ('out.npz', 'gpu_out.npz', 'cpu_out.npz', 'vertex_out.npz', 'glow_out.npz', 'diffuse.npz', 'glow_pages.npz')
SOURCE = 'source.scenario_structure_bsp'
TEMPLATE = 'template.bitmap'
SCENARIO = 'scenario.scenario'
CHARTS = 'charts.npz'
RESULT = 'result.npz'
GLOW = 'glow.npz'

# diffuse_pages is as big as the pages and the finish never reads it
_UNKEPT = ('diffuse_pages',)
_LISTS = ('pages', 'written')
_INTS = ('vertex_rays_cast', 'samples_used', 'texel_samples_used')


@dataclass
class Manifest:
    tags: str
    scenario: str
    bsp: str
    input_path: str
    has_template: bool
    vertex_device: str
    glow_device: str
    glow_reason: str
    settings: dict
    kept_at: str


def kept_root() -> Path:
    return frozen.app_dir() / KEPT_FOLDER


def level_folder(root: Path, tags: Path, scenario: str, bsp: str) -> Path:
    digest = hashlib.sha256(f'{Path(tags).resolve()}|{scenario}|{bsp}'.encode()).hexdigest()[:16]
    return root / f'{bsp}-{digest}'


def new_work(root: Path, tags: Path, scenario: str, bsp: str) -> Path:
    return level_folder(root, tags, scenario, bsp) / f'{datetime.now():%Y%m%d-%H%M%S}'


def create_work(root: Path, tags: Path, scenario: str, bsp: str) -> Path:
    first = new_work(root, tags, scenario, bsp)
    first.parent.mkdir(parents=True, exist_ok=True)
    work, number = first, 1
    while True:
        try:
            work.mkdir()
        except FileExistsError:
            # (the folder names are timestamps by the second: a second kept bake of the level inside it must not share a folder)
            number += 1
            work = first.with_name(f'{first.name}-{number}')
        else:
            return work


def kept_dir(work: Path) -> Path:
    return work / 'kept'


def write_manifest(work: Path, manifest: Manifest) -> None:
    (work / MANIFEST).write_text(json.dumps({'version': VERSION, **asdict(manifest)}, indent=1))


def _refusal(work: Path) -> KeptBakeError:
    return KeptBakeError(f'{work} is not a kept bake OpenTool can write again.')


def read_manifest(work: Path) -> Manifest:
    refusal = _refusal(work)
    try:
        data = json.loads((work / MANIFEST).read_text())
        if not isinstance(data, dict):
            raise refusal
        if data.pop('version', None) != VERSION:
            raise refusal
        manifest = Manifest(**data)
    except (OSError, ValueError, TypeError) as exc:
        raise refusal from exc
    needed = [SOURCE, SCENARIO, CHARTS, RESULT] + ([TEMPLATE] if manifest.has_template else [])
    if not all((kept_dir(work) / name).is_file() for name in needed):
        raise refusal
    return manifest


def latest(root: Path, tags: Path, scenario: str, bsp: str) -> Path | None:
    level = level_folder(root, tags, scenario, bsp)
    if not level.is_dir():
        return None
    # the folder names are timestamps, so the name order is the age order
    for work in sorted((p for p in level.iterdir() if p.is_dir()), key=lambda p: p.name, reverse=True):
        try:
            read_manifest(work)
        except KeptBakeError:
            continue
        return work
    return None


def prune(work: Path) -> None:
    for other in work.parent.iterdir():
        if other != work and other.is_dir():
            failed = []

            def note(function, path, error):
                failed.append(path)

            # (a folder another program holds open must not fail a bake that already succeeded: it stays, named in the log)
            if sys.version_info >= (3, 12):
                shutil.rmtree(other, onexc=note)
            else:
                shutil.rmtree(other, onerror=note)
            if failed:
                logger.warning('could not remove the older kept bake %s', other)


def save_sources(work: Path, input_file: Path, template_file: Path | None, scenario_file: Path, charts_file: Path) -> None:
    folder = kept_dir(work)
    shutil.copyfile(input_file, folder / SOURCE)
    if template_file is not None:
        shutil.copyfile(template_file, folder / TEMPLATE)
    shutil.copyfile(scenario_file, folder / SCENARIO)
    shutil.copyfile(charts_file, folder / CHARTS)


def save_result(work: Path, result: BakeResult) -> None:
    arrays = {}
    for field in fields(BakeResult):
        value = getattr(result, field.name)
        if field.name in _UNKEPT or value is None:
            continue
        if field.name in _LISTS:
            arrays[f'{field.name}_count'] = np.array(len(value))
            arrays.update({f'{field.name}_{i}': np.asarray(item) for i, item in enumerate(value)})
        elif field.name == 'stages':
            arrays[field.name] = np.array(json.dumps(value))
        elif field.name == 'device':
            arrays[field.name] = np.array(str(value))
        else:
            arrays[field.name] = np.asarray(value)
    np.savez(kept_dir(work) / RESULT, **arrays)


def load_result(work: Path) -> BakeResult:
    try:
        return _load_result(work)
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as exc:
        raise _refusal(work) from exc


def _load_result(work: Path) -> BakeResult:
    values = {field.name: None for field in fields(BakeResult) if field.name not in _UNKEPT}
    with np.load(kept_dir(work) / RESULT, allow_pickle=False) as data:
        for field in fields(BakeResult):
            if field.name in _UNKEPT:
                continue
            if field.name in _LISTS:
                # (pages and written are not optional: a missing count is a damaged file, not an empty list)
                values[field.name] = [data[f'{field.name}_{i}'] for i in range(int(data[f'{field.name}_count']))]
            elif field.name in data:
                value = data[field.name]
                if field.name == 'stages':
                    value = json.loads(str(value))
                elif field.name == 'device':
                    value = str(value)
                elif field.name in _INTS:
                    value = int(value)
                values[field.name] = value
    if values['device'] is None:
        raise KeyError('device')
    return BakeResult(**values)


def save_glow(work: Path, pages: list) -> None:
    np.savez(kept_dir(work) / GLOW, **{f'page_{i}': np.asarray(page) for i, page in enumerate(pages)})


def load_glow(work: Path) -> list[np.ndarray]:
    try:
        with np.load(kept_dir(work) / GLOW, allow_pickle=False) as data:
            return [data[f'page_{i}'] for i in range(len(data.files))]
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        raise _refusal(work) from exc


def drop_superseded(work: Path) -> None:
    for name in SUPERSEDED:
        (work / name).unlink(missing_ok=True)
    for lamps in work.glob('lamps_*_out.npz'):
        lamps.unlink()


def setting_differences(kept: dict, wanted: Settings) -> list[str]:
    differences = []
    for name, value in asdict(wanted).items():
        if name in ALLOWED_CHANGES:
            continue
        if name not in kept:
            differences.append(f'{name}: not in the kept bake')
        # (the kept dict came through JSON, which turns tuples into lists)
        elif kept[name] != json.loads(json.dumps(value)):
            differences.append(f'{name}: {kept[name]!r} when kept, {value!r} now')
    return differences
