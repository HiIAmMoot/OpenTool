from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json
import logging
from pathlib import Path, PureWindowsPath
import tempfile
from typing import Callable

from lightmapper import pipeline
from lightmapper.progress import BlenderLogFollower, parse_line
from lightmapper.settings import Settings
from lightmapper.tagio.tagfile import read_tag, resolve_reference_path, write_tag
from lightmapper.validate import validate_bsp
from .errors import explain
from .form import bsp_tag_path, new_names

logger = logging.getLogger(__name__)


@dataclass
class BakeRequest:
    tags: Path
    scenario: str
    bsp: str
    settings: Settings
    new_names: bool
    work: Path | None = None
    verbose: bool = False
    backups: bool = True
    charts: Path | None = None


@dataclass
class BakeOutcome:
    status: str
    message: str
    problems: list[str] = field(default_factory=list)
    output_folder: Path | None = None
    work: Path = Path()
    device: str | None = None
    devices: dict | None = None
    rays: dict | None = None
    warnings: list[str] = field(default_factory=list)


def default_work(bsp: str) -> Path:
    return Path(tempfile.gettempdir()) / 'halo_lightmapper' / f'{bsp}-{datetime.now():%Y%m%d-%H%M%S}'


def chart_cache_path(tags: Path, scenario: str, bsp: str) -> Path:
    digest = hashlib.sha256(f'{Path(tags).resolve()}|{scenario}|{bsp}'.encode()).hexdigest()[:16]
    return Path(tempfile.gettempdir()) / 'halo_lightmapper' / 'charts' / f'{digest}.npz'


class _EventHandler(logging.Handler):
    def __init__(self, emit, verbose):
        super().__init__(logging.INFO)
        self.emit_event = emit
        self.verbose = verbose
        self.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))

    def filter(self, record):
        return (record.levelno >= logging.WARNING or self.verbose
                or record.getMessage().startswith('stage: '))

    def emit(self, record):
        try:
            self.emit_event(('log', self.format(record)))
        except Exception:
            pass


def write_lit_scenario(tags, scenario, bsp, out_bsp, out_scenario, backups=True):
    path = resolve_reference_path(tags, 'scnr', out_scenario)
    tag = read_tag(resolve_reference_path(tags, 'scnr', scenario))
    entries = tag.root.block('structure_bsps')
    for entry in entries:
        _, original_bsp = entry.reference('structure_bsp')
        # (bsp is the name the modder picked, as find_bsp matches it, not the tag path)
        if PureWindowsPath(original_bsp).name == bsp:
            lit_bsp = out_bsp
        else:
            earlier_lit_bsp = f'{original_bsp}_lit'
            if resolve_reference_path(tags, 'sbsp', earlier_lit_bsp).is_file():
                lit_bsp = earlier_lit_bsp
            else:
                lit_bsp = original_bsp
        entry.set_reference('structure_bsp', 'sbsp', lit_bsp)
    # The lit scenario must match the scenario the bake read; earlier lit BSPs are found by name, not position.
    write_tag(tag, path, backup=backups)


def read_summary(work: Path) -> tuple[dict | None, dict | None, list[str]]:
    try:
        summary = json.loads((work / 'summary.json').read_text())
        devices, rays, warnings = summary['devices'], summary.get('rays'), summary.get('warnings', [])
        if (isinstance(devices, dict) and {'page_bake', 'vertex_pass', 'denoise'} <= devices.keys()
                and isinstance(warnings, list)):
            return devices, rays if isinstance(rays, dict) else None, [str(text) for text in warnings]
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None, None, []


def run_bake(request: BakeRequest, emit: Callable[[tuple], None]) -> BakeOutcome:
    work = Path(request.work) if request.work is not None else default_work(request.bsp)
    root = logging.getLogger()
    old_level = root.level
    event_handler = None
    file_handler = None
    follower = None
    outcome = None
    device = None

    def safe_emit(event):
        try:
            emit(event)
        except Exception:
            logger.exception('Could not report bake event')

    def progress(stage):
        nonlocal follower
        try:
            if follower is not None:
                follower.stop(); follower.join(); follower = None
            safe_emit(('stage', stage))
            if stage == 'blender':
                try:
                    total = len(json.loads((work / 'job.json').read_text())['pages'])
                except (OSError, ValueError, KeyError, TypeError):
                    total = 0

                def line(text):
                    nonlocal device
                    logging.getLogger('opentool.blender').info(text)
                    parsed = parse_line(text)
                    if parsed and parsed[0] == 'device':
                        device = parsed[1]; safe_emit(parsed)
                    elif parsed and parsed[0] == 'page':
                        safe_emit(('pages', parsed[1] + 1, total))

                follower = BlenderLogFollower(blender_log, line)
                try:
                    follower.start()
                except Exception:
                    logger.exception('Could not start Blender log follower')
                    follower = None
        except Exception:
            logger.exception('Could not report bake progress')

    try:
        work.mkdir(parents=True, exist_ok=True)
        blender_log = work / 'blender.log'
        try:
            # a reused work folder would otherwise replay progress from the previous run.
            blender_log.unlink()
        except FileNotFoundError:
            pass
        event_handler = _EventHandler(emit, request.verbose)
        file_handler = logging.FileHandler(work / 'opentool.log', mode='w', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        root.setLevel(logging.INFO)
        root.addHandler(event_handler)
        root.addHandler(file_handler)
        out_bsp, out_scenario = new_names(request.scenario, bsp_tag_path(request.tags, request.scenario, request.bsp)) if request.new_names else (None, None)
        # keep pipeline reads on the modder's original tags; runner writes the lit scenario so references for other BSPs survive.
        result = pipeline.bake(request.tags, request.scenario, request.bsp, request.settings,
                               out_bsp=out_bsp, out_scenario=None, work=work, progress=progress,
                               backups=request.backups, charts=request.charts)
        if out_scenario is not None:
            write_lit_scenario(request.tags, request.scenario, request.bsp, out_bsp, out_scenario,
                               backups=request.backups)
        safe_emit(('stage', 'validate'))
        written_bsp = out_bsp or bsp_tag_path(request.tags, request.scenario, request.bsp)
        bsp_tag = read_tag(resolve_reference_path(request.tags, 'sbsp', written_bsp))
        bitmap_path = resolve_reference_path(request.tags, 'bitm', bsp_tag.root.reference('lightmaps_bitmap')[1])
        bitmap_tag = read_tag(bitmap_path)
        problems = validate_bsp(bsp_tag, bitmap_tag)
        status = 'problems' if problems else 'done'
        message = ('Finished with problems. The tags were written; the problems are listed in the log.' if problems
                   else f'Done. The lit level is {out_scenario or request.scenario}.')
        output_folder = resolve_reference_path(request.tags, 'sbsp', written_bsp).parent
        outcome = BakeOutcome(status, message, problems, output_folder, work, device, *read_summary(work))
    except Exception as exc:
        logger.exception('Bake failed')
        outcome = BakeOutcome('failed', explain(exc, work), [], None, work, device)
    finally:
        if follower is not None:
            try:
                follower.stop(); follower.join()
            except Exception:
                logger.exception('Could not stop Blender log follower')
        if event_handler is not None:
            root.removeHandler(event_handler)
            event_handler.close()
        if file_handler is not None:
            root.removeHandler(file_handler)
            file_handler.close()
        root.setLevel(old_level)
    safe_emit(('done', outcome))
    return outcome
