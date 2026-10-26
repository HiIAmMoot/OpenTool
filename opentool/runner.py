from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
from pathlib import Path
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


@dataclass
class BakeOutcome:
    status: str
    message: str
    problems: list[str] = field(default_factory=list)
    output_folder: Path | None = None
    work: Path = Path()
    device: str | None = None


def default_work(bsp: str) -> Path:
    return Path(tempfile.gettempdir()) / 'halo_lightmapper' / f'{bsp}-{datetime.now():%Y%m%d-%H%M%S}'


class _EventHandler(logging.Handler):
    def __init__(self, emit):
        super().__init__(logging.INFO)
        self.emit_event = emit
        self.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))

    def emit(self, record):
        try:
            self.emit_event(('log', self.format(record)))
        except Exception:
            pass


def write_lit_scenario(tags, scenario, bsp, out_bsp, out_scenario):
    _, index, _ = pipeline.find_bsp(tags, scenario, bsp)
    path = resolve_reference_path(tags, 'scnr', out_scenario)
    if path.is_file():
        tag = read_tag(path)
        entries = tag.root.block('structure_bsps')
        if len(entries) <= index:
            logger.warning('Existing lit scenario has fewer BSP entries; rebuilding it from the original scenario.')
            tag = read_tag(resolve_reference_path(tags, 'scnr', scenario))
    else:
        tag = read_tag(resolve_reference_path(tags, 'scnr', scenario))
    entries = tag.root.block('structure_bsps')
    entries[index].set_reference('structure_bsp', 'sbsp', out_bsp)
    write_tag(tag, path)


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
                    safe_emit(('log', text))
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
        event_handler = _EventHandler(safe_emit)
        file_handler = logging.FileHandler(work / 'opentool.log', mode='w', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        root.setLevel(logging.INFO)
        root.addHandler(event_handler)
        root.addHandler(file_handler)
        out_bsp, out_scenario = new_names(request.scenario, bsp_tag_path(request.tags, request.scenario, request.bsp)) if request.new_names else (None, None)
        # keep pipeline reads on the modder's original tags; runner writes the lit scenario so references for other BSPs survive.
        result = pipeline.bake(request.tags, request.scenario, request.bsp, request.settings,
                               out_bsp=out_bsp, out_scenario=None, work=work, progress=progress)
        if out_scenario is not None:
            write_lit_scenario(request.tags, request.scenario, request.bsp, out_bsp, out_scenario)
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
        outcome = BakeOutcome(status, message, problems, output_folder, work, device)
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
