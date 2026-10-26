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
            logging.warning('Existing lit scenario has fewer BSP entries; rebuilding it from the original scenario.')
            tag = read_tag(resolve_reference_path(tags, 'scnr', scenario))
    else:
        tag = read_tag(resolve_reference_path(tags, 'scnr', scenario))
    entries = tag.root.block('structure_bsps')
    entries[index].set_reference('structure_bsp', 'sbsp', out_bsp)
    write_tag(tag, path)


def run_bake(request: BakeRequest, emit: Callable[[tuple], None]) -> BakeOutcome:
    work = Path(request.work) if request.work is not None else default_work(request.bsp)
    work.mkdir(parents=True, exist_ok=True)
    blender_log = work / 'blender.log'
    try:
        blender_log.unlink()
    except FileNotFoundError:
        pass
    root = logging.getLogger()
    old_level = root.level
    event_handler = _EventHandler(emit)
    file_handler = logging.FileHandler(work / 'opentool.log', mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    root.setLevel(logging.INFO)
    root.addHandler(event_handler)
    root.addHandler(file_handler)
    follower = None
    outcome = None
    device = None

    def progress(stage):
        nonlocal follower
        if follower is not None:
            follower.stop(); follower.join(); follower = None
        emit(('stage', stage))
        if stage == 'blender':
            try:
                total = len(json.loads((work / 'job.json').read_text())['pages'])
            except (OSError, ValueError, KeyError, TypeError):
                total = 0

            def line(text):
                nonlocal device
                emit(('log', text))
                parsed = parse_line(text)
                if parsed and parsed[0] == 'device':
                    device = parsed[1]; emit(parsed)
                elif parsed and parsed[0] == 'page':
                    emit(('pages', parsed[1] + 1, total))

            follower = BlenderLogFollower(blender_log, line)
            follower.start()

    try:
        out_bsp, out_scenario = new_names(request.scenario, bsp_tag_path(request.tags, request.scenario, request.bsp)) if request.new_names else (None, None)
        result = pipeline.bake(request.tags, request.scenario, request.bsp, request.settings,
                               out_bsp=out_bsp, out_scenario=None, work=work, progress=progress)
        if out_scenario is not None:
            write_lit_scenario(request.tags, request.scenario, request.bsp, out_bsp, out_scenario)
        emit(('stage', 'validate'))
        problems = validate_bsp(result.bsp_tag, result.bitmap_tag)
        status = 'problems' if problems else 'done'
        outcome = BakeOutcome(status, 'Finished with problems.' if problems else 'Bake finished.', problems,
                              Path(request.tags), work, device)
    except Exception as exc:
        logging.exception('Bake failed')
        outcome = BakeOutcome('failed', explain(exc, work), [], None, work, device)
    finally:
        if follower is not None:
            follower.stop(); follower.join()
        root.removeHandler(event_handler); root.removeHandler(file_handler)
        event_handler.close(); file_handler.close(); root.setLevel(old_level)
    emit(('done', outcome))
    return outcome
