from pathlib import Path, PureWindowsPath

from lightmapper.pipeline import find_bsp
from lightmapper.tagio.tagfile import read_tag, resolve_reference_path


LIT_SUFFIX = '_lit'


def tags_folder_problem(tags: Path) -> str | None:
    tags = Path(tags)
    if not tags.is_dir():
        return 'That folder does not exist.'
    if not (tags / 'globals').is_dir():
        return "This doesn't look like an editing kit tags folder (it has no globals folder). Pick the folder named tags inside the editing kit."
    return None


def scenario_tag_path(tags: Path, file: Path) -> str:
    root = Path(tags).resolve()
    scenario = Path(file).resolve()
    try:
        relative = scenario.relative_to(root)
    except ValueError:
        raise ValueError('The scenario has to be inside the tags folder you picked.') from None
    return str(relative.with_suffix('')).replace('/', '\\')


def list_bsps(tags: Path, scenario: str) -> list[str]:
    tag = read_tag(resolve_reference_path(tags, 'scnr', scenario))
    return [PureWindowsPath(entry.reference('structure_bsp')[1]).name
            for entry in tag.root.block('structure_bsps')]


def bsp_tag_path(tags: Path, scenario: str, bsp: str) -> str:
    return find_bsp(tags, scenario, bsp)[2]


def new_names(scenario: str, bsp_path: str) -> tuple[str, str]:
    def lit(path: str) -> str:
        return path if path.endswith(LIT_SUFFIX) else path + LIT_SUFFIX
    return lit(bsp_path), lit(scenario)
