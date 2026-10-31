from dataclasses import asdict, dataclass, field, fields
import json
import os
from pathlib import Path
import tempfile

from .presets import ADVANCED, CUSTOM, DEVICES, IGNORE_NORMALS, PRESET_NAMES


@dataclass
class UserSettings:
    tags: str = ''
    scenario: str = ''
    bsp: str = ''
    blender: str = ''
    preset: str = 'Normal'
    new_names: bool = False
    backups: bool = True
    lightmap_32bit: bool = False
    adaptive_sampling: bool = True
    ignore_normals_mode: str = 'tool'
    advanced: dict = field(default_factory=dict)
    advanced_open: bool = False
    verbose: bool = False
    device: str = 'auto'
    # set by load alone, never saved: 0.2.1 wrote no adaptive_sampling, and saved 'enhanced' for every preset
    legacy_0_2_1: bool = field(default=False, compare=False)


def settings_path() -> Path:
    return Path(os.environ.get('APPDATA') or Path.home()) / 'OpenTool' / 'settings.json'


def load(path: Path | None = None) -> UserSettings:
    defaults = UserSettings()
    try:
        data = json.loads(Path(path or settings_path()).read_text())
    except (OSError, ValueError, RecursionError):
        return defaults
    if not isinstance(data, dict):
        return defaults
    values = {}
    for item in fields(UserSettings):
        if item.name == 'legacy_0_2_1':
            values[item.name] = 'adaptive_sampling' not in data
            continue
        if item.name not in data:
            continue
        value = data[item.name]
        default = getattr(defaults, item.name)
        if item.name == 'advanced':
            if isinstance(value, dict):
                values[item.name] = {key: entry for key, entry in value.items()
                                     if key in ADVANCED and isinstance(entry, str)}
            continue
        else:
            valid = type(value) is type(default)
        if valid:
            values[item.name] = value
    if values.get('preset', defaults.preset) not in (*PRESET_NAMES, CUSTOM):
        values['preset'] = defaults.preset
    if values.get('device', defaults.device) not in tuple(value for _, value in DEVICES):
        values['device'] = defaults.device
    if values.get('ignore_normals_mode', defaults.ignore_normals_mode) not in tuple(value for _, value in IGNORE_NORMALS):
        values['ignore_normals_mode'] = defaults.ignore_normals_mode
    return UserSettings(**values)


def save(us: UserSettings, path: Path | None = None) -> None:
    target = Path(path or settings_path())
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=target.parent, delete=False) as temp:
            temp_path = Path(temp.name)
            # the adaptive_sampling key also marks the file as written by 0.2.2 or later (0.2.1 files lack it)
            saved = {key: value for key, value in asdict(us).items() if key != 'legacy_0_2_1'}
            temp.write(json.dumps(saved, indent=2))
        os.replace(temp_path, target)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
