from dataclasses import asdict, dataclass, field, fields
import json
import os
from pathlib import Path
import tempfile

from .presets import ADVANCED, PRESET_NAMES


@dataclass
class UserSettings:
    tags: str = ''
    scenario: str = ''
    bsp: str = ''
    blender: str = ''
    preset: str = 'Normal'
    new_names: bool = True
    advanced: dict = field(default_factory=dict)
    advanced_open: bool = False


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
    if values.get('preset', defaults.preset) not in PRESET_NAMES:
        values['preset'] = defaults.preset
    return UserSettings(**values)


def save(us: UserSettings, path: Path | None = None) -> None:
    target = Path(path or settings_path())
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=target.parent, delete=False) as temp:
            temp_path = Path(temp.name)
            temp.write(json.dumps(asdict(us), indent=2))
        os.replace(temp_path, target)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
