"""Read the light tags attached to scenery and give them as tool's light records.

Tool reads only these fields of a light tag: `flags` (the dynamic bit), the radiosity intensity and
colour, and the falloff and cutoff angles. The light's game colour and radius are not read, so a light
bakes as radiosity colour times radiosity intensity whatever it looks like in the game.
"""

from dataclasses import dataclass
import logging
import math
from pathlib import Path

import numpy as np

from ..tagio.tagfile import read_tag, resolve_reference_path


logger = logging.getLogger(__name__)
DYNAMIC = 0x1
POINT = 2
SPOT = 3
# tool's constants at 0x9b63a8 and 0x9b67dc
LAST_LEVEL = np.float32(1 / 255)
PI = math.pi


@dataclass(frozen=True)
class LightDefinition:
    dynamic: bool
    intensity: float
    colour: tuple[float, float, float]
    falloff: float
    cutoff: float


@dataclass
class SceneLight:
    position: np.ndarray
    axis: np.ndarray
    rgb: np.ndarray
    kind: int
    reach: np.float32
    cos_cutoff: np.float32
    cos_falloff: np.float32
    owner: int
    tag: str


def read_light(tags_root: Path, path: str, cache: dict) -> LightDefinition | None:
    if path not in cache:
        cache[path] = None
        light_path = resolve_reference_path(tags_root, 'ligh', path)
        if not light_path.is_file():
            logger.warning('light tag is missing: %s', light_path)
            return None
        root = read_tag(light_path).root
        cache[path] = LightDefinition(
            bool(root.get('flags') & DYNAMIC), float(root.get('radiosity.intensity')),
            tuple(float(x) for x in root.get('radiosity.color')),
            float(root.get('shape.falloff_angle')), float(root.get('shape.cutoff_angle')))
    return cache[path]


def light_kind(falloff: float, cutoff: float) -> int:
    # tool's record rule (found in the executable): a cutoff that is not positive, or both angles at pi, is a point light
    return POINT if cutoff <= 0 or (falloff >= PI and cutoff >= PI) else SPOT


def reach(intensity: float) -> np.float32:
    # the distance where intensity / d^2 drops under one 8-bit level, as tool computes it in float32
    return np.float32(np.sqrt(np.float64(np.float32(intensity) / LAST_LEVEL)))


def bakes(light: LightDefinition) -> bool:
    return not light.dynamic and light.intensity > 0 and max(light.colour) > 0


def ramp(s: float, cos_cutoff: float, cos_falloff: float) -> float:
    # the cutoff test comes first: a falloff wider than the cutoff leaves a step at the cutoff
    if s <= cos_cutoff:
        return 0.0
    if s >= cos_falloff:
        return 1.0
    return (s - cos_cutoff) / (cos_falloff - cos_cutoff)


def _cos32(angle: float) -> np.float32:
    return np.float32(np.cos(np.float64(np.float32(angle))))


def scene_light(definition: LightDefinition, position, axis, owner: int, tag: str) -> SceneLight:
    return SceneLight(
        np.asarray(position, np.float64), np.asarray(axis, np.float64),
        np.asarray(definition.colour, np.float64) * definition.intensity,
        light_kind(definition.falloff, definition.cutoff), reach(definition.intensity),
        _cos32(definition.cutoff), _cos32(definition.falloff), owner, tag)
