"""Gather stock sky lighting, surface ambient classes and shader emission.

Cluster sky indices retain scenario order, including repeated sky references.
Unclustered surfaces borrow classes by exact positions across source materials.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
import logging
import math
from pathlib import Path
import struct

import numpy as np

from .formats.bsp import BspGeometry, NO_CLUSTER
from .tagio.tagfile import Tag, read_tag, resolve_reference_path


SIMPLE_PARAMETERIZATION = 0x1
IGNORE_NORMALS = 0x2
# every shader group starts with the same base fields; these groups have a layout of their own
LAYOUT_GROUPS = ('senv', 'sgla', 'soso', 'schi', 'scex')
# these have none, so the base fields are read straight from the tag's first bytes
BASE_ONLY_GROUPS = ('swat', 'smet', 'spla', 'sotr')
# shader groups tool sees light through: a surface of one that is not a cut-out casts no solid shadow
TRANSPARENT_GROUPS = ('sgla', 'schi', 'scex', 'swat', 'smet', 'spla')
TAG_HEADER_SIZE = 64
# flags u16, detail level s16, power, emitted colour, tint colour (big-endian, as in the tag)
BASE_RADIOSITY = struct.Struct('>Hhf3f3f')
logger = logging.getLogger(__name__)


@dataclass
class Sun:
    direction: np.ndarray
    angle: float
    color: np.ndarray
    power: float
    classes: tuple[int, ...]
    # the sky light's own direction, shared by its grid samples (None for a lone sun)
    centre: np.ndarray | None = None


@dataclass
class Lighting:
    suns: list[Sun]
    ambient: dict[int, np.ndarray]


@dataclass
class ShaderRadiosity:
    flags: int
    detail_level: int
    power: float
    emitted: np.ndarray
    tint: np.ndarray


def sky_direction(yaw: float, pitch: float) -> np.ndarray:
    return np.array([math.cos(pitch) * math.cos(yaw),
                     math.cos(pitch) * math.sin(yaw), math.sin(pitch)])


# each grid sample is a point direction: the spread comes from the grid, not from the sun's disc
GRID_SUN_ANGLE = 0.001


def sky_samples(light, classes: tuple[int, ...], n: int) -> list[Sun]:
    if not classes:
        return []
    yaw, pitch = light.get('radiosity.direction')
    half = light.get('radiosity.diameter')
    color, power = np.array(light.get('radiosity.color')), light.get('radiosity.power')
    if power < 0:
        logger.warning('sky light has negative radiosity power %s; clamping it to zero', power)
        power = 0.0
    centre = sky_direction(yaw, pitch)
    if n == 1:
        return [Sun(centre, GRID_SUN_ANGLE, color, power, classes, centre)]
    return [Sun(sky_direction(yaw + half * (2 * i / (n - 1) - 1), pitch + half * (2 * j / (n - 1) - 1)),
                GRID_SUN_ANGLE, color, power / n ** 2, classes, centre)
            for i in range(n) for j in range(n)]


def gather_lighting(scenario: Tag, tags_root: Path, sky_grid: int = 4) -> Lighting:
    suns, ambient = [], {}
    for index, reference in enumerate(scenario.root.block('skies')):
        group, path = reference.reference('sky')
        if group != 'sky ' or not path:
            continue
        sky = read_tag(resolve_reference_path(tags_root, group, path)).root
        ambient[index] = (sky.get('outdoor_ambient_radiosity.power')
                          * np.array(sky.get('outdoor_ambient_radiosity.color')))
        if index == 0:
            ambient[-1] = (sky.get('indoor_ambient_radiosity.power')
                           * np.array(sky.get('indoor_ambient_radiosity.color')))
        for light in sky.block('lights'):
            flags = light.get('radiosity.flags')
            classes = (index,) if flags & 1 else ()
            if index == 0 and flags & 2:
                classes += (-1,)
            suns.extend(sky_samples(light, classes, sky_grid))
    return Lighting(suns, ambient)


def _shader_radiosity(values) -> ShaderRadiosity:
    flags, detail_level, power, *colours = values
    return ShaderRadiosity(flags, detail_level, power, np.array(colours[:3]), np.array(colours[3:]))


def _read_base_radiosity(file: Path, group: str) -> ShaderRadiosity:
    with open(file, 'rb') as handle:
        header = handle.read(TAG_HEADER_SIZE)
        data = handle.read(BASE_RADIOSITY.size)
    if len(header) != TAG_HEADER_SIZE or len(data) != BASE_RADIOSITY.size:
        raise ValueError(f'{file} is too short to hold the shader base fields')
    if header[36:40] != group.encode():
        raise ValueError(f'{file} is a {header[36:40].decode("latin-1")} tag, not {group}')
    return _shader_radiosity(BASE_RADIOSITY.unpack(data))


def read_radiosity(tags_root: Path, group: str, path: str) -> ShaderRadiosity:
    if group not in LAYOUT_GROUPS + BASE_ONLY_GROUPS:
        raise ValueError(f'unsupported radiosity shader group {group!r}')
    file = resolve_reference_path(tags_root, group, path)
    if group in BASE_ONLY_GROUPS:
        try:
            result = _read_base_radiosity(file, group)
        except (OSError, ValueError, struct.error) as error:
            logger.warning('cannot read %s shader %s: %s; treating its emission as zero', group, path, error)
            return ShaderRadiosity(0, 1, 0.0, np.zeros(3), np.ones(3))
    else:
        root = read_tag(file).root
        result = _shader_radiosity((root.get('radiosity.flags'), root.get('radiosity.detail_level'),
                                    root.get('radiosity.power'),
                                    *root.get('radiosity.color_of_emitted_light'),
                                    *root.get('radiosity.tint_color')))
    if result.flags & IGNORE_NORMALS:
        logger.warning('shader %s has "ignore normals" set, which is not supported: its receivers '
                       'keep their cosine', path)
    return result


def emission(r: ShaderRadiosity) -> np.ndarray:
    return r.power * np.asarray(r.emitted) if r.power > 0 else np.zeros(3)


def _position_subsets(positions):
    points = sorted(set(tuple(p) for p in positions))
    for size in range(1, len(points) + 1):
        for subset in combinations(points, size):
            yield subset, 1 if size % 2 else -1


def surface_classes(geometry: BspGeometry, sky_count: int) -> np.ndarray:
    for cluster, sky in enumerate(geometry.cluster_sky):
        if sky < -1 or sky >= sky_count:
            raise ValueError(f'cluster {cluster} has sky index {sky}; scenario has {sky_count} skies')
    assigned = geometry.surface_cluster != NO_CLUSTER
    result = np.full(len(geometry.surface_cluster), -1, dtype=np.int16)
    result[assigned] = geometry.cluster_sky[geometry.surface_cluster[assigned]]
    outdoor = Counter(int(c) for c in result[assigned] if c >= 0)
    fallback = min(outdoor, key=lambda c: (-outdoor[c], c)) if outdoor else -1
    result[~assigned] = fallback

    # inclusion-exclusion counts each neighbouring surface once, even across three
    # shared vertices, without scanning high-valence adjacency lists per receiver
    votes = defaultdict(Counter)
    for group in geometry.groups:
        for tri, surface in zip(group.tris, group.surface_ids):
            if assigned[surface]:
                for subset, _ in _position_subsets(group.env['position'][tri]):
                    votes[subset][int(result[surface])] += 1
    neighbours = 0
    for group in geometry.groups:
        for tri, surface in zip(group.tris, group.surface_ids):
            if assigned[surface]:
                continue
            counts = Counter()
            for subset, sign in _position_subsets(group.env['position'][tri]):
                for light_class, count in votes.get(subset, {}).items():
                    counts[light_class] += sign * count
            if counts:
                result[surface] = min(counts, key=lambda c: (-counts[c], c))
                neighbours += 1
    logger.info('surface classes: clustered=%d neighbours=%d fallback=%d',
                np.count_nonzero(assigned), neighbours,
                np.count_nonzero(~assigned) - neighbours)
    return result
