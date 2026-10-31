"""Turn scenery placements into world-space occluder triangles from render models.

The shadow casters are the highest-detail gbxmodel parts, as the player sees
them, never the collision model. Vertices are already in the default pose, so
a static placement needs only its own transform and node weights are ignored.

A light attached to a scenery sits on a model marker. A marker is a rotation and translation on a
node, so its model-space frame is its node's chain of default poses composed with it, and it turns
with the placement like the geometry does. Each region's chosen permutation brings its own markers.
"""

from collections.abc import Iterator
from dataclasses import dataclass
import logging
import math
from pathlib import Path
import re
import struct

import numpy as np

from ..tagio.tagfile import Element, Tag, read_tag, resolve_reference_path
from .lights import SceneLight, bakes, read_light, scene_light


logger = logging.getLogger(__name__)
NOT_PLACED_AUTOMATICALLY = 0x1
CANNOT_BE_CHOSEN_RANDOMLY = 0x1
NONE = -1


@dataclass
class Occluders:
    positions: np.ndarray
    tris: np.ndarray
    uv: np.ndarray
    tri_shader: np.ndarray
    shaders: list[tuple[str, str, float, float]]
    tri_owner: np.ndarray


@dataclass
class ModelMarkers:
    node0: np.ndarray
    named: dict[str, np.ndarray]


@dataclass
class ModelGeometry:
    positions: np.ndarray
    tris: np.ndarray
    uv: np.ndarray
    tri_shader: np.ndarray
    shaders: list[tuple[str, str, float, float]]
    markers: ModelMarkers | None = None


@dataclass
class SceneryDefinition:
    origin_offset: np.ndarray
    model: tuple[str, str]
    lights: tuple[tuple[str, str], ...] = ()


@dataclass
class Placement:
    index: int
    position: np.ndarray
    rotation: tuple
    definition: SceneryDefinition
    desired: int


def decode_strip(indices: np.ndarray) -> np.ndarray:
    s = np.asarray(indices, np.int64)
    if len(s) < 3:
        return np.zeros((0, 3), np.int32)
    a, b, c = s[:-2], s[1:-1], s[2:]
    odd = (np.arange(len(a)) & 1) == 1
    # odd triangles swap their last two so every one keeps the same winding
    tris = np.stack([a, np.where(odd, c, b), np.where(odd, b, c)], axis=1)
    keep = (tris >= 0).all(axis=1) & (a != b) & (b != c) & (a != c)
    return tris[keep].astype(np.int32)


def placement_rotation(yaw: float, pitch: float, roll: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([
        [cy * cp, sy * cr - cy * sp * sr, cy * sp * cr + sy * sr],
        [-(sy * cp), sy * sp * sr + cy * cr, cy * sr - sy * sp * cr],
        [-sp, -(cp * sr), cp * cr]])


def quaternion_matrix(q) -> np.ndarray:
    # columns forward, left, up; the engine turns the opposite way to the textbook quaternion matrix
    x, y, z, w = (float(v) for v in q)
    # (the engine normalises, so an exporter's non-unit quaternion still turns pure)
    length = math.sqrt(x * x + y * y + z * z + w * w)
    if length > 0:
        x, y, z, w = x / length, y / length, z / length, w / length
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]).T


def _frame(rotation, translation) -> np.ndarray:
    return np.hstack([quaternion_matrix(rotation), np.asarray(translation, np.float64).reshape(3, 1)])


def _compose(outer: np.ndarray, inner: np.ndarray) -> np.ndarray:
    return np.hstack([outer[:, :3] @ inner[:, :3], outer[:, :3] @ inner[:, 3:] + outer[:, 3:]])


def _node_frames(nodes) -> list[np.ndarray | None]:
    frames: list[np.ndarray | None] = [None] * len(nodes)

    def frame(index: int, depth: int = 0) -> np.ndarray | None:
        # (a parent loop in a bad tag ends at the depth bound instead of recursing forever)
        if frames[index] is None and depth <= len(nodes):
            node = nodes[index]
            local = _frame(node.get('default_rotation'), node.get('default_translation'))
            parent = node.get('parent_node')
            if 0 <= parent < len(nodes) and parent != index:
                above = frame(parent, depth + 1)
                local = _compose(above, local) if above is not None else local
            frames[index] = local
        return frames[index]

    for index in range(len(nodes)):
        frame(index)
    return frames


def model_markers(root: Element, picks: list[tuple[Element, int]]) -> ModelMarkers:
    frames = _node_frames(root.block('nodes'))
    named: dict[str, np.ndarray] = {}
    for region, pick in picks:
        permutations = region.block('permutations')
        if not 0 <= pick < len(permutations):
            continue
        for marker in permutations[pick].block('markers'):
            node = marker.get('node_index')
            if not 0 <= node < len(frames) or frames[node] is None:
                continue
            name = marker.get('name').split(b'\0')[0].decode('ascii', 'replace').lower()
            named.setdefault(name, _compose(frames[node], _frame(marker.get('rotation'), marker.get('translation'))))
    return ModelMarkers(frames[0] if frames else np.hstack([np.eye(3), np.zeros((3, 1))]), named)


def marker_frame(markers: ModelMarkers, name: str) -> tuple[np.ndarray, np.ndarray, bool]:
    # an empty name asks for node 0 and an unknown one falls back to it (the caller says which)
    frame = markers.named.get(name.lower()) if name else None
    found = frame is not None or not name
    frame = markers.node0 if frame is None else frame
    return frame[:, 3], frame[:, 0], found


def variant_number(permutation: Element) -> int:
    # permutation_number is cache-only (0 in tag files): the cache builder reads it from the name's trailing digits
    name = permutation.get('name').split(b'\0')[0].decode('ascii', 'replace')
    match = re.search(r'\d+$', name)
    return struct.unpack('<h', struct.pack('<H', int(match.group()) & 0xFFFF))[0] if match else 0


def _candidates(region: Element, variant: int) -> list[int]:
    return [index for index, permutation in enumerate(region.block('permutations'))
            if not permutation.get('flags') & CANNOT_BE_CHOSEN_RANDOMLY
            and (variant_number(permutation) == variant
                 or (variant == NONE and variant_number(permutation) < 100))]


def _select(regions: list[Element], variant: int) -> tuple[list[int], bool]:
    picks, complete = [], True
    for region in regions:
        found = _candidates(region, variant)
        if not found and variant != NONE:
            found = _candidates(region, 0)
        # the engine picks one candidate at random; the first keeps the bake repeatable
        picks.append(found[0] if found else 0)
        complete = complete and bool(found)
    return picks, complete


def choose_permutations(regions: list[Element], desired: int) -> list[int]:
    if desired > 0:
        picks, complete = _select(regions, desired)
        if complete:
            return picks
    picks, _ = _select(regions, NONE)
    chosen = 0
    for region, pick in zip(regions, picks):
        if pick < len(region.block('permutations')):
            chosen = variant_number(region.block('permutations')[pick])
            if chosen:
                break
    return _select(regions, chosen)[0] if chosen > 0 else picks


def choose_permutation(region: Element, variant: int) -> int:
    return choose_permutations([region], variant)[0]


def _vertices(part: Element) -> tuple[np.ndarray, np.ndarray]:
    elements = part.block('uncompressed_vertices')
    if not elements:
        return np.zeros((0, 3), np.float32), np.zeros((0, 2), np.float32)
    layout = elements[0].layout
    offsets = {f.key: f.offset for f in layout.fields if f.kind != 'pad'}
    p, t = offsets['position'], offsets['texture_coords']
    raw = np.frombuffer(b''.join(bytes(e.body) for e in elements), np.uint8).reshape(len(elements), layout.size)
    position = np.ascontiguousarray(raw[:, p:p + 12]).view('>f4').reshape(-1, 3)
    uv = np.ascontiguousarray(raw[:, t:t + 8]).view('>f4').reshape(-1, 2)
    return position.astype(np.float32), uv.astype(np.float32)


def _strip(part: Element) -> np.ndarray:
    elements = part.block('triangle_data')
    if not elements:
        return np.zeros(0, np.int16)
    # each element carries three consecutive strip indices
    return np.frombuffer(b''.join(bytes(e.body) for e in elements), '>i2')


def model_geometry(model: Tag, desired: int) -> ModelGeometry:
    root = model.root
    scale_u, scale_v = root.get('base_map_u_scale'), root.get('base_map_v_scale')
    shaders = []
    for entry in root.block('shaders'):
        group, path = entry.reference('shader')
        shaders.append((group or '', path, scale_u, scale_v))
    unbound = len(shaders)
    geometries = root.block('geometries')
    positions, uvs, tris, tri_shader = [], [], [], []
    base = 0
    regions = list(root.block('regions'))
    if desired > 0 and not any(variant_number(p) == desired for r in regions for p in r.block('permutations')):
        logger.warning('no permutation is numbered %d, the model falls back to its default', desired)
    picks = choose_permutations(regions, desired)
    for region, pick in zip(regions, picks):
        permutations = region.block('permutations')
        if not permutations:
            continue
        geometry = permutations[pick].get('super_high')
        if not 0 <= geometry < len(geometries):
            continue
        for part in geometries[geometry].block('parts'):
            position, uv = _vertices(part)
            triangles = decode_strip(_strip(part))
            inside = (triangles < len(position)).all(axis=1)
            if not inside.all():
                logger.warning('%d triangles with out-of-range indices dropped', int((~inside).sum()))
            triangles = triangles[inside]
            shader = part.get('shader_index')
            positions.append(position)
            uvs.append(uv)
            tris.append(triangles + base)
            tri_shader.append(np.full(len(triangles), shader if 0 <= shader < unbound else unbound, np.int32))
            base += len(position)
    if any((t == unbound).any() for t in tri_shader):
        shaders.append(('', '', scale_u, scale_v))
    return ModelGeometry(
        np.concatenate(positions) if positions else np.zeros((0, 3), np.float32),
        np.concatenate(tris).astype(np.int32) if tris else np.zeros((0, 3), np.int32),
        np.concatenate(uvs) if uvs else np.zeros((0, 2), np.float32),
        np.concatenate(tri_shader) if tri_shader else np.zeros(0, np.int32), shaders,
        model_markers(root, list(zip(regions, picks))))


def _load_scenery(tags_root: Path, path: str, cache: dict) -> SceneryDefinition | None:
    if path not in cache:
        cache[path] = None
        scenery_path = resolve_reference_path(tags_root, 'scen', path)
        if not scenery_path.is_file():
            logger.warning('scenery tag is missing: %s', scenery_path)
            return None
        root = read_tag(scenery_path).root
        model_group, model_path = root.reference('model')
        if model_group not in ('mode', 'mod2') or not model_path:
            logger.warning('scenery %s has no render model', path)
            return None
        origin_offset = np.array(root.get('origin_offset'), np.float64)
        # shared by every placement of the scenery, so no one may change it in place
        origin_offset.setflags(write=False)
        lights = []
        for attachment in root.block('attachments'):
            group, light = attachment.reference('type')
            if group == 'ligh' and light:
                lights.append((light, attachment.get('marker').split(b'\0')[0].decode('ascii', 'replace')))
        cache[path] = SceneryDefinition(origin_offset, (model_group, model_path), tuple(lights))
    return cache[path]


def _load_model(tags_root: Path, model: tuple[str, str], desired: int, cache: dict) -> ModelGeometry | None:
    key = (*model, desired)
    if key not in cache:
        cache[key] = None
        if model in cache:
            return None
        model_file = resolve_reference_path(tags_root, *model)
        if not model_file.is_file():
            # remembered under the bare model, so another permutation of it does not warn again
            cache[model] = None
            logger.warning('model tag is missing: %s', model_file)
            return None
        cache[key] = model_geometry(read_tag(model_file), desired)
    return cache[key]


def iter_placements(scenario: Tag, tags_root: Path, bsp_index: int, cache: dict | None = None) -> Iterator[Placement]:
    palette = scenario.root.block('scenery_palette')
    # one dict serves both loaders: scenery paths are strings and model keys are tuples
    cache = {} if cache is None else cache
    for index, placement in enumerate(scenario.root.block('scenery')):
        if placement.get('placement.not_placed') & NOT_PLACED_AUTOMATICALLY:
            continue
        mask = placement.get('bsp_indices')
        # a zero mask is every BSP: all placements of the stock levels carry it and are in play
        if mask and not mask >> bsp_index & 1:
            continue
        type_index = placement.get('type')
        if not 0 <= type_index < len(palette):
            continue
        group, path = palette[type_index].reference('name')
        if group != 'scen' or not path:
            continue
        definition = _load_scenery(tags_root, path, cache)
        if definition is None:
            continue
        yield Placement(index, np.array(placement.get('placement.position'), np.float64),
                        placement.get('placement.rotation'), definition,
                        placement.get('placement.desired_permutation'))


def iter_placed(scenario: Tag, tags_root: Path, bsp_index: int, owners: list[int] | None = None):
    # owners lines up with the yielded geometry, which skips placements whose model has no triangles: the index of a
    # yielded one is how a light or occluder is tied back to its scenario placement
    cache = {}
    for placement in iter_placements(scenario, tags_root, bsp_index, cache):
        geometry = _load_model(tags_root, placement.definition.model, placement.desired, cache)
        if geometry is not None and len(geometry.tris):
            if owners is not None:
                owners.append(placement.index)
            yield (placement.position, placement.rotation, placement.definition.origin_offset, geometry)


def world_positions(geometry: ModelGeometry, position: np.ndarray, rotation, origin_offset: np.ndarray) -> np.ndarray:
    # world = position + x*forward + y*left + z*up, the matrix rows being forward, left, up;
    # the scenery tag's origin offset is in model space, so it turns with the placement
    return ((geometry.positions.astype(np.float64) + origin_offset) @ placement_rotation(*rotation)
            + position).astype(np.float32)


def gather_occluders(scenario: Tag, tags_root: Path, bsp_index: int) -> Occluders:
    positions, uvs, tris, tri_shader, tri_owner = [], [], [], [], []
    remap: dict[tuple, int] = {}
    base = 0
    owners: list[int] = []
    for position, rotation, origin_offset, geometry in iter_placed(scenario, tags_root, bsp_index, owners):
        local = np.array([remap.setdefault(s, len(remap)) for s in geometry.shaders], np.int32)
        positions.append(world_positions(geometry, position, rotation, origin_offset))
        uvs.append(geometry.uv)
        tris.append(geometry.tris + base)
        tri_shader.append(local[geometry.tri_shader])
        tri_owner.append(np.full(len(geometry.tris), owners[len(tris) - 1], np.int32))
        base += len(geometry.positions)
    if not tris:
        return Occluders(np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32),
                         np.zeros((0, 2), np.float32), np.zeros(0, np.int32), [], np.zeros(0, np.int32))
    return Occluders(np.concatenate(positions), np.concatenate(tris).astype(np.int32),
                     np.concatenate(uvs), np.concatenate(tri_shader).astype(np.int32), list(remap),
                     np.concatenate(tri_owner))


def gather_lights(scenario: Tag, tags_root: Path, bsp_index: int) -> tuple[list[SceneLight], dict[str, int]]:
    counts = dict.fromkeys(('attached', 'dynamic', 'dark', 'missing_light', 'missing_model',
                            'missing_marker', 'placed'), 0)
    lights: list[SceneLight] = []
    cache, light_cache, warned = {}, {}, set()
    for placement in iter_placements(scenario, tags_root, bsp_index, cache):
        for light_path, marker in placement.definition.lights:
            counts['attached'] += 1
            definition = read_light(tags_root, light_path, light_cache)
            if definition is None:
                counts['missing_light'] += 1
                continue
            if definition.dynamic:
                counts['dynamic'] += 1
                continue
            if not bakes(definition):
                counts['dark'] += 1
                continue
            geometry = _load_model(tags_root, placement.definition.model, placement.desired, cache)
            if geometry is None:
                counts['missing_model'] += 1
                continue
            position, forward, found = marker_frame(geometry.markers, marker)
            if not found:
                counts['missing_marker'] += 1
                # (marker names match without case, so one model's spellings of a missing marker are one warning)
                if (placement.definition.model, marker.lower()) not in warned:
                    warned.add((placement.definition.model, marker.lower()))
                    logger.warning('model %s has no marker %r, its light sits on node 0',
                                   placement.definition.model[1], marker)
            rotation = placement_rotation(*placement.rotation)
            lights.append(scene_light(
                definition, (position + placement.definition.origin_offset) @ rotation + placement.position,
                forward @ rotation, placement.index, light_path))
            counts['placed'] += 1
    return lights, counts
