"""Chart each source group with xatlas and pack the charts into power-of-two pages.

Ordinary groups are charted from their geometry. Groups whose shader asks for
simple parameterization keep their texture layout (islands scaled to world
density) and go through a second atlas, because xatlas cannot mix meshes and uv
meshes in one atlas; their chart rectangles share our final packing. Every stock cap is checked here: a layout over a cap is
split where a material can be, and refused otherwise, never truncated.
"""

from dataclasses import dataclass, replace
import json
import logging
import math
import os
from pathlib import Path
import tempfile
from zipfile import BadZipFile

import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from . import xatlas_c
from .formats.bsp import OutMaterial, OutPage, SourceGroup
from .formats.vertices import ENV, LM
from .gather import SIMPLE_PARAMETERIZATION, ShaderRadiosity


logger = logging.getLogger(__name__)
# identical input bytes and settings can have incompatible layouts under a different packing algorithm
CACHE_VERSION = 15
# the hard cap on a page side, and so on what 'auto' may come to (the most MCC accepts)
AUTO_PAGE_MAX = 4096
# the smallest side 'auto' settles on, however small the biggest chart is
AUTO_PAGE_MIN = 256
# Auto steps its page shape up the ladder until the written pages are this many or fewer (owner decision)
AUTO_TARGET_PAGES = 10
# xatlas always charts at this side, whatever the pages end up being, so the layout of a chart never depends on page_max
CHART_SIDE = 4096
# the bake side the search for the smallest page side starts from
FIRST_BAKE_SIDE = 512


class PackedPages(list):
    # the Max resolution the layout was made for: the fixed value, or what 'auto' came to for this level and density.
    # The written pages are capped at it, so the combine at write needs it and a cached layout has to carry it
    page_max: int = 0
    # the shape every written page but the last is, so the write repacks onto what the estimate counted
    page_shape: tuple[int, int] = ()
    # the shape the estimate arrives at for the same charts, counted on rectangles alone
    estimate_shape: tuple[int, int] = ()


def save_charts(path: Path, pages: list[OutPage], key: str):
    arrays = {'meta': json.dumps({'version': CACHE_VERSION, 'key': key, 'page_max': getattr(pages, 'page_max', 0),
                                  'page_shape': list(getattr(pages, 'page_shape', ())),
                                  'estimate_shape': list(getattr(pages, 'estimate_shape', ())),
                                  'pages': [[p.width, p.height, len(p.materials)] for p in pages]})}
    for i, page in enumerate(pages):
        for j, material in enumerate(page.materials):
            prefix = f'p{i}_m{j}_'
            for name, dtype in [('group', np.int32), ('env', ENV), ('lm', LM),
                                ('tris', np.uint16), ('surface_ids', np.int32)]:
                arrays[prefix + name] = np.asarray(getattr(material, name), dtype=dtype)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        # the old cache stays readable until a complete archive replaces it; close before replacing on Windows.
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + '.', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            np.savez(stream, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def cache_version(path: Path) -> int | None:
    # the layout version a charts file was saved under, or None for one that cannot be read
    try:
        with np.load(path, allow_pickle=False) as data:
            return int(json.loads(str(data['meta']))['version'])
    except (OSError, ValueError, KeyError, TypeError, EOFError, BadZipFile):
        return None


def load_charts(path: Path, key: str) -> list[OutPage] | None:
    try:
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data['meta']))
            if meta['version'] != CACHE_VERSION or meta['key'] != key:
                return None
            pages = []
            for i, (width, height, count) in enumerate(meta['pages']):
                materials = []
                for j in range(count):
                    prefix = f'p{i}_m{j}_'
                    materials.append(OutMaterial(int(data[prefix + 'group']),
                        *(data[prefix + name].copy() for name in ('env', 'lm', 'tris', 'surface_ids'))))
                pages.append(OutPage(width, height, materials))
            pages = PackedPages(pages)
            pages.page_max = int(meta['page_max'])
            pages.page_shape = tuple(int(side) for side in meta['page_shape'])
            pages.estimate_shape = tuple(int(side) for side in meta['estimate_shape'])
    except (OSError, ValueError, KeyError, TypeError, EOFError, BadZipFile):
        # an interrupted or damaged cache is disposable; gather and charts can rebuild it.
        return None
    return pages


class CapError(ValueError):
    pass


class TriangleTooWide(CapError):
    def __init__(self, shader: str, width: int, page_max: int):
        super().__init__(f'shader {shader} has a triangle {width} texels wide with padding, over the largest '
                         f'page side ({page_max}): lower texels_per_unit')
        self.shader, self.width, self.page_max = shader, width, page_max


@dataclass
class ChartSettings:
    # the global texels_per_unit is the level 1 density; levels 0 (high) to 3 (turd) multiply it with
    # tool's final-quality start-edge ratios 2.4 / (0.9, 2.4, 4, 80)
    texels_per_unit: float = 2.0
    detail_level_scale: tuple = (2.667, 1.0, 0.6, 0.03)
    padding: int = 4
    page_max: int | str = 'auto'
    page_min: int = 8
    max_vertices: int = 64000
    max_surfaces: int = 20000
    max_materials_per_page: int = 2048
    max_pages: int = 127
    # xatlas lets a chart grow across a hard normal seam unless the seam weight tops 1000, and tool splits at every crease
    respect_normal_seams: bool = True

    def __post_init__(self):
        if self.page_max != 'auto' and not (type(self.page_max) is int and self.page_max > 0
                                            and not self.page_max & (self.page_max - 1)):
            raise ValueError(f"page_max must be 'auto' or a power of two, not {self.page_max!r}")
        if self.page_min < 1 or self.page_min & (self.page_min - 1):
            raise ValueError(f'page_min must be a power of two, not {self.page_min}')
        if self.page_min > self.page_cap:
            raise ValueError('page_min exceeds page_max')
        # (a settings file brings the list back as a list, which would not equal the default)
        self.detail_level_scale = tuple(float(v) for v in self.detail_level_scale)
        if len(self.detail_level_scale) != 4 or not all(v > 0 for v in self.detail_level_scale):
            raise ValueError(f'detail_level_scale needs four positive multipliers, not {self.detail_level_scale}')

    @property
    def page_cap(self) -> int:
        # (the limit a page side may reach, not the Max resolution: for 'auto' that is found per level, and
        # chart_and_pack hands it back as PackedPages.page_max)
        return AUTO_PAGE_MAX if self.page_max == 'auto' else self.page_max

    def density_scale(self, shader: ShaderRadiosity) -> float:
        if not 0 <= shader.detail_level < len(self.detail_level_scale):
            raise ValueError(f'radiosity detail level {shader.detail_level} is not 0 to 3')
        return self.detail_level_scale[shader.detail_level]


def next_pow2(n: int) -> int:
    return 1 << max(int(n) - 1, 0).bit_length()


def _in_triangle(px, py, a, b, c):
    d1 = (px - b[0]) * (a[1] - b[1]) - (a[0] - b[0]) * (py - b[1])
    d2 = (px - c[0]) * (b[1] - c[1]) - (b[0] - c[0]) * (py - c[1])
    d3 = (px - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (py - a[1])
    return ((d1 > 0) & (d2 > 0) & (d3 > 0)) | ((d1 < 0) & (d2 < 0) & (d3 < 0))


def _texel_cover(texels: np.ndarray, tri: np.ndarray, low: np.ndarray):
    # the (rows, columns) of the texels whose centre a triangle holds, counted from `low`
    a, b, c = texels[tri[0]] - low, texels[tri[1]] - low, texels[tri[2]] - low
    x0, y0 = np.floor(np.minimum(np.minimum(a, b), c)).astype(np.int64)
    x1, y1 = np.ceil(np.maximum(np.maximum(a, b), c)).astype(np.int64)
    px, py = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
    ys, xs = np.nonzero(_in_triangle(px, py, a, b, c))
    return ys + y0, xs + x0


def _island_overlaps(texels: np.ndarray, own: np.ndarray, tris: np.ndarray) -> bool:
    # one grid per island, sized to the island, so tiling uvs cost nothing extra
    low = np.floor(texels[own].min(axis=0)).astype(np.int64)
    size = np.ceil(texels[own].max(axis=0)).astype(np.int64) - low + 1
    grid = np.zeros((size[1], size[0]), np.uint8)
    for tri in tris:
        ys, xs = _texel_cover(texels, tri, low)
        grid[ys, xs] = np.minimum(grid[ys, xs] + 1, 2)
    return bool(grid.max() > 1)


def _subgroup(group: SourceGroup, rows: np.ndarray) -> SourceGroup:
    used = np.unique(group.tris[rows])
    return SourceGroup(group.key, group.template, group.env[used].copy(),
                       np.searchsorted(used, group.tris[rows]).astype(np.int32), group.surface_ids[rows])


@dataclass
class SimpleLayout:
    kept: SourceGroup | None
    uvs: np.ndarray | None
    recharted: SourceGroup | None
    split: int
    recharted_islands: int
    reasons: list[str]


def _padded_area(sizes) -> int:
    return sum(int(w) * int(h) for w, h in sizes)


def _simple_layout(group: SourceGroup, density: float, s: ChartSettings) -> SimpleLayout:
    """Island uvs at world scale in a row of their own, per island; those that cannot keep a layout are charted normally."""
    uv = group.env['texcoord'].astype(np.float64)
    tris = group.tris
    # xatlas joins uv-mesh faces that share a uv value, not only a vertex index
    node = _uv_nodes(uv)
    count = node.max() + 1
    edges = np.concatenate([node[tris[:, [0, 1]]], node[tris[:, [1, 2]]]])
    graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(count, count))
    islands, component = connected_components(graph, directed=False)
    labels = component[node]
    a, b, c = (uv[tris[:, k]] for k in range(3))
    uv_area = 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
    p, q, r = (group.env['position'][tris[:, k]].astype(np.float64) for k in range(3))
    world_area = 0.5 * np.linalg.norm(np.cross(q - p, r - p), axis=1)
    tri_island = labels[tris[:, 0]]
    flat = np.bincount(tri_island, (world_area > 1e-12) & (uv_area <= 1e-12), islands) > 0
    uv_sum = np.bincount(tri_island, uv_area, islands)
    world_sum = np.bincount(tri_island, world_area, islands)
    used = np.bincount(tri_island, minlength=islands) > 0
    empty = used & ~flat & ((uv_sum <= 0) | (world_sum <= 0))
    scale = np.ones(islands)
    good = used & ~flat & ~empty
    scale[good] = density * np.sqrt(world_sum[good] / uv_sum[good])
    scaled = uv * scale[labels][:, None]
    tpu = s.texels_per_unit
    reasons, split, recharted = [], 0, []
    kept_env, kept_tris, kept_ids, kept_uv = [], [], [], []
    cursor, offset = 0.0, 0

    def place(rows, local_tris, mine, uv_scaled):
        nonlocal cursor, offset
        kept_uv.append(uv_scaled)
        kept_env.append(group.env[mine])
        kept_tris.append(local_tris.astype(np.int32) + offset)
        kept_ids.append(group.surface_ids[rows])
        offset += len(mine)
        cursor = uv_scaled[:, 0].max() + 1.0

    for island in np.flatnonzero(used):
        rows = np.flatnonzero(tri_island == island)
        mine = np.unique(tris[rows])
        local = np.searchsorted(mine, tris[rows])
        reason = ('no texture area' if flat[island] else
                  'no texture or world area' if empty[island] else None)
        if reason is None:
            # (each island moves into a row of its own so no two share a uv value or overlap by accident)
            low = scaled[mine].min(axis=0)
            row_uv = scaled[mine] + [cursor - low[0], -low[1]]
            if not _island_overlaps(row_uv * tpu, np.arange(len(mine)), local):
                place(rows, local, mine, row_uv)
                continue
            texels = row_uv * tpu
            cut = _split_self_overlap(texels, local)
            sizes = [_padded_size(texels[np.unique(local[part])], s.padding) for part in cut]
            alone = _density_checked_pieces([(0, _subgroup(group, rows), None, density)], False, s, join_creases=True)
            rechart = [_padded_size(piece[3], s.padding) for piece in alone]
            if _padded_area(sizes) > _padded_area(rechart) or len(cut) > len(rechart):
                reason = 'it overlaps itself'
            else:
                split += 1
                for part in cut:
                    verts = np.unique(local[part])
                    piece = texels[verts]
                    # (a cut piece moves by whole texels, so the coverage test that cut it still holds where it lands)
                    shift = [np.ceil(cursor * tpu) - np.floor(piece[:, 0].min()), -np.floor(piece[:, 1].min())]
                    place(rows[part], np.searchsorted(verts, local[part]), mine[verts], (piece + shift) / tpu)
                continue
        recharted.append(rows)
        if reason not in reasons:
            reasons.append(reason)
    kept = uvs = None
    if kept_uv:
        kept = SourceGroup(group.key, group.template, np.concatenate(kept_env), np.concatenate(kept_tris),
                           np.concatenate(kept_ids))
        uvs = np.concatenate(kept_uv).astype(np.float32)
    again = _subgroup(group, np.sort(np.concatenate(recharted))) if recharted else None
    return SimpleLayout(kept, uvs, again, split, len(recharted), reasons)


def _split(vertex_count: int, tris: np.ndarray, s: ChartSettings):
    # greedy in triangle order; yields (triangle rows, local vertex ids) per material
    if vertex_count <= s.max_vertices and len(tris) <= s.max_surfaces:
        yield np.arange(len(tris)), np.unique(tris)
        return
    rows, seen = [], set()
    for t, tri in enumerate(tris):
        fresh = {int(v) for v in tri} - seen
        if rows and (len(seen) + len(fresh) > s.max_vertices or len(rows) + 1 > s.max_surfaces):
            yield np.array(rows), np.array(sorted(seen))
            rows, seen = [], set()
            fresh = {int(v) for v in tri}
        rows.append(t)
        seen |= fresh
    if rows:
        yield np.array(rows), np.array(sorted(seen))


def _run_atlas(entries, uv_mode: bool, s: ChartSettings, *, limit_charts=True, join_creases=False):
    # entries: (group index, group, scaled uvs or None, density multiplier); pieces keep xatlas texel uvs until page sizes are known
    atlas = xatlas_c.Atlas()
    # xatlas has one density for the whole atlas, so a group's own density is a scaled copy of its
    # positions (or world-scaled uvs). The scale is relative to the run's smallest multiplier,
    # which moves into the atlas density: no group shrinks, so a thin triangle never falls
    # under xatlas's zero-area cut
    lowest = min((d for _, _, _, d in entries), default=1.0)
    for _, group, uvs, density in entries:
        if uv_mode:
            atlas.add_uv_mesh(uvs / np.float32(lowest), group.tris)
        else:
            atlas.add_mesh(group.env['position'] * np.float32(density / lowest), group.tris, group.env['normal'])
    texels_per_unit = s.texels_per_unit * lowest
    # (xatlas starts a chart at texel 0 and lets it reach the last texel: _finish adds a margin
    # on both sides, so the atlas gets that much less)
    # an automatic-size reference cannot silently shrink a chart; only the candidate is constrained, and always to
    # the same side, so a chart's shape does not change with page_max (oversized ones are cut afterwards)
    atlas.generate(xatlas_c.PackSettings(resolution=CHART_SIDE - 2 * s.padding if limit_charts else 0,
                                          texels_per_unit=texels_per_unit,
                                          padding=s.padding, bilinear=True, rotate_charts=True,
                                          brute_force=False,
                                          normal_seam_weight=1001.0 if s.respect_normal_seams and not join_creases else 4.0,
                                          # 64 texels keeps terrain pieces below the 128-page jump;
                                          # a boundary cap also stops long thin charts defeating the area cap
                                          max_chart_area=(64 / texels_per_unit) ** 2 if limit_charts and not uv_mode else 0,
                                          max_boundary_length=256 / texels_per_unit if limit_charts and not uv_mode else 0))
    page_count = max(atlas.page_count, 1)
    pages = [[] for _ in range(page_count)]
    extent = np.zeros((page_count, 2))
    for m, (index, group, _, _) in enumerate(entries):
        verts = atlas.vertices(m)
        local = atlas.indices(m)
        if len(local) != len(group.tris):
            raise RuntimeError('xatlas dropped triangles')
        unplaced = verts['atlas'][local] < 0
        if np.any(unplaced.any(axis=1) & ~unplaced.all(axis=1)):
            raise RuntimeError('xatlas left part of a triangle outside every page')
        # (xatlas charts no degenerate triangle: they get page 0 and uv 0, which covers no texel)
        tri_page = np.where(unplaced[:, 0], 0, verts['atlas'][local[:, 0]])
        labels = verts['chart'][local[:, 0]]
        order = np.argsort(labels, kind='stable')
        for rows in np.split(order, np.flatnonzero(np.diff(labels[order])) + 1):
            page = tri_page[rows[0]]
            subsets = [rows] if uv_mode else _bounded_charts(verts['uv'], local, rows)
            for rows in subsets:
                tris = local[rows]
                for part, used in _split(len(verts), tris, s):
                    pick = verts[used]
                    extent[page] = np.maximum(extent[page], pick['uv'].max(axis=0))
                    pages[page].append((index, group, pick['xref'].astype(np.int64), pick['uv'].copy(),
                                        np.searchsorted(used, tris[part]).astype(np.int32), group.surface_ids[rows[part]]))
    return pages, extent


def _bounded_charts(uv, tris, rows):
    # xatlas's planar fast path bypasses its chart limits. Splitting its UV layout
    # at 120 texels leaves two four-texel margins in a 128 side, without rescaling faces
    pending = [rows]
    while pending:
        rows = pending.pop()
        corners = uv[tris[rows]]
        span = np.ptp(corners.reshape(-1, 2), axis=0)
        if span.max() <= 120 or len(rows) == 1:
            yield rows
            continue
        axis = int(np.argmax(span))
        order = np.argsort(corners[:, :, axis].mean(axis=1), kind='stable')
        middle = len(rows) // 2
        pending.extend((rows[order[middle:]], rows[order[:middle]]))


def _uv_nodes(uv: np.ndarray) -> np.ndarray:
    return np.unique(uv, axis=0, return_inverse=True)[1].reshape(-1)


def _components(uv: np.ndarray, tris: np.ndarray, rows: np.ndarray) -> list[np.ndarray]:
    # triangles are joined by a shared uv value at a corner, as xatlas joins uv-mesh faces
    rows = np.sort(np.asarray(rows))
    if not len(rows):
        return []
    used, local = np.unique(tris[rows], return_inverse=True)
    node = _uv_nodes(uv[used])[local.reshape(-1, 3)]
    count = node.max() + 1
    edges = np.concatenate([node[:, [0, 1]], node[:, [1, 2]]])
    graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(count, count))
    label = connected_components(graph, directed=False)[1][node[:, 0]]
    order = np.argsort(label, kind='stable')
    found = np.split(rows[order], np.flatnonzero(np.diff(label[order])) + 1)
    return sorted(found, key=lambda part: part[0])


def _padded_size(uv: np.ndarray, padding: int) -> np.ndarray:
    return np.ceil(uv.max(axis=0) - np.floor(uv.min(axis=0))).astype(int) + 2 * padding


def _bisect_piece(piece, side: int, padding: int) -> list:
    index, group, xref, uv, tris, surface_ids = piece
    if len(tris) == 1 or _padded_size(uv, padding).max() <= side:
        return [piece]
    parts = []
    pending = [np.arange(len(tris))]
    while pending:
        rows = pending.pop()
        used = np.unique(tris[rows])
        if len(rows) == 1 or _padded_size(uv[used], padding).max() <= side:
            parts.append((index, group, xref[used], uv[used].copy(),
                          np.searchsorted(used, tris[rows]).astype(np.int32), surface_ids[rows]))
            continue
        axis = int(np.argmax(np.ptp(uv[used], axis=0)))
        rows = rows[np.argsort(uv[tris[rows]].mean(axis=1)[:, axis], kind='stable')]
        middle = len(rows) // 2
        halves = [part for half in (rows[:middle], rows[middle:]) for part in _components(uv, tris, half)]
        pending.extend(reversed(halves))
    return parts


def _split_oversized(pieces, side: int, padding: int) -> tuple[list, int, int]:
    parts, split, made = [], 0, 0
    for piece in pieces:
        cut = _bisect_piece(piece, side, padding)
        parts.extend(cut)
        if cut[0] is not piece:
            split += 1
            made += len(cut)
    return parts, split, made


def next_step(shape: tuple[int, int]) -> tuple[int, int]:
    # one step down the page ladder: the longer side halves, so a square drops to half its height first.
    # Shapes are kept wide (width >= height); a chart turns to fit
    width, height = shape
    return (width, height // 2) if width == height else (height, height)


def step_up(shape: tuple[int, int]) -> tuple[int, int]:
    # one step up the page ladder, the reverse of next_step
    width, height = shape
    return (width, width) if width != height else (2 * width, width)


def written_pages(pages, shape: tuple[int, int], s: ChartSettings, blocks=None) -> int:
    # how many pages the write repacks this layout onto at this shape, the count the Auto target is judged by.
    # (imported here: combine builds on this module)
    from .combine import chart_blocks, plan_repack
    blocks = chart_blocks(pages, s.padding) if blocks is None else blocks
    try:
        return len(plan_repack(blocks, shape, s.page_min, s.max_materials_per_page, s.max_pages))
    except CapError:
        return s.max_pages + 1


def ladder_shape(sizes, top: int, bottom: int) -> tuple[int, int]:
    # the smallest step down from top x top that holds every (w, h) either way round, its longer side at least bottom
    longest = max(max(size) for size in sizes)
    shortest = max(min(size) for size in sizes)
    shape = (top, top)
    while True:
        step = next_step(shape)
        if step[0] < bottom or step[1] < 1 or longest > step[0] or shortest > step[1]:
            return shape
        shape = step


def auto_page_max(parts, s: ChartSettings) -> tuple[int, int]:
    # Auto is the page shape that holds the biggest chart whole with its padding (AUTO_PAGE_MIN up, within the hard
    # cap); the bake side is its longer side. parts are the charts after anything over the cap is cut
    return ladder_shape([tuple(_padded_size(part[3], s.padding)) for part in parts], AUTO_PAGE_MAX,
                        max(AUTO_PAGE_MIN, s.page_min))


def _widest_triangle(pieces, padding: int) -> tuple[int, str]:
    widest, shader = 0, ''
    for _, group, _, uv, tris, _ in pieces:
        corners = uv[tris]
        size = np.ceil(corners.max(axis=1) - np.floor(corners.min(axis=1))).astype(int) + 2 * padding
        if len(size) and size.max() > widest:
            widest, shader = int(size.max()), group.key[1]
    return widest, shader


def _split_self_overlap(texels: np.ndarray, tris: np.ndarray) -> list[np.ndarray]:
    # coverage is the texel centres a triangle holds, as _island_overlaps counts them
    corners = texels[np.unique(tris)]
    low = np.floor(corners.min(axis=0)).astype(np.int64)
    size = np.ceil(corners.max(axis=0)).astype(np.int64) - low + 1
    cover = []
    for tri in tris:
        ys, xs = _texel_cover(texels, tri, low)
        cover.append(ys * size[0] + xs)
    taken = np.zeros(size[0] * size[1], bool)
    node = _uv_nodes(texels)[tris]
    flat = node.reshape(-1)
    order = np.argsort(flat, kind='stable')
    starts = np.searchsorted(flat[order], np.arange(flat.max() + 2))
    owner = order // 3

    def neighbours(t):
        return sorted({int(u) for n in node[t] for u in owner[starts[n]:starts[n + 1]]})

    assigned = np.zeros(len(tris), bool)
    pieces = []
    for first in range(len(tris)):
        if assigned[first]:
            continue
        rows = [first]
        assigned[first] = True
        taken[cover[first]] = True
        for t in rows:
            for n in neighbours(t):
                if assigned[n] or taken[cover[n]].any():
                    continue
                assigned[n] = True
                taken[cover[n]] = True
                rows.append(n)
        for t in rows:
            taken[cover[t]] = False
        pieces.append(np.array(sorted(rows)))
    return pieces


def _finish(pages, extent, s: ChartSettings) -> list[OutPage]:
    result = []
    for pieces, ext in zip(pages, extent):
        used = ext + 2 * s.padding
        w = min(max(next_pow2(math.ceil(used[0])), s.page_min), s.page_cap)
        h = min(max(next_pow2(math.ceil(used[1])), s.page_min), s.page_cap)
        assert used[0] <= w and used[1] <= h, 'a chart reaches into its page padding'
        by_group = {}
        for index, group, xref, uv, tris, surface_ids in pieces:
            lm = np.zeros(len(xref), LM)
            lm['texcoord'] = (uv + s.padding) / np.array([w, h], dtype=np.float32)
            by_group.setdefault(index, []).append((group.env[xref], lm, tris, surface_ids))
        materials = []
        for index, parts in by_group.items():
            offsets = np.cumsum([0] + [len(p[0]) for p in parts[:-1]])
            env = np.concatenate([p[0] for p in parts])
            lm = np.concatenate([p[1] for p in parts])
            # equal source vertices on separate charts need independent lightmap UVs
            tris = np.concatenate([p[2] + offset for p, offset in zip(parts, offsets)])
            surface_ids = np.concatenate([p[3] for p in parts])
            for rows, vertices in _split(len(env), tris, s):
                materials.append(OutMaterial(index, env[vertices], lm[vertices],
                    np.searchsorted(vertices, tris[rows]).astype(np.int32), surface_ids[rows]))
        result.append(OutPage(w, h, materials))
    return result


def _page_max_hint(s) -> str:
    return ' or allow a larger page_max' if s.page_max != 'auto' and s.page_max < AUTO_PAGE_MAX else ''


def _check_caps(pages, s):
    if len(pages) > s.max_pages:
        raise CapError(f'{len(pages)} lightmap pages needed, over max_pages ({s.max_pages}): '
                       'lower texels_per_unit' + _page_max_hint(s))
    for page in pages:
        if len(page.materials) > s.max_materials_per_page:
            raise CapError(f'{len(page.materials)} materials on one page, over max_materials_per_page '
                           f'({s.max_materials_per_page}): lower texels_per_unit' + _page_max_hint(s))


def _packing_areas(pages):
    groups, surfaces, areas = [], [], []
    for page in pages:
        for material in page.materials:
            uv = material.lm['texcoord'].astype(np.float64) * [page.width, page.height]
            a, b, c = (uv[material.tris[:, k]] for k in range(3))
            ab, ac = b - a, c - a
            areas.append(np.abs(ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0]))
            groups.append(np.full(len(a), material.group))
            surfaces.append(material.surface_ids)
    order = np.lexsort((np.concatenate(surfaces), np.concatenate(groups)))
    return np.concatenate(areas)[order]


def _piece_areas(pieces):
    groups, surfaces, areas = [], [], []
    for group, _, _, uv, tris, surface_ids in pieces:
        corners = uv[tris].astype(np.float64)
        ab, ac = corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]
        areas.append(np.abs(ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0]))
        groups.append(np.full(len(tris), group))
        surfaces.append(surface_ids)
    order = np.lexsort((np.concatenate(surfaces), np.concatenate(groups)))
    return np.concatenate(areas)[order]


def _density_change(reference, candidate):
    if reference.shape != candidate.shape or not np.isfinite(candidate).all():
        return math.inf, math.inf
    positive = reference > 1e-12
    if np.any(candidate[~positive] > 1e-6):
        return math.inf, math.inf
    median = float(np.median(np.abs(candidate[positive] / reference[positive] - 1))) if positive.any() else 0.0
    total = abs(float(candidate.sum() / reference.sum()) - 1) if reference.sum() > 1e-12 else 0.0
    return median, total


def _density_checked_pieces(entries, uv_mode, s, join_creases=False):
    reference = [piece for page in _run_atlas(entries, uv_mode, s, limit_charts=False, join_creases=join_creases)[0]
                 for piece in page]
    candidate = [piece for page in _run_atlas(entries, uv_mode, s, join_creases=join_creases)[0] for piece in page]
    median, total = _density_change(_piece_areas(reference), _piece_areas(candidate))
    # splitting can re-unfold a face, but cannot buy smaller pages by reducing
    # requested density: enforce both the typical face and the total area budget
    if median <= 0.01 and total <= 0.02:
        return candidate
    logger.info('rejecting chart candidate: median area change %.4f%%, total %.4f%%', median * 100, total * 100)
    return reference


class _FreeSpace:
    def __init__(self, capacity):
        self.base = next_pow2(max(1, capacity))
        self.tree = np.zeros(2 * self.base, np.int64)
        self.owners = [0] * (2 * self.base)

    def set(self, index, value, owner=1):
        index += self.base
        self.tree[index] = value
        self.owners[index] = owner if value else 0
        while index > 1:
            index //= 2
            value = max(self.tree[2 * index], self.tree[2 * index + 1])
            owners = self.owners[2 * index] | self.owners[2 * index + 1]
            if self.tree[index] == value and self.owners[index] == owners:
                break
            self.tree[index] = value
            self.owners[index] = owners

    def first(self, needed, allowed=None):
        def find(index):
            if self.tree[index] < needed or (allowed is not None and not self.owners[index] & allowed):
                return None
            if index >= self.base:
                return index - self.base
            left = find(2 * index)
            return left if left is not None else find(2 * index + 1)
        return find(1)

    def remaining(self, index):
        return int(self.tree[self.base + index])


def _piece_usage(piece, previous, s):
    batches, vertices, surfaces = previous
    if not batches:
        batches = 1
    if vertices + len(piece[2]) <= s.max_vertices and surfaces + len(piece[4]) <= s.max_surfaces:
        return batches, vertices + len(piece[2]), surfaces + len(piece[4])
    seen = set()
    for tri in piece[4]:
        fresh = set(map(int, tri)) - seen
        if surfaces and (vertices + len(fresh) > s.max_vertices or surfaces + 1 > s.max_surfaces):
            batches += 1
            vertices = surfaces = 0
            seen = set()
            fresh = set(map(int, tri))
        vertices += len(fresh)
        surfaces += 1
        seen.update(fresh)
    return batches, vertices, surfaces


def _shelf_pack(rectangles, order, width, height, transpose, s):
    row_space, page_space = _FreeSpace(len(rectangles)), _FreeSpace(s.max_pages)
    pages, extents, used_height, rows = [], [], [], []
    usage, material_counts = [], []
    for i in order:
        piece, uv, size = rectangles[i]
        w, h = size[::-1] if transpose else size
        w, h = max(1, w), max(1, h)
        if w > width or h > height:
            return None
        allowed = (1 << s.max_pages) - 1
        while True:
            row = row_space.first(w, allowed)
            page = rows[row][0] if row is not None else page_space.first(h, allowed)
            if page is None:
                break
            prior = usage[page].get(piece[0], (0, 0, 0))
            updated = _piece_usage(piece, prior, s)
            if material_counts[page] + updated[0] - prior[0] <= s.max_materials_per_page:
                break
            # a full draw-batch budget may still accept another chart of an existing group
            allowed &= ~(1 << page)
        if row is None:
            if page is None:
                if len(pages) >= s.max_pages:
                    return None
                page = len(pages)
                pages.append([])
                extents.append(np.zeros(2))
                used_height.append(0)
                usage.append({})
                material_counts.append(0)
            row = len(rows)
            rows.append((page, used_height[page]))
            used_height[page] += h
            row_space.set(row, width, 1 << page)
            page_space.set(page, height - used_height[page], 1 << page)
        page, y = rows[row]
        prior = usage[page].get(piece[0], (0, 0, 0))
        updated = _piece_usage(piece, prior, s)
        material_counts[page] += updated[0] - prior[0]
        if material_counts[page] > s.max_materials_per_page:
            return None
        usage[page][piece[0]] = updated
        free = row_space.remaining(row)
        moved = (uv[:, ::-1] if transpose else uv) + [width - free, y]
        pages[page].append((*piece[:3], moved, *piece[4:]))
        extents[page] = np.maximum(extents[page], moved.max(axis=0))
        row_space.set(row, free - w, 1 << page)
    return pages, extents


def _pack_rectangles(pieces, s):
    rectangles = []
    for piece in pieces:
        uv = piece[3]
        low = np.floor(uv.min(axis=0))
        size = _padded_size(uv, s.padding)
        if np.any(size > s.page_cap):
            raise CapError('a chart exceeds page_max including padding')
        rectangles.append((piece, uv - low, tuple(size)))
    # decreasing heights let every earlier shelf accept a later chart's height;
    # indexed first-fit fills those gaps without a quadratic scan of earlier rows
    best = None
    total_area = sum(w * h for _, _, (w, h) in rectangles)
    for transpose in (False, True):
        order = sorted(range(len(rectangles)), key=lambda i: (
            -rectangles[i][2][1 if not transpose else 0],
            -rectangles[i][2][0 if not transpose else 1], i))
        for width in (1 << k for k in range(s.page_min.bit_length() - 1, s.page_cap.bit_length())):
            for height in (1 << k for k in range(s.page_min.bit_length() - 1, s.page_cap.bit_length())):
                if width * height * s.max_pages < total_area:
                    continue
                packed = _shelf_pack(rectangles, order, width, height, transpose, s)
                if packed is not None:
                    pages, extents = packed
                    cost = sum(max(s.page_min, next_pow2(math.ceil(e[0] + 2 * s.padding))) *
                               max(s.page_min, next_pow2(math.ceil(e[1] + 2 * s.padding))) for e in extents)
                    # fewer textures and material splits outweigh extra texels;
                    # area only chooses between layouts with the same page count
                    if best is None or (len(pages), cost) < best[:2]:
                        best = (len(pages), cost, pages, extents)
    if best is None:
        raise CapError(f'packing exceeds max_pages ({s.max_pages}), page_max, or '
                       f'max_materials_per_page ({s.max_materials_per_page})')
    _, _, pages, extents = best
    return _finish(pages, extents, s)


def _fit(source: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    # (squared residual, scale) of the least-squares map target = scale * source + shift
    design = np.column_stack([source, np.ones(len(source))])
    fit = np.linalg.lstsq(design, target, rcond=None)[0]
    return float(((design @ fit - target) ** 2).sum()), float(fit[0])


def _stretch_plausible(source: np.ndarray, scale: float) -> bool:
    # xatlas only stretches a chart up to whole texels, so a negative scale (a mirror) or one beyond that is a misread
    extent = float(np.ptp(source))
    return extent < 1e-9 or 0.999 <= scale <= 1 + 1 / extent + 1e-3


def _nest_input(part) -> np.ndarray:
    return (part[3] - part[3].min(axis=0)).astype(np.float32)


def _nest_units(parts) -> list:
    # xatlas makes one chart of each uv island of a part and places each on its own (a part can be several islands
    # after _bounded_charts or the bisection), so the islands are what moves. They are found on the values xatlas
    # is given: an island goes to xatlas in its own frame, where float32 can keep apart two corners that rounded
    # together in the part's, so each island is split again on its own values until nothing splits
    units, pending = [], list(reversed(parts))
    while pending:
        unit = pending.pop()
        index, group, xref, uv, tris, surface_ids = unit
        found = _components(_nest_input(unit), tris, np.arange(len(tris)))
        if len(found) == 1:
            units.append(unit)
            continue
        split = []
        for rows in found:
            used = np.unique(tris[rows])
            split.append((index, group, xref[used], uv[used].copy(),
                          np.searchsorted(used, tris[rows]).astype(np.int32), surface_ids[rows]))
        pending.extend(reversed(split))
    return units


def _nest_at(parts, side: int, s: ChartSettings, margin: int):
    atlas = xatlas_c.Atlas()
    for part in parts:
        atlas.add_uv_mesh(_nest_input(part), part[4])
    # (margin is xatlas's padding; _finish still adds s.padding, and the extent check below holds either way)
    atlas.generate(xatlas_c.PackSettings(resolution=side - 2 * margin, texels_per_unit=1.0, padding=margin,
                                         bilinear=True, rotate_charts=True, rotate_charts_to_axis=False,
                                         brute_force=False))
    pages = [[] for _ in range(max(atlas.page_count, 1))]
    extent = np.zeros((len(pages), 2))
    empty = []
    first = None
    for m, part in enumerate(parts):
        verts, indices = atlas.vertices(m), atlas.indices(m)
        placed = (verts['atlas'][indices] >= 0).all(axis=1)
        if not placed.any():
            # (a part of degenerate triangles only covers no texel: it rides along with the first placed part)
            if (verts['atlas'][indices] >= 0).any():
                return None
            empty.append(part)
            continue
        if np.any((verts['atlas'][indices] >= 0).any(axis=1) & ~placed):
            return None
        used = np.unique(indices[placed])
        # (a part moves as one: a second chart or page would need the rectangle layout)
        if len(np.unique(verts['atlas'][used])) != 1 or len(np.unique(verts['chart'][used])) != 1:
            return None
        page = int(verts['atlas'][used[0]])
        uv = part[3].astype(np.float64)
        source, target = uv[verts['xref'][used]], verts['uv'][used].astype(np.float64)
        (sx, scale_sx), (sy, scale_sy) = _fit(source[:, 0], target[:, 0]), _fit(source[:, 1], target[:, 1])
        (tx, scale_tx), (ty, scale_ty) = _fit(source[:, 1], target[:, 0]), _fit(source[:, 0], target[:, 1])
        straight, turned = sx + sy, tx + ty
        # (xatlas swaps x and y for a turned chart and records nothing, and it stretches a chart to whole texels:
        # only the swap and the position are taken, never the stretch)
        if turned < straight:
            uv, source = uv[:, ::-1], source[:, ::-1]
            scales = (scale_tx, scale_ty)
        else:
            scales = (scale_sx, scale_sy)
        if not all(_stretch_plausible(source[:, k], scales[k]) for k in (0, 1)):
            return None
        moved = (uv + (target.min(axis=0) - source.min(axis=0))).astype(part[3].dtype)
        pages[page].append((*part[:3], moved, *part[4:]))
        extent[page] = np.maximum(extent[page], moved.max(axis=0))
        if first is None:
            first = page, moved.min(axis=0)
    if first is None:
        return None
    for part in empty:
        uv = part[3]
        pages[first[0]].append((*part[:3], (uv - uv.min(axis=0) + first[1]).astype(uv.dtype), *part[4:]))
    keep = [i for i, pieces in enumerate(pages) if pieces]
    if np.any(extent[keep] + 2 * s.padding > side):
        return None
    return _finish([pages[i] for i in keep], extent[keep], replace(s, page_max=side))


def _nest_pack(parts, s: ChartSettings, margin: int | None = None) -> list[OutPage] | None:
    margin = s.padding if margin is None else margin
    top = s.page_cap
    side = max(s.page_min, next_pow2(max(int(_padded_size(part[3], s.padding).max()) for part in parts)))
    best = None
    while side <= top:
        candidate = _nest_at(parts, side, s, margin)
        if candidate is not None:
            key = (len(candidate), _pages_area(candidate))
            if best is None or key < best[0]:
                best = key, candidate
        side *= 2
    return None if best is None else best[1]


def _blocks_fit(pages, shape, s) -> bool:
    from .combine import chart_blocks
    return all(max(b.width, b.height) <= shape[0] and min(b.width, b.height) <= shape[1]
               for b in chart_blocks(pages, s.padding))


def _pages_area(pages) -> int:
    return sum(page.width * page.height for page in pages)


def _touched_cells(a, b, c, eps=1e-3):
    # the texels whose open cell meets the open triangle (a separating-axis test; an edge on a cell boundary touches
    # only the cell it covers). eps absorbs the float32 rounding of the stored uvs
    low, high = np.minimum(np.minimum(a, b), c), np.maximum(np.maximum(a, b), c)
    x0, y0 = np.floor(low + eps).astype(np.int64)
    x1, y1 = np.ceil(high - eps).astype(np.int64)
    if x1 <= x0 or y1 <= y0:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    xs, ys = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
    xs, ys = xs.ravel(), ys.ravel()
    keep = np.ones(len(xs), bool)
    for p, q, r in ((a, b, c), (b, c, a), (c, a, b)):
        normal = np.array([p[1] - q[1], q[0] - p[0]])
        reach = sorted((normal @ p, normal @ r))
        tol = eps * np.hypot(*normal)
        base = normal[0] * xs + normal[1] * ys
        keep &= (base + min(0, normal[0]) + min(0, normal[1]) < reach[1] - tol) & \
                (base + max(0, normal[0]) + max(0, normal[1]) > reach[0] + tol)
    return xs[keep], ys[keep]


def _charts_separated(pages: list[OutPage], s: ChartSettings, placed) -> bool:
    # placed: the pieces that were placed, each one chart. A chart is named by its triangles (group, surface id),
    # not found from the pages: its triangles can share no vertex (a source seam, or a split at a vertex cap) and
    # still touch, and two charts can share a uv value only where they wrongly touch
    reach = 2 * s.padding
    big = np.iinfo(np.int32).max
    owner = {}
    for number, piece in enumerate(placed, 1):
        for surface in piece[5]:
            if owner.setdefault((piece[0], int(surface)), number) != number:
                return False
    for page in pages:
        label = np.zeros((page.height, page.width), np.int32)
        for material in page.materials:
            texels = material.lm['texcoord'].astype(np.float64) * [page.width, page.height]
            for tri, surface in zip(material.tris, material.surface_ids):
                id_ = owner.get((material.group, int(surface)))
                if id_ is None:
                    return False
                a, b, c = texels[tri]
                # (corners on one line hold no texel or sub-texel centre, so the bake gives the triangle nothing;
                # xatlas does not place such an island and it rides along wherever the first chart starts)
                if (b[0] - a[0]) * (c[1] - a[1]) == (b[1] - a[1]) * (c[0] - a[0]):
                    continue
                xs, ys = _touched_cells(a, b, c)
                if len(xs) and (xs.min() < 0 or ys.min() < 0 or xs.max() >= page.width or ys.max() >= page.height):
                    return False
                if np.any((label[ys, xs] != 0) & (label[ys, xs] != id_)):
                    return False
                label[ys, xs] = id_
        size = 2 * reach + 1
        # (a texel with another chart's label anywhere within the reach shows in the filtered maximum or minimum)
        highest = maximum_filter(label, size=size, mode='constant', cval=0)
        lowest = minimum_filter(np.where(label == 0, big, label), size=size, mode='constant', cval=big)
        if np.any((label != 0) & ((highest != label) | (lowest != label))):
            return False
    return True


def chart_and_pack(groups: list[SourceGroup], radiosity: list[ShaderRadiosity],
                   s: ChartSettings, nest: bool = False) -> list[OutPage]:
    try:
        return _chart_and_pack(groups, radiosity, s, nest)
    except CapError as error:
        # (the message for the user depends on whether Auto would have helped; TriangleTooWide names its own)
        if not hasattr(error, 'page_max'):
            error.page_max = s.page_max
        raise


def _chart_and_pack(groups, radiosity, s: ChartSettings, nest: bool) -> list[OutPage]:
    if len(radiosity) != len(groups):
        raise ValueError('one shader radiosity per source group is required')
    top = s.page_cap
    if top - 2 * s.padding <= 0:
        raise ValueError('page_max must exceed twice the padding')
    ordinary, simple, fallback = [], [], []
    for index, (group, shader) in enumerate(zip(groups, radiosity)):
        if not len(group.tris):
            continue
        density = s.density_scale(shader)
        if shader.flags & SIMPLE_PARAMETERIZATION:
            layout = _simple_layout(group, density, s)
            if layout.kept is not None:
                simple.append((index, layout.kept, layout.uvs, density))
            if layout.recharted is not None:
                fallback.append((index, layout.recharted, None, density))
            if layout.split or layout.recharted_islands:
                logger.warning('shader %s asks for simple parameterization: %d island(s) split where they overlap '
                               'themselves, %d island(s) charted normally (%s)', group.key[1], layout.split,
                               layout.recharted_islands, ', '.join(layout.reasons) or 'none')
            continue
        ordinary.append((index, group, None, density))
    pieces = []
    if ordinary:
        pieces.extend(_density_checked_pieces(ordinary, False, s))
    if fallback:
        # (a separate atlas, because xatlas has one normal seam weight per generate call: tool lays these shaders
        # out from texture uvs that run on across creases, so their charts must not split there)
        pieces.extend(_density_checked_pieces(fallback, False, s, join_creases=True))
    if simple:
        pieces.extend(_density_checked_pieces(simple, True, s))
    if not pieces:
        return []
    widest, shader = _widest_triangle(pieces, s.padding)
    if widest > top:
        raise TriangleTooWide(shader, widest, top)
    if s.page_max == 'auto':
        # (cut at the cap first, so what is left is what 'auto' must fit whole)
        pieces, split, made = _split_oversized(pieces, top, s.padding)
        logger.info('auto Max resolution: %d piece(s) over %d cut into %d parts', split, top, made)
        return _auto_pack(pieces, widest, s, nest)
    side = max(min(FIRST_BAKE_SIDE, top), s.page_min, next_pow2(widest))
    while True:
        side_s = replace(s, page_max=side)
        parts, split, made = _split_oversized(pieces, side, s.padding)
        shape = (s.page_max, s.page_max)
        try:
            pages = _pack_rectangles(parts, side_s)
            _check_caps(pages, side_s)
            # (the write repacks onto pages of the Max resolution, and a bitmap holds max_pages of them: found out
            # here rather than after hours of baking)
            _check_written(pages, shape, s)
        except CapError:
            # (the last side's error is the one worth reading: it has the most room)
            if side >= top:
                raise
            side *= 2
            continue
        logger.info('bake side %d: %d piece(s) split into %d parts', side, split, made)
        if nest:
            nested = _nested_or(pages, parts, side_s)
            if nested is not pages and written_pages(nested, shape, s) <= s.max_pages:
                pages = nested
        pages = PackedPages(pages)
        pages.page_max = s.page_max
        pages.page_shape = (s.page_max, s.page_max)
        pages.estimate_shape = pages.page_shape
        return pages


def _check_written(pages, shape, s: ChartSettings) -> int:
    count = written_pages(pages, shape, s)
    if count > s.max_pages:
        raise CapError(f'the written lightmap needs more than {s.max_pages} pages of {shape[0]}x{shape[1]}')
    return count


def _rect_layout(pieces, widest: int, shape, s: ChartSettings):
    side = max(max(shape), next_pow2(widest))
    side_s = replace(s, page_max=side)
    parts, split, made = _split_oversized(pieces, side, s.padding)
    pages = _pack_rectangles(parts, side_s)
    _check_caps(pages, side_s)
    return pages, parts, side_s


def _auto_pack(pieces, widest: int, s: ChartSettings, nest: bool) -> PackedPages:
    # Auto starts from the shape the biggest chart needs, which never shrinks, so no chart is cut. While the write
    # would make more than AUTO_TARGET_PAGES pages of it, or the caps fail, the shape steps up the ladder, to the cap.
    # The bake counts on the layout it will write, nested where nesting wins; the estimate, which never nests, counts
    # on rectangles, so its shape can come out larger (owner decision: the safe side for its memory figure)
    cap = (s.page_cap, s.page_cap)
    shape = auto_page_max(pieces, s)
    estimate_shape = None
    nests, layouts, block_cache = {}, {}, {}

    def count_of(layout, shape):
        # (two shapes of one bake side count the same layout, so its blocks are worked out once)
        from .combine import chart_blocks
        if id(layout) not in block_cache:
            block_cache[id(layout)] = (layout, chart_blocks(layout, s.padding))
        return written_pages(layout, shape, s, block_cache[id(layout)][1])

    def rect_layout(shape):
        # (shapes with one longer side share a bake side, and so the same pack, or the same cap failure)
        side = max(max(shape), next_pow2(widest))
        if side not in layouts:
            try:
                layouts[side] = _rect_layout(pieces, widest, shape, s)
            except CapError as error:
                layouts[side] = error
        if isinstance(layouts[side], CapError):
            raise layouts[side]
        return layouts[side]

    while True:
        try:
            rectangles, parts, side_s = rect_layout(shape)
        except CapError:
            if shape == cap:
                raise
            shape = step_up(shape)
            continue
        count = count_of(rectangles, shape)
        if estimate_shape is None and (count <= min(AUTO_TARGET_PAGES, s.max_pages) or shape == cap):
            estimate_shape = shape
        layout = rectangles
        if nest:
            # (shapes with one longer side share a bake side, and so the same nest)
            if side_s.page_max not in nests:
                nests[side_s.page_max] = _nested_or(rectangles, parts, side_s)
            nested = nests[side_s.page_max]
            if nested is not rectangles and shape != cap and not _blocks_fit(nested, shape, s):
                # (nested charts move as one block at the write, and a half-height page may not hold it: the next
                # step, a square of the same side, does, so nesting is kept rather than given up for this shape)
                logger.info('auto: a nested block does not fit %dx%d, trying the next shape', *shape)
                shape = step_up(shape)
                continue
            if nested is not rectangles and _blocks_fit(nested, shape, s):
                layout, count = nested, count_of(nested, shape)
        if count > min(AUTO_TARGET_PAGES, s.max_pages) and shape != cap:
            shape = step_up(shape)
            continue
        if count > s.max_pages:
            # (at the cap the target may be missed, but not the page count a bitmap holds)
            raise CapError(f'the written lightmap needs more than {s.max_pages} pages of {shape[0]}x{shape[1]}')
        break
    probe = shape
    while estimate_shape is None:
        # (the bake settled first; the estimate's shape is where the rectangles alone get within the target)
        probe = step_up(probe)
        try:
            within = count_of(rect_layout(probe)[0], probe) <= min(AUTO_TARGET_PAGES, s.max_pages)
        except CapError:
            within = False
        if within or probe == cap:
            estimate_shape = probe
    logger.info('auto page shape %dx%d: %d written page(s), bake side %d; the estimate counts %dx%d',
                *shape, count, side_s.page_max, *estimate_shape)
    pages = PackedPages(layout)
    pages.page_max = side_s.page_max
    pages.page_shape = shape
    pages.estimate_shape = estimate_shape
    return pages


def _nested_or(rectangles, parts, s):
    units = _nest_units(parts)
    # xatlas places each chart stretched to whole texels and we keep it unstretched, so a hollow chart's inner edge
    # can sit up to a texel nearer what nests in it than xatlas left. A texel more of xatlas padding on every chart
    # covers that; it costs packing, so it is only the second try, when the plain padding fell short
    for margin in (s.padding, s.padding + 1):
        nested = _nest_pack(units, s, margin)
        if nested is None:
            reason = 'a chart could not be placed whole'
            break
        try:
            _check_caps(nested, s)
        except CapError as error:
            reason, nested = str(error), None
            break
        if _charts_separated(nested, s, units):
            break
        reason = 'charts too close'
        logger.info('nesting: charts too close at xatlas padding %d', margin)
        nested = None
    if nested is None:
        logger.info('nesting: %s, keeping the rectangle layout', reason)
        return rectangles
    # (a tie keeps the rectangle layout: nesting moves charts for nothing)
    if (len(nested), _pages_area(nested)) >= (len(rectangles), _pages_area(rectangles)):
        logger.info('nesting: %s the rectangle layout (%d page(s), %d texels against %d, %d), keeping it',
                    'ties with' if (len(nested), _pages_area(nested)) == (len(rectangles), _pages_area(rectangles))
                    else 'not smaller than', len(nested), _pages_area(nested), len(rectangles),
                    _pages_area(rectangles))
        return rectangles
    return nested
