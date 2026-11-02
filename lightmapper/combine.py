"""Repack the baked charts onto the written pages when the bitmap is written.

The bake keeps its pages; only the written bitmap sees the new layout. Each chart's
baked texel block (the chart with its padding) moves whole, by a translation and at
most an x/y swap, so a repack changes where a texel sits and nothing it holds.
"""

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .charts import AUTO_PAGE_MIN, CapError, ladder_shape, next_pow2, next_step
from .formats.bsp import OutMaterial, OutPage

# (the charts' own tolerance for float32 texcoords sitting on a texel edge)
EDGE = 1e-3


@dataclass(frozen=True)
class Block:
    page: int
    # the texels copied: the chart's cells with the padding round them
    x: int
    y: int
    width: int
    height: int
    # indices of the page's materials with triangles here (the per-page material cap counts these)
    materials: tuple[int, ...]


@dataclass(frozen=True)
class Move:
    page: int
    x: int
    y: int
    width: int
    height: int
    to_x: int
    to_y: int
    # a turned block is transposed: its texel (x, y) lands at (y, x)
    swapped: bool


@dataclass(frozen=True)
class WrittenPage:
    width: int
    height: int
    moves: tuple[Move, ...]


def _chart_boxes(page: OutPage, padding: int):
    # (material, lowest cell, highest cell + 1) per connected chart; a chart's triangles share vertices only within
    # one material, and charts of one material that touch only merge into a bigger block, which is safe
    for index, material in enumerate(page.materials):
        texels = material.lm['texcoord'].astype(np.float64) * [page.width, page.height]
        n = len(material.env)
        edges = np.concatenate([material.tris[:, [0, 1]], material.tris[:, [1, 2]]])
        label = connected_components(coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n)),
                                     directed=False)[1][material.tris[:, 0]]
        corners = texels[material.tris]
        low = np.floor(corners.min(axis=1) + EDGE)
        high = np.ceil(corners.max(axis=1) - EDGE)
        # (a zero-area triangle still needs a home: it gets the cell it sits in)
        high = np.maximum(high, low + 1)
        for chart in np.unique(label):
            rows = label == chart
            yield index, low[rows].min(axis=0).astype(int) - padding, high[rows].max(axis=0).astype(int) + padding


def chart_blocks(pages: list[OutPage], padding: int) -> list[Block]:
    # the padded boxes of charts that overlap (nested charts, or a chart in another's concavity) move as one block,
    # so no box is cut and no two blocks share a texel
    blocks = []
    for number, page in enumerate(pages):
        boxes = [(low, high, {index}) for index, low, high in _chart_boxes(page, padding)]
        while True:
            label = np.full((page.height + 2 * padding, page.width + 2 * padding), -1, np.int64)
            parent = list(range(len(boxes)))

            def find(k):
                while parent[k] != k:
                    parent[k] = parent[parent[k]]
                    k = parent[k]
                return k
            for k, (low, high, _) in enumerate(boxes):
                region = label[low[1] + padding:high[1] + padding, low[0] + padding:high[0] + padding]
                for other in np.unique(region[region >= 0]):
                    parent[find(int(other))] = find(k)
                region[...] = k
            merged = {}
            for k, (low, high, materials) in enumerate(boxes):
                root = find(k)
                if root in merged:
                    m_low, m_high, m_materials = merged[root]
                    merged[root] = (np.minimum(m_low, low), np.maximum(m_high, high), m_materials | materials)
                else:
                    merged[root] = (low, high, set(materials))
            done = len(merged) == len(boxes)
            boxes = [merged[root] for root in sorted(merged)]
            if done:
                break
        for low, high, materials in boxes:
            # (a real layout keeps every chart a padding from the page edge, so a block is cut only in a hand-made
            # one, and then moves as it is)
            x0, y0 = max(int(low[0]), 0), max(int(low[1]), 0)
            x1, y1 = min(int(high[0]), page.width), min(int(high[1]), page.height)
            blocks.append(Block(number, x0, y0, x1 - x0, y1 - y0, tuple(sorted(materials))))
    return blocks


def written_shape(blocks: list[Block], page_max: int, auto: bool) -> tuple[int, int]:
    # Auto is the ladder step that holds the biggest block; a fixed Max resolution is its own square
    if not auto:
        return page_max, page_max
    return ladder_shape([(b.width, b.height) for b in blocks], page_max, AUTO_PAGE_MIN)


def _fits(span, shape):
    return max(span) <= shape[0] and min(span) <= shape[1]


class _Page:
    def __init__(self, width, height):
        self.width, self.height = width, height
        # free rectangles (x, y, w, h), a guillotine split of the page
        self.free = np.array([[0, 0, width, height]], np.int64)
        self.moves = []
        self.materials = set()
        self.reach = np.zeros(2, np.int64)

    def best(self, span):
        # (leftover short side, free rectangle, swapped) of the tightest fit, or None
        found = None
        for swapped in (False, True):
            w, h = (span[1], span[0]) if swapped else span
            if swapped and w == h:
                continue
            ok = (self.free[:, 2] >= w) & (self.free[:, 3] >= h)
            if not ok.any():
                continue
            short = np.minimum(self.free[:, 2] - w, self.free[:, 3] - h)
            short = np.where(ok, short, np.iinfo(np.int64).max)
            k = int(np.argmin(short))
            if found is None or short[k] < found[0]:
                found = (int(short[k]), k, swapped)
        return found

    def place(self, k, w, h):
        fx, fy, fw, fh = self.free[k]
        # (the split keeps the larger leftover whole, so later blocks find room)
        if (fw - w) * fh > fw * (fh - h):
            parts = [(fx + w, fy, fw - w, fh), (fx, fy + h, w, fh - h)]
        else:
            parts = [(fx + w, fy, fw - w, h), (fx, fy + h, fw, fh - h)]
        parts = [part for part in parts if part[2] > 0 and part[3] > 0]
        self.free = np.concatenate([np.delete(self.free, k, axis=0), np.array(parts, np.int64).reshape(-1, 4)])
        return int(fx), int(fy)


def plan_repack(blocks: list[Block], shape: tuple[int, int], page_min: int = 8, max_materials: int = 2048,
                max_pages: int = 127) -> list[WrittenPage]:
    step = next_step(shape)
    order = sorted(range(len(blocks)), key=lambda k: (-blocks[k].width * blocks[k].height,
                                                      -max(blocks[k].width, blocks[k].height), k))
    spans = [(b.width, b.height) for b in blocks]
    # a block too big for a step down needs the full shape; the rest may go a step down
    big = [k for k in order if not _fits(spans[k], step)]
    small = [k for k in order if _fits(spans[k], step)]
    written = []

    def put(k, candidates, open_new):
        block = blocks[k]
        keys = {(block.page, m) for m in block.materials}
        best = None
        for number, page in candidates:
            if len(page.materials | keys) > max_materials:
                continue
            found = page.best(spans[k])
            if found is not None and (best is None or found[0] < best[0][0]):
                best = (found, number, page)
        if best is None:
            if open_new is None:
                return False
            if len(keys) > max_materials:
                raise CapError(f'one chart holds {len(keys)} materials, over max_materials_per_page ({max_materials})')
            page = open_new()
            found = page.best(spans[k])
            if found is None:
                raise CapError('a chart is bigger than the written page')
            best = (found, None, page)
        (_, slot, swapped), _, page = best
        w, h = (spans[k][1], spans[k][0]) if swapped else spans[k]
        x, y = page.place(slot, w, h)
        page.moves.append(Move(block.page, block.x, block.y, block.width, block.height, x, y, swapped))
        page.reach = np.maximum(page.reach, (x + w, y + h))
        page.materials |= keys
        return True

    def opener(size, into):
        def open_new():
            if len(written) >= max_pages:
                raise CapError(f'The bake needs more than {max_pages} lightmap pages and a bitmap holds {max_pages}.')
            page = _Page(*size)
            written.append(page)
            into.append((len(written) - 1, page))
            return page
        return open_new

    auto_pages, step_pages = [], []
    for k in big:
        put(k, auto_pages, opener(shape, auto_pages))
    # (the rest fill the space the big charts left, biggest first; with nothing big, which only a fixed Max
    # resolution allows, no page of the full shape opens)
    rest = [k for k in small if not (big and put(k, auto_pages, None))]
    for k in rest:
        put(k, step_pages, opener(step, step_pages))
    plan = []
    for page in written:
        # (a page only as big as what it holds needs, in powers of two, as the bake pages are)
        width = min(max(next_pow2(int(page.reach[0])), page_min), page.width)
        height = min(max(next_pow2(int(page.reach[1])), page_min), page.height)
        plan.append(WrittenPage(width, height, tuple(page.moves)))
    return plan


def whole_pages(pages: list[OutPage]) -> list[WrittenPage]:
    return [WrittenPage(p.width, p.height, (Move(i, 0, 0, p.width, p.height, 0, 0, False),))
            for i, p in enumerate(pages)]


def _owners(page: OutPage, moves):
    # the move holding each triangle of a bake page, found by the cell its centre lies in
    grid = np.full((page.height, page.width), -1, np.int64)
    for k, move in moves:
        grid[move.y:move.y + move.height, move.x:move.x + move.width] = k
    owners = []
    for material in page.materials:
        centre = (material.lm['texcoord'].astype(np.float64)[material.tris].mean(axis=1) * [page.width, page.height])
        cell = np.clip(np.floor(centre).astype(int), 0, [page.width - 1, page.height - 1])
        owner = grid[cell[:, 1], cell[:, 0]]
        if np.any(owner < 0):
            raise RuntimeError('a triangle lies outside every moved block')
        owners.append(owner)
    return owners


def repack_layout(pages: list[OutPage], plan: list[WrittenPage]) -> list[OutPage]:
    moves = [(number, k, move) for number, page in enumerate(plan) for k, move in enumerate(page.moves)]
    by_page = {}
    for number, k, move in moves:
        by_page.setdefault(move.page, []).append(((number, k), move))
    owner_of = {}
    for source, entries in by_page.items():
        flat = [(i, move) for i, (_, move) in enumerate(entries)]
        owner_of[source] = ([key for key, _ in entries], _owners(pages[source], flat))
    out = []
    for number, written in enumerate(plan):
        size = np.array([written.width, written.height], np.float64)
        materials = []
        for source in sorted({move.page for move in written.moves}):
            keys, owners = owner_of[source]
            page = pages[source]
            for index, material in enumerate(page.materials):
                picked = [(keys[o], o) for o in np.unique(owners[index]) if keys[o][0] == number]
                if not picked:
                    continue
                rows = np.flatnonzero(np.isin(owners[index], [o for _, o in picked]))
                used = np.unique(material.tris[rows])
                lm = material.lm[used].copy()
                texels = material.lm['texcoord'][used].astype(np.float64) * [page.width, page.height]
                vertex_owner = np.empty(len(material.lm), np.int64)
                vertex_owner[material.tris[rows].ravel()] = np.repeat(owners[index][rows], 3)
                for (_, k), o in picked:
                    move = written.moves[k]
                    mine = vertex_owner[used] == o
                    local = texels[mine] - [move.x, move.y]
                    if move.swapped:
                        local = local[:, ::-1]
                    # float64 so the offset adds exactly; the field stays float32
                    lm['texcoord'][mine] = (local + [move.to_x, move.to_y]) / size
                materials.append(OutMaterial(material.group, material.env[used], lm,
                                             np.searchsorted(used, material.tris[rows]).astype(np.int32),
                                             material.surface_ids[rows]))
        out.append(OutPage(written.width, written.height, materials))
    return out


def repack_arrays(arrays: list[np.ndarray], plan: list[WrittenPage]) -> list[np.ndarray]:
    out = []
    for written in plan:
        first = arrays[written.moves[0].page]
        page = np.zeros((written.height, written.width, *first.shape[2:]), first.dtype)
        for move in written.moves:
            block = arrays[move.page][move.y:move.y + move.height, move.x:move.x + move.width]
            if move.swapped:
                block = block.swapaxes(0, 1)
            h, w = block.shape[:2]
            page[move.to_y:move.to_y + h, move.to_x:move.to_x + w] = block
        out.append(page)
    return out
