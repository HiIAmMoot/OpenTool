"""Combine bake pages into fewer, larger ones when the bitmap is written.

The bake keeps its pages; only the written bitmap sees the combined ones, so a
combine changes where a page sits and nothing a texel holds.
"""

from dataclasses import dataclass

import numpy as np

from .charts import CapError
from .formats.bsp import OutMaterial, OutPage


@dataclass(frozen=True)
class Placement:
    width: int
    height: int
    # (bake page index, x offset, y offset) of each page placed in this written page
    sources: tuple[tuple[int, int, int], ...]


def plan_combine(sizes: list[tuple[int, int]], material_counts: list[int], page_max: int,
                 max_materials: int = 2048, max_pages: int = 127) -> list[Placement]:
    entries = [(Placement(w, h, ((i, 0, 0),)), material_counts[i]) for i, (w, h) in enumerate(sizes)]
    changed = True
    while changed:
        changed = False
        # fours first, then pairs, until nothing more joins
        for n in (4, 2):
            by_size = {}
            for entry in entries:
                by_size.setdefault((entry[0].width, entry[0].height), []).append(entry)
            result = []
            for (w, h) in sorted(by_size, key=lambda s: (-s[0] * s[1], -s[0], -s[1])):
                same = by_size[(w, h)]
                if n == 4:
                    big_w, big_h, offsets = 2 * w, 2 * h, [(0, 0), (w, 0), (0, h), (w, h)]
                elif w <= h:
                    big_w, big_h, offsets = 2 * w, h, [(0, 0), (w, 0)]
                else:
                    big_w, big_h, offsets = w, 2 * h, [(0, 0), (0, h)]
                if max(big_w, big_h) > page_max:
                    result += same
                    continue
                k = 0
                while k + n <= len(same):
                    run = same[k:k + n]
                    if sum(m for _, m in run) <= max_materials:
                        sources = tuple((i, x + ox, y + oy) for (p, _), (ox, oy) in zip(run, offsets)
                                        for i, x, y in p.sources)
                        result.append((Placement(big_w, big_h, sources), sum(m for _, m in run)))
                        changed = True
                    else:
                        # a run over the material cap stays as it is
                        result += run
                    k += n
                result += same[k:]
            entries = result
    plan = [p for p, _ in entries]
    if len(plan) > max_pages:
        raise CapError(f'The bake needs {len(plan)} lightmap pages and a bitmap holds {max_pages}.')
    return plan


def combine_layout(pages: list[OutPage], plan: list[Placement]) -> list[OutPage]:
    out = []
    for placement in plan:
        size = np.array([placement.width, placement.height], np.float64)
        materials = []
        for i, ox, oy in placement.sources:
            page = pages[i]
            for material in page.materials:
                lm = material.lm.copy()
                # float64 so the offset adds exactly; the field stays float32
                lm['texcoord'] = (material.lm['texcoord'].astype(np.float64)
                                  * (page.width, page.height) + (ox, oy)) / size
                materials.append(OutMaterial(material.group, material.env, lm, material.tris, material.surface_ids))
        out.append(OutPage(placement.width, placement.height, materials))
    return out


def combine_arrays(arrays: list[np.ndarray], plan: list[Placement]) -> list[np.ndarray]:
    out = []
    for placement in plan:
        first = arrays[placement.sources[0][0]]
        combined = np.zeros((placement.height, placement.width, *first.shape[2:]), first.dtype)
        for i, ox, oy in placement.sources:
            h, w = arrays[i].shape[:2]
            combined[oy:oy + h, ox:ox + w] = arrays[i]
        out.append(combined)
    return out
