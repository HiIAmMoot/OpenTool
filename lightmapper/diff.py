"""Compare our lightmaps with tool's, surface by surface, each in its own layout.

The two layouts share nothing but the input surfaces, so a surface is sampled at
its centroid in the lightmap UVs of each side's own tag and page, never at one
side's UVs on the other's page. Surfaces smaller than a texel in either layout
are dropped: a bilinear sample there reads its neighbours.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path, PureWindowsPath

import numpy as np

from .formats.vertices import decode_r5g6b5, unpack_uncompressed
from .raster import sample_bilinear
from .tagio.tagfile import Tag, read_tag, resolve_reference_path


LUMA = np.array([0.2126, 0.7152, 0.0722])
MIN_MATERIAL_SAMPLES = 50
ACCEPT_PEARSON = 0.6
# tool's draft and final lighting can differ, so acceptance allows quality-preset variation
ACCEPT_MEDIAN_ERROR = 0.09
ACCEPT_MATERIAL_RATIO = 0.15
# a material whose fitted light is more than this share emission sits next to emitters, where
# tool's bounce-free emission model and ours part ways, so its brightness ratio is not checked
EMITTER_SHARE = 0.25


@dataclass
class SurfaceTable:
    # one row per surface, in the tag's surface order; page is -1 where unlit
    page: np.ndarray
    material: np.ndarray
    uv: np.ndarray
    position: np.ndarray
    normal: np.ndarray
    vector: np.ndarray
    vertex: np.ndarray


@dataclass
class Samples:
    tool_rgb: np.ndarray
    ours_rgb: np.ndarray
    surface_area: np.ndarray
    material: np.ndarray
    tool_vec: np.ndarray
    our_vec: np.ndarray
    tool_normal: np.ndarray
    our_normal: np.ndarray
    surface: np.ndarray
    layout: dict = field(default_factory=dict)
    # largest channel before any clip, where the caller has it; ours_rgb alone cannot show clipping
    ours_peak: np.ndarray | None = None


@dataclass
class Calibration:
    w_sun: float
    w_emit: float
    w_amb: float
    gamma: float
    radiosity_mode: str = 'coherence'
    radiosity_scale: float = 1.0
    median_abs_error: float = 0.0
    pearson_r: float = 0.0
    samples_used: int = 0


def bitmap_pages(tag: Tag) -> list[np.ndarray]:
    data = tag.root.data('processed_pixel_data')
    pages = []
    for bitmap in tag.root.block('bitmaps'):
        width, height = bitmap.get('width'), bitmap.get('height')
        if bitmap.get('format') != 6:
            raise ValueError(f'lightmap page format {bitmap.get("format")} is not r5g6b5')
        start = bitmap.get('pixel_data_offset')
        raw = np.frombuffer(data, '<u2', count=width * height, offset=start)
        pages.append(decode_r5g6b5(raw.reshape(height, width)))
    return pages


def surface_table(bsp: Tag) -> SurfaceTable:
    root = bsp.root
    triangles = np.array([np.frombuffer(s.body, '>u2', count=3) for s in root.block('surfaces')],
                         dtype=np.int64).reshape(-1, 3)
    count = len(triangles)
    page = np.full(count, -1, np.int32)
    material = np.full(count, '', object)
    uv = np.full((count, 3, 2), np.nan)
    position = np.full((count, 3, 3), np.nan)
    normal = np.full((count, 3, 3), np.nan)
    vector = np.full((count, 3, 3), np.nan)
    vertex = np.full((count, 3), -1, np.intp)
    offset = 0
    for element in root.block('lightmaps'):
        if element.get('bitmap') < 0:
            continue
        for item in element.block('materials'):
            env, lm = unpack_uncompressed(item.data('uncompressed_vertices'),
                                          item.get('rendered_vertices.vertex_count'),
                                          item.get('lightmap_vertices.vertex_count'))
            if len(env) != len(lm):
                raise ValueError('a lightmapped material has unequal vertex buffers')
            start = item.get('surfaces')
            rows = np.arange(start, start + item.get('surface_count'))
            corners = triangles[rows]
            page[rows] = element.get('bitmap')
            material[rows] = PureWindowsPath(item.reference('shader')[1]).name
            uv[rows] = lm['texcoord'][corners]
            position[rows] = env['position'][corners]
            normal[rows] = env['normal'][corners]
            vector[rows] = lm['incident_radiosity'][corners]
            vertex[rows] = corners + offset
            offset += len(env)
    return SurfaceTable(page, material, uv, position, normal, vector, vertex)


def by_input_order(table: SurfaceTable, new_to_input: np.ndarray) -> SurfaceTable:
    order = np.asarray(new_to_input)
    if len(order) != len(table.page) or not np.array_equal(np.sort(order), np.arange(len(order))):
        raise ValueError('new_to_input is not a permutation of the surfaces')
    result = {}
    for name, array in vars(table).items():
        placed = np.empty_like(array)
        placed[order] = array
        result[name] = placed
    return SurfaceTable(**result)


def _texel_area(uv: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    a, b, c = uv[:, 0], uv[:, 1], uv[:, 2]
    cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    return np.abs(cross) / 2 * sizes[:, 0] * sizes[:, 1]


def _sizes(page: np.ndarray, pages: list[np.ndarray]) -> np.ndarray:
    shapes = np.array([(p.shape[1], p.shape[0]) for p in pages] or [(0, 0)], np.float64)
    return shapes[np.clip(page, 0, len(shapes) - 1)]


def _world_area(position: np.ndarray) -> np.ndarray:
    a, b, c = (position[:, i].astype(np.float64) for i in range(3))
    return np.linalg.norm(np.cross(b - a, c - a), axis=1) / 2


def _sample_pages(table: SurfaceTable, pages: list[np.ndarray], rows: np.ndarray) -> np.ndarray:
    result = np.zeros((len(table.page), 3))
    centroid = table.uv.mean(axis=1)
    for p, image in enumerate(pages):
        selected = rows & (table.page == p)
        if selected.any():
            result[selected] = sample_bilinear(image, centroid[selected])
    return result[rows]


def _layout_stats(table: SurfaceTable, pages: list[np.ndarray], rows: np.ndarray) -> dict:
    lit = table.page >= 0
    sizes = _sizes(table.page, pages)
    texel_area = np.where(lit, _texel_area(np.nan_to_num(table.uv), sizes), 0)
    total = float(sum(p.shape[0] * p.shape[1] for p in pages))
    world = _world_area(np.nan_to_num(table.position))[rows].sum()
    return {'pages': len(pages),
            'utilisation': float(texel_area[lit].sum() / total) if total else 0.0,
            'texels_per_unit': float(np.sqrt(texel_area[rows].sum() / world)) if world else 0.0}


def paired_samples(tool_bsp: Tag, tool_bitmap: Tag, ours: list[np.ndarray], our_bsp: Tag,
                   new_to_input: np.ndarray) -> Samples:
    tool = surface_table(tool_bsp)
    mine = by_input_order(surface_table(our_bsp), new_to_input)
    tool_pages = bitmap_pages(tool_bitmap)
    ours = [np.asarray(p, np.float64) for p in ours]
    rows = (tool.page >= 0) & (mine.page >= 0)
    rows &= _texel_area(np.nan_to_num(tool.uv), _sizes(tool.page, tool_pages)) >= 1
    rows &= _texel_area(np.nan_to_num(mine.uv), _sizes(mine.page, ours)) >= 1
    tool_rgb = _sample_pages(tool, tool_pages, rows)
    ours_rgb = _sample_pages(mine, ours, rows)
    return Samples(
        tool_rgb, ours_rgb, _world_area(tool.position[rows]), tool.material[rows],
        tool.vector[rows].reshape(-1, 3), mine.vector[rows].reshape(-1, 3),
        tool.normal[rows].reshape(-1, 3), mine.normal[rows].reshape(-1, 3),
        np.flatnonzero(rows),
        {'tool': _layout_stats(tool, tool_pages, rows), 'ours': _layout_stats(mine, ours, rows)})


def load_pages(tags: Path, bsp: Tag) -> Tag:
    group, path = bsp.root.reference('lightmaps_bitmap')
    if group != 'bitm' or not path:
        raise ValueError('the BSP has no lightmap bitmap')
    return read_tag(resolve_reference_path(tags, group, path))


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.ravel(a).astype(np.float64), np.ravel(b).astype(np.float64)
    if len(a) < 2 or a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _vector_angles(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    la, lb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    ok = (la > 0) & (lb > 0)
    cosine = np.einsum('ij,ij->i', a[ok], b[ok]) / (la[ok] * lb[ok])
    return np.degrees(np.arccos(np.clip(cosine, -1, 1)))


def emitter_shares(material: np.ndarray, emit_term: np.ndarray, total: np.ndarray) -> dict:
    # per material over the light before gamma; only materials the report would judge
    emit_lum, total_lum = np.asarray(emit_term, float) @ LUMA, np.asarray(total, float) @ LUMA
    shares = {}
    for name in sorted(set(material)):
        rows = material == name
        if rows.sum() >= MIN_MATERIAL_SAMPLES and total_lum[rows].sum() > 0:
            share = float(emit_lum[rows].sum() / total_lum[rows].sum())
            if share > EMITTER_SHARE:
                shares[str(name)] = share
    return shares


def report(samples: Samples, fitted: Calibration | None = None, excluded: dict | None = None) -> dict:
    tool, ours = np.asarray(samples.tool_rgb, float), np.asarray(samples.ours_rgb, float)
    result = {'samples': len(tool), 'fit': asdict(fitted) if fitted else None,
              'median_abs_error': float(np.median(np.abs(tool - ours))) if len(tool) else float('nan'),
              'pearson_r': pearson(tool, ours),
              'clipped_fraction': (None if samples.ours_peak is None
                                   else float(np.mean(np.asarray(samples.ours_peak) > 1))),
              'layout': samples.layout}
    materials = {}
    tool_lum, ours_lum = tool @ LUMA, ours @ LUMA
    for name in sorted(set(samples.material)):
        rows = samples.material == name
        if rows.sum() >= MIN_MATERIAL_SAMPLES:
            t, o = float(tool_lum[rows].mean()), float(ours_lum[rows].mean())
            materials[str(name)] = {'count': int(rows.sum()), 'tool': t, 'ours': o,
                                    'ratio': o / t if t else float('inf')}
    result['materials'] = materials
    result['excluded'] = {name: share for name, share in (excluded or {}).items() if name in materials}
    lengths = (np.linalg.norm(samples.tool_vec, axis=1) - np.linalg.norm(samples.our_vec, axis=1))
    angles = _vector_angles(np.asarray(samples.tool_vec, float), np.asarray(samples.our_vec, float))
    result['vector'] = {
        'corners': len(samples.tool_vec),
        'median_angle_deg': float(np.median(angles)) if len(angles) else float('nan'),
        'median_length_difference': float(np.median(np.abs(lengths))) if len(lengths) else float('nan')}
    result['acceptance'] = {
        'pearson_r': result['pearson_r'] >= ACCEPT_PEARSON,
        'median_abs_error': result['median_abs_error'] <= ACCEPT_MEDIAN_ERROR,
        'materials': all(abs(m['ratio'] - 1) <= ACCEPT_MATERIAL_RATIO
                      for name, m in materials.items() if name not in result['excluded'])}
    return result


def render_markdown(data: dict) -> str:
    lines = ['# Lightmap differential', '',
             f'- samples: {data["samples"]}',
             f'- median absolute error: {data["median_abs_error"]:.4f}',
             f'- Pearson r: {data["pearson_r"]:.4f}']
    if data['clipped_fraction'] is not None:
        lines.append(f'- clipped fraction (ours above 1): {data["clipped_fraction"]:.4f}')
    for side, stats in data['layout'].items():
        lines.append(f'- {side}: {stats["pages"]} pages, utilisation {stats["utilisation"]:.3f}, '
                     f'{stats["texels_per_unit"]:.3f} texels/unit')
    vector = data['vector']
    lines.append(f'- vectors: {vector["corners"]} corners, median angle {vector["median_angle_deg"]:.2f} deg, '
                 f'median length difference {vector["median_length_difference"]:.4f}')
    if data['fit']:
        lines += ['', '## Fit', ''] + [f'- {k}: {v}' for k, v in data['fit'].items()]
    lines += ['', '## Mean luminance per material', '', '| material | samples | tool | ours | ratio |',
              '|---|---|---|---|---|']
    for name, m in data['materials'].items():
        lines.append(f'| {name} | {m["count"]} | {m["tool"]:.4f} | {m["ours"]:.4f} | {m["ratio"]:.3f} |')
    if data['excluded']:
        lines += ['', '## Excluded from the material check', '',
                  f'Emission is more than {EMITTER_SHARE:.0%} of the fitted light:', '']
        lines += [f'- {name}: emission share {share:.2f}' for name, share in data['excluded'].items()]
    lines += ['', '## Acceptance', ''] + [f'- {k}: {"pass" if v else "FAIL"}'
                                         for k, v in data['acceptance'].items()]
    return '\n'.join(lines) + '\n'
