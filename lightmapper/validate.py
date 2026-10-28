"""Check stock loader invariants without changing tag data.

Strict compression additionally checks our writer's byte-exact encoding and
lightmap buffer type; older retail tools produced valid buffers that differ.
"""

import numpy as np

from .formats.vertices import compress, unpack_uncompressed
from .tagio.tagfile import Tag

# bytes per texel of the page formats a lightmap may use (r5g6b5, x8r8g8b8)
PAGE_TEXEL_BYTES = {6: 2, 10: 4}

def validate_bsp(bsp: Tag, bitmap: Tag | None, strict_compression: bool = False) -> list[str]:
    """Our own outputs must also match tool's byte layout in strict_compression mode."""
    root = bsp.root
    lightmaps = root.block('lightmaps')
    surfaces = root.block('surfaces')
    surface_count = len(surfaces)
    problems = []

    def report(rule, lightmap='all', material='all'):
        problems.append(f'lightmap {lightmap}, material {material}: {rule}')

    if len(lightmaps) > 128:
        report('lightmap element count exceeds 128 (including the final element)')
    if not lightmaps or lightmaps[-1].get('bitmap') != -1:
        report('last element must have bitmap -1')
    if surface_count > 131072:
        report(f'BSP surface count {surface_count} exceeds 131072')
    lightmapped = any(page.get('bitmap') >= 0 for page in lightmaps[:-1])
    bitmap_group, bitmap_path = root.reference('lightmaps_bitmap')
    if (bitmap_group is not None or bitmap_path) and bitmap_group != 'bitm':
        report('lightmaps_bitmap group must be bitm')
    if lightmapped and bitmap is None:
        report('missing bitmap tag for lightmapped BSP')
    if lightmapped and not bitmap_path:
        report('lightmaps_bitmap reference must not be empty for lightmapped BSP')

    # the XML calls these signed shorts, but loader vertex indices are unsigned
    triangles = np.frombuffer(b''.join(s.body for s in surfaces), dtype='>u2').reshape(-1, 3)
    next_surface = 0
    for p, page in enumerate(lightmaps):
        final = p == len(lightmaps) - 1
        index = page.get('bitmap')
        if not final:
            if index != p:
                report(f'page bitmap index {index} must equal {p}', p)
            if bitmap is not None and not 0 <= index < len(bitmap.root.block('bitmaps')):
                report(f'page bitmap index {index} is outside bitmap count', p)
        materials = page.block('materials')
        if len(materials) > 2048:
            report('material count exceeds 2048', p)
        for m, material in enumerate(materials):
            for rule in _material_problems(material, final, strict_compression):
                report(rule, p, m)
            start = material.get('surfaces')
            count = material.get('surface_count')
            end = start + count
            in_bounds = 0 <= start <= end <= surface_count
            contiguous = start == next_surface
            if not in_bounds:
                report(f'surface range [{start}, {end}) is outside [0, {surface_count})', p, m)
            if not contiguous:
                report(f'surface range must be contiguous and ascending: expected {next_surface}, got {start}', p, m)
            # never rescan overlapping ranges in malformed tags
            if in_bounds and contiguous:
                vc = material.get('rendered_vertices.vertex_count')
                if np.any(triangles[start:end] >= vc):
                    report('surface vertex index must be less than vertex count', p, m)
            if in_bounds:
                next_surface = max(next_surface, end)
    if next_surface != surface_count:
        report(f'material surface ranges must cover [0, {surface_count}); end is {next_surface}')

    for i, ref in enumerate(root.block('leaf_surfaces')):
        if not 0 <= ref.get('surface') < surface_count:
            report(f'leaf surfaces {i}: surface index outside [0, {surface_count})')
    for c, cluster in enumerate(root.block('clusters')):
        for group, rule in _cluster_group_problems(cluster, lightmaps):
            report(f'cluster {c}, group {group}: {rule}')
        for s, subcluster in enumerate(cluster.block('subclusters')):
            for i, ref in enumerate(subcluster.block('surface_indices')):
                if not 0 <= ref.get('index') < surface_count:
                    report(f'cluster {c}, subcluster {s}, reference {i}: surface index outside [0, {surface_count})')

    if bitmap is not None:
        for p, rule in _bitmap_problems(bitmap):
            report(rule, p)
    return problems


def _cluster_group_problems(cluster, lightmaps):
    # unlike subclusters, cluster lists interleave material headers with surfaces
    refs = cluster.block('surface_indices')
    offset = group = 0
    while offset < len(refs):
        if len(refs) - offset < 3:
            yield group, 'incomplete group header (lightmap index, material index, count)'
            break
        lightmap_index, material_index, count = (ref.get('index') for ref in refs[offset:offset + 3])
        offset += 3
        if count < 0 or count > len(refs) - offset:
            yield group, 'surface count must be non-negative and fit the group list'
            break
        end = offset + count
        if not 0 <= lightmap_index < len(lightmaps):
            yield group, 'lightmap index outside lightmaps block'
        elif not 0 <= material_index < len(lightmaps[lightmap_index].block('materials')):
            yield group, 'material index outside selected lightmap'
        else:
            material = lightmaps[lightmap_index].block('materials')[material_index]
            start = material.get('surfaces')
            stop = start + material.get('surface_count')
            if any(not start <= ref.get('index') < stop for ref in refs[offset:end]):
                yield group, f'surface index outside material surface range [{start}, {stop})'
        offset = end
        group += 1


def _material_problems(material, final, strict_compression):
    problems = []
    group, path = material.reference('shader')
    if group is None or not path:
        problems.append('shader must be non-null')
    buffers = ('rendered_vertices', 'lightmap_vertices') if strict_compression else ('rendered_vertices',)
    for buffer in buffers:
        if material.get(buffer + '.vertex_type') != 0:
            problems.append(f'{buffer} vertex type must be 0')
    vc = material.get('rendered_vertices.vertex_count')
    lc = material.get('lightmap_vertices.vertex_count')
    if vc > 64000:
        problems.append(f'vertex count {vc} exceeds 64000')
    expected_lc = 0 if final else vc
    if lc != expected_lc:
        problems.append(f'lightmap vertex count {lc} must equal {expected_lc}')
    if material.get('surface_count') > 20000:
        problems.append('material surface count exceeds 20000')
    uncompressed = material.data('uncompressed_vertices')
    compressed = material.data('compressed_vertices')
    expected_uc = vc * 56 + lc * 20
    expected_c = vc * 32 + lc * 8
    if len(uncompressed) != expected_uc:
        problems.append(f'uncompressed_vertices size {len(uncompressed)} must equal {expected_uc}')
    if len(compressed) != expected_c:
        problems.append(f'compressed_vertices size {len(compressed)} must equal {expected_c}')
    if len(uncompressed) != expected_uc:
        return problems

    env, lm = unpack_uncompressed(uncompressed, vc, lc)
    finite = True
    for field, values in [('normal', env['normal']), ('binormal', env['binormal']),
                          ('tangent', env['tangent']), ('incident_radiosity', lm['incident_radiosity'])]:
        is_finite = bool(np.all(np.isfinite(values)))
        finite = finite and is_finite
        if not is_finite or np.any(np.abs(values) > np.float32(1.005)):
            problems.append(f'{field} components must be finite and within +/-1.005')
    uv = lm['texcoord']
    finite_uv = bool(np.all(np.isfinite(uv)))
    if not finite_uv or np.any((uv < 0) | (uv > 1)):
        problems.append('lightmap UV must be finite and in [0,1]')
    # non-finite inputs already fail; do not cast their NaNs to compressed integers
    if strict_compression and finite and finite_uv and len(compressed) == expected_c:
        if compressed != compress(env, lm):
            problems.append('compressed vertices differ from recompression of uncompressed vertices')
    return problems


def _bitmap_problems(bitmap):
    problems = []
    pages = bitmap.root.block('bitmaps')
    offset = 0
    for p, page in enumerate(pages):
        texel_bytes = PAGE_TEXEL_BYTES.get(page.get('format'))
        if texel_bytes is None:
            problems.append((p, 'bitmap page must use format 6 (r5g6b5) or 10 (x8r8g8b8)'))
            texel_bytes = 2
        if page.get('type') != 0:
            problems.append((p, 'bitmap page must be type 0 (2D)'))
        if page.get('depth') != 1:
            problems.append((p, 'bitmap page must have depth 1'))
        width, height = page.get('width'), page.get('height')
        if any(side <= 0 or side & (side - 1) for side in (width, height)):
            problems.append((p, 'bitmap page sides must be positive power-of-two sizes'))
        if page.get('mipmap_count') != 0:
            problems.append((p, 'bitmap page mipmap count must be 0'))
        if page.get('pixel_data_offset') != offset:
            problems.append((p, f'bitmap pixel data offset must equal running offset {offset}'))
        offset += width * height * texel_bytes
    if len(bitmap.root.data('processed_pixel_data')) != offset:
        problems.append(('all', f'bitmap total pixel data size must equal the sum of the page sizes ({offset})'))
    if len(bitmap.root.block('sequences')) != len(pages) + 1:
        problems.append(('all', 'bitmap sequences count must equal pages + 1'))
    return problems
