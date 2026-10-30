"""Extract welded BSP sources and replace their stock lightmap layout.

Source identities retain shader references verbatim. Surface provenance survives
page packing so visibility lists can be remapped without changing their meaning.
"""

from dataclasses import dataclass, field
import struct

import numpy as np

from .vertices import ENV, compress, pack_uncompressed, unpack_uncompressed
from ..tagio.tagfile import Element, Tag, new_element


NO_CLUSTER = -32768
LIGHTMAPPED_LIGHTING = {
    'ambient_color': tuple(np.float32(x) for x in (0.2, 0.2, 0.2)),
    'distant_light_count': 2,
    'distant_light_0_color': tuple(np.float32(x) for x in (1, 1, 1)),
    'distant_light_0_direction': tuple(np.float32(x) for x in (-0.577, -0.577, -0.577)),
    'distant_light_1_color': tuple(np.float32(x) for x in (0.4, 0.4, 0.5)),
    'distant_light_1_direction': tuple(np.float32(x) for x in (0, 0, 1)),
    'reflection_tint': tuple(np.float32(x) for x in (0.5, 1, 1, 1)),
    'shadow_vector': tuple(np.float32(x) for x in (0, 0, -1)),
    'shadow_color': tuple(np.float32(x) for x in (0, 0, 0)),
}


@dataclass
class SourceGroup:
    key: (tuple[str, str, int, int, int]
          | tuple[str, str, int, int, int, tuple[float, ...], tuple[float, ...]])
    template: Element
    env: np.ndarray
    tris: np.ndarray
    surface_ids: np.ndarray


@dataclass
class BspGeometry:
    groups: list[SourceGroup]
    passthrough: Element
    surface_cluster: np.ndarray
    cluster_sky: np.ndarray
    # every surface's vertex indices, which the unlit group's materials index into
    surface_vertices: np.ndarray = field(default_factory=lambda: np.empty((0, 3), np.int32))


@dataclass
class OutMaterial:
    group: int
    env: np.ndarray
    lm: np.ndarray
    tris: np.ndarray
    surface_ids: np.ndarray


@dataclass
class OutPage:
    width: int
    height: int
    materials: list[OutMaterial]


def read_geometry(bsp: Tag) -> BspGeometry:
    root = bsp.root
    pages = root.block('lightmaps')
    if not pages:
        raise ValueError('lightmaps block is empty')
    if pages[-1].get('bitmap') != -1:
        raise ValueError('last lightmaps element has a bitmap')
    surfaces = root.block('surfaces')
    # vertex indices are unsigned on disk despite the XML's signed short fields
    triangles = np.array([struct.unpack('>3H', s.body) for s in surfaces],
                         dtype=np.int32).reshape(-1, 3)
    sources = {}
    for page in pages[:-1]:
        for material in page.block('materials'):
            key = (*material.reference('shader'), material.get('shader_permutation'),
                    material.get('flags'), material.get('breakable_surface'))
            if material.get('flags') & 1:
                key += (material.get('plane'), material.get('centroid'))
            if key not in sources:
                sources[key] = (material, [], [], [])
            _, envs, tris, surface_ids = sources[key]
            env, _ = unpack_uncompressed(material.data('uncompressed_vertices'),
                                        material.get('rendered_vertices.vertex_count'),
                                        material.get('lightmap_vertices.vertex_count'))
            start = material.get('surfaces')
            indices = np.arange(start, start + material.get('surface_count'), dtype=np.int32)
            envs.append(env)
            tris.append(triangles[indices])
            surface_ids.append(indices)
    groups = []
    for key, (template, envs, tris, surface_ids) in sources.items():
        offsets = np.cumsum([0] + [len(env) for env in envs[:-1]])
        joined = np.concatenate(envs)
        unique, inverse = np.unique(joined.view('V56'), return_inverse=True)
        indices = np.concatenate([tri + offset for tri, offset in zip(tris, offsets)])
        groups.append(SourceGroup(key, template.copy(), unique.view(ENV).copy(),
                                  inverse[indices].astype(np.int32), np.concatenate(surface_ids)))

    clusters = root.block('clusters')
    surface_cluster = np.full(len(surfaces), NO_CLUSTER, dtype=np.int16)
    for c, cluster in enumerate(clusters):
        for sub in cluster.block('subclusters'):
            for ref in sub.block('surface_indices'):
                s = ref.get('index')
                if surface_cluster[s] == NO_CLUSTER:
                    surface_cluster[s] = c
    fallback = np.full(len(surfaces), NO_CLUSTER, dtype=np.int16)
    leaf_surfaces = root.block('leaf_surfaces')
    for leaf in root.block('leaves'):
        c = leaf.get('cluster')
        if c < 0:
            continue
        start = leaf.get('surface_references')
        for ref in leaf_surfaces[start:start + leaf.get('surface_reference_count')]:
            s = ref.get('surface')
            if fallback[s] == NO_CLUSTER or c < fallback[s]:
                fallback[s] = c
    missing = surface_cluster == NO_CLUSTER
    surface_cluster[missing] = fallback[missing]
    return BspGeometry(groups, pages[-1].copy(), surface_cluster,
                       np.array([c.get('sky') for c in clusters], dtype=np.int16), triangles)


def apply_layout(bsp: Tag, geometry: BspGeometry, pages: list[OutPage],
                 bitmap_path: str) -> np.ndarray:
    root = bsp.root
    # grouped lists bind surfaces to materials, not just to the old surface order
    for c, cluster in enumerate(root.block('clusters')):
        if cluster.block('surface_indices'):
            raise ValueError(f'cluster {c} has grouped surface_indices; layout requires empty cluster lists')
    old_surfaces = root.block('surfaces')
    if len(pages) > 127:
        raise ValueError('layout exceeds 127 lightmap pages')
    if len(old_surfaces) > 131072:
        raise ValueError('layout exceeds 131072 BSP surfaces')
    passthrough_materials = geometry.passthrough.block('materials')
    if len(passthrough_materials) > 2048:
        raise ValueError('layout exceeds 2048 materials per page')
    for material in passthrough_materials:
        if (material.get('surface_count') > 20000
                or material.get('rendered_vertices.vertex_count') > 64000
                or material.get('lightmap_vertices.vertex_count') > 65535):
            raise ValueError('layout exceeds stock material limits')
    page_layout = next(f.layout for f in root.layout.fields if f.key == 'lightmaps')
    surface_layout = next(f.layout for f in root.layout.fields if f.key == 'surfaces')
    new_pages, new_surfaces, provenance = [], [], []
    for p, page in enumerate(pages):
        if len(page.materials) > 2048:
            raise ValueError('layout exceeds 2048 materials per page')
        element = new_element(page_layout)
        element.set('bitmap', p)
        materials = []
        for out in page.materials:
            if len(out.env) > 64000 or len(out.tris) > 20000:
                raise ValueError('layout exceeds stock material limits')
            if len(out.env) != len(out.lm) or len(out.tris) != len(out.surface_ids):
                raise ValueError('layout vertex or surface counts disagree')
            if not 0 <= out.group < len(geometry.groups):
                raise ValueError('layout group index out of range')
            if np.any(out.tris >= len(out.env)):
                raise ValueError('layout triangle index out of range')
            material = geometry.groups[out.group].template.copy()
            material.set('surfaces', len(new_surfaces))
            material.set('surface_count', len(out.tris))
            for prefix in ('rendered_vertices', 'lightmap_vertices'):
                material.set(prefix + '.vertex_count', len(out.env))
                material.set(prefix + '.vertex_type', 0)
            material.set_data('uncompressed_vertices', pack_uncompressed(out.env, out.lm))
            material.set_data('compressed_vertices', compress(out.env, out.lm))
            for field, value in LIGHTMAPPED_LIGHTING.items():
                material.set(field, value)
            materials.append(material)
            for tri in out.tris:
                surface = new_element(surface_layout)
                surface.body[:] = struct.pack('>3H', *tri)
                new_surfaces.append(surface)
            provenance.append(np.asarray(out.surface_ids, dtype=np.int32))
        element.set_block('materials', materials)
        new_pages.append(element)
    passthrough = geometry.passthrough.copy()
    for material in passthrough.block('materials'):
        start = material.get('surfaces')
        count = material.get('surface_count')
        material.set('surfaces', len(new_surfaces))
        new_surfaces.extend(s.copy() for s in old_surfaces[start:start + count])
        provenance.append(np.arange(start, start + count, dtype=np.int32))
    new_pages.append(passthrough)
    new_to_input = np.concatenate(provenance) if provenance else np.empty(0, dtype=np.int32)
    if not np.array_equal(np.sort(new_to_input), np.arange(len(old_surfaces))):
        raise ValueError('layout must contain every input surface exactly once')
    inverse = np.empty(len(old_surfaces), dtype=np.int32)
    inverse[new_to_input] = np.arange(len(new_to_input), dtype=np.int32)
    for ref in root.block('leaf_surfaces'):
        ref.set('surface', int(inverse[ref.get('surface')]))
    for cluster in root.block('clusters'):
        for sub in cluster.block('subclusters'):
            for ref in sub.block('surface_indices'):
                ref.set('index', int(inverse[ref.get('index')]))
    root.set_block('surfaces', new_surfaces)
    root.set_block('lightmaps', new_pages)
    root.set_reference('lightmaps_bitmap', 'bitm', bitmap_path)
    return new_to_input


# the cluster of each point by the collision BSP walk the engine's scenario code does, -1 outside any
def point_clusters(bsp_tag: Tag, points: np.ndarray) -> np.ndarray:
    root = bsp_tag.root
    result = np.full(len(points), -1, dtype=np.int32)
    collision = root.block('collision_bsp')
    if not collision:
        return result
    nodes = [(n.get('plane'), n.get('back_child'), n.get('front_child'))
             for n in collision[0].block('bsp3d_nodes')]
    planes = [p.get('plane') for p in collision[0].block('planes')]
    leaf_clusters = [leaf.get('cluster') for leaf in root.block('leaves')]
    cluster_count = len(root.block('clusters'))
    for i, point in enumerate(points):
        current = 0
        # (a cycle in a malformed tree would never reach a leaf)
        for _ in range(len(nodes) + 1):
            if current == 0xFFFFFFFF:
                break
            if current & 0x80000000:
                leaf = current & 0x7FFFFFFF
                if leaf < len(leaf_clusters) and 0 <= leaf_clusters[leaf] < cluster_count:
                    result[i] = leaf_clusters[leaf]
                break
            if current >= len(nodes) or nodes[current][0] >= len(planes):
                break
            plane, back, front = nodes[current]
            *normal, d = planes[plane]
            side = float(np.dot(normal, point)) - d
            current = front if side >= 0 else back
    return result
