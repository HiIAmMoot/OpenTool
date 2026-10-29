"""Halo winds BSP and model triangles clockwise relative to their normals; Blender wants counter-clockwise."""

from pathlib import PureWindowsPath

import numpy as np
import pytest

from conftest import HCEEK_TAGS
from lightmapper import pipeline
from lightmapper.charts import ChartSettings, chart_and_pack
from lightmapper.formats import scenery
from lightmapper.formats.bsp import read_geometry
from lightmapper.gather import read_radiosity, surface_classes
from lightmapper.tagio.tagfile import read_tag, resolve_reference_path


AGREE = .99


def agreement(positions, normals, tris):
    winding = np.cross(positions[tris[:, 1]] - positions[tris[:, 0]], positions[tris[:, 2]] - positions[tris[:, 0]])
    mean = normals[tris].mean(axis=1)
    norm = np.linalg.norm(winding, axis=1) * np.linalg.norm(mean, axis=1)
    keep = norm > 1e-12
    return float(np.mean((winding[keep] * mean[keep]).sum(axis=1) / norm[keep] > .5))


def build_scene(tags, scenario_path, bsp_path):
    scenario = read_tag(resolve_reference_path(tags, 'scnr', scenario_path))
    _, index, path = pipeline.find_bsp(tags, scenario_path, PureWindowsPath(bsp_path).name)
    geometry = read_geometry(read_tag(resolve_reference_path(tags, 'sbsp', path)))
    radiosity = [read_radiosity(tags, *g.key[:2]) for g in geometry.groups]
    pages = chart_and_pack(geometry.groups, radiosity, ChartSettings(texels_per_unit=.05))
    classes = surface_classes(geometry, len(scenario.root.block('skies')))
    return pipeline._scene(tags, scenario, index, geometry, pages, radiosity, classes)[0]


@pytest.mark.ek
def test_bloodgulch_occluder_winding_matches_normals(monkeypatch):
    # the model's own vertex normals are read beside the positions, then rotated like them
    seen = []
    recorded = []
    real_vertices = scenery._vertices

    def vertices(part):
        elements = part.block('uncompressed_vertices')
        if elements:
            layout = elements[0].layout
            offset = {f.key: f.offset for f in layout.fields if f.kind != 'pad'}['normal']
            raw = np.frombuffer(b''.join(bytes(e.body) for e in elements), np.uint8).reshape(len(elements), layout.size)
            recorded.append(np.ascontiguousarray(raw[:, offset:offset + 12]).view('>f4').reshape(-1, 3).astype(np.float64))
        return real_vertices(part)

    real_model = scenery.model_geometry

    def model_geometry(model, desired):
        recorded.clear()
        geometry = real_model(model, desired)
        geometry.normals = np.concatenate(recorded) if recorded else np.zeros((0, 3))
        return geometry

    real_placed = scenery.iter_placed

    def placed(*args):
        for position, rotation, origin_offset, geometry in real_placed(*args):
            seen.append(geometry.normals @ scenery.placement_rotation(*rotation))
            yield position, rotation, origin_offset, geometry

    monkeypatch.setattr(scenery, '_vertices', vertices)
    monkeypatch.setattr(scenery, 'model_geometry', model_geometry)
    monkeypatch.setattr(scenery, 'iter_placed', placed)
    scene = build_scene(HCEEK_TAGS, r'levels\test\bloodgulch\bloodgulch', r'levels\test\bloodgulch\bloodgulch')
    normals = np.concatenate(seen)
    # the scene lists each triangle's corners separately, so corner normals are those of the model's indices
    occ = scenery.gather_occluders(read_tag(resolve_reference_path(HCEEK_TAGS, 'scnr', r'levels\test\bloodgulch\bloodgulch')),
                                   HCEEK_TAGS, 0)
    # the level's own unlit emitters follow the scenery
    assert len(scene.occ_tris) >= len(occ.tris) > 1000
    positions = scene.occ_positions.reshape(-1, 3)[:len(occ.tris) * 3]
    corner_normals = normals[occ.tris].reshape(-1, 3)
    assert agreement(positions, corner_normals, scene.occ_tris[:len(occ.tris)]) >= AGREE
