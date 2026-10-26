"""Check scenery occluders against the engine's rotation formula and permutation rules."""

import math

import numpy as np
import pytest

from conftest import HCEEK_TAGS as TAGS
from lightmapper.formats import scenery
from lightmapper.formats.scenery import (choose_permutation, choose_permutations, decode_strip,
                                         gather_occluders,
                                         placement_rotation)
from lightmapper.tagio.tagfile import new_element, new_tag, read_tag, write_tag


def child(parent, key):
    return new_element(next(f.layout for f in parent.layout.fields if f.key == key))


def add(parent, key, **values):
    element = child(parent, key)
    for name, value in values.items():
        element.set(name.replace('__', '.'), value)
    parent.block(key).append(element)
    return element


def test_decode_strip():
    assert decode_strip(np.array([0, 1, 2, 3, -1, -1], np.int16)).tolist() == [[0, 1, 2], [1, 3, 2]]


def test_decode_strip_skips_degenerate_and_short():
    assert decode_strip(np.array([0, 1, 1, 2, 3], np.int16)).tolist() == [[1, 2, 3]]
    assert decode_strip(np.array([0, 1], np.int16)).shape == (0, 3)


def test_rotation_yaw():
    m = placement_rotation(math.pi / 2, 0, 0)
    assert np.allclose(m[0], [0, 1, 0]) and np.allclose(m[2], [0, 0, 1])


def test_rotation_matches_engine_formula():
    rng = np.random.default_rng(7)
    for yaw, pitch, roll in rng.uniform(-math.pi, math.pi, (20, 3)):
        cr, sr, cp, sp, cy, sy = (math.cos(roll), math.sin(roll), math.cos(pitch),
                                  math.sin(pitch), math.cos(yaw), math.sin(yaw))
        want = [[cy * cp, sy * cr - cy * sp * sr, cy * sp * cr + sy * sr],
                [-(sy * cp), sy * sp * sr + cy * cr, cy * sr - sy * sp * cr],
                [-sp, -(cp * sr), cp * cr]]
        assert np.allclose(placement_rotation(yaw, pitch, roll), want, atol=1e-6)


def regions_of(*layouts):
    root = new_tag('mod2').root
    for names in layouts:
        reg = add(root, 'regions')
        for name, flags in names:
            add(reg, 'permutations', name=name.encode(), flags=flags)
    return list(root.block('regions'))


def test_desired_zero_is_none():
    # NONE takes the first candidate ("base 2") then re-picks by its number; a literal variant 0 would take "base"
    assert choose_permutations(regions_of([('base 2', 0), ('base', 0)]), 0) == [0]


def test_desired_number_picks_matching_suffix():
    assert choose_permutations(regions_of([('base', 0), ('base 2', 0)]), 2) == [1]


def test_region_without_variant_retries_variant_zero():
    assert choose_permutations(regions_of([('a', 0), ('a 2', 0)], [('b', 0)]), 2) == [1, 0]


def test_unmatched_desired_falls_back():
    assert choose_permutations(regions_of([('base', 0), ('base 2', 0)]), 7) == [0]


def test_first_region_variant_drives_the_others():
    assert choose_permutations(regions_of([('a 3', 0), ('a', 0)], [('b', 0), ('b 3', 0)]), 0) == [0, 1]


def test_flagged_permutation_is_never_chosen():
    assert choose_permutations(regions_of([('x 2', 1), ('x 2', 0)]), 2) == [1]
    assert choose_permutations(regions_of([('x', 1), ('y', 0)]), -1) == [1]


def test_choose_permutation_single_region():
    assert choose_permutation(regions_of([('x', 1), ('y', 0)])[0], -1) == 1


def test_variant_number_from_name_digits():
    reg = regions_of([('base', 0), ('tree 12', 0), ('x7', 0), ('y 70000', 0)])[0]
    assert [scenery.variant_number(p) for p in reg.block('permutations')] == [0, 12, 7, 70000 - 65536]


def write_model(tags, name, lods, shaders=(('shdr', 'a'),), scale=(2.0, 3.0)):
    """lods: geometry index -> list of (positions, strip) parts; the permutation uses them in order."""
    model = new_tag('mod2')
    root = model.root
    root.set('base_map_u_scale', scale[0])
    root.set('base_map_v_scale', scale[1])
    for group, path in shaders:
        shader = add(root, 'shaders')
        shader.set_reference('shader', group, path)
    reg = add(root, 'regions')
    perm = add(reg, 'permutations', permutation_number=0)
    for i, parts in enumerate(lods):
        geo = add(root, 'geometries')
        for shader_index, positions, strip in parts:
            part = add(geo, 'parts', shader_index=shader_index)
            for k, p in enumerate(positions):
                add(part, 'uncompressed_vertices', position=tuple(p), texture_coords=(k / 8, 0.5))
            padded = list(strip) + [-1] * (-len(strip) % 3)
            for j in range(0, len(padded), 3):
                add(part, 'triangle_data', indices=padded[j], indices_1=padded[j + 1],
                    indices_2=padded[j + 2])
    perm.set('super_low', 0)
    perm.set('super_high', len(lods) - 1)
    write_tag(model, tags / (name + '.gbxmodel'))


def write_scenery(tags, name, model):
    tag = new_tag('scen')
    tag.root.set_reference('model', 'mode', model)
    write_tag(tag, tags / (name + '.scenery'))


QUAD = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)]


def scene(tmp_path, placements):
    tags = tmp_path / 'tags'
    tags.mkdir()
    write_model(tags, 'm', [[(0, QUAD, [0, 1, 2])], [(0, QUAD, [0, 1, 2, 3])]])
    write_scenery(tags, 's', 'm')
    scn = new_tag('scnr')
    ref = add(scn.root, 'scenery_palette')
    ref.set_reference('name', 'scen', 's')
    for kw in placements:
        add(scn.root, 'scenery', **kw)
    for mask in (0, 0):
        add(scn.root, 'structure_bsps')
    return scn, tags


def test_highest_lod(tmp_path):
    scn, tags = scene(tmp_path, [dict(placement__position=(0, 0, 0))])
    occ = gather_occluders(scn, tags, 0)
    assert len(occ.tris) == 2  # the super high strip (two triangles), not the one-triangle super low


def test_placement_transform_uv_and_shader(tmp_path):
    scn, tags = scene(tmp_path, [dict(placement__position=(10, 20, 30),
                                      placement__rotation=(math.pi / 2, 0, 0))])
    occ = gather_occluders(scn, tags, 0)
    # yaw 90 degrees: model x -> world y, model y -> world -x
    assert np.allclose(occ.positions[1], [10, 21, 30]) and np.allclose(occ.positions[2], [9, 20, 30])
    assert np.allclose(occ.uv[3], [3 / 8, 0.5])
    assert occ.shaders == [('shdr', 'a', 2.0, 3.0)]
    assert occ.tri_shader.tolist() == [0, 0] and occ.tris.dtype == np.int32


def test_not_placed_and_other_bsp_skipped(tmp_path):
    scn, tags = scene(tmp_path, [
        dict(placement__position=(0, 0, 0)),
        dict(placement__position=(5, 0, 0), placement__not_placed=1),
        dict(placement__position=(9, 0, 0), bsp_indices=0b01),
        dict(placement__position=(13, 0, 0), bsp_indices=0b10),
    ])
    occ = gather_occluders(scn, tags, 1)
    xs = sorted({round(float(x)) for x in occ.positions[:, 0] if abs(x - round(x)) < 1e-6} & {0, 5, 9, 13})
    assert xs == [0, 13]


def test_missing_tags_warn_without_triangles(tmp_path, caplog):
    scn, tags = scene(tmp_path, [dict(placement__position=(0, 0, 0))])
    (tags / 's.scenery').unlink()
    occ = gather_occluders(scn, tags, 0)
    assert len(occ.tris) == 0 and occ.positions.shape == (0, 3) and occ.uv.shape == (0, 2)
    assert 'scenery' in caplog.text


@pytest.mark.ek
def test_bloodgulch_occluders():
    scn = read_tag(TAGS / 'levels/test/bloodgulch/bloodgulch.scenario')
    occ = gather_occluders(scn, TAGS, 0)
    assert len(occ.tris) > 0 and occ.tris.max() < len(occ.positions)
    assert len(occ.uv) == len(occ.positions) and len(occ.tri_shader) == len(occ.tris)
    assert occ.tri_shader.max() < len(occ.shaders)
    # every placement's world bounding box centre lies within its model's farthest vertex of its position
    count = 0
    for pos, rot, geometry in scenery.iter_placed(scn, TAGS, 0):
        world = scenery.world_positions(geometry, pos, rot).astype(np.float64)
        centre = (world.min(axis=0) + world.max(axis=0)) / 2
        radius = np.linalg.norm(geometry.positions.astype(np.float64), axis=1).max()
        assert np.linalg.norm(centre - pos) <= radius + 1e-3
        count += 1
    assert count > 0


def test_unmatched_desired_permutation_warns(tmp_path, caplog):
    scn, tags = scene(tmp_path, [dict(placement__position=(0, 0, 0), placement__desired_permutation=7)])
    assert len(gather_occluders(scn, tags, 0).tris) == 2
    assert 'no permutation is numbered 7' in caplog.text


def test_out_of_range_triangles_warn(tmp_path, caplog):
    tags = tmp_path / 'tags'
    tags.mkdir()
    write_model(tags, 'm', [[(0, QUAD, [0, 1, 2, 9])]])
    write_scenery(tags, 's', 'm')
    scn = new_tag('scnr')
    add(scn.root, 'scenery_palette').set_reference('name', 'scen', 's')
    add(scn.root, 'scenery', placement__position=(0, 0, 0))
    assert len(gather_occluders(scn, tags, 0).tris) == 1
    assert '1 triangles with out-of-range indices dropped' in caplog.text


def test_oblique_placement_matches_hand_transform(tmp_path):
    tags = tmp_path / 'tags'
    tags.mkdir()
    verts = [(1, 2, 3), (4, -5, 6), (-7, 8, 10)]
    write_model(tags, 'm', [[(0, verts, [0, 1, 2])]])
    write_scenery(tags, 's', 'm')
    yaw, pitch, roll = 0.3, -0.4, 0.5
    scn = new_tag('scnr')
    add(scn.root, 'scenery_palette').set_reference('name', 'scen', 's')
    add(scn.root, 'scenery', placement__position=(10, 20, 30), placement__rotation=(yaw, pitch, roll))
    occ = gather_occluders(scn, tags, 0)
    cr, sr, cp, sp, cy, sy = (math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch),
                              math.cos(yaw), math.sin(yaw))
    forward = (cy * cp, sy * cr - cy * sp * sr, cy * sp * cr + sy * sr)
    left = (-(sy * cp), sy * sp * sr + cy * cr, cy * sr - sy * sp * cr)
    up = (-sp, -(cp * sr), cp * cr)
    for got, (x, y, z) in zip(occ.positions, verts):
        want = [(10, 20, 30)[k] + x * forward[k] + y * left[k] + z * up[k] for k in range(3)]
        assert np.allclose(got, want, atol=1e-4)
