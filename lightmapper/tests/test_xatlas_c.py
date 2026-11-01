import os
import sys

import numpy as np
import pytest

from lightmapper import xatlas_c

CUBE = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], np.float32) * 4
CUBE_TRIS = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                      [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])


def test_cube_charts_and_packs():
    atlas = xatlas_c.Atlas()
    atlas.add_mesh(CUBE, CUBE_TRIS)
    atlas.generate(xatlas_c.PackSettings(resolution=64, texels_per_unit=2.0, padding=4))
    assert atlas.page_count >= 1
    verts = atlas.vertices(0)
    assert len(verts) >= 8
    assert np.all(verts['atlas'] >= 0)
    assert np.all(verts['uv'] >= 0) and np.all(verts['uv'] <= atlas.width)
    assert np.all(verts['xref'] < len(CUBE))
    indices = atlas.indices(0)
    assert indices.shape == (12, 3) and indices.max() < len(verts)
    for tri in indices:
        assert len({int(verts['atlas'][i]) for i in tri}) == 1


def test_uv_mesh_atlas_keeps_every_vertex():
    uvs = np.array([[0, 0], [8, 0], [0, 8]], np.float32)
    atlas = xatlas_c.Atlas()
    atlas.add_uv_mesh(uvs, np.array([[0, 1, 2]]))
    atlas.generate(xatlas_c.PackSettings(resolution=64, texels_per_unit=1.0, padding=2))
    verts = atlas.vertices(0)
    assert np.all(verts['atlas'] >= 0)


def test_mixed_mesh_kinds_refused():
    atlas = xatlas_c.Atlas()
    atlas.add_mesh(CUBE, CUBE_TRIS)
    with pytest.raises(ValueError):
        atlas.add_uv_mesh(np.zeros((3, 2), np.float32), np.array([[0, 1, 2]]))


def test_frozen_loads_the_shipped_library_without_building(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, 'frozen', True, raising=False)
    shipped = tmp_path / 'xatlas.dll'
    shipped.write_bytes(b'')
    os.utime(shipped, (0, 0))
    monkeypatch.setattr(xatlas_c, '_library_path', lambda: shipped)
    monkeypatch.setattr(xatlas_c, '_find_clang', lambda: pytest.fail('frozen must not compile'))
    assert xatlas_c.library() == shipped


def test_frozen_missing_library_says_unzip_again(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, 'frozen', True, raising=False)
    monkeypatch.setattr(xatlas_c, '_library_path', lambda: tmp_path / 'xatlas.dll')
    monkeypatch.setattr(xatlas_c, '_find_clang', lambda: pytest.fail('frozen must not compile'))
    with pytest.raises(RuntimeError, match='unzip the download again'):
        xatlas_c.library()


def rotated_quad(degrees):
    turn = np.radians(degrees)
    rotation = np.array([[np.cos(turn), -np.sin(turn)], [np.sin(turn), np.cos(turn)]])
    return (np.array([[0, 0], [200, 0], [200, 100], [0, 100]], float) @ rotation.T + 200).astype(np.float32), np.array([[0, 1, 2], [0, 2, 3]])


def packed_quad(**options):
    # (xatlas stretches a chart to whole texels, hence the large quad: the angle moves by about 1/200)
    uvs, tris = rotated_quad(30)
    atlas = xatlas_c.Atlas()
    atlas.add_uv_mesh(uvs, tris)
    atlas.generate(xatlas_c.PackSettings(resolution=512, texels_per_unit=1.0, padding=2, **options))
    verts = atlas.vertices(0)
    out = verts['uv'][atlas.indices(0)]
    source = uvs[verts['xref'][atlas.indices(0)]]
    return source, out


def test_rotate_charts_to_axis_off_keeps_the_chart_angle():
    source, out = packed_quad(rotate_charts_to_axis=False)
    for before, after in zip(source[:, 1] - source[:, 0], out[:, 1] - out[:, 0]):
        angle = np.arctan2(before[1], before[0])
        assert min(abs(np.arctan2(after[1], after[0]) - angle), abs(np.arctan2(after[0], after[1]) - angle)) < 1e-3
    kept = np.ptp(out.reshape(-1, 2), axis=0)
    _, aligned = packed_quad()
    assert not np.allclose(np.sort(np.ptp(aligned.reshape(-1, 2), axis=0)), np.sort(kept), atol=1.0)
