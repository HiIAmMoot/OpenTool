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
