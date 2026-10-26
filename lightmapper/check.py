import numpy as np

from . import blender_find, pages, raster, resources, xatlas_c


def run_checks() -> list[tuple[str, bool, str]]:
    results = []

    try:
        absent = resources.missing()
        results.append(('files', not absent, 'missing: ' + ', '.join(absent) if absent else 'all required files are present'))
    except Exception as exc:
        results.append(('files', False, str(exc)))

    try:
        xatlas_c.library()
        positions = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                              [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], dtype=np.float32)
        triangles = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
                              [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
                              [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]], dtype=np.int32)
        atlas = xatlas_c.Atlas()
        atlas.add_mesh(positions, triangles)
        atlas.generate(xatlas_c.PackSettings(resolution=32, padding=1))
        if atlas.page_count < 1:
            raise RuntimeError('xatlas generated no atlas pages')
        del atlas
        results.append(('xatlas', True, 'library loaded and cube charted'))
    except Exception as exc:
        results.append(('xatlas', False, str(exc)))

    try:
        image = np.zeros((4, 4, 3), dtype=np.float32)
        mask = np.zeros((4, 4), dtype=np.bool_)
        mask[1, 1] = True
        classes = np.zeros((4, 4), dtype=np.int16)
        raster.dilate(image, mask, classes, 1)
        results.append(('numba', True, 'raster.dilate ran on a 4x4 page'))
    except Exception as exc:
        results.append(('numba', False, str(exc)))

    try:
        page = np.full((32, 32, 3), 0.5, dtype=np.float32)
        denoised = pages._denoise(page, 120)
        if not np.allclose(denoised, page, atol=0.02):
            raise RuntimeError('denoised constant page differs by more than 0.02')
        results.append(('denoiser', True, 'OIDN denoised a constant 32x32 page'))
    except Exception as exc:
        results.append(('denoiser', False, str(exc)))

    try:
        blender = blender_find.find_blender()
        major, minor = blender.version
        results.append(('blender', True, f'Blender {major}.{minor} at {blender.path}'))
    except Exception as exc:
        results.append(('blender', False, str(exc)))
    return results
