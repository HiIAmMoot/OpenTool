"""Real Blender bakes of tiny scenes: orientation, shadows, light linking, emission, glass, cut-outs, bounce."""

import os
import json
import re
from dataclasses import replace
from pathlib import Path
import subprocess

import numpy as np
import pytest

from lightmapper.blender_job import BAKE_SCRIPT, BlenderError, Scene, run_blender, write_job
from lightmapper.formats.lights import SceneLight
from lightmapper.gather import Sun
from lightmapper.settings import Settings


UP = np.float64([0, 0, 1])
CORNERS_UV = ((0, 0), (1, 0), (1, 1), (0, 1))


def quad(corners, normal, lm=CORNERS_UV, tex=CORNERS_UV):
    return np.float32(corners), np.float32(lm), np.float32(tex), np.float32([normal] * 4)


def floor(x0, x1, y0, y1, z=0.0, up=True, lm=CORNERS_UV):
    # counter-clockwise seen from the side the normal points to
    corners = [(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)]
    if up:
        return quad(corners, (0, 0, 1), lm, CORNERS_UV)
    return quad(corners[::-1], (0, 0, -1), lm[::-1], CORNERS_UV[::-1])


def wall_x(x, y0, y1, z0, z1):
    # faces +x
    return quad([(x, y0, z0), (x, y1, z0), (x, y1, z1), (x, y0, z1)], (1, 0, 0))


def build(receivers, occluders=()):
    """Receivers and occluders are dicts of a quad plus page, cls, casts, emission, texture, cutout."""
    cols = {k: [] for k in ('p', 'lm', 'tex', 'nrm', 'tris', 'page', 'cls', 'casts', 'emit', 'texture', 'cut', 'tint', 'ignore')}
    base = 0
    for r in receivers:
        p, lm, tex, nrm = r['quad']
        cols['p'].append(p)
        cols['nrm'].append(nrm)
        cols['lm'].append(lm)
        cols['tex'].append(tex)
        cols['tris'] += [(base, base + k, base + k + 1) for k in range(1, len(p) - 1)]
        for _ in range(len(p) - 2):
            cols['page'].append(r.get('page', 0))
            cols['cls'].append(r.get('cls', 0))
            cols['casts'].append(r.get('casts', True))
            cols['tint'].append(r.get('tint', (1, 1, 1)))
            cols['emit'].append(r.get('emission', (0, 0, 0)))
            cols['texture'].append(r.get('texture', -1))
            cols['cut'].append(r.get('cutout', False))
            cols['ignore'].append(r.get('ignore_normals', False))
        base += len(p)
    o = {k: [] for k in ('p', 'uv', 'tris', 'texture', 'cut', 'casts', 'emit', 'tint')}
    base = 0
    for r in occluders:
        p, _, tex, _ = r['quad']
        o['p'].append(p)
        o['uv'].append(tex)
        o['tris'] += [(base, base + k, base + k + 1) for k in range(1, len(p) - 1)]
        for _ in range(len(p) - 2):
            o['texture'].append(r.get('texture', -1))
            o['cut'].append(r.get('cutout', False))
            o['casts'].append(r.get('casts', True))
            o['emit'].append(r.get('emission', (0, 0, 0)))
            o['tint'].append(r.get('tint', (1, 1, 1)))
        base += len(p)
    n = len(cols['tris'])
    return Scene(
        positions=np.concatenate(cols['p']), normals=np.concatenate(cols['nrm']),
        lm_uv=np.concatenate(cols['lm']), tex_uv=np.concatenate(cols['tex']),
        tris=np.int32(cols['tris']), tri_page=np.int16(cols['page']), tri_class=np.int16(cols['cls']),
        tri_casts_shadow=np.bool_(cols['casts']), tri_emission=np.float32(cols['emit']).reshape(n, 3),
        tri_texture=np.int32(cols['texture']), tri_cutout=np.bool_(cols['cut']),
        tri_tint=np.float32(cols['tint']).reshape(n, 3),
        occ_positions=np.concatenate(o['p']) if o['p'] else np.zeros((0, 3), np.float32),
        occ_uv=np.concatenate(o['uv']) if o['uv'] else np.zeros((0, 2), np.float32),
        occ_tris=np.int32(o['tris']).reshape(-1, 3), occ_tri_texture=np.int32(o['texture']),
        occ_tri_cutout=np.bool_(o['cut']), occ_tri_casts_shadow=np.bool_(o['casts']),
        occ_tri_emission=np.float32(o['emit']).reshape(-1, 3), occ_tri_tint=np.float32(o['tint']).reshape(-1, 3),
        query_pos=np.float32([[0, 0, 0]]), query_normal=np.float32([[0, 0, 1]]),
        query_class=np.int16([0]),
        tri_ignore_normals=np.bool_(cols['ignore']) if any(cols['ignore']) else None,
    )


def sun(direction=UP, classes=(0,), strength=3.0):
    direction = np.float64(direction)
    return Sun(direction / np.linalg.norm(direction), 0.01, np.float64([1, 1, 1]), strength, classes)


def bake(tmp_path, receivers, suns, pages=((16, 16),), occluders=(), textures=None, lights=(), **settings):
    s = Settings(**{**dict(samples=32, bounces=1, w_sun=1.0, w_emit=1.0, w_amb=1.0, albedo_scale=1.0,
                           albedo_fallback=0.5, ignore_normals_mode='tool'), **settings})
    write_job(tmp_path, build(receivers, occluders), list(pages), suns, {}, s, 'all', textures=textures,
              lights=list(lights))
    return run_blender(tmp_path, s)


@pytest.mark.blender
def test_blender_reports_its_stage_seconds(tmp_path):
    result = bake(tmp_path, [full_quad()], [sun()])
    assert set(result.stages) == {'start', 'scene', 'pages', 'vertex'}
    assert all(value >= 0 for value in result.stages.values())


def full_quad(**extra):
    return dict(quad=floor(0, 4, 0, 4), **extra)


@pytest.mark.blender
def test_ignore_normals_receiver_takes_light_from_behind(tmp_path):
    receivers = [dict(quad=floor(0, 4, 0, 4), ignore_normals=True),
                 dict(quad=floor(10, 14, 0, 4, up=False), page=1)]
    panels = [dict(quad=floor(-100, 104, -100, 104, z=-1), emission=(4, 4, 4), casts=False)]
    result = bake(tmp_path, receivers, [], pages=((16, 16), (16, 16)), occluders=panels)
    ratio = result.diffuse_pages[0][2:-2, 2:-2].mean() / result.diffuse_pages[1][2:-2, 2:-2].mean()
    print(f'BACK_SIDE_MEASUREMENT ratio={ratio}')
    assert ratio == pytest.approx(0.5, rel=0.1)


@pytest.mark.blender
def test_ignore_normals_receiver_gets_no_cycles_sun(tmp_path):
    result = bake(tmp_path, [full_quad(ignore_normals=True),
                            dict(quad=floor(10, 14, 0, 4), page=1)], [sun()],
                  pages=((16, 16), (16, 16)))
    assert result.pages[0][2:-2, 2:-2].max() < 1e-6
    np.testing.assert_allclose(result.pages[1][2:-2, 2:-2], 3 / np.pi, rtol=0.05)


@pytest.mark.blender
@pytest.mark.parametrize('radiance', [0, 4])
def test_mixed_texel_corrects_only_its_flagged_diffuse(tmp_path, radiance):
    left = ((0, 0), (2 / 3, 0), (2 / 3, 1), (0, 1))
    right = ((2 / 3, 0), (1, 0), (1, 1), (2 / 3, 1))
    receivers = [dict(quad=floor(0, 4, 0, 4, lm=left), ignore_normals=True),
                 dict(quad=floor(10, 14, 0, 4, lm=right))]
    panels = [dict(quad=floor(-100, 104, -100, 104, z=-1), emission=(radiance,) * 3, casts=False)] if radiance else []
    settings = Settings(samples=1024, vertex_rays=1, bounces=1, w_sun=1, w_emit=1,
                        albedo_scale=1, albedo_fallback=0.5, ignore_normals_mode='tool')
    write_job(tmp_path, build(receivers, panels), [(1, 1)], [sun(strength=np.pi)], {}, settings)
    result = run_blender(tmp_path, settings)
    # six flagged samples take the panel, three unflagged samples take the sun; the vertex page keeps half the panel
    np.testing.assert_allclose(result.pages[0], 2 / 3 * radiance + 1 / 3, rtol=0.05)
    np.testing.assert_allclose(result.diffuse_pages[0], 1 / 3 * radiance + 1 / 3, rtol=0.05)


TWIN_DEVICES = [
    'CPU',
    pytest.param('OPTIX', marks=pytest.mark.gpu_nvidia),
    pytest.param('HIP', marks=pytest.mark.gpu_amd),
]


def twin_sheet_scene(translation=(0, 0, 0), reverse=False, control=False, blocker=False):
    lower_uv = ((0, 0), (0.5, 0), (0.5, 1), (0, 1))
    upper_uv = ((0.5, 0), (1, 0), (1, 1), (0.5, 1))
    p, lm, tex, nrm = floor(16.45, 16.75, -23.4140568, -21.880722, z=-3.61228895,
                            up=False, lm=lower_uv)
    lower = dict(quad=(p, lm, tex, nrm))
    # reversing about corner zero keeps the diagonal, so these are exact reversed triangle twins
    order = [0, 3, 2, 1]
    upper = dict(quad=(p[order], np.float32(upper_uv)[order], tex[order], -nrm[order]))
    receivers = [lower] if control else ([upper, lower] if reverse else [lower, upper])
    panel = dict(quad=floor(16.5, 16.7, -23.3640568, -21.930722, z=-3.69436),
                 casts=False, emission=(84.70593, 63.52938, 56.11761))
    occluders = [panel]
    if blocker:
        occluders.append(dict(quad=floor(15, 18, -25, -20, z=-3.65)))
    scene = build(receivers, occluders)
    scene.positions += np.float32(translation)
    scene.occ_positions += np.float32(translation)
    scene.query_pos = scene.positions[:1].copy()
    scene.query_normal = np.float32([[0, 0, -1]])
    return scene


def bake_twin_sheet(tmp_path, device, **scene_options):
    from lightmapper.gpu import choose
    s = Settings(samples=288, bounces=1, vertex_rays=1, w_emit=1, albedo_scale=1,
                 albedo_fallback=0.5)
    choice = replace(choose('auto', None), cycles=() if device == 'CPU' else (device,),
                     required=device != 'CPU')
    write_job(tmp_path, twin_sheet_scene(**scene_options), [(16, 8)], [], {}, s,
              choice=choice, vertex_pass='skip')
    result = run_blender(tmp_path, s)
    assert result.device == device
    return result.pages[0]


@pytest.fixture(scope='module')
def twin_device_pages():
    return {}


@pytest.mark.blender
@pytest.mark.parametrize('device', TWIN_DEVICES)
@pytest.mark.parametrize('translation', [(0, 0, 0), (400, -400, 100), (400, -400, 1400)],
                         ids=['origin', 'far', 'stock-height'])
@pytest.mark.parametrize('reverse', [False, True], ids=['lower-first', 'upper-first'])
def test_twin_sheet_bakes_light_from_a_panel_on_every_device(tmp_path, device, translation, reverse,
                                                           twin_device_pages):
    options = dict(translation=translation, reverse=reverse)
    control = bake_twin_sheet(tmp_path / 'control', device, control=True, **options)
    page = bake_twin_sheet(tmp_path / 'twins', device, **options)
    lower = page[1:-1, 1:7]
    reference = control[1:-1, 1:7]
    assert reference.mean() > 20, 'the control must see the bright panel'
    np.testing.assert_allclose(lower, reference, rtol=0.02,
                               err_msg='legitimate panel lighting must not be self-shadowed')
    assert page[1:-1, 9:15].max() < 0.01, 'the upper side must retain its own dark island'
    if device != 'CPU':
        cpu = bake_twin_sheet(tmp_path / 'cpu', 'CPU', **options)
        np.testing.assert_allclose(lower, cpu[1:-1, 1:7], rtol=0.02)
    seen = twin_device_pages.setdefault((translation, reverse), {})
    for other_device, other in seen.items():
        np.testing.assert_allclose(lower, other, rtol=0.02, err_msg=f'{device} vs {other_device}')
    seen[device] = lower.copy()
    print(f'TWIN_MEASUREMENT {device} {translation} reverse={reverse} '
          f'lower={lower.mean(axis=(0, 1)).tolist()} control={reference.mean(axis=(0, 1)).tolist()}')


@pytest.mark.blender
@pytest.mark.parametrize('device', TWIN_DEVICES)
def test_twin_separation_keeps_a_real_blocker(tmp_path, device):
    page = bake_twin_sheet(tmp_path, device, blocker=True)
    assert page[1:-1, 1:7].max() < 0.01


def inspect_twin_geometry(tmp_path, blender_exe, scene):
    s = Settings(device='cpu', samples=1, vertex_rays=1)
    write_job(tmp_path, scene, [(16, 8)], [], {}, s)
    original = (tmp_path / 'scene.npz').read_bytes()
    done = subprocess.run(
        [str(blender_exe), '-b', '--factory-startup', '--python-exit-code', '1', '--python',
         str(Path(__file__).with_name('twin_geometry_driver.py')), '--', str(tmp_path)],
        capture_output=True, text=True, stdin=subprocess.DEVNULL,
        env=dict(os.environ, LIGHTMAPPER_BAKE_DIR=str(BAKE_SCRIPT.parent)))
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-1000:]
    assert (tmp_path / 'scene.npz').read_bytes() == original
    with np.load(tmp_path / 'geometry.npz') as data:
        return {key: data[key] for key in data.files}


@pytest.mark.blender
@pytest.mark.parametrize('coordinate', [1400, 4096, -1400, -4096])
@pytest.mark.parametrize('tilted', [False, True], ids=['horizontal', 'tilted'])
def test_twin_separation_survives_float32_at_large_coordinates(tmp_path, blender_exe, coordinate, tilted):
    if tilted:
        base = np.float32([coordinate] * 3)
        u, v = np.float32([2, -1, 0]), np.float32([3, 0, -1])
        normal = np.float64([1, 2, 3]) / np.sqrt(14)
    else:
        base = np.float32([0, 0, coordinate])
        u, v = np.float32([2, 0, 0]), np.float32([0, 2, 0])
        normal = np.float64([0, 0, 1])
    positions = np.array([base, base + u, base + u + v, base + v])
    order = [0, 3, 2, 1]
    scene = build([dict(quad=quad(positions, normal)), dict(quad=quad(positions[order], -normal))])
    mesh = inspect_twin_geometry(tmp_path, blender_exe, scene)
    original = scene.positions[scene.tris].astype(np.float64)
    moved = mesh['positions'][mesh['tris']].astype(np.float64)
    normals = np.array([normal, normal, -normal, -normal])
    outward = ((moved - original) * normals[:, None]).sum(axis=2)
    assert (outward > 0).all(), 'every float32 corner must remain separated outward'
    # four float32 steps at these magnitudes are exact powers of two, independent of the bake implementation
    distance = {1400: 1 / 2048, 4096: 1 / 512}[abs(coordinate)]
    expected = (original + distance * normals[:, None]).astype(np.float32)
    np.testing.assert_array_equal(moved, expected,
                                  err_msg='separation must retain the four-step rounding margin')
    print(f'TWIN_FLOAT32 coordinate={coordinate} tilted={tilted} '
          f'distance={distance} min_outward={outward.min()} max_outward={outward.max()}')


@pytest.mark.blender
@pytest.mark.parametrize('triangles', [4, 20000])
def test_twin_separation_copies_only_bake_corners(tmp_path, blender_exe, triangles):
    scene = twin_sheet_scene()
    if triangles == 20000:
        count = triangles // 4
        offsets = np.float32([(i % 100, i // 100, 0) for i in range(count)])
        scene.positions = (scene.positions[None] + offsets[:, None]).reshape(-1, 3)
        scene.tris = (scene.tris[None] + 8 * np.arange(count)[:, None, None]).reshape(-1, 3)
        for name in ('normals', 'lm_uv', 'tex_uv', 'tri_page', 'tri_class', 'tri_casts_shadow',
                     'tri_emission', 'tri_texture', 'tri_cutout', 'tri_tint'):
            value = getattr(scene, name)
            setattr(scene, name, np.tile(value, (count,) + (1,) * (value.ndim - 1)))
    else:
        # this unpaired neighbour shares corners with a twin but must not move with it
        scene.positions = np.concatenate([scene.positions, np.float32([[16.4, -23.5, -3.5]])])
        scene.normals = np.concatenate([scene.normals, np.float32([[0, 0, -1]])])
        scene.tex_uv = np.concatenate([scene.tex_uv, np.float32([[0.3, 0.7]])])
        scene.lm_uv = np.concatenate([scene.lm_uv, np.float32([[0.1, 0.9]])])
        scene.tris = np.concatenate([scene.tris, [[0, 1, 8]]])
        for name in ('tri_page', 'tri_class', 'tri_casts_shadow', 'tri_emission',
                     'tri_texture', 'tri_cutout', 'tri_tint'):
            value = getattr(scene, name)
            setattr(scene, name, np.concatenate([value, value[:1]]))
    # smooth normals deliberately disagree with geometric normals
    scene.normals[:] = [0.6, 0, 0.8]
    mesh = inspect_twin_geometry(tmp_path, blender_exe, scene)
    with np.load(tmp_path / 'scene.npz') as source:
        paired = source['tri_twin'][:len(scene.tris)] >= 0
        assert paired.sum() == triangles
        assert len(mesh['positions']) == len(np.unique(scene.tris)) + 3 * triangles
        local = mesh['tris']
        assert len(np.unique(local[paired])) == 3 * triangles
        assert not np.isin(local[paired], local[~paired]).any()
        corners = scene.positions[scene.tris].astype(np.float64)
        normal = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
        normal /= np.linalg.norm(normal, axis=1)[:, None]
        assert mesh['separation'] > 0
        expected = corners + paired[:, None, None] * normal[:, None] * mesh['separation']
        np.testing.assert_array_equal(mesh['positions'][local], expected.astype(np.float32))
        for name, uv in [('texture', scene.tex_uv), ('lightmap', scene.lm_uv)]:
            flipped = uv[scene.tris].copy()
            flipped[:, :, 1] = 1 - flipped[:, :, 1]
            np.testing.assert_array_equal(mesh[name], flipped)
        np.testing.assert_allclose(mesh['normals'], scene.normals[scene.tris], atol=1e-4)


@pytest.mark.blender
def test_emit_only_pane_back_face_transmits_the_light_behind_it(tmp_path):
    roof = dict(quad=floor(-100, 100, -100, 100, z=2, up=False), emission=(4, 4, 4), texture=0)
    pane = dict(quad=floor(-100, 100, -100, 100, z=1), casts=False, emission=(4, 4, 4))
    # a black roof cannot bounce the pane's front emission back down into this transmission check
    black = [np.uint8([[[0, 0, 0, 255]]])]
    bare = bake(tmp_path / 'bare', [full_quad()], [], occluders=[roof], textures=black).pages[0]
    screened = bake(tmp_path / 'pane', [full_quad()], [], occluders=[roof, pane], textures=black).pages[0]
    assert bare[2:-2, 2:-2].mean() > 3.5
    np.testing.assert_allclose(screened[2:-2, 2:-2].mean(), bare[2:-2, 2:-2].mean(), rtol=0.05)


@pytest.mark.blender
def test_seed_sets_the_cycles_seed(tmp_path):
    s = Settings(samples=8, vertex_rays=1, seed=2)
    write_job(tmp_path, build([full_quad()]), [(16, 16)], [], {}, s)
    run_blender(tmp_path, s)
    assert 'cycles seed 2000' in (tmp_path / 'blender.log').read_text()


def image_probe(tmp_path, blender_exe, samples=32):
    quarter = dict(quad=floor(0, 4, 0, 4, lm=((0, 0), (0.5, 0), (0.5, 0.5), (0, 0.5))))
    s = Settings(samples=samples, bounces=1, vertex_rays=1)
    write_job(tmp_path, build([quarter]), [(16, 8)], [sun()], {}, s)
    done = subprocess.run(
        [str(blender_exe), '-b', '--factory-startup', '--python-exit-code', '1', '--python',
         str(Path(__file__).with_name('bake_driver.py')), '--', str(tmp_path)],
        capture_output=True, text=True, stdin=subprocess.DEVNULL,
        env=dict(os.environ, LIGHTMAPPER_BAKE_DIR=str(BAKE_SCRIPT.parent)))
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-1000:]
    return np.load(tmp_path / 'probe.npz')


@pytest.mark.blender
def test_use_clear_false_preserves_unwritten_sentinel_in_both_passes(tmp_path, blender_exe):
    with image_probe(tmp_path, blender_exe) as data:
        for name, sentinel in [('DIFFUSE', -1), ('EMIT', -2)]:
            rgb = data[name]
            h, w = rgb.shape[:2]
            np.testing.assert_array_equal(rgb[h // 2:, w // 2:], sentinel)
            assert (rgb[:h // 2, :w // 2] >= 0).all()
        assert data['DIFFUSE'][:2, :2].min() > 0
        np.testing.assert_array_equal(data['EMIT'][:2, :2], 0)


@pytest.mark.blender
@pytest.mark.parametrize(('samples', 'want'), [(32, 4), (1, 1), (256, 28)])
def test_subtexel_image_size_and_sample_budget(tmp_path, blender_exe, samples, want):
    with image_probe(tmp_path, blender_exe, samples) as data:
        np.testing.assert_array_equal(data['size'], [48, 24])
        assert data['samples'] == want


@pytest.mark.blender
def test_tiny_triangle_on_texel_centres_gets_baked(tmp_path):
    scene = build([full_quad()])
    # translated from bloodgulch's (384.5,247.5),(384.5,245.5),(383.5,245.5)
    scene.lm_uv[:3] = np.float32([[2.5, 3.5], [2.5, 1.5], [1.5, 1.5]]) / 8
    scene.tris = scene.tris[:1]
    for name in ('tri_page', 'tri_class', 'tri_casts_shadow', 'tri_emission', 'tri_texture', 'tri_cutout', 'tri_tint'):
        setattr(scene, name, getattr(scene, name)[:1])
    s = Settings(samples=32, bounces=1, vertex_rays=1)
    write_job(tmp_path, scene, [(8, 8)], [sun()], {}, s)
    result = run_blender(tmp_path, s)
    assert result.pages[0][2, 2].min() > 0
    assert result.written[0][2, 2]
    assert not result.written[0][7, 7]


@pytest.mark.blender
def test_lit_quad_and_orientation(tmp_path):
    # the quad's lightmap UVs cover u, v in [0, 0.5] only, so page rows 0..7 and columns 0..7
    quarter = dict(quad=floor(0, 4, 0, 4, lm=((0, 0), (0.5, 0), (0.5, 0.5), (0, 0.5))))
    result = bake(tmp_path, [quarter], [sun()])
    page = result.pages[0]
    assert page.shape == (16, 16, 3)
    assert page[:8, :8].mean() > 0 and page[8:, :].max() == 0 and page[:, 8:].max() == 0
    assert result.device in ('OPTIX', 'CUDA', 'HIP', 'ONEAPI', 'CPU')


@pytest.mark.blender
@pytest.mark.parametrize('kinds,expected', [
    pytest.param(['OPTIX', 'CUDA'], ('OPTIX', 'CUDA'), marks=pytest.mark.gpu_nvidia),
    pytest.param(['HIP'], ('HIP',), marks=pytest.mark.gpu_amd),
    ([], ('CPU',))])
def test_cycles_bakes_on_the_chosen_vendor(tmp_path, kinds, expected):
    from lightmapper.gpu import choose
    s = Settings(samples=8, vertex_rays=1)
    choice = replace(choose('auto', None), cycles=tuple(kinds), required=bool(kinds))
    write_job(tmp_path, build([full_quad()]), [(16, 16)], [sun()], {}, s, choice=choice)
    result = run_blender(tmp_path, s)
    assert result.device in expected and result.pages[0][2:-2, 2:-2].mean() > 0


@pytest.mark.blender
def test_cpu_setting_bakes_on_the_cpu_without_a_warning(tmp_path):
    s = Settings(device='cpu', samples=8, vertex_rays=1)
    write_job(tmp_path, build([full_quad()]), [(16, 16)], [sun()], {}, s)
    result = run_blender(tmp_path, s)
    assert result.device == 'CPU' and result.pages[0][2:-2, 2:-2].mean() > 0
    assert 'no GPU device found' not in (tmp_path / 'blender.log').read_text()


@pytest.mark.blender
def test_cpu_bake_uses_the_cpu_counts_in_both_stages(tmp_path, blender_exe):
    s = Settings(blender=str(blender_exe), device='cpu', samples=90, cpu_samples=18,
                 vertex_rays=64, cpu_vertex_rays=16)
    work = tmp_path
    write_job(work, build([full_quad()]), [(8, 8)], [], {}, s)
    result = run_blender(work, s)
    assert (result.samples_used, result.texel_samples_used) == (18, 18)
    with np.load(work / 'out.npz') as out:
        assert int(out['vertex_rays_used']) == 16
        assert int(out['vertex_rays_cast']) == 16
    assert 'samples 18, texel samples 18' in (work / 'blender.log').read_text()


@pytest.mark.blender
def test_required_cycles_device_that_is_missing_fails_the_bake(tmp_path):
    s = Settings(samples=8, vertex_rays=1)
    path = write_job(tmp_path, build([full_quad()]), [(16, 16)], [sun()], {}, s)
    job = json.loads(path.read_text())
    job.update(cycles_devices=['NOPE'], cycles_required=True, hiprt=False)
    path.write_text(json.dumps(job))
    with pytest.raises(BlenderError, match='none of'):
        run_blender(tmp_path, s)


@pytest.mark.blender
def test_shadow_from_occluder(tmp_path):
    # the occluder covers the half of the quad with x < 2, which is the left half of the page
    cover = dict(quad=floor(-1, 2, -1, 5, z=1.0))
    page = bake(tmp_path, [full_quad()], [sun()], occluders=[cover]).pages[0]
    shadowed, lit = page[2:14, 1:6].mean(), page[2:14, 10:15].mean()
    assert lit > 0 and shadowed < 0.2 * lit


@pytest.mark.blender
def test_light_linking(tmp_path):
    near = dict(quad=floor(0, 2, 0, 2), page=0, cls=0)
    far = dict(quad=floor(10, 12, 0, 2), page=1, cls=1)
    result = bake(tmp_path, [near, far], [sun(classes=(0,))], pages=[(8, 8), (8, 8)])
    assert result.pages[0].mean() > 0
    assert result.pages[1].mean() < 0.05 * result.pages[0].mean()


@pytest.mark.blender
def test_emission(tmp_path):
    glow = dict(quad=floor(0, 2, 0, 2, z=1.0, up=False), page=0, emission=(4.0, 4.0, 4.0))
    plain = dict(quad=floor(0, 2, 0, 2), page=1)
    result = bake(tmp_path, [glow, plain], [], pages=[(8, 8), (8, 8)])
    assert result.pages[0].mean() > 0 and result.pages[1].mean() > 0


@pytest.mark.blender
@pytest.mark.parametrize('emitter_kind', ['receiver', 'occluder'])
def test_emitting_cutout_holes_do_not_light_a_separate_receiver(tmp_path, emitter_kind):
    results = []
    for name, texture in [('opaque', np.full((2, 2, 4), 255, np.uint8)),
                          ('checker', checker()), ('holes', np.zeros((2, 2, 4), np.uint8))]:
        glow = dict(quad=floor(0, 4, 0, 4, z=1, up=False), page=1,
                    emission=(4, 4, 4), texture=0, cutout=True)
        receivers = [full_quad()]
        occluders = []
        if emitter_kind == 'receiver':
            receivers.append(glow)
        else:
            occluders.append(glow)
        scene = build(receivers, occluders)
        scene.query_pos = np.float32([[2, 2, 0]])
        s = Settings(samples=64, vertex_rays=256, w_emit=1.)
        work = tmp_path / name
        write_job(work, scene, [(16, 16), (8, 8)], [], {}, s, textures=[texture])
        result = run_blender(work, s)
        results.append((result.pages[0].mean(), result.vertex_irradiance[0]))
    opaque, checked, holes = np.array(results)
    assert opaque.min() > 0.1
    np.testing.assert_allclose(holes, 0, atol=1e-6)
    assert ((checked > 0.1 * opaque) & (checked < 0.9 * opaque)).all()


@pytest.mark.blender
@pytest.mark.parametrize('sources, weight', [('sun', 1.), ('all', 0.)])
def test_source_disabled_emitter_pane_keeps_diffuse_rays_transparent(tmp_path, sources, weight):
    roof = dict(quad=floor(-100, 100, -100, 100, z=2, up=False),
                page=1, cls=1, texture=0)
    pane = dict(quad=floor(-100, 100, -100, 100, z=1), casts=False, emission=(4, 4, 4), texture=0)
    white = [np.full((1, 1, 4), 255, np.uint8)]
    s = Settings(samples=64, vertex_rays=64, w_emit=weight, w_sun=1., albedo_scale=1.)
    means = []
    for name, occ in [('bare', []), ('pane', [pane])]:
        work = tmp_path / name
        scene = build([full_quad(), roof], occ)
        write_job(work, scene, [(16, 16), (8, 8)], [sun(-UP, classes=(1,))], {},
                  s, sources=sources, textures=white)
        means.append(run_blender(work, s).pages[0][2:-2, 2:-2].mean())
    assert means[0] > 0.1
    assert means[1] == pytest.approx(means[0], rel=0.1)


@pytest.mark.blender
def test_emit_only_occluder_lights_a_receiver_and_blocks_nothing(tmp_path):
    glow = dict(quad=floor(0, 4, 0, 4, z=1.0, up=False), casts=False, emission=(4.0, 4.0, 4.0))
    dark = dict(glow, emission=(0, 0, 0))
    plain = dict(quad=floor(0, 4, 0, 4), page=0)
    lit = bake(tmp_path / 'glow', [plain], [], occluders=[glow]).pages[0].mean()
    unlit = bake(tmp_path / 'dark', [plain], [], occluders=[dark]).pages[0].mean()
    assert lit > 0 and unlit == 0
    # the sun reaches the floor through the emitter's pane
    through = bake(tmp_path / 'sun', [plain], [sun()], occluders=[glow]).pages[0].mean()
    assert through > lit


@pytest.mark.blender
def test_emit_only_occluder_lights_only_its_front_side(tmp_path):
    def lit(up, name):
        glow = dict(quad=floor(0, 4, 0, 4, z=1.0, up=up), casts=False, emission=(4.0, 4.0, 4.0))
        return bake(tmp_path / name, [dict(quad=floor(0, 4, 0, 4), page=0)], [], occluders=[glow]).pages[0].mean()
    facing, away = lit(False, 'facing'), lit(True, 'away')
    assert facing > 0 and away < 0.01 * facing


@pytest.mark.blender
def test_glass_casts_no_shadow(tmp_path):
    glass = dict(quad=floor(-1, 5, -1, 5, z=1.0), page=1, casts=False)
    open_page = bake(tmp_path / 'open', [full_quad()], [sun()]).pages[0]
    under = bake(tmp_path / 'glass', [full_quad(), glass], [sun()], pages=[(16, 16), (8, 8)]).pages[0]
    assert open_page.mean() > 0
    assert under.mean() == pytest.approx(open_page.mean(), rel=0.1)


def half_cylinder(facets=6, radius=2.0):
    # a strip along y over the upper half-turn, one quad per facet, vertex normals pointing out of the axis
    quads = []
    for k in range(facets):
        a0, a1 = np.pi * k / facets, np.pi * (k + 1) / facets
        top = [(radius * np.cos(a), y, radius * np.sin(a)) for a in (a0, a1) for y in (0, 1)]
        normals = [(np.cos(a), 0, np.sin(a)) for a in (a0, a1) for y in (0, 1)]
        corners = [top[0], top[1], top[3], top[2]]
        nrm = [normals[0], normals[1], normals[3], normals[2]]
        lm = [(k / facets, 0), (k / facets, 1), ((k + 1) / facets, 1), ((k + 1) / facets, 0)]
        quads.append(dict(quad=(np.float32(corners), np.float32(lm), np.float32(CORNERS_UV), np.float32(nrm))))
    return quads


@pytest.mark.blender
def test_curved_surface_is_smooth_across_facets(tmp_path):
    facets, texels = 6, 8
    page = bake(tmp_path, half_cylinder(facets), [sun()], pages=[(facets * texels, 8)]).pages[0]
    profile = page[:, :, 0].mean(axis=0)
    steps = np.abs(np.diff(profile))
    edges = np.arange(1, facets) * texels - 1
    # the true irradiance changes by well under a tenth of its peak between two texels
    assert profile.max() > 0 and steps[edges].max() < 0.1 * profile.max()


@pytest.mark.blender
def test_page_holds_irradiance_not_albedo(tmp_path):
    red = np.uint8([[[255, 0, 0, 255]] * 2] * 2)
    page = bake(tmp_path, [full_quad(texture=0)], [sun()], textures=[red]).pages[0]
    r, g, b = page.mean(axis=(0, 1))
    assert g > 0 and r == pytest.approx(g, rel=0.05) and b == pytest.approx(g, rel=0.05)


def checker():
    opaque, hole = [200, 200, 200, 255], [200, 200, 200, 0]
    return np.uint8([[opaque, hole], [hole, opaque]])


@pytest.mark.blender
def test_cutout_checker_shadow(tmp_path):
    screen = dict(quad=floor(0, 4, 0, 4, z=1.0), texture=0, cutout=True)
    page = bake(tmp_path, [full_quad()], [sun()], occluders=[screen], textures=[checker()]).pages[0]
    # texture row 0 is v = 0: opaque where x < 2 and y < 2, and where x > 2 and y > 2; page row follows y
    under_opaque = np.mean([page[2:6, 2:6].mean(), page[10:14, 10:14].mean()])
    under_holes = np.mean([page[2:6, 10:14].mean(), page[10:14, 2:6].mean()])
    assert under_holes > 0 and under_opaque < 0.2 * under_holes


@pytest.mark.blender
def test_bounce_takes_texture_colour(tmp_path):
    # the sun is linked to the wall's class only, so the ground gets nothing but the wall's bounce
    def run(colour, name):
        wall = dict(quad=wall_x(0, 0, 4, 0, 4), page=1, cls=1, texture=0)
        ground = dict(quad=floor(0, 4, 0, 4), page=0, cls=0)
        texture = np.uint8([[colour + [255]] * 2] * 2)
        result = bake(tmp_path / name, [ground, wall], [sun((1, 0, 1), classes=(1,), strength=6.0)],
                      pages=[(16, 16), (8, 8)], textures=[texture])
        return result.pages[0].mean(axis=(0, 1))
    red = run([255, 0, 0], 'red')
    grey = run([128, 128, 128], 'grey')
    assert red[0] > 0 and red[0] > 3 * red[1]
    assert grey[0] > 0 and grey[0] == pytest.approx(grey[1], rel=0.1)


@pytest.mark.blender
def test_albedo_is_read_raw_not_linearised(tmp_path):
    # a 128/255 texel reflects 0.502 of a 255 texel's bounce; linearised from sRGB it would reflect 0.216
    def run(value, name):
        wall = dict(quad=wall_x(0, 0, 4, 0, 4), page=1, cls=1, texture=0)
        ground = dict(quad=floor(0, 4, 0, 4), page=0, cls=0)
        texture = np.uint8([[[value] * 3 + [255]] * 2] * 2)
        result = bake(tmp_path / name, [ground, wall], [sun((1, 0, 1), classes=(1,), strength=6.0)],
                      pages=[(16, 16), (8, 8)], textures=[texture])
        return result.pages[0].mean()
    assert run(128, 'grey') / run(255, 'white') == pytest.approx(128 / 255, rel=0.12)


def pane(tint, **extra):
    # a see-through pane above the floor (it casts no solid shadow) that tints the light crossing it
    return dict(quad=floor(-1, 5, -1, 5, z=1.0), page=1, casts=False, tint=tint, **extra)


def under_pane(tmp_path, tint, name):
    return bake(tmp_path / name, [full_quad(), pane(tint)], [sun()], pages=[(16, 16), (8, 8)]).pages[0]


@pytest.mark.blender
def test_black_tint_pane_blocks_the_sun(tmp_path):
    open_page = bake(tmp_path / 'open', [full_quad()], [sun()]).pages[0]
    blocked = under_pane(tmp_path, (0, 0, 0), 'black')
    assert open_page.mean() > 0
    assert blocked[2:14, 2:14].mean() < 0.05 * open_page.mean()


@pytest.mark.blender
def test_white_tint_pane_passes_the_sun(tmp_path):
    open_page = bake(tmp_path / 'open', [full_quad()], [sun()]).pages[0]
    assert under_pane(tmp_path, (1, 1, 1), 'white').mean() == pytest.approx(open_page.mean(), rel=0.1)


@pytest.mark.blender
def test_coloured_tint_pane_colours_the_light_it_passes(tmp_path):
    open_page = bake(tmp_path / 'open', [full_quad()], [sun()]).pages[0]
    r, g, b = under_pane(tmp_path, (1, 0.5, 0), 'orange')[2:14, 2:14].mean(axis=(0, 1))
    full = open_page[2:14, 2:14].mean()
    assert r == pytest.approx(full, rel=0.1) and g == pytest.approx(0.5 * full, rel=0.1) and b < 0.05 * full


@pytest.mark.blender
def test_pane_still_shades_as_diffuse_for_bake_rays(tmp_path):
    # the pane's own page is lit by the sun from above: tint only changes what shadow rays see
    result = bake(tmp_path, [full_quad(), pane((0, 0, 0))], [sun()], pages=[(16, 16), (8, 8)])
    assert result.pages[1].mean() > 0


@pytest.mark.blender
def test_emitter_lights_only_its_front_side(tmp_path):
    def lit(up, name):
        glow = dict(quad=floor(0, 4, 0, 4, z=1.0, up=up), page=0, emission=(4.0, 4.0, 4.0))
        plain = dict(quad=floor(0, 4, 0, 4), page=1)
        return bake(tmp_path / name, [glow, plain], [], pages=[(8, 8), (8, 8)]).pages[1].mean()
    facing, away = lit(False, 'facing'), lit(True, 'away')
    assert facing > 0
    assert away < 0.01 * facing


@pytest.mark.blender
def test_only_the_page_being_baked_holds_a_sub_texel_buffer(tmp_path, blender_exe):
    import json

    pages = [(16, 16), (16, 16), (8, 8)]
    receivers = [dict(quad=floor(0, 4, 0, 4, z=n), page=n, casts=False) for n in range(3)]
    s = Settings(samples=8, bounces=1, vertex_rays=1)
    write_job(tmp_path, build(receivers), pages, [sun()], {}, s)
    done = subprocess.run(
        [str(blender_exe), '-b', '--factory-startup', '--python-exit-code', '1', '--python',
         str(Path(__file__).with_name('page_images_driver.py')), '--', str(tmp_path)],
        capture_output=True, text=True, stdin=subprocess.DEVNULL,
        env=dict(os.environ, LIGHTMAPPER_BAKE_DIR=str(BAKE_SCRIPT.parent)))
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-1000:]
    sizes = json.loads((tmp_path / 'sizes.json').read_text())
    full = [9 * w * h for w, h in pages]
    assert len(sizes['during']) == 6
    for k, seen in enumerate(sizes['during']):
        page = k // 2
        assert seen[page] == full[page]
        assert all(size < full[other] // 9 for other, size in enumerate(seen) if other != page), seen
    assert max(sizes['after']) <= 4


def bake_texels(tmp_path, texel_samples, receivers, suns):
    s = Settings(samples=32, bounces=1, w_sun=1.0, w_emit=1.0, w_amb=1.0, texel_samples=texel_samples)
    # a quad whose corners sit on texel centres owns few sub-texels at its edge
    write_job(tmp_path, build(receivers), [(16, 16)], suns, {}, s, 'all')
    return run_blender(tmp_path, s).pages[0]


@pytest.mark.blender
def test_top_up_keeps_the_page_and_does_not_mask_the_emission_pass(tmp_path):
    edge = dict(quad=floor(0, 4, 0, 4, lm=((1.5 / 16, 1.5 / 16), (9.5 / 16, 1.5 / 16),
                                          (9.5 / 16, 9.5 / 16), (1.5 / 16, 9.5 / 16))),
                emission=(2.0, 2.0, 2.0))
    off = bake_texels(tmp_path / 'off', 0, [edge], [sun()])
    on = bake_texels(tmp_path / 'on', 256, [edge], [sun()])
    covered = off.sum(axis=2) > 0
    # a masked pass must neither lose a texel nor leave the page darker or brighter (the edge texels differ by noise only)
    assert (on.sum(axis=2) > 0).sum() == covered.sum()
    np.testing.assert_allclose(on[covered].mean(), off[covered].mean(), rtol=0.03)


@pytest.mark.blender
@pytest.mark.parametrize('atlas_path', [False, True], ids=['page-path', 'atlas-path'])
def test_a_mask_is_full_size_only_while_its_page_tops_up(tmp_path, blender_exe, atlas_path):
    import json

    edge = ((1.5 / 16, 1.5 / 16), (9.5 / 16, 1.5 / 16), (9.5 / 16, 9.5 / 16), (1.5 / 16, 9.5 / 16))
    pages = [(16, 16)] * 3
    receivers = [dict(quad=floor(0, 4, 0, 4, z=n, lm=edge), page=n, casts=False) for n in range(3)]
    s = Settings(samples=32, bounces=1, vertex_rays=1)
    write_job(tmp_path, build(receivers), pages, [sun()], {}, s)
    if not atlas_path:
        job = json.loads((tmp_path / 'job.json').read_text())
        job['test_atlas_page_path'] = True
        (tmp_path / 'job.json').write_text(json.dumps(job))
    done = subprocess.run(
        [str(blender_exe), '-b', '--factory-startup', '--python-exit-code', '1', '--python',
         str(Path(__file__).with_name('mask_images_driver.py')), '--', str(tmp_path)],
        capture_output=True, text=True, stdin=subprocess.DEVNULL,
        env=dict(os.environ, LIGHTMAPPER_BAKE_DIR=str(BAKE_SCRIPT.parent)))
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-1000:]
    sizes = json.loads((tmp_path / 'mask_sizes.json').read_text())
    assert all(size == [1, 1] for size in sizes['before'])
    full_reads = 0
    for seen in sizes['during']:
        for other, size in enumerate(seen['masks']):
            if other != seen['page']:
                assert size == [1, 1], seen
        full_reads += seen['masks'][seen['page']] == [16, 16]
    if atlas_path:
        # the tiers bake into the atlas, so the page's reads (the first bake and the emit pass) never see a mask
        assert full_reads == 0
        assert len(sizes['during']) == 3 * 2
    else:
        # the edge quad needs top-up tiers, and only those reads see a full-size mask
        assert full_reads >= 3
        # (two reads per page, the first bake and the emit pass, see a one-texel mask)
        assert full_reads == len(sizes['during']) - 3 * 2
    assert sizes['atlas'] == [1, 1]
    # a mask that is back to one texel must still read as all ones, or the emit pass would be masked
    assert all(a['size'] == [1, 1] and a['min'] == 1.0 for a in sizes['after']), sizes['after']


@pytest.mark.blender
def test_a_masked_out_receiver_still_shadows_its_page_neighbour(tmp_path):
    # the roof owns whole texels, so no top-up tier includes it and its surface is masked out in every one
    edge = ((1.5 / 16, 1.5 / 16), (9.5 / 16, 1.5 / 16), (9.5 / 16, 9.5 / 16), (1.5 / 16, 9.5 / 16))
    whole = ((10 / 16, 10 / 16), (14 / 16, 10 / 16), (14 / 16, 14 / 16), (10 / 16, 14 / 16))
    ground = dict(quad=floor(0, 4, 0, 4, lm=edge), casts=False)
    roof = dict(quad=floor(0, 4, 0, 4, z=1, lm=whole))
    lit = bake_texels(tmp_path / 'lit', 0, [ground], [sun()])
    covered = lit.sum(axis=2) > 0
    shadowed = bake_texels(tmp_path / 'shadowed', 256, [ground, roof], [sun()])
    # only the camera-ray gate lets the roof stay in the way of the ground's masked-in rays
    assert shadowed[covered].max() < 0.05 * lit[covered].mean()


@pytest.mark.blender
def test_flagged_glow_lights_a_neighbour(tmp_path):
    from lightmapper.blender_job import run_blender_glow
    scene = build([dict(quad=floor(-8, 8, -8, 8, z=1), ignore_normals=True),
                   dict(quad=floor(-4, -2, 2, 4), page=1), dict(quad=floor(-4, -2, -4, -2), page=2)])
    s = Settings(samples=256, vertex_rays=1, albedo_fallback=.25, device='cpu', ignore_normals_mode='enhanced')
    write_job(tmp_path, scene, [(8, 8), (8, 8), (8, 8)], [], {}, s)
    bright = np.zeros((8, 8, 3), np.float32)
    bright[:4] = 4
    glow, _ = run_blender_glow(tmp_path, s, [bright, np.zeros((8, 8, 3), np.float32), np.zeros((8, 8, 3), np.float32)])
    np.testing.assert_array_equal(glow[0], 0)
    # (the page's first rows are the low-v side of the quad, so the neighbour on the -y side gets their light)
    assert glow[2][2:-2, 2:-2].mean() > 0.5
    assert glow[1][2:-2, 2:-2].mean() < 0.1


@pytest.mark.blender
@pytest.mark.parametrize('received', [0., 1.])
def test_mixed_glow_source_uses_only_flagged_samples(tmp_path, received):
    from lightmapper.blender_job import run_blender_glow
    from lightmapper.raster import flagged_source

    left = ((0, 0), (2/3, 0), (2/3, 1), (0, 1))
    right = ((2/3, 0), (1, 0), (1, 1), (2/3, 1))
    scene = build([dict(quad=floor(-100, 100, -100, 100, z=1, lm=left), ignore_normals=True),
                   dict(quad=floor(300, 400, 300, 400, z=1, lm=right)),
                   dict(quad=floor(-1, 1, -1, 1), page=1)])
    s = Settings(samples=256, vertex_rays=1, albedo_fallback=.25, device='cpu', ignore_normals_mode='enhanced')
    write_job(tmp_path, scene, [(1, 1), (8, 8)], [], {}, s)
    with np.load(tmp_path / 'scene.npz') as data:
        arrays = dict(data)
    assert len(arrays['texel_pos']) == 6
    sources = [flagged_source(arrays, i, w, h, np.full((6, 3), received, np.float32))
               for i, (w, h) in enumerate([(1, 1), (8, 8)])]
    glow, _ = run_blender_glow(tmp_path, s, sources)
    np.testing.assert_allclose(glow[1][2:-2, 2:-2], .25 * received, atol=.01)


@pytest.mark.blender
def test_page_buffer_is_refilled_before_every_bake(tmp_path, blender_exe):
    edge = ((1.5 / 16, 1.5 / 16), (9.5 / 16, 1.5 / 16), (9.5 / 16, 9.5 / 16), (1.5 / 16, 9.5 / 16))
    receivers = [dict(quad=floor(0, 4, 0, 4, lm=edge), casts=False)]
    s = Settings(samples=32, bounces=1, vertex_rays=1, texel_samples=256)
    write_job(tmp_path, build(receivers), [(16, 16)], [sun()], {}, s)
    done = subprocess.run(
        [str(blender_exe), '-b', '--factory-startup', '--python-exit-code', '1', '--python',
         str(Path(__file__).with_name('page_buffer_driver.py')), '--', str(tmp_path)],
        capture_output=True, text=True, stdin=subprocess.DEVNULL,
        env=dict(os.environ, LIGHTMAPPER_BAKE_DIR=str(BAKE_SCRIPT.parent)))
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-1000:]
    reads = np.load(tmp_path / 'reads.npz')
    count = int(reads['count'])
    assert count >= 3  # base, at least one tier, emit
    # every read is a view of the one page buffer
    assert bool(reads['shared'])
    for n in range(count):
        rgb, written = reads[f'read_{n}'], reads[f'set_{n}']
        np.testing.assert_array_equal(rgb[written], n + 1)
        np.testing.assert_array_equal(rgb[~written], -1)  # not the previous bake's value


def run_scene_settings_driver(tmp_path, blender_exe, clamp_indirect=None, **settings):
    s = Settings(samples=8, vertex_rays=1, **settings)
    write_job(tmp_path, build([full_quad()]), [(16, 16)], [], {}, s)
    if clamp_indirect is not None:
        job = json.loads((tmp_path / 'job.json').read_text())
        job['test_clamp_indirect'] = clamp_indirect
        (tmp_path / 'job.json').write_text(json.dumps(job))
    done = subprocess.run(
        [str(blender_exe), '-b', '--factory-startup', '--python-exit-code', '1', '--python',
         str(Path(__file__).with_name('scene_settings_driver.py')), '--', str(tmp_path)],
        capture_output=True, text=True, stdin=subprocess.DEVNULL,
        env=dict(os.environ, LIGHTMAPPER_BAKE_DIR=str(BAKE_SCRIPT.parent)))
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-1000:]
    return json.loads((tmp_path / 'clamps.json').read_text())


@pytest.mark.blender
def test_cycles_clamps_are_off(tmp_path, blender_exe):
    recorded = run_scene_settings_driver(tmp_path, blender_exe)
    assert {k: recorded[k] for k in ('sample_clamp_direct', 'sample_clamp_indirect')} == {
        'sample_clamp_direct': 0.0, 'sample_clamp_indirect': 0.0}


@pytest.mark.blender
@pytest.mark.parametrize('adaptive', [True, False])
def test_adaptive_sampling_follows_the_setting(tmp_path, blender_exe, adaptive):
    assert run_scene_settings_driver(tmp_path, blender_exe, adaptive_sampling=adaptive)['use_adaptive_sampling'] is adaptive


@pytest.mark.blender
def test_bright_bounce_is_not_clamped(tmp_path, blender_exe):
    # the floor sees only the wall's bounce of the patch: the patch faces the wall, away from the floor
    def floor_mean(name, clamp_indirect=None):
        root = tmp_path / name
        wall = dict(quad=wall_x(0, 0, 4, 0, 4), page=1)
        patch = dict(quad=quad([(0.1, 0, 0), (0.1, 0, 4), (0.1, 4, 4), (0.1, 4, 0)], (-1, 0, 0)),
                     casts=False, emission=(400.0, 400.0, 400.0))
        s = Settings(samples=64, bounces=1, vertex_rays=1)
        write_job(root, build([full_quad(page=0), wall], [patch]), [(16, 16), (16, 16)], [], {}, s)
        if clamp_indirect is not None:
            job = json.loads((root / 'job.json').read_text())
            job['test_clamp_indirect'] = clamp_indirect
            (root / 'job.json').write_text(json.dumps(job))
        return run_blender(root, s).pages[0].mean()
    assert floor_mean('off') > 1.2 * floor_mean('clamped', clamp_indirect=10)


LAMP_PAGE = 32
LAMP_SIDE = 8.0
TEXEL = LAMP_SIDE / LAMP_PAGE


def lamp_light(position, kind=2, axis=(0, 0, -1), reach=100.0, cutoff=0.0, falloff=0.0):
    axis = np.float64(axis) / np.linalg.norm(axis)
    return SceneLight(np.float64(position), axis, np.float64([1, 1, 1]), kind, np.float32(reach),
                      np.float32(np.cos(cutoff)), np.float32(np.cos(falloff)), 0, 'lamp')


def bake_lamp(tmp_path, light):
    result = bake(tmp_path, [dict(quad=floor(0, LAMP_SIDE, 0, LAMP_SIDE))], [], pages=((LAMP_PAGE, LAMP_PAGE),),
                  lights=[light], samples=256, w_light=1.0, vertex_rays=1, device='cpu')
    centres = (np.arange(LAMP_PAGE) + 0.5) * TEXEL
    x, y = np.meshgrid(centres, centres)
    offset = np.stack([x, y, np.zeros_like(x)], axis=-1) - light.position
    distance = np.linalg.norm(offset, axis=-1)
    cosine = -offset[..., 2] / distance
    return result.pages[0][..., 0], distance, cosine, offset / distance[..., None]


@pytest.mark.blender
def test_point_lamp_bakes_strength_over_pi_d_squared_with_the_cosine(tmp_path):
    page, distance, cosine, _ = bake_lamp(tmp_path, lamp_light((4, 4, 1)))
    assert np.median(page / (cosine / (np.pi * distance ** 2))) == pytest.approx(1, rel=0.01)


@pytest.mark.blender
def test_spot_lamp_with_no_falloff_band_is_a_step_at_the_cutoff(tmp_path):
    # equal angles make the Blender tree use a GREATER_THAN step rather than a map range
    light = lamp_light((4, 4, 4), 3, (0, 0, -1), cutoff=0.5, falloff=0.5)
    page, distance, cosine, direction = bake_lamp(tmp_path, light)
    angle = np.arccos(np.clip(direction @ light.axis, -1, 1))
    margin = TEXEL / distance
    inside, outside = angle < 0.5 - margin, angle > 0.5 + margin
    assert inside.sum() > 20 and outside.sum() > 20
    assert np.median(page[inside] / (cosine / (np.pi * distance ** 2))[inside]) == pytest.approx(1, rel=0.02)
    assert page[outside].max() == 0


@pytest.mark.blender
def test_lamp_cutoff_ends_at_reach(tmp_path):
    page, distance, _, _ = bake_lamp(tmp_path, lamp_light((0, 0, 0.25), reach=3))
    assert (page[distance > 3 + TEXEL] == 0).all()
    assert (page[distance < 3 - TEXEL] > 0).all()


@pytest.mark.blender
@pytest.mark.parametrize('axis', [(0, 0, -1), (1, 0, -1)])
def test_spot_lamp_follows_tools_cosine_ramp(tmp_path, axis):
    cutoff, falloff = 0.6, 0.3
    light = lamp_light((4, 4, 4), 3, axis, cutoff=cutoff, falloff=falloff)
    page, distance, cosine, direction = bake_lamp(tmp_path, light)
    angle = np.arccos(np.clip(direction @ light.axis, -1, 1))
    # one texel seen from the lamp, in angle
    margin = TEXEL / distance
    band = (angle > falloff + margin) & (angle < cutoff - margin)
    assert band.sum() > 20
    ramp = (direction @ light.axis - light.cos_cutoff) / (light.cos_falloff - light.cos_cutoff)
    expected = cosine * ramp / (np.pi * distance ** 2)
    assert np.median(page[band] / expected[band]) == pytest.approx(1, rel=0.02)
    assert page[angle > cutoff + margin].max() < 2e-4


def owned_scene(receivers, occluders, owners):
    scene = build(receivers, occluders)
    return replace(scene, occ_tri_owner=np.repeat(np.int32(owners), 2))


def box_faces(low, high):
    (x0, y0, z0), (x1, y1, z1) = low, high
    return [floor(x0, x1, y0, y1, z0), floor(x0, x1, y0, y1, z1),
            quad([(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)], (0, -1, 0)),
            quad([(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)], (0, 1, 0)),
            quad([(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)], (-1, 0, 0)),
            quad([(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)], (1, 0, 0))]


LAMP_SETTINGS = dict(samples=256, bounces=1, w_sun=1.0, w_emit=1.0, w_amb=1.0, albedo_scale=1.0,
                     albedo_fallback=0.5, w_light=1.0, vertex_rays=1, device='cpu')


def bake_owned(tmp_path, receivers, occluders, owners, lights):
    s = Settings(**LAMP_SETTINGS)
    write_job(tmp_path, owned_scene(receivers, occluders, owners), [(LAMP_PAGE, LAMP_PAGE)], [], {}, s, 'all',
              lights=lights)
    return run_blender(tmp_path, s)


def run_lamp_objects_driver(tmp_path, blender_exe, receivers, occluders, owners, lights, edit_job=None, suns=()):
    s = Settings(samples=8, vertex_rays=1)
    write_job(tmp_path, owned_scene(receivers, occluders, owners), [(16, 16)], list(suns), {}, s, 'all', lights=lights)
    if edit_job:
        job = json.loads((tmp_path / 'job.json').read_text())
        edit_job(job)
        (tmp_path / 'job.json').write_text(json.dumps(job))
    done = subprocess.run(
        [str(blender_exe), '-b', '--factory-startup', '--python-exit-code', '1', '--python',
         str(Path(__file__).with_name('lamp_objects_driver.py')), '--', str(tmp_path)],
        capture_output=True, text=True, stdin=subprocess.DEVNULL,
        env=dict(os.environ, LIGHTMAPPER_BAKE_DIR=str(BAKE_SCRIPT.parent)))
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-1000:]
    return json.loads((tmp_path / 'lamp_objects.json').read_text())


def walls(count):
    # one small wall per placement, all alike, so only the owner tells them apart
    return [dict(quad=wall_x(0.1 + 0.1 * n, 0, 1, 0, 1)) for n in range(count)]


@pytest.mark.blender
def test_owner_placement_gets_its_own_objects(tmp_path, blender_exe):
    def occluder_names(name, lights):
        out = run_lamp_objects_driver(tmp_path / name, blender_exe, [full_quad()], walls(2), [0, 1], lights)
        return [o['name'] for o in out['objects'] if o['name'].startswith('occ_')]
    assert len(occluder_names('lit', [lamp_light((1, 1, 1))])) == 2
    assert len(occluder_names('dark', [])) == 1


@pytest.mark.blender
def test_lamps_on_one_placement_share_a_blocker_set(tmp_path, blender_exe):
    lights = [lamp_light((1, 1, 1)), lamp_light((1, 2, 1)), replace(lamp_light((1, 3, 1)), owner=1)]
    out = run_lamp_objects_driver(tmp_path, blender_exe, [full_quad()], walls(2), [0, 1], lights)
    lamps = {lamp['name']: lamp for lamp in out['lamps']}
    assert lamps['lamp_0']['blockers'] == lamps['lamp_1']['blockers'] != lamps['lamp_2']['blockers']
    assert len({lamp['receivers'] for lamp in lamps.values()}) == 1
    blocked = {o['owner']: [c for c in o['collections'] if c.startswith('lamp_blockers')]
               for o in out['objects'] if o['name'].startswith('occ_')}
    assert blocked[0] == [lamps['lamp_2']['blockers']] and blocked[1] == [lamps['lamp_0']['blockers']]


@pytest.mark.blender
def test_cycles_blocker_set_excludes_only_the_owner(tmp_path):
    box = [dict(quad=q) for q in box_faces((1, 1, 1), (3, 3, 3))]
    wall = [dict(quad=quad([(3.5, 0, 0), (3.5, LAMP_SIDE, 0), (3.5, LAMP_SIDE, 4), (3.5, 0, 4)], (1, 0, 0)))]
    result = bake_owned(tmp_path, [dict(quad=floor(0, LAMP_SIDE, 0, LAMP_SIDE))], box + wall,
                        [0] * 6 + [1], [lamp_light((2, 2, 2))])
    page = result.pages[0][..., 0]
    centres = (np.arange(LAMP_PAGE) + 0.5) * TEXEL
    lit, shadowed = page[:, centres < 3.0], page[:, centres > 4.5]
    assert lit.mean() > 0.01
    assert shadowed.mean() < 0.05 * lit.mean()


@pytest.mark.blender
def test_flagged_receivers_are_not_lit_by_cycles_lamps(tmp_path):
    # (tool mode only: enhanced mode gives a flagged receiver zero albedo, which would pass whatever the links did)
    receivers = [full_quad(ignore_normals=True), dict(quad=floor(10, 14, 0, 4), page=1)]
    s = Settings(**{**LAMP_SETTINGS, 'samples': 64, 'ignore_normals_mode': 'tool'})
    write_job(tmp_path, build(receivers), [(16, 16), (16, 16)], [], {}, s, 'all',
              lights=[lamp_light((2, 2, 2)), lamp_light((12, 2, 2))])
    result = run_blender(tmp_path, s)
    assert result.pages[0][2:-2, 2:-2].max() < 1e-6
    assert result.pages[1][2:-2, 2:-2].mean() > 0.01


@pytest.mark.blender
def test_tinted_pane_dims_a_cycles_lamp(tmp_path):
    pane = dict(quad=floor(-1, LAMP_SIDE + 1, -1, LAMP_SIDE + 1, z=2.0), casts=False, tint=(0.5, 1, 1))
    result = bake_owned(tmp_path, [dict(quad=floor(0, LAMP_SIDE, 0, LAMP_SIDE))], [pane], [-1],
                        [lamp_light((4, 4, 4))])
    r, g, _ = result.pages[0][8:24, 8:24].mean(axis=(0, 1))
    assert r / g == pytest.approx(0.5, rel=0.05)


@pytest.mark.blender
def test_link_budget_guard(tmp_path, blender_exe):
    def lamps(count):
        return [replace(lamp_light((1 + 0.1 * (n % 5), 1, 1)), owner=n) for n in range(count)]
    receivers = [dict(quad=floor(0, 4, 0, 4), cls=n, page=n) for n in range(4)]
    suns = [sun(classes=(n,)) for n in range(4)]
    s = Settings(samples=4, bounces=1, w_sun=1.0, w_light=1.0, vertex_rays=1, device='cpu')
    scene = replace(build(receivers, walls(62)), occ_tri_owner=np.repeat(np.arange(62, dtype=np.int32), 2))
    write_job(tmp_path / 'fits', scene, [(8, 8)] * 4, suns, {}, s, 'all', lights=lamps(62))
    run_blender(tmp_path / 'fits', s)
    assert 'light linking sets' not in (tmp_path / 'fits' / 'blender.log').read_text()

    def one_batch(job):
        for lamp in job['lights']:
            lamp['batch'] = 0
    out = run_lamp_objects_driver(tmp_path / 'over', blender_exe, [full_quad()], walls(63), list(range(63)),
                                  lamps(63), edit_job=one_batch)
    assert '64 shadow' in out['error']


@pytest.mark.blender
@pytest.mark.parametrize('classes, fits', [(62, True), (63, False)])
def test_write_job_and_blender_agree_on_the_light_side_link_budget(tmp_path, blender_exe, classes, fits):
    # a receiver per class, each lit by its own sun: one link pattern each, plus the occluders' every-set pattern
    receivers = [dict(quad=floor(0, 4, 0, 4), cls=n) for n in range(classes)]
    suns = [sun(classes=(n,)) for n in range(classes)]
    s = Settings(samples=8, vertex_rays=1)
    write_job(tmp_path / 'parent', owned_scene(receivers, walls(1), [0]), [(16, 16)], suns, {}, s, 'all',
              lights=[lamp_light((1, 1, 1))])
    assert json.loads((tmp_path / 'parent' / 'job.json').read_text())['lights'][0]['batch'] == (0 if fits else 1)

    def main_run(job):
        job['lights'][0]['batch'] = 0
    out = run_lamp_objects_driver(tmp_path / 'blender', blender_exe, receivers, walls(1), [0], [lamp_light((1, 1, 1))],
                                  edit_job=main_run, suns=suns)
    assert (out['error'] is None) == fits and (fits or '64 light' in out['error'])


@pytest.mark.blender
def test_batched_lamps_equal_one_batch(tmp_path, monkeypatch):
    from lightmapper import blender_job
    boxes = [dict(quad=q) for n in range(3) for q in box_faces((1.5 + 2 * n, 0.5, 0.2), (2.0 + 2 * n, 1.0, 0.7))]
    owners = [n for n in range(3) for _ in range(6)]
    lights = [replace(lamp_light((1.75 + 2 * n, 3, 2.5)), owner=n) for n in range(3)]
    suns = [sun((1, 1, 2), strength=1.0)]
    s = Settings(**{**LAMP_SETTINGS, 'samples': 1024, 'bounces': 1})
    receivers = [dict(quad=floor(0, 8, 0, 8))]
    scene = owned_scene(receivers, boxes, owners)

    def run(name, seed=0, per_batch=None):
        root = tmp_path / name
        t = replace(s, seed=seed)
        with monkeypatch.context() as patch:
            if per_batch:
                patch.setattr(blender_job, 'lamp_batches', lambda owners, suns: [[o] for o in dict.fromkeys(owners)])
            write_job(root, scene, [(LAMP_PAGE, LAMP_PAGE)], suns, {}, t, 'all', lights=lights)
        total = run_blender(root, t).pages[0]
        if per_batch:
            for k in (1, 2):
                total = total + blender_job.run_blender_lamp_batch(root, t, k)[0][0]
        return total

    one, other, batched = run('one'), run('other', seed=1), run('batched', per_batch=True)
    assert batched.mean() == pytest.approx(one.mean(), rel=0.02)
    noise = (one - other).std()
    assert (np.abs(batched - one) < 3 * noise).mean() > 0.97


def split_lamp_bakes(tmp_path, monkeypatch, pane):
    # one bake with the lamp in the main run, and the same lamp run as batch 1 of a main run that has none
    from lightmapper import blender_job
    s = Settings(**{**LAMP_SETTINGS, 'samples': 256})
    scene = owned_scene([dict(quad=floor(0, LAMP_SIDE, 0, LAMP_SIDE))], pane, [-1] * len(pane))
    lights = [lamp_light((4, 4, 3))]

    def write(name, split):
        with monkeypatch.context() as patch:
            if split:
                patch.setattr(blender_job, 'lamp_batches', lambda owners, room: [[], list(dict.fromkeys(owners))])
            write_job(tmp_path / name, scene, [(LAMP_PAGE, LAMP_PAGE)], [], {}, s, 'all', lights=lights)
        return tmp_path / name

    whole = run_blender(write('whole', False), s).pages[0]
    main = run_blender(write('split', True), s)
    batch = blender_job.run_blender_lamp_batch(tmp_path / 'split', s, 1)[0][0]
    return whole, main.pages[0], batch


@pytest.mark.blender
def test_a_lamp_batch_adds_the_lamp_only_not_the_emission_again(tmp_path, monkeypatch):
    glow = [dict(quad=floor(-1, LAMP_SIDE + 1, -1, LAMP_SIDE + 1, z=5.0, up=False), casts=False,
                 emission=(0.5, 0.5, 0.5))]
    whole, main, batch = split_lamp_bakes(tmp_path, monkeypatch, glow)
    assert main.mean() > 0.1 * whole.mean()
    assert (main + batch).mean() == pytest.approx(whole.mean(), rel=0.03)


@pytest.mark.blender
def test_a_lamp_batch_takes_its_own_noise_seed(tmp_path, monkeypatch):
    whole, _, batch = split_lamp_bakes(tmp_path, monkeypatch, [])
    # (the same lamp on the same scene, so only the seed tells the two bakes' noise apart)
    assert not np.array_equal(whole, batch)
    assert batch.mean() == pytest.approx(whole.mean(), rel=0.03)


@pytest.mark.blender
def test_page_bake_point_light_matches_the_vertex_pass_on_a_plane(tmp_path):
    # the page bakes strength cos / (pi d^2) and the vertex pass irradiance strength cos / d^2 (the pass works in pi x
    # page units), so a page texel times pi is the irradiance at the same point
    point = ((16 + .5) * TEXEL, (16 + .5) * TEXEL, 0)
    scene = replace(build([dict(quad=floor(0, LAMP_SIDE, 0, LAMP_SIDE))]), query_pos=np.float32([point]))
    s = Settings(**LAMP_SETTINGS)
    write_job(tmp_path, scene, [(LAMP_PAGE, LAMP_PAGE)], [], {}, s, 'all', lights=[lamp_light((4, 4, 1))])
    result = run_blender(tmp_path, s)
    assert result.vertex_irradiance[0] == pytest.approx(
        np.pi * result.pages[0][16, 16, 0], rel=0.05)
    assert result.vertex_irradiance[0] > 0.3


def run_receiver_groups(root, blender_exe, edit_job=None):
    job = json.loads((root / 'job.json').read_text())
    if edit_job:
        edit_job(job)
        (root / 'job.json').write_text(json.dumps(job))
    done = subprocess.run(
        [str(blender_exe), '-b', '--factory-startup', '--python-exit-code', '1', '--python',
         str(Path(__file__).with_name('receiver_groups_driver.py')), '--', str(root)],
        capture_output=True, text=True, stdin=subprocess.DEVNULL,
        env=dict(os.environ, LIGHTMAPPER_BAKE_DIR=str(BAKE_SCRIPT.parent)))
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-1000:]
    return json.loads((root / 'groups.json').read_text())


def flagged_and_plain_floors():
    # (side by side on the page: the pipeline never puts flagged and unflagged triangles in one chart, as charts are built
    # per shader group with padding)
    left = ((0, 0), (0.5, 0), (0.5, 1), (0, 1))
    right = ((0.5, 0), (1, 0), (1, 1), (0.5, 1))
    return [dict(quad=floor(0, 4, 0, 4, lm=left), ignore_normals=True), dict(quad=floor(10, 14, 0, 4, lm=right))]


def slot_of(group, flagged_triangle):
    # (the flagged floor is the scene's first two triangles)
    return group['slots'][group['slot_of_triangle'][0 if flagged_triangle else 2]]


@pytest.mark.blender
@pytest.mark.parametrize('mode, choice', [('enhanced', 'zero_albedo'), ('tool', 'translucent')])
def test_flagged_and_unflagged_share_an_object_without_lamps(tmp_path, blender_exe, mode, choice):
    # (the only sun lights class 1, so no receiver here has a sun bit)
    s = Settings(samples=8, vertex_rays=1, ignore_normals_mode=mode)
    write_job(tmp_path, build(flagged_and_plain_floors()), [(16, 16)], [sun(classes=(1,))], {}, s)
    groups = run_receiver_groups(tmp_path, blender_exe)
    assert len(groups) == 1 and groups[0]['triangles'] == 4 and groups[0]['flagged']
    flagged, plain = slot_of(groups[0], True), slot_of(groups[0], False)
    assert flagged[choice] and flagged['glow_image'] is None
    assert not any(plain.values())


@pytest.mark.blender
def test_flag_stays_in_the_object_key_when_the_job_has_lamps(tmp_path, blender_exe):
    s = Settings(samples=8, vertex_rays=1, w_light=1.0, ignore_normals_mode='enhanced')
    write_job(tmp_path, build(flagged_and_plain_floors()), [(16, 16)], [sun(classes=(1,))], {}, s, 'all',
              lights=[lamp_light((2, 2, 3))])

    def lamp_in_batch_one(job):
        job['lights'][0]['batch'] = 1

    groups = run_receiver_groups(tmp_path, blender_exe, lamp_in_batch_one)
    assert sorted((g['flagged'], g['triangles']) for g in groups) == [(False, 2), (True, 2)]
    for group in groups:
        assert group['slots'][0]['zero_albedo'] == group['flagged']


@pytest.mark.blender
def test_glow_run_merges_flagged_receivers(tmp_path, blender_exe):
    s = Settings(samples=8, vertex_rays=1, ignore_normals_mode='enhanced')
    write_job(tmp_path, build(flagged_and_plain_floors()), [(16, 16)], [], {}, s)
    np.savez(tmp_path / 'glow_pages.npz', page_0=np.ones((2, 2, 3), np.float32),
             box_0=np.array([0, 0, 2, 2, 16, 16], np.int32))

    def as_glow_run(job):
        job.update(glow=True, glow_pages='glow_pages.npz', suns=[], lights=[])

    groups = run_receiver_groups(tmp_path, blender_exe, as_glow_run)
    assert len(groups) == 1 and groups[0]['triangles'] == 4
    flagged, plain = slot_of(groups[0], True), slot_of(groups[0], False)
    assert flagged['glow_image'] == 'glow_0' and plain['glow_image'] is None


@pytest.mark.blender
def test_flagged_slot_of_a_page_without_glow_holds_the_black_image(tmp_path, blender_exe):
    s = Settings(samples=8, vertex_rays=1, ignore_normals_mode='enhanced')
    write_job(tmp_path, build(flagged_and_plain_floors()), [(16, 16)], [], {}, s)
    # (no entry for page 0: the page gathered nothing)
    np.savez(tmp_path / 'glow_pages.npz')

    def as_glow_run(job):
        job.update(glow=True, glow_pages='glow_pages.npz', suns=[], lights=[])

    groups = run_receiver_groups(tmp_path, blender_exe, as_glow_run)
    flagged = slot_of(groups[0], True)
    assert flagged['glow_image'] == 'glow_black' and flagged['pure_emission'] and not flagged['zero_albedo']


@pytest.mark.blender
@pytest.mark.parametrize('emissive', [False, True])
@pytest.mark.parametrize('mode', ['tool', 'enhanced'])
def test_merged_object_bakes_what_two_objects_baked(tmp_path, mode, emissive):
    s = Settings(samples=64, bounces=1, w_sun=1.0, w_emit=1.0, w_amb=1.0, vertex_rays=1, device='cpu',
                 albedo_fallback=0.5, ignore_normals_mode=mode)
    floors = flagged_and_plain_floors()
    if emissive:
        # (an unflagged emitter without sun bits merges with the flagged floor in the main run)
        low = ((0.5, 0), (1, 0), (1, 0.5), (0.5, 0.5))
        high = ((0.5, 0.5), (1, 0.5), (1, 1), (0.5, 1))
        floors[1] = dict(quad=floor(10, 14, 0, 4, lm=low))
        floors.append(dict(quad=floor(20, 24, 0, 4, lm=high), emission=(2, 2, 2)))
    panels = [dict(quad=floor(-100, 104, -100, 104, z=-1), emission=(4, 4, 4), casts=False)]

    def baked(name, split):
        root = tmp_path / name
        write_job(root, build(floors, panels), [(16, 16)], [sun(classes=(1,))], {}, s)
        if split:
            job = json.loads((root / 'job.json').read_text())
            job['test_split_flagged'] = True
            (root / 'job.json').write_text(json.dumps(job))
        return run_blender(root, s)

    two, one = baked('two', True), baked('one', False)
    assert two.written[0].any()
    np.testing.assert_array_equal(one.written[0], two.written[0])
    np.testing.assert_allclose(one.pages[0], two.pages[0], atol=1e-6, rtol=0)


def quantise8(page):
    from lightmapper.pages import quantise
    return quantise(page, Settings(lightmap_32bit=True, dither=False))


def glow_run(work, scene, pages, sources, uncropped):
    from lightmapper.blender_job import _run_blender, run_blender_glow
    s = Settings(samples=256, vertex_rays=1, albedo_fallback=.25, device='cpu', ignore_normals_mode='enhanced')
    write_job(work, scene, pages, [], {}, s)
    glow, _ = run_blender_glow(work, s, sources)
    if not uncropped:
        return glow
    # (the test-only key makes the run read each page as a full image, as it did before cropping)
    job = json.loads((work / 'job_glow.json').read_text())
    job['test_glow_uncropped'] = True
    (work / 'job_glow.json').write_text(json.dumps(job))
    _, out = _run_blender(work, s, 'job_glow.json', 'blender_glow_full.log')
    with np.load(out) as data:
        return [np.array(data[f'page_{i}'], np.float32) for i in range(len(pages))]


def assert_rounding_only(cropped, full):
    # (an 8-bit step is the smallest change that reaches a page, so only float rounding may differ)
    for got, want in zip(cropped, full):
        assert np.array_equal(quantise8(got), quantise8(want))
        assert np.abs(got - want).max() <= 1e-6 * max(np.abs(want).max(), 1e-12)


@pytest.mark.blender
def test_cropped_glow_bakes_what_the_full_image_baked(tmp_path):
    scene = build([dict(quad=floor(-8, 8, -8, 8, z=1), ignore_normals=True),
                   dict(quad=floor(-4, -2, 2, 4), page=1), dict(quad=floor(-4, -2, -4, -2), page=2)])
    source = np.zeros((8, 8, 3), np.float32)
    source[1:4, 2:5] = 1 + np.arange(9, dtype=np.float32).reshape(3, 3, 1) * np.array([.3, .2, .1], np.float32)
    sources = [source, np.zeros((8, 8, 3), np.float32), np.zeros((8, 8, 3), np.float32)]
    cropped = glow_run(tmp_path / 'cropped', scene, [(8, 8)] * 3, sources, False)
    full = glow_run(tmp_path / 'full', scene, [(8, 8)] * 3, sources, True)
    assert full[2].max() > 0.05
    assert np.array_equal(quantise8(cropped[2]), quantise8(full[2]))
    assert np.abs(cropped[2] - full[2]).max() <= 1e-6 * np.abs(full[2]).max()
    assert_rounding_only(cropped, full)


@pytest.mark.blender
def test_cropped_glow_reads_black_outside_its_box(tmp_path):
    left = ((0, 0), (.5, 0), (.5, 1), (0, 1))
    right = ((.5, 0), (1, 0), (1, 1), (.5, 1))
    # (the lit half is far from the receiver, so only the dark half's glow could reach it)
    scene = build([dict(quad=floor(300, 400, 300, 400, z=1, lm=left), ignore_normals=True),
                   dict(quad=floor(-4, 4, -4, 4, z=1, lm=right), ignore_normals=True),
                   dict(quad=floor(-1, 1, -1, 1), page=1)])
    source = np.zeros((8, 8, 3), np.float32)
    source[:, :4] = 4 + np.arange(32, dtype=np.float32).reshape(8, 4, 1)
    sources = [source, np.zeros((8, 8, 3), np.float32)]
    cropped = glow_run(tmp_path / 'cropped', scene, [(8, 8), (8, 8)], sources, False)
    full = glow_run(tmp_path / 'full', scene, [(8, 8), (8, 8)], sources, True)
    assert_rounding_only(cropped, full)
    # (nothing from the dark half: an edge-extended crop would carry the box's last column across it)
    assert cropped[1][2:-2, 2:-2].max() < 0.05
    np.testing.assert_array_equal(cropped[0][:, 4:], 0)


@pytest.mark.blender
def test_page_without_glow_keeps_its_flagged_receivers_black(tmp_path):
    scene = build([dict(quad=floor(-8, 8, -8, 8, z=1), ignore_normals=True),
                   dict(quad=floor(-4, -2, 2, 4), page=1, ignore_normals=True),
                   dict(quad=floor(-4, -2, -4, -2), page=2)])
    bright = np.zeros((8, 8, 3), np.float32)
    bright[:4] = 4
    sources = [bright, np.zeros((8, 8, 3), np.float32), np.zeros((8, 8, 3), np.float32)]
    cropped = glow_run(tmp_path / 'cropped', scene, [(8, 8)] * 3, sources, False)
    full = glow_run(tmp_path / 'full', scene, [(8, 8)] * 3, sources, True)
    assert full[2].max() > 0.05
    assert_rounding_only(cropped, full)


# ---------- edge top-ups on the atlas

ATLAS_DRIVER = Path(__file__).with_name('atlas_driver.py')
TIER_LINE = re.compile(r'atlas tier (\d+): slots (\d+), side (\d+), atlas pixels baked (\d+) \(([\d.]+)\)')
FALLBACK_LINE = re.compile(r'atlas tier (\d+): page bake \((.*)\)')
ATLAS_RUN = {'name': 'A', 'job': {'seed': 0}}


def page_run(name, seed=0, **job):
    return {'name': name, 'job': {'seed': seed, 'test_atlas_page_path': True, **job}}


def device_choice(device):
    from lightmapper.gpu import choose
    return replace(choose('auto', None), cycles=() if device == 'CPU' else (device,), required=device != 'CPU')


class AtlasBake:
    def __init__(self, root, runs):
        self.root = root
        self.report = json.loads((root / 'atlas.json').read_text())

    def out(self, name):
        with np.load(self.root / f'out_{name}.npz') as data:
            return {key: data[key] for key in data.files}

    def topped_up(self, name, page=0):
        with np.load(self.root / f'masks_{name}.npz') as data:
            mask = data[f'mask_{page}']
        return mask & self.out(name)[f'written_{page}']

    def log(self, name):
        return self.report[name]['log']

    def tiers(self, name):
        return [tuple(int(v) for v in m.groups()[:4]) for m in map(TIER_LINE.search, self.log(name)) if m]

    def fallbacks(self, name):
        return [(int(m.group(1)), m.group(2)) for m in map(FALLBACK_LINE.search, self.log(name)) if m]

    def tier_calls(self, name):
        tiers, base_next = [], True
        for call in self.report[name]['calls']:
            if call['type'] == 'EMIT':
                base_next = True
            elif base_next:
                base_next = False
            else:
                tiers.append(call)
        return tiers


def atlas_bake(tmp_path, blender_exe, scene, pages, suns, runs, device='CPU', samples=72 * 16, texel_samples=72 * 16,
               textures=None, lights=(), **settings):
    s = Settings(**{**dict(samples=samples, texel_samples=texel_samples, bounces=1, adaptive_sampling=False,
                           w_sun=1.0, w_emit=1.0, w_amb=1.0, w_light=1.0, albedo_scale=1.0, albedo_fallback=0.5,
                           ignore_normals_mode='tool', vertex_rays=1), **settings})
    write_job(tmp_path, scene, list(pages), suns, {}, s, 'all', textures=textures, choice=device_choice(device),
              vertex_pass='skip', lights=list(lights))
    (tmp_path / 'driver.json').write_text(json.dumps({'runs': runs}))
    done = subprocess.run(
        [str(blender_exe), '-b', '--factory-startup', '--python-exit-code', '1', '--python', str(ATLAS_DRIVER),
         '--', str(tmp_path)],
        capture_output=True, text=True, stdin=subprocess.DEVNULL,
        env=dict(os.environ, LIGHTMAPPER_BAKE_DIR=str(BAKE_SCRIPT.parent)))
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-1000:]
    return AtlasBake(tmp_path, runs)


def equivalence(a, p0, p1, selected):
    """The spike's criteria on texels: (a) the bias of A - P0 and (b) A against P1 as P0 is, over 16 groups."""
    A, P0, P1 = a[selected], p0[selected], p1[selected]
    n = len(A)
    assert n >= 32, f'only {n} topped-up texels to judge'
    diff = A - P0
    error = diff.std(axis=0, ddof=1) / np.sqrt(n)
    result = {'n': n, 'bias_z': (np.abs(diff.mean(axis=0)) / np.maximum(error, 1e-30)).tolist(),
              'bias_ok': bool((np.abs(diff.mean(axis=0)) <= 3 * error).all())}
    after, before = np.abs(A - P1).sum(axis=1), np.abs(P0 - P1).sum(axis=1)
    group = np.arange(n) % 16
    for name, stat in (('mean', np.mean), ('p99', lambda x: np.percentile(x, 99))):
        delta = stat(after) - stat(before)
        per = [stat(after[group == g]) - stat(before[group == g]) for g in range(16)]
        se = np.std(per, ddof=1) / 4
        result[name] = {'delta': float(delta), 'se': float(se), 'ratio': float(delta / se) if se > 0 else 0.0,
                        'ok': bool(delta <= 3 * se + 1e-6 * stat(before))}
    return result


def assert_equivalent(label, a, p0, p1, selected):
    result = equivalence(a, p0, p1, selected)
    print(f'ATLAS_EQUIVALENCE {label} {json.dumps(result)}')
    assert result['bias_ok'], (label, result)
    assert result['mean']['ok'] and result['p99']['ok'], (label, result)
    return result


def compare_runs(label, bake, selected=None):
    a, p0, p1 = (bake.out(name)['page_0'] for name in ('A', 'P0', 'P1'))
    return assert_equivalent(label, a, p0, p1, bake.topped_up('A') if selected is None else selected)


def chart32(theta, centre, half, page=32):
    c, s = np.cos(np.radians(theta)), np.sin(np.radians(theta))
    return tuple((float((centre[0] + half * (c * x - s * y)) / page), float((centre[1] + half * (s * x + c * y)) / page))
                 for x, y in ((-1, -1), (1, -1), (1, 1), (-1, 1)))


def soft_sun(direction=(0.4, 0.2, 1), classes=(0,), strength=3.0, angle=0.12):
    direction = np.float64(direction)
    return Sun(direction / np.linalg.norm(direction), angle, np.float64([1, 1, 1]), strength, classes)


def edge_scene(**extra):
    # a rotated chart on a 32 x 32 page, so its edge texels own few sub-texels, with a roof throwing a penumbra on it
    ground = dict(quad=floor(0, 8, 0, 8, lm=chart32(23, (16, 16), 9.3)), **extra)
    return build([ground], [dict(quad=floor(2, 5, 1, 4, z=1.5))])


@pytest.mark.blender
def test_top_ups_bake_only_the_atlas(tmp_path, blender_exe):
    bake = atlas_bake(tmp_path, blender_exe, edge_scene(), [(32, 32)], [soft_sun()], [ATLAS_RUN, page_run('P0')])
    tiers = bake.tiers('A')
    calls = bake.tier_calls('A')
    assert tiers and not bake.fallbacks('A'), bake.log('A')
    assert len(calls) == len(tiers)
    for call, (n, slots, side, baked) in zip(calls, tiers):
        assert all(name.startswith('atlas') for name in call['selected']), call['selected']
        assert call['images']['atlas'] == [side, side]
        assert call['images']['page_0'] == [1, 1]
        for info in call['objects'].values():
            assert info['visible'] == {'visible_camera': True, 'visible_diffuse': False, 'visible_glossy': False,
                                       'visible_shadow': False, 'visible_transmission': False,
                                       'visible_volume_scatter': False}
        assert sum(info['triangles'] for info in call['objects'].values()) == slots
    # the first and the last bake still see the page's own receivers, at full size then
    first, last = bake.report['A']['calls'][0], bake.report['A']['calls'][-1]
    assert all(name.startswith('recv') for name in first['selected'] + last['selected'])
    assert first['images']['page_0'] == last['images']['page_0'] == [96, 96]
    for name in ('A', 'P0'):
        print(f'ATLAS_TIMING {name} page seconds {float(bake.out(name)["seconds_pages"]):.2f} '
              f'tiers {len(bake.tiers(name))} samples {[c["samples"] for c in bake.tier_calls(name)]}')
    assert not bake.tiers('P0')


@pytest.mark.blender
@pytest.mark.parametrize('device', TWIN_DEVICES)
@pytest.mark.parametrize('slivers', [False, True], ids=['R1-R4', 'R5'])
def test_atlas_top_up_matches_page_top_up(tmp_path, blender_exe, bake_module, device, slivers):
    from atlas_scenes import PAGE, spike_scene

    scene, regions, suns = spike_scene(slivers)
    k = 32 if device == 'CPU' else 128
    bake = atlas_bake(tmp_path, blender_exe, scene, [(PAGE, PAGE)], suns,
                      [ATLAS_RUN, page_run('P0', 0), page_run('P1', 1)], device=device, samples=72 * k,
                      texel_samples=72 * k, vertex_rays=16)
    seconds = {name: bake.report[name]['seconds'] for name in ('A', 'P0', 'P1')}
    tiers = bake.tiers('A')
    print(f'ATLAS_RUN_TIME {device} {"R5" if slivers else "R1-R4"} {json.dumps(seconds)} tiers {tiers} '
          f'fallbacks {bake.fallbacks("A")}')
    assert tiers
    if not slivers:
        assert not bake.fallbacks('A'), bake.log('A')
    selected = bake.topped_up('A')
    a, p0, p1 = (bake.out(name)['page_0'] for name in ('A', 'P0', 'P1'))
    # the regions' (a) statistic, for diagnosis only
    with np.load(tmp_path / 'scene.npz') as data:
        owner = bake_module.subtexel_owner({key: data[key] for key in data.files}, 0, PAGE, PAGE)
    texel_owner = owner.reshape(PAGE, 3, PAGE, 3).max(axis=(1, 3))[::-1]
    names = np.array(regions)
    by_region = {}
    for region in sorted(set(regions)):
        mine = selected & (texel_owner >= 0) & (names[np.maximum(texel_owner, 0)] == region)
        if mine.sum() > 2:
            diff = (a - p0)[mine]
            by_region[region] = {'n': int(mine.sum()), 'mean': diff.mean(axis=0).tolist(),
                                 'bias_z': (np.abs(diff.mean(axis=0)) / (diff.std(axis=0, ddof=1) / np.sqrt(len(diff)) + 1e-30)).tolist()}
    report = json.loads((tmp_path / 'atlas.json').read_text())
    report['regions_a'] = by_region
    (tmp_path / 'atlas.json').write_text(json.dumps(report))
    print(f'ATLAS_REGIONS {device} {json.dumps(by_region)}')
    assert_equivalent(f'{device} {"R5" if slivers else "R1-R4"}', a, p0, p1, selected)
    interior = bake.out('A')['written_0'] & ~selected
    if device == 'CPU':
        np.testing.assert_array_equal(a[interior], p0[interior])
    else:
        # (two GPU bakes of one seed differ by about 1e-6, the spike's noise floor, before any atlas is involved)
        np.testing.assert_allclose(a[interior], p0[interior], rtol=1e-5)


@pytest.mark.blender
def test_back_only_material_keeps_only_the_translucent_closure(tmp_path, blender_exe):
    texture = np.zeros((16, 16, 4), np.uint8)
    texture[..., 0] = np.linspace(40, 255, 16)[None, :]
    texture[..., 1] = np.linspace(40, 255, 16)[:, None]
    texture[..., 2:] = 128, 255
    scene = build([dict(quad=floor(0, 8, 0, 8, lm=chart32(23, (16, 16), 9.3)), ignore_normals=True, texture=0)])
    bake = atlas_bake(tmp_path, blender_exe, scene, [(32, 32)], [soft_sun()], [ATLAS_RUN], textures=[texture])
    assert bake.tiers('A') and not bake.fallbacks('A'), bake.log('A')
    materials = [m for call in bake.tier_calls('A') for info in call['objects'].values() for m in info['materials']]
    back = [m for m in materials if 'BSDF_TRANSLUCENT' in m['types']]
    front = [m for m in materials if 'BSDF_DIFFUSE' in m['types']]
    assert back and front
    assert all('BSDF_DIFFUSE' not in m['types'] for m in back)
    assert all('BSDF_TRANSLUCENT' not in m['types'] for m in front)
    # both halves take their colour from the same texture
    assert all('TEX_IMAGE' in m['colours']['BSDF_TRANSLUCENT'] for m in back)
    assert all('TEX_IMAGE' in m['colours']['BSDF_DIFFUSE'] for m in front)


def cutout_texture():
    texture = np.zeros((16, 16, 4), np.uint8)
    texture[..., 0] = np.linspace(40, 255, 16)[None, :]
    texture[..., 1] = np.linspace(40, 255, 16)[:, None]
    texture[..., 2:] = 128, 255
    # the hole is one band of rows (the diffuse pass divides the colour out, so only the cut-out shows the texture), which
    # the rotated chart's edge texels cross, and a mirrored v would put it on the opposite side
    texture[:4, :, 3] = 0
    return texture


@pytest.mark.blender
def test_atlas_top_up_reads_textures_and_cutouts(tmp_path, blender_exe):
    scene = build([dict(quad=floor(0, 8, 0, 8, lm=chart32(23, (16, 16), 9.3)), texture=0, cutout=True)])
    bake = atlas_bake(tmp_path, blender_exe, scene, [(32, 32)], [soft_sun()],
                      [ATLAS_RUN, page_run('P0', 0), page_run('P1', 1)], textures=[cutout_texture()])
    assert bake.tiers('A') and not bake.fallbacks('A'), bake.log('A')
    a, p0 = bake.out('A')['page_0'], bake.out('P0')['page_0']
    selected = bake.topped_up('A')
    hole = selected & (p0.sum(axis=2) < 1e-6)
    assert hole.sum() > 4, 'the cut-out must cross the topped-up texels'
    np.testing.assert_allclose(a[hole], p0[hole], atol=1e-6)
    compare_runs('cutout', bake, selected & ~hole)


@pytest.mark.blender
def test_atlas_top_up_keeps_tint(tmp_path, blender_exe):
    pane = dict(quad=floor(0, 8, 0, 8, z=1, lm=chart32(-17, (24, 16), 4.7)), casts=False, tint=(0.5, 0.8, 1))
    ground = dict(quad=floor(0, 8, 0, 8, lm=chart32(23, (8, 16), 4.7)))
    scene = build([ground, pane])
    bake = atlas_bake(tmp_path, blender_exe, scene, [(32, 32)], [soft_sun((0.5, 0.1, 1))],
                      [ATLAS_RUN, page_run('P0', 0), page_run('P1', 1)])
    assert bake.tiers('A') and not bake.fallbacks('A'), bake.log('A')
    # (the bake ray itself is no shadow ray, so the tint cannot show in the pane copy's own value: its material is checked
    # for the shadow-ray switch, so the copy is built as its source is)
    materials = [m for call in bake.tier_calls('A') for info in call['objects'].values() for m in info['materials']]
    assert any('LIGHT_PATH' in m['types'] and 'BSDF_TRANSPARENT' in m['types'] for m in materials)
    selected = bake.topped_up('A')
    columns = np.arange(32)[None, :] >= 16
    compare_runs('tint floor', bake, selected & ~columns)
    compare_runs('tint pane', bake, selected & columns)


@pytest.mark.blender
def test_atlas_top_up_in_a_lamp_batch(tmp_path, blender_exe):
    wall = dict(quad=wall_x(7, -2, 8, 0, 3))
    ground = dict(quad=floor(0, 8, 0, 8, lm=chart32(23, (16, 16), 9.3)))
    lamp_job = {'lamps_only': True, 'lamp_batch': 0, 'suns': [], 'ambient': {}, 'vertex_pass': 'skip'}
    bake = atlas_bake(tmp_path, blender_exe, build([ground], [wall]), [(32, 32)], [],
                      [{**ATLAS_RUN, 'job': {**ATLAS_RUN['job'], **lamp_job}},
                       page_run('P0', 0, **lamp_job), page_run('P1', 1, **lamp_job)],
                      lights=[lamp_light((3, 4, 1.5))])
    assert bake.tiers('A') and not bake.fallbacks('A'), bake.log('A')
    selected = bake.topped_up('A')
    assert bake.out('P0')['page_0'][selected].mean() > 0.003, 'the lamp must light the page'
    compare_runs('lamp batch', bake)


@pytest.mark.blender
def test_atlas_objects_keep_their_light_links(tmp_path, blender_exe):
    lit = dict(quad=floor(0, 8, 0, 8, lm=chart32(23, (8, 16), 4.7)))
    unlit = dict(quad=floor(10, 18, 0, 8, lm=chart32(-11, (24, 16), 4.7)), cls=1)
    bake = atlas_bake(tmp_path, blender_exe, build([lit, unlit]), [(32, 32)], [soft_sun()],
                      [ATLAS_RUN, page_run('P0')])
    assert bake.tiers('A') and not bake.fallbacks('A'), bake.log('A')
    a, p0 = bake.out('A')['page_0'], bake.out('P0')['page_0']
    selected = bake.topped_up('A')
    columns = np.arange(32)[None, :] >= 16
    assert selected[:, :16].any() and selected[:, 16:].any()
    # the sun lights class 0 only, so the class 1 copies stay dark and the class 0 copies are lit
    assert a[selected & columns].max() < 1e-3
    assert p0[selected & columns].max() < 1e-3
    np.testing.assert_allclose(a[selected & ~columns].mean(), p0[selected & ~columns].mean(), rtol=0.03)
    assert a[selected & ~columns].mean() > 0.3


@pytest.mark.blender
def test_atlas_objects_keep_lamp_links(tmp_path, blender_exe):
    flagged = dict(quad=floor(0, 8, 0, 8, lm=chart32(23, (8, 16), 4.7)), ignore_normals=True)
    plain = dict(quad=floor(10, 18, 0, 8, lm=chart32(-11, (24, 16), 4.7)))
    bake = atlas_bake(tmp_path, blender_exe, build([flagged, plain]), [(32, 32)], [],
                      [ATLAS_RUN, page_run('P0')], lights=[lamp_light((14, 4, 1.5))])
    assert bake.tiers('A') and not bake.fallbacks('A'), bake.log('A')
    objects = [(name, info) for call in bake.tier_calls('A') for name, info in call['objects'].items()]
    assert any(info['flagged'] for _, info in objects) and any(not info['flagged'] for _, info in objects)
    for name, info in objects:
        assert ('lamp_receivers' in info['collections']) == (not info['flagged']), (name, info)
    a, p0 = bake.out('A')['page_0'], bake.out('P0')['page_0']
    selected = bake.topped_up('A')
    columns = np.arange(32)[None, :] >= 16
    # the flagged receiver takes no lamp light either way, the plain one does
    assert a[selected & ~columns].max() < 1e-6 and p0[selected & ~columns].max() < 1e-6
    assert p0[selected & columns].mean() > 0.003
    np.testing.assert_allclose(a[selected & columns].mean(), p0[selected & columns].mean(), rtol=0.05)


@pytest.mark.blender
def test_atlas_object_casts_no_shadow(tmp_path, blender_exe):
    # the roof owns whole texels, so only the ground tops up; its shadow must reach the ground's copies and no copy may cast one
    whole = chart32(0, (28, 28), 3)
    ground = dict(quad=floor(0, 8, 0, 8, lm=chart32(23, (13, 13), 8.5)), casts=False)
    roof = dict(quad=floor(1, 6, 1, 6, z=1, lm=whole))
    bake = atlas_bake(tmp_path, blender_exe, build([ground, roof]), [(32, 32)], [soft_sun((0.1, 0.05, 1))],
                      [ATLAS_RUN, page_run('P0', 0), page_run('P1', 1)])
    assert bake.tiers('A') and not bake.fallbacks('A'), bake.log('A')
    for call in bake.tier_calls('A'):
        assert all(not info['visible']['visible_shadow'] for info in call['objects'].values())
    compare_runs('shadow', bake)


@pytest.mark.blender
@pytest.mark.parametrize('radiance', [0, 4])
def test_flagged_edges_keep_their_correction(tmp_path, blender_exe, radiance):
    left = ((0, 0), (2 / 3, 0), (2 / 3, 1), (0, 1))
    right = ((2 / 3, 0), (1, 0), (1, 1), (2 / 3, 1))
    receivers = [dict(quad=floor(0, 4, 0, 4, lm=left), ignore_normals=True),
                 dict(quad=floor(10, 14, 0, 4, lm=right))]
    panels = [dict(quad=floor(-100, 104, -100, 104, z=-1), emission=(radiance,) * 3, casts=False)] if radiance else []
    bake = atlas_bake(tmp_path, blender_exe, build(receivers, panels), [(1, 1)], [sun(strength=np.pi)],
                      [ATLAS_RUN], samples=1024, texel_samples=2048)
    assert bake.tiers('A') and not bake.fallbacks('A'), bake.log('A')
    out = bake.out('A')
    # six flagged samples take the panel, three unflagged samples take the sun; the vertex page keeps half the panel
    np.testing.assert_allclose(out['page_0'], 2 / 3 * radiance + 1 / 3, rtol=0.05)
    np.testing.assert_allclose(out['diffuse_0'], 1 / 3 * radiance + 1 / 3, rtol=0.05)


@pytest.mark.blender
def test_glow_run_atlas_has_one_copy_per_slot(tmp_path, blender_exe):
    from lightmapper.blender_job import glow_box

    inset = ((1.5 / 16, 1.5 / 16), (13.2 / 16, 1.5 / 16), (13.2 / 16, 13.2 / 16), (1.5 / 16, 13.2 / 16))
    scene = build([dict(quad=floor(-8, 8, -8, 8, z=1, lm=inset), ignore_normals=True),
                   dict(quad=floor(-4, 2, -4, 2, lm=inset), page=1)])
    glow = np.zeros((16, 16, 3), np.float32)
    glow[:8] = 4
    box = glow_box(glow)
    x0, y0, x1, y1 = box
    np.savez(tmp_path / 'glow_pages.npz', page_0=glow[y0:y1, x0:x1], box_0=np.array([x0, y0, x1, y1, 16, 16], np.int32))
    glow_job = {'glow': True, 'glow_pages': 'glow_pages.npz', 'suns': [], 'ambient': {}, 'lights': [], 'bounces': 0,
                'vertex_pass': 'skip'}
    bake = atlas_bake(tmp_path, blender_exe, scene, [(16, 16), (16, 16)], [],
                      [{**ATLAS_RUN, 'job': {**ATLAS_RUN['job'], **glow_job}},
                       page_run('P0', 0, **glow_job), page_run('P1', 1, **glow_job)],
                      ignore_normals_mode='enhanced', samples=72 * 32, texel_samples=72 * 32)
    assert bake.tiers('A') and not bake.fallbacks('A'), bake.log('A')
    for call, (n, slots, side, baked) in zip(bake.tier_calls('A'), bake.tiers('A')):
        assert sum(info['triangles'] for info in call['objects'].values()) == slots
        assert all('BSDF_TRANSLUCENT' not in m['types'] for info in call['objects'].values() for m in info['materials'])
    # (every slot is a front copy: no back copies in Enhanced mode)
    assert bake.report['A']['roles'] and all(r[1] == r[2] == 0 for r in bake.report['A']['roles'])
    selected = bake.topped_up('A', 1)
    assert selected.sum() >= 32 and bake.out('P0')['page_1'][selected].mean() > 0.05
    assert_equivalent('glow', bake.out('A')['page_1'], bake.out('P0')['page_1'], bake.out('P1')['page_1'], selected)


@pytest.mark.blender
def test_unowned_subtexel_tier_bakes_through_the_mask(tmp_path, blender_exe):
    bake = atlas_bake(tmp_path, blender_exe, edge_scene(), [(32, 32)], [soft_sun()],
                      [{**ATLAS_RUN, 'unowned': True}, page_run('P0')])
    assert not bake.tiers('A')
    assert len(bake.fallbacks('A')) == len(bake.tier_calls('A')) >= 1, bake.log('A')
    for call in bake.tier_calls('A'):
        assert call['images']['page_0'] == [96, 96] and call['images']['atlas'] == [1, 1]
        assert all(name.startswith('recv') for name in call['selected'])
    # (the same seeds and samples through the same masks as the page path)
    np.testing.assert_array_equal(bake.out('A')['page_0'], bake.out('P0')['page_0'])


@pytest.mark.blender
def test_unwritten_atlas_slot_bakes_the_tier_through_the_mask(tmp_path, blender_exe):
    bake = atlas_bake(tmp_path, blender_exe, edge_scene(), [(32, 32)], [soft_sun()],
                      [{**ATLAS_RUN, 'unwritten': True}, page_run('P0')])
    assert len(bake.fallbacks('A')) == len(bake.tiers('A')) >= 1, bake.log('A')
    assert all('unwritten' in reason for _, reason in bake.fallbacks('A'))
    np.testing.assert_array_equal(bake.out('A')['page_0'], bake.out('P0')['page_0'])


@pytest.mark.blender
def test_atlas_objects_are_removed_after_each_tier(tmp_path, blender_exe):
    bake = atlas_bake(tmp_path, blender_exe, edge_scene(ignore_normals=True), [(32, 32)], [soft_sun()], [ATLAS_RUN])
    run = bake.report['A']
    assert len(bake.tiers('A')) >= 2, bake.log('A')
    before = run['before']
    assert run['after'] == before
    for call in bake.tier_calls('A'):
        assert call['counts']['objects'] - before['objects'] == len(call['selected'])
        assert call['counts']['meshes'] - before['meshes'] == len(call['selected'])
        assert call['counts']['materials'] - before['materials'] == sum(
            len(info['materials']) for info in call['objects'].values())
