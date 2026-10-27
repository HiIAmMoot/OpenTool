"""Real Blender bakes of tiny scenes: orientation, shadows, light linking, emission, glass, cut-outs, bounce."""

import os
import json
from dataclasses import replace
from pathlib import Path
import subprocess

import numpy as np
import pytest

from lightmapper.blender_job import BAKE_SCRIPT, BlenderError, Scene, run_blender, write_job
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
    cols = {k: [] for k in ('p', 'lm', 'tex', 'nrm', 'tris', 'page', 'cls', 'casts', 'emit', 'texture', 'cut', 'tint')}
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
    )


def sun(direction=UP, classes=(0,), strength=3.0):
    direction = np.float64(direction)
    return Sun(direction / np.linalg.norm(direction), 0.01, np.float64([1, 1, 1]), strength, classes)


def bake(tmp_path, receivers, suns, pages=((16, 16),), occluders=(), textures=None):
    s = Settings(samples=32, bounces=1, w_sun=1.0, w_emit=1.0, w_amb=1.0, albedo_scale=1.0,
                 albedo_fallback=0.5)
    write_job(tmp_path, build(receivers, occluders), list(pages), suns, {}, s, 'all', textures=textures)
    return run_blender(tmp_path, s)


@pytest.mark.blender
def test_blender_reports_its_stage_seconds(tmp_path):
    result = bake(tmp_path, [full_quad()], [sun()])
    assert set(result.stages) == {'start', 'scene', 'pages', 'vertex'}
    assert all(value >= 0 for value in result.stages.values())


def full_quad(**extra):
    return dict(quad=floor(0, 4, 0, 4), **extra)


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
    assert result.device in ('OPTIX', 'CUDA', 'HIP', 'CPU')


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
def test_a_mask_is_full_size_only_while_its_page_tops_up(tmp_path, blender_exe):
    import json

    edge = ((1.5 / 16, 1.5 / 16), (9.5 / 16, 1.5 / 16), (9.5 / 16, 9.5 / 16), (1.5 / 16, 9.5 / 16))
    pages = [(16, 16)] * 3
    receivers = [dict(quad=floor(0, 4, 0, 4, z=n, lm=edge), page=n, casts=False) for n in range(3)]
    s = Settings(samples=32, bounces=1, vertex_rays=1)
    write_job(tmp_path, build(receivers), pages, [sun()], {}, s)
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
    # the edge quad needs top-up tiers, and only those reads see a full-size mask
    assert full_reads >= 3
    # (two reads per page, the first bake and the emit pass, see a one-texel mask)
    assert full_reads == len(sizes['during']) - 3 * 2
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
