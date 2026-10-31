"""Exercise orchestration on synthetic tags and scenes."""

from dataclasses import replace
import importlib.util
import json
import struct
import subprocess
import sys
import time

import numpy as np
import pytest

from conftest import LIGHTMAPPER_BLENDER
from lightmapper.blender_job import BakeResult
from lightmapper.formats.vertices import ENV, LM, compress, pack_uncompressed, unpack_uncompressed
from lightmapper.settings import Settings
from lightmapper.tagio.tagfile import new_element, new_tag, read_tag, resolve_reference_path, write_tag
from lightmapper.validate import validate_bsp
from lightmapper import gpu, cpu_vertex_job as vertex_cpu, vertex_gpu, blender_job, memory
from test_blender_job import fake


NV = gpu.Adapter('NVIDIA GeForce RTX 4060 Laptop GPU', gpu.NVIDIA, (1,), 'vulkan', True)
GPU_CHOICE = gpu.choose('auto', [NV])


@pytest.mark.parametrize('module', ['lightmapper.pipeline', 'opentool.runner'])
def test_parent_import_does_not_load_compiled_kernels(module):
    code = (f'import importlib, sys; importlib.import_module({module!r}); '
            'assert "lightmapper.vertex_cpu" not in sys.modules')
    done = subprocess.run([sys.executable, '-c', code], capture_output=True, timeout=60)
    assert done.returncode == 0, done.stderr.decode(errors='replace')


def child(parent, name):
    return new_element(next(f.layout for f in parent.layout.fields if f.key == name))


def api():
    assert importlib.util.find_spec('lightmapper.pipeline') is not None, 'bake pipeline is missing'
    from lightmapper import pipeline
    return pipeline


def put(tags, group, path, tag):
    target = resolve_reference_path(tags, group, path)
    target.parent.mkdir(parents=True, exist_ok=True)
    write_tag(tag, target)


@pytest.fixture
def tiny(tmp_path, monkeypatch):
    tags = tmp_path / 'tags'
    scenario, bsp, shader = new_tag('scnr'), new_tag('sbsp'), new_tag('senv')
    ref = child(scenario.root, 'structure_bsps')
    ref.set_reference('structure_bsp', 'sbsp', 'room')
    scenario.root.set_block('structure_bsps', [ref])
    page, final = child(bsp.root, 'lightmaps'), child(bsp.root, 'lightmaps')
    final.set('bitmap', -1)
    material = child(page, 'materials')
    material.set_reference('shader', 'senv', 'wall')
    env, lm = np.zeros(6, ENV), np.zeros(6, LM)
    env['position'] = [[0, 0, 0], [4, 0, 0], [0, 4, 0], [10, 0, 0], [14, 0, 0], [10, 4, 0]]
    env['normal'] = [0, 0, 1]
    env['tangent'] = [1, 0, 0]
    env['binormal'] = [0, 1, 0]
    env['texcoord'] = [[0, 0], [1, 0], [0, 1]] * 2
    for prefix in ('rendered_vertices', 'lightmap_vertices'):
        material.set(prefix + '.vertex_count', 6)
    material.set('surface_count', 2)
    material.set_data('uncompressed_vertices', pack_uncompressed(env, lm))
    material.set_data('compressed_vertices', compress(env, lm))
    page.set_block('materials', [material])
    bsp.root.set_block('lightmaps', [page, final])
    surfaces = [child(bsp.root, 'surfaces') for _ in range(2)]
    surfaces[0].body[:] = struct.pack('>3H', 0, 1, 2)
    surfaces[1].body[:] = struct.pack('>3H', 3, 4, 5)
    bsp.root.set_block('surfaces', surfaces)
    put(tags, 'scnr', 'level', scenario)
    put(tags, 'sbsp', 'room', bsp)
    put(tags, 'senv', 'wall', shader)
    pipeline = api()

    def fake_blender(work, s):
        job = json.loads((work / 'job.json').read_text())
        with np.load(work / 'scene.npz') as scene:
            vectors = scene['query_normal'] * (scene['query_pos'][:, 0:1] / 100 + .25)
        pages = [np.zeros((h, w, 3), np.float32) for w, h in job['pages']]
        return BakeResult(pages,
                          vectors, np.ones(len(vectors), np.float32), 'test CPU',
                          [np.ones((h, w), bool) for w, h in job['pages']],
                          stages={'start': 0, 'scene': 2, 'pages': 3, 'vertex': 0}, diffuse_pages=pages)

    monkeypatch.setattr(pipeline, 'run_blender', fake_blender)

    def compiled(work, pages, timeout):
        job = json.loads((work / 'job.json').read_text())
        with np.load(work / 'scene.npz') as scene:
            vectors = scene['query_normal'] * (scene['query_pos'][:, 0:1] / 100 + .25)
        return vertex_gpu.VertexPassResult(vectors, np.ones(len(vectors), np.float32),
            None, None, len(vectors) * job['cpu_vertex_rays']), cpu_info()

    monkeypatch.setattr(vertex_cpu, 'run_in_child', compiled)
    return tags, Settings(texels_per_unit=.25, page_max=64, denoise=False), tmp_path / 'work'


def test_memory_guard_warns_and_starts_blender(tiny, monkeypatch, caplog):
    tags, settings, work = tiny
    calls = []
    blender = api().run_blender
    monkeypatch.setattr(memory, 'free_physical_bytes', lambda: 1)

    def record(*args):
        calls.append(args)
        return blender(*args)

    monkeypatch.setattr(api(), 'run_blender', record)
    api().bake(tags, 'level', 'room', settings, work=work)
    assert len(calls) == 1 and (work / 'job.json').exists()
    summary = json.loads((work / 'summary.json').read_text())
    message = memory.out_of_memory_message(memory.SceneCounts(2, 6), summary['sizes'], settings.page_max, 1)
    assert message in summary['warnings']
    assert ('WARNING', message) in [(r.levelname, r.getMessage()) for r in caplog.records]
    assert 'needs about 0.0 GB' in message and 'has 0.0 GB free' in message


def test_memory_guard_sees_the_scenery(tiny, monkeypatch):
    from test_scenery import write_model, write_scenery

    tags, settings, work = tiny
    write_model(tags, 'model', [[(0, [(0, 0, 0), (1, 0, 0), (0, 1, 0)], [0, 1, 2])]],
                shaders=(('senv', 'wall'),))
    write_scenery(tags, 'prop', 'model')
    scenario = read_tag(tags / 'level.scenario')
    palette = child(scenario.root, 'scenery_palette')
    palette.set_reference('name', 'scen', 'prop')
    scenario.root.set_block('scenery_palette', [palette])
    placement = child(scenario.root, 'scenery')
    placement.set('bsp_indices', 1)
    scenario.root.set_block('scenery', [placement])
    put(tags, 'scnr', 'level', scenario)
    seen = []
    check = memory.check_bake_memory
    monkeypatch.setattr(memory, 'free_physical_bytes', lambda: 64 * memory.GB)

    def spy(counts, pages, page_max):
        seen.append((counts, pages, page_max))
        return check(counts, pages, page_max)

    monkeypatch.setattr(memory, 'check_bake_memory', spy)
    api().bake(tags, 'level', 'room', settings, work=work, dry_run=True)
    assert len(seen) == 1
    seen_counts, pages, page_max = seen[0]
    assert seen_counts.triangles == 2 + 1
    assert seen_counts.vertices == 6 + 3
    job = json.loads((work / 'job.json').read_text())
    assert pages == [tuple(page) for page in job['pages']]
    assert page_max == settings.page_max


def place_scenery(tags):
    from test_scenery import write_model, write_scenery

    write_model(tags, 'model', [[(0, [(0, 0, 0), (1, 0, 0), (0, 1, 0)], [0, 1, 2])]],
                shaders=(('senv', 'wall'),))
    write_scenery(tags, 'prop', 'model')
    scenario = read_tag(tags / 'level.scenario')
    palette = child(scenario.root, 'scenery_palette')
    palette.set_reference('name', 'scen', 'prop')
    scenario.root.set_block('scenery_palette', [palette])
    placement = child(scenario.root, 'scenery')
    placement.set('bsp_indices', 1)
    scenario.root.set_block('scenery', [placement])
    put(tags, 'scnr', 'level', scenario)


def test_estimate_matches_what_the_bake_sees(tiny, monkeypatch, tmp_path):
    tags, settings, work = tiny
    place_scenery(tags)
    monkeypatch.setattr(memory, 'free_physical_bytes', lambda: 64 * memory.GB)
    estimate = api().estimate_scene(tags, 'level', 'room', settings, tmp_path / 'charts.npz')
    seen = []
    check = memory.check_bake_memory

    def spy(counts, pages, page_max):
        seen.append((counts, pages))
        return check(counts, pages, page_max)

    monkeypatch.setattr(memory, 'check_bake_memory', spy)
    api().bake(tags, 'level', 'room', settings, work=work, charts=tmp_path / 'charts.npz')
    assert (estimate.counts, estimate.pages) == seen[0]
    assert estimate.counts == memory.SceneCounts(3, 9)


def test_estimate_leaves_a_chart_cache_the_bake_reuses(tiny, monkeypatch, tmp_path):
    tags, settings, work = tiny
    api().estimate_scene(tags, 'level', 'room', settings, tmp_path / 'charts.npz')

    def refuse(*args, **kwargs):
        raise AssertionError('charted again')

    monkeypatch.setattr(api(), 'chart_and_pack', refuse)
    api().bake(tags, 'level', 'room', settings, work=work, charts=tmp_path / 'charts.npz')


def test_estimate_starts_no_blender(tiny, monkeypatch, tmp_path):
    tags, settings, work = tiny

    def refuse(*args, **kwargs):
        raise AssertionError('estimate reached the bake')

    for name in ('run_blender', 'write_job'):
        monkeypatch.setattr(api(), name, refuse)
    monkeypatch.setattr(gpu, 'probe_choice', refuse)
    estimate = api().estimate_scene(tags, 'level', 'room', settings, tmp_path / 'charts.npz')
    assert estimate.pages


def test_estimate_raises_cap_error_over_max_pages(tiny, tmp_path):
    from lightmapper.charts import CapError

    tags, settings, work = tiny
    settings = replace(settings, texels_per_unit=1, page_max=32, max_pages=1)
    with pytest.raises(CapError, match='pages needed, over max_pages|packing exceeds max_pages'):
        api().estimate_scene(tags, 'level', 'room', settings, tmp_path / 'charts.npz')


def test_estimate_refuses_over_the_surface_cap(tiny, monkeypatch, tmp_path):
    tags, settings, work = tiny
    monkeypatch.setattr(api(), 'MAX_BSP_SURFACES', 1)
    with pytest.raises(ValueError, match='exceed stock cap'):
        api().estimate_scene(tags, 'level', 'room', settings, tmp_path / 'charts.npz')


def test_dry_run_warns_instead_of_refusing(tiny, monkeypatch, caplog):
    tags, settings, work = tiny
    settings = replace(settings, texels_per_unit=2)
    before = {p: p.read_bytes() for p in tags.rglob('*') if p.is_file()}
    monkeypatch.setattr(memory, 'free_physical_bytes', lambda: 1)
    out = api().bake(tags, 'level', 'room', settings, work=work, dry_run=True)
    summary = json.loads((work / 'summary.json').read_text())
    counts = memory.SceneCounts(2, 6)
    pages = summary['sizes']
    message = memory.out_of_memory_message(counts, pages, settings.page_max, 1)
    assert message in summary['warnings']
    assert ('WARNING', message) in [(r.levelname, r.getMessage()) for r in caplog.records]
    assert summary['memory'] == {'estimate_gb': round(memory.bake_bytes(counts, pages) / memory.GB, 2),
                                 'free_gb': round(1 / memory.GB, 2)}
    assert out.work == work
    assert {p: p.read_bytes() for p in tags.rglob('*') if p.is_file()} == before


@pytest.mark.parametrize('free', [64 * memory.GB, None])
def test_summary_records_the_memory_check(tiny, monkeypatch, free):
    tags, settings, work = tiny
    settings = replace(settings, texels_per_unit=2)
    monkeypatch.setattr(memory, 'free_physical_bytes', lambda: free)
    api().bake(tags, 'level', 'room', settings, work=work)
    summary = json.loads((work / 'summary.json').read_text())
    estimate = memory.bake_bytes(memory.SceneCounts(2, 6), summary['sizes'])
    assert summary['memory'] == {'estimate_gb': round(estimate / memory.GB, 2),
                                 'free_gb': 64.0 if free is not None else None}


def cpu_info():
    return {'compile': .125, 'trace': .25, 'threads': 2, 'threading_layer': 'workqueue'}


def test_pipeline_outputs(tiny):
    tags, settings, work = tiny
    out = api().bake(tags, 'level', 'room', settings, out_bsp='copy', out_scenario='copy_level', work=work)
    assert validate_bsp(out.bsp_tag, out.bitmap_tag, strict_compression=True) == []
    np.testing.assert_array_equal(np.sort(out.new_to_input), [0, 1])
    for page in out.bsp_tag.root.block('lightmaps')[:-1]:
        for m in page.block('materials'):
            env, lm = unpack_uncompressed(m.data('uncompressed_vertices'),
                                        m.get('rendered_vertices.vertex_count'), m.get('lightmap_vertices.vertex_count'))
            expected = np.zeros((len(lm), 3), np.float32)
            expected[:, 2] = env['position'][:, 0] / 100 + .25
            np.testing.assert_array_equal(lm['incident_radiosity'], expected)
    assert read_tag(tags / 'copy_level.scenario').root.block('structure_bsps')[0].reference('structure_bsp') == ('sbsp', 'copy')
    assert read_tag(tags / 'copy.scenario_structure_bsp').root.reference('lightmaps_bitmap') == ('bitm', 'copy')
    assert (tags / 'copy.bitmap').is_file()
    np.testing.assert_array_equal(np.load(work / 'new_to_input.npy'), out.new_to_input)
    with np.load(work / 'float_pages.npz', allow_pickle=False) as data:
        assert len(data.files) == len(out.float_pages)
        assert all(data[k].dtype == np.float32 for k in data.files)
    summary = json.loads((work / 'summary.json').read_text())
    assert summary['device'] == 'test CPU'
    assert summary['pages'] == len(out.float_pages)
    assert summary['seconds'] and summary['texels_per_unit'] == .25
    assert {'blender_start', 'blender_scene', 'page_bake', 'denoise'} <= summary['seconds'].keys()
    assert summary['settings']['stitch_seams'] is True
    assert summary['settings']['respect_normal_seams'] is True
    assert summary['settings']['texel_samples'] is None
    assert summary['texel_samples'] == settings.resolved_texel_samples()
    assert summary['devices']['denoise'] == 'off'


def test_summary_splits_the_blender_step(tiny, fake, monkeypatch):
    tags, settings, work = tiny
    monkeypatch.setattr(api(), 'run_blender', blender_job.run_blender)
    monkeypatch.setenv('FAKE_BLENDER_MODE', 'stage_timings')
    api().bake(tags, 'level', 'room', replace(settings, blender=fake), out_bsp='timed', work=work)
    seconds = json.loads((work / 'summary.json').read_text())['seconds']
    assert 1.5 <= seconds['blender_start'] < 6  # the fake's sleep plus however slow the interpreter starts under load
    assert (seconds['blender_scene'], seconds['page_bake'], seconds['blender_vertex']) == (2, 3, 0)
    assert 'denoise' in seconds and seconds['denoise'] == 0 and seconds['finish'] >= 0


def test_finish_excludes_the_denoise_worker_start_and_pages(tiny, monkeypatch):
    tags, settings, work = tiny

    def finish_seconds(delay):
        class Worker:
            device = 'CPU'
            warnings = []
            seconds = delay

            def close(self): pass

        def start(cls, device, timeout, strict):
            time.sleep(delay)
            return Worker()

        monkeypatch.setattr(api().DenoiseWorker, 'start', classmethod(start))
        api().bake(tags, 'level', 'room', replace(settings, denoise=True), work=work, dry_run=True)
        return json.loads((work / 'summary.json').read_text())['seconds']

    finish_seconds(0)  # the first bake pays one-time costs that would hide the start-up
    baseline = finish_seconds(0)
    slow = finish_seconds(2.0)
    assert slow['denoise'] == 2.0
    # the start-up sits inside the finish stage's wall time, so it must be taken out of finish
    assert 0 <= slow['finish'] < baseline['finish'] + 1.0


def test_denoise_startup_fallback_is_logged_and_summarized(tiny, monkeypatch, caplog):
    from lightmapper.pages import DenoiseWorker

    tags, settings, work = tiny
    real_start = DenoiseWorker.start.__func__
    monkeypatch.setattr(DenoiseWorker, 'start', classmethod(
        lambda cls, device, timeout, strict: real_start(cls, 'NOPE', timeout, strict=False)))
    out = api().bake(tags, 'level', 'room', replace(settings, denoise=True), work=work, dry_run=True)
    summary = json.loads((work / 'summary.json').read_text())
    assert summary['devices']['denoise'] == 'CPU'
    assert any('OIDN NOPE failed' in warning for warning in summary['warnings'])
    assert 'OIDN NOPE failed' in caplog.text
    assert out.work == work


def test_denoise_mid_page_fallback_is_logged_and_summarized(tiny, monkeypatch, caplog):
    tags, settings, work = tiny
    module = api()
    warning = 'OIDN CUDA failed (mid-page filter error); denoising on the CPU'

    class Worker:
        device = 'CUDA'
        warnings = []
        seconds = 0.0

        def denoise(self, page):
            self.device = 'CPU'
            self.warnings.append(warning)
            module.logger.warning('%s', warning)
            return page

        def close(self): pass

    worker = Worker()
    monkeypatch.setattr(module.DenoiseWorker, 'start', classmethod(
        lambda cls, device, timeout, strict: worker))
    blender = module.run_blender

    def lit(work, s):
        result = blender(work, s)
        result.pages = [np.full_like(page, 0.5) for page in result.pages]
        return result

    monkeypatch.setattr(module, 'run_blender', lit)
    module.bake(tags, 'level', 'room', replace(settings, denoise=True), work=work, dry_run=True)
    summary = json.loads((work / 'summary.json').read_text())
    assert summary['devices']['denoise'] == 'CPU'
    assert warning in summary['warnings']
    assert warning in caplog.text


def test_required_amd_starts_hyp_worker_non_strict(tiny, monkeypatch):
    tags, settings, work = tiny
    choice = gpu.choose('amd', None)
    monkeypatch.setattr(gpu, 'probe_choice', lambda setting: choice)
    seen = []

    class Worker:
        device = 'CPU'
        warnings = []
        seconds = 0.0
        def close(self): pass

    monkeypatch.setattr(api().DenoiseWorker, 'start', classmethod(
        lambda cls, device, timeout, strict: seen.append((device, strict)) or Worker()))
    api().bake(tags, 'level', 'room', replace(settings, denoise=True, device='amd'), work=work, dry_run=True)
    assert seen == [('HIP', False)]


@pytest.fixture
def gpu_bake(tiny, fake, monkeypatch):
    tags, settings, work = tiny
    monkeypatch.setattr(api(), 'run_blender', blender_job.run_blender)
    monkeypatch.setattr(gpu, 'probe_choice', lambda setting: GPU_CHOICE)
    return tags, replace(settings, blender=fake), work


def test_gpu_choice_skips_the_blender_vertex_pass(gpu_bake, monkeypatch):
    from test_blender_job import page
    from test_blender_bake import sun
    tags, settings, work = gpu_bake
    bsp = read_tag(tags / 'room.scenario_structure_bsp')
    material = bsp.root.block('lightmaps')[0].block('materials')[0]
    env, lm = unpack_uncompressed(material.data('uncompressed_vertices'), 6, 6)
    # a shared fold gives the real boundary builder non-empty edge probes
    env['position'] = [[0, 0, 0], [4, 0, 0], [0, 4, 0], [0, 0, 0], [0, 4, 0], [0, 4, 4]]
    material.set_data('uncompressed_vertices', pack_uncompressed(env, lm))
    material.set_data('compressed_vertices', compress(env, lm))
    put(tags, 'sbsp', 'room', bsp)
    lighting = api().gather_lighting
    monkeypatch.setattr(api(), 'gather_lighting', lambda *args:
                        replace(lighting(*args), suns=[sun((.8, 0, .6), classes=(-1,))]))
    seen_pages, terms, boundaries = [], [], []
    expected = []
    real_prepare, real_term = api().prepare_boundaries, api().sun_term

    def run(work, pages, adapter, timeout):
        assert adapter == NV and timeout == settings.timeout_s
        seen_pages.extend(pages)
        with np.load(work / 'scene.npz') as scene:
            q, e = len(scene['query_pos']), len(scene['edge_pos'])
        assert q >= 3 and e >= 3, 'both masks need a non-empty every-third-query pattern'
        vp = vertex_gpu.VertexPassResult(np.full((q, 3), .25, np.float32), np.ones(q, np.float32),
            (np.arange(q) % 3 == 0)[:, None], (np.arange(e) % 3 == 0)[:, None], 12345)
        expected.append(vp)
        return vp

    def prepare(*args):
        boundaries.append(args[3])
        return real_prepare(*args)

    def term(*args, **kwargs):
        terms.append((args[2], kwargs['edge_sun_lit']))
        return real_term(*args, **kwargs)

    monkeypatch.setattr(vertex_gpu, 'run_in_child', run, raising=False)
    monkeypatch.setattr(api(), 'prepare_boundaries', prepare)
    monkeypatch.setattr(api(), 'sun_term', term)
    out = api().bake(tags, 'level', 'room', settings, work=work, dry_run=True)
    job = json.loads((work / 'job.json').read_text())
    summary = json.loads((work / 'summary.json').read_text())
    assert job['vertex_pass'] == 'skip'
    assert job['cycles_devices'] == ['OPTIX', 'CUDA']
    assert summary['devices']['vertex_pass'] == 'NVIDIA GeForce RTX 4060 Laptop GPU (vulkan)'
    assert summary['vertex_rays_cast'] == 12345
    assert 'vertex_pass' in summary['seconds'] and 'blender' in summary['seconds']
    assert len(seen_pages) == len(job['pages'])
    for i, (w, h) in enumerate(job['pages']):
        np.testing.assert_array_equal(seen_pages[i], page(i, w, h))
    assert terms and len(boundaries) == 1
    for lit, edge in terms:
        np.testing.assert_array_equal(lit, expected[0].sun_lit)
        np.testing.assert_array_equal(edge, expected[0].edge_sun_lit)
    for p in out.bsp_tag.root.block('lightmaps')[:-1]:
        for m in p.block('materials'):
            _, lm = unpack_uncompressed(m.data('uncompressed_vertices'),
                m.get('rendered_vertices.vertex_count'), m.get('lightmap_vertices.vertex_count'))
            np.testing.assert_array_equal(lm['incident_radiosity'], .25)


@pytest.mark.parametrize('error_type', [vertex_gpu.GpuPassError, gpu.DeviceError])
def test_gpu_failure_falls_back_to_the_compiled_pass(gpu_bake, monkeypatch, caplog, error_type):
    tags, settings, work = gpu_bake
    calls = []
    compiled = vertex_cpu.run_in_child

    def run(*args):
        calls.append(args)
        return compiled(*args)

    def fail(*args):
        raise error_type('gather: device lost')

    monkeypatch.setattr(vertex_gpu, 'run_in_child', fail, raising=False)
    monkeypatch.setattr(vertex_cpu, 'run_in_child', run)
    out = api().bake(tags, 'level', 'room', settings, work=work, dry_run=True)
    summary = json.loads((work / 'summary.json').read_text())
    assert 'falling back to the CPU pass' in caplog.text
    assert any('device lost' in w for w in summary['warnings'])
    assert summary['devices']['vertex_pass'] == 'cpu'
    assert calls and not (out.work / 'job_vertex.json').exists()


def test_named_vendor_gpu_failure_fails_before_any_tag_write(gpu_bake, monkeypatch):
    tags, settings, work = gpu_bake
    monkeypatch.setattr(gpu, 'probe_choice', lambda setting: gpu.choose('nvidia', [NV]))
    before = {p: p.read_bytes() for p in tags.rglob('*') if p.is_file()}

    def fail(*args):
        raise vertex_gpu.GpuPassError('gather: device lost')

    monkeypatch.setattr(vertex_gpu, 'run_in_child', fail, raising=False)
    with pytest.raises(api().BakeError, match='nvidia vertex pass failed'):
        api().bake(tags, 'level', 'room', settings, out_bsp='copy', work=work)
    assert {p: p.read_bytes() for p in tags.rglob('*') if p.is_file()} == before
    assert not (work / 'job_vertex.json').exists()


def test_cpu_device_runs_the_compiled_pass_after_blender(gpu_bake, monkeypatch):
    tags, settings, work = gpu_bake
    calls = []
    compiled = vertex_cpu.run_in_child

    def run(*args):
        assert (work / 'out.npz').is_file()
        calls.append(args)
        return compiled(*args)

    monkeypatch.setattr(gpu, 'probe_choice', lambda setting: gpu.choose(setting, []))
    monkeypatch.setattr(vertex_gpu, 'run_in_child', lambda *a: pytest.fail('GPU called for CPU'), raising=False)
    monkeypatch.setattr(vertex_cpu, 'run_in_child', run)
    api().bake(tags, 'level', 'room', replace(settings, device='cpu'), work=work, dry_run=True)
    job = json.loads((work / 'job.json').read_text())
    summary = json.loads((work / 'summary.json').read_text())
    assert job['vertex_pass'] == 'skip' and calls and calls[0][2] == settings.timeout_s
    assert calls[0][0] == work and len(calls[0][1]) == len(job['pages'])
    from test_blender_job import page
    for i, (w, h) in enumerate(job['pages']):
        np.testing.assert_array_equal(calls[0][1][i], page(i, w, h))
    assert job['cycles_devices'] == []
    assert summary['seconds']['blender_vertex'] == 0 and 'vertex_compile' in summary['seconds']
    assert summary['seconds']['vertex_compile'] == .125
    assert not (work / 'job_vertex.json').exists()


def test_summary_records_cpu_for_the_compiled_pass(gpu_bake, monkeypatch, caplog):
    tags, settings, work = gpu_bake
    caplog.set_level('INFO')
    monkeypatch.setattr(gpu, 'probe_choice', lambda setting: gpu.choose(setting, []))
    api().bake(tags, 'level', 'room', replace(settings, device='cpu'), work=work, dry_run=True)
    summary = json.loads((work / 'summary.json').read_text())
    assert summary['devices']['vertex_pass'] == 'cpu' and summary['warnings'] == []
    assert 'compiled CPU vertex pass: 2 threads, workqueue layer, 0.1 s compile' in caplog.text


def test_failed_child_falls_back_to_the_blender_pass_with_a_warning(gpu_bake, monkeypatch, caplog):
    tags, settings, work = gpu_bake
    monkeypatch.setattr(gpu, 'probe_choice', lambda setting: gpu.choose(setting, []))

    def fail(*args):
        raise vertex_cpu.CpuPassError('vertex: process died (exit 3): boom')

    monkeypatch.setattr(vertex_cpu, 'run_in_child', fail)
    out = api().bake(tags, 'level', 'room', replace(settings, device='cpu'), work=work, dry_run=True)
    summary = json.loads((work / 'summary.json').read_text())
    warning = ("compiled CPU vertex pass failed (vertex: process died (exit 3): boom); "
               "falling back to Blender's")
    assert any(w.startswith(warning) for w in summary['warnings'])
    assert warning in caplog.text
    assert summary['devices']['vertex_pass'] == 'cpu'
    assert (work / 'job_vertex.json').is_file()
    with np.load(work / 'scene.npz') as scene:
        expected = scene['query_pos'][:, [1, 2, 0]] * 2
    with np.load(work / 'vertex_out.npz') as data:
        np.testing.assert_array_equal(data['vertex_vector'], expected)
    for page in out.bsp_tag.root.block('lightmaps')[:-1]:
        for material in page.block('materials'):
            env, lm = unpack_uncompressed(material.data('uncompressed_vertices'),
                material.get('rendered_vertices.vertex_count'), material.get('lightmap_vertices.vertex_count'))
            raw = env['position'][:, [1, 2, 0]].astype(np.float64) * 2
            energy = env['position'].sum(axis=1) - 1
            length = np.linalg.norm(raw, axis=1)
            want = raw / np.maximum(np.maximum(length, energy), 1)[:, None]
            want[energy <= 0] = 0
            np.testing.assert_array_equal(lm['incident_radiosity'], want.astype(np.float32))


def test_both_cpu_passes_failing_stops_the_bake(gpu_bake, monkeypatch):
    tags, settings, work = gpu_bake
    monkeypatch.setattr(gpu, 'probe_choice', lambda setting: gpu.choose(setting, []))
    before = {p: p.read_bytes() for p in tags.rglob('*') if p.is_file()}

    def fail(*args):
        raise vertex_cpu.CpuPassError('vertex: process died (exit 3): boom')

    monkeypatch.setattr(vertex_cpu, 'run_in_child', fail)
    monkeypatch.setenv('FAKE_BLENDER_MODE', 'vertex_fail')
    with pytest.raises(blender_job.BlenderError):
        api().bake(tags, 'level', 'room', replace(settings, device='cpu'), out_bsp='copy', work=work)
    assert {p: p.read_bytes() for p in tags.rglob('*') if p.is_file()} == before


def test_blender_backend_runs_the_vertex_only_relaunch(gpu_bake, monkeypatch):
    tags, settings, work = gpu_bake
    monkeypatch.setattr(gpu, 'probe_choice', lambda setting: gpu.choose(setting, []))
    monkeypatch.setattr(vertex_cpu, 'run_in_child', lambda *a: pytest.fail('compiled pass called'))
    api().bake(tags, 'level', 'room', replace(settings, device='cpu', cpu_vertex_backend='blender'),
               work=work, dry_run=True)
    assert json.loads((work / 'job.json').read_text())['vertex_pass'] == 'skip'
    assert (work / 'job_vertex.json').is_file()
    summary = json.loads((work / 'summary.json').read_text())
    assert summary['warnings'] == [] and summary['devices']['vertex_pass'] == 'cpu'
    assert 'vertex_compile' not in summary['seconds']


def test_gpu_stages_use_the_gpu_counts(gpu_bake, monkeypatch):
    tags, s, work = gpu_bake
    s = replace(s, samples=90, cpu_samples=18, vertex_rays=64, cpu_vertex_rays=16)

    def run(work, pages, adapter, timeout):
        with np.load(work / 'scene.npz') as scene:
            q, e = len(scene['query_pos']), len(scene['edge_pos'])
        return vertex_gpu.VertexPassResult(np.zeros((q, 3)), np.ones(q), np.zeros((q, 0), bool),
                                          np.zeros((e, 0), bool), q * 64)

    monkeypatch.setattr(vertex_gpu, 'run_in_child', run)
    api().bake(tags, 'level', 'room', s, work=work, dry_run=True)
    summary = json.loads((work / 'summary.json').read_text())
    assert summary['rays'] == {'samples': s.samples, 'texel_samples': s.resolved_texel_samples(), 'vertex_rays': s.vertex_rays}


def test_compiled_pass_uses_the_cpu_ray_count(tiny, fake, monkeypatch):
    tags, s, work = tiny
    calls = []
    compiled = vertex_cpu.run_in_child

    def run(*args):
        calls.append(args)
        return compiled(*args)

    s = replace(s, blender=fake, device='cpu', samples=90, cpu_samples=18, vertex_rays=64, cpu_vertex_rays=16)
    monkeypatch.setattr(api(), 'run_blender', blender_job.run_blender)
    monkeypatch.setattr(vertex_cpu, 'run_in_child', run)
    api().bake(tags, 'level', 'room', s, work=work, dry_run=True)
    summary = json.loads((work / 'summary.json').read_text())
    assert summary['rays'] == {'samples': 18, 'texel_samples': 18, 'vertex_rays': 16}
    assert calls
    with np.load(work / 'scene.npz') as scene:
        assert summary['vertex_rays_cast'] == len(scene['query_pos']) * 16


def test_gpu_vertex_failure_reruns_on_the_cpu_with_the_cpu_rays(gpu_bake, monkeypatch):
    tags, s, work = gpu_bake
    s = replace(s, samples=90, cpu_samples=18, vertex_rays=64, cpu_vertex_rays=16)

    def fail(*args):
        raise vertex_gpu.GpuPassError('gather: device lost')

    monkeypatch.setattr(vertex_gpu, 'run_in_child', fail)
    api().bake(tags, 'level', 'room', s, work=work, dry_run=True)
    summary = json.loads((work / 'summary.json').read_text())
    assert not (work / 'job_vertex.json').exists()
    assert summary['rays']['vertex_rays'] == 16 and any('GPU vertex pass failed' in w for w in summary['warnings'])
    assert summary['rays']['samples'] == 90 and summary['rays']['texel_samples'] == 90
    with np.load(work / 'scene.npz') as scene:
        assert summary['vertex_rays_cast'] == len(scene['query_pos']) * 16


def test_counts_without_cpu_values_stay_the_same_on_the_cpu(tiny, fake, monkeypatch):
    tags, s, work = tiny
    s = replace(s, blender=fake, device='cpu')
    monkeypatch.setattr(api(), 'run_blender', blender_job.run_blender)
    api().bake(tags, 'level', 'room', s, work=work, dry_run=True)
    summary = json.loads((work / 'summary.json').read_text())
    assert summary['rays'] == {'samples': s.samples, 'texel_samples': s.resolved_texel_samples(), 'vertex_rays': s.vertex_rays}


def test_device_probe_failure_precedes_charting_and_tag_writes(tiny, monkeypatch):
    tags, settings, work = tiny

    def fail(setting):
        raise gpu.DeviceError('nvidia: unavailable')

    monkeypatch.setattr(gpu, 'probe_choice', fail)
    monkeypatch.setattr(api(), 'chart_and_pack', lambda *a: pytest.fail('charted before probe'))
    with pytest.raises(gpu.DeviceError, match='unavailable'):
        api().bake(tags, 'level', 'room', settings, work=work)


def test_cycles_cpu_fallback_is_reported(gpu_bake, monkeypatch, caplog):
    tags, settings, work = gpu_bake
    real = api().run_blender

    def cpu(*args):
        return replace(real(*args), device='CPU')

    def run(work, pages, adapter, timeout):
        with np.load(work / 'scene.npz') as scene:
            q, e = len(scene['query_pos']), len(scene['edge_pos'])
        return vertex_gpu.VertexPassResult(np.zeros((q, 3)), np.ones(q), np.zeros((q, 0), bool),
                                            np.zeros((e, 0), bool), 1)

    monkeypatch.setattr(api(), 'run_blender', cpu)
    monkeypatch.setattr(vertex_gpu, 'run_in_child', run, raising=False)
    api().bake(tags, 'level', 'room', settings, work=work, dry_run=True)
    summary = json.loads((work / 'summary.json').read_text())
    assert 'cycles fell back to the CPU' in caplog.text
    assert 'cycles fell back to the CPU' in summary['warnings']


def test_the_probe_is_its_own_stage(tiny):
    tags, settings, work = tiny
    api().bake(tags, 'level', 'room', settings, work=work, dry_run=True)
    assert 'probe' in json.loads((work / 'summary.json').read_text())['seconds']


def test_cycles_cpu_fallback_is_reported_without_a_chosen_adapter(tiny, monkeypatch, caplog):
    tags, settings, work = tiny
    choice = gpu.choose('auto', [])
    assert choice.adapter is None and choice.cycles
    monkeypatch.setattr(gpu, 'probe_choice', lambda setting: choice)
    fake_blender = api().run_blender
    monkeypatch.setattr(api(), 'run_blender', lambda *args: replace(fake_blender(*args), device='CPU'))
    api().bake(tags, 'level', 'room', settings, work=work, dry_run=True)
    summary = json.loads((work / 'summary.json').read_text())
    assert summary['warnings'].count('cycles fell back to the CPU') == 1
    assert 'cycles fell back to the CPU' in caplog.text


def test_no_cycles_cpu_warning_when_cycles_is_not_offered_a_gpu(tiny, monkeypatch):
    tags, settings, work = tiny
    monkeypatch.setattr(gpu, 'probe_choice', lambda setting: gpu.choose('cpu', []))
    fake_blender = api().run_blender
    monkeypatch.setattr(api(), 'run_blender', lambda *args: replace(fake_blender(*args), device='CPU'))
    api().bake(tags, 'level', 'room', replace(settings, device='cpu'), work=work, dry_run=True)
    assert json.loads((work / 'summary.json').read_text())['warnings'] == []


def test_unpaired_coincident_triangle_groups_are_a_summary_warning(tiny, caplog):
    tags, settings, work = tiny
    bsp = read_tag(tags / 'room.scenario_structure_bsp')
    surfaces = [child(bsp.root, 'surfaces') for _ in range(3)]
    for surface in surfaces:
        surface.body[:] = struct.pack('>3H', 0, 1, 2)
    bsp.root.set_block('surfaces', surfaces)
    bsp.root.block('lightmaps')[0].block('materials')[0].set('surface_count', 3)
    put(tags, 'sbsp', 'room', bsp)
    api().bake(tags, 'level', 'room', settings, work=work, dry_run=True)
    summary = json.loads((work / 'summary.json').read_text())
    live = [record.getMessage() for record in caplog.records if record.name == 'lightmapper.twins']
    assert live == ['1 position groups with more than two triangles left unpaired']
    assert summary['warnings'] == live


def test_no_twin_warning_when_every_group_pairs(tiny):
    tags, settings, work = tiny
    api().bake(tags, 'level', 'room', settings, work=work, dry_run=True)
    assert not any('left unpaired' in w for w in json.loads((work / 'summary.json').read_text())['warnings'])


def test_pipeline_stitches_before_quantisation(tiny, monkeypatch):
    from test_seams import fixture
    from lightmapper.pages import quantise
    from lightmapper.diff import bitmap_pages

    tags, settings, work = tiny
    pages, raw, _, _ = fixture()
    for page in pages:
        page.materials[0].group = 0
    module = api()
    monkeypatch.setattr(module, 'chart_and_pack', lambda *args: pages)
    original = module.run_blender

    def with_step(work, s):
        result = original(work, s)
        result.pages = [r.copy() for r in raw]
        return result

    monkeypatch.setattr(module, 'run_blender', with_step)
    out = module.bake(tags, 'level', 'room', settings, work=work, dry_run=True)
    for image in out.float_pages:
        np.testing.assert_allclose(image[2, 2:11], 0.5, atol=1e-5)
    from lightmapper.formats.vertices import decode_r5g6b5
    for image, pixels in zip(out.float_pages, bitmap_pages(out.bitmap_tag)):
        np.testing.assert_array_equal(pixels, decode_r5g6b5(quantise(image, settings)))


def test_pipeline_hands_simple_parameterization_groups_to_stitching(tiny, monkeypatch):
    tags, settings, work = tiny
    module = api()
    seen = []
    original_read, original_finish = module.read_radiosity, module.finish_pages
    monkeypatch.setattr(module, 'read_radiosity', lambda *args: replace(original_read(*args), flags=module.SIMPLE_PARAMETERIZATION))

    def spy(*args, **kwargs):
        seen.append(kwargs.get('joined_groups'))
        return original_finish(*args, **kwargs)

    monkeypatch.setattr(module, 'finish_pages', spy)
    module.bake(tags, 'level', 'room', settings, work=work, dry_run=True)
    assert seen == [{0}]


def test_pipeline_fills_covered_hole_using_blender_written_mask(tiny, monkeypatch):
    from lightmapper.raster import coverage
    tags, settings, work = tiny
    module = api()
    original = module.run_blender
    holes = []

    def with_holes(work, s):
        result = original(work, s)
        with np.load(work / 'scene.npz') as scene:
            for i, raw in enumerate(result.pages):
                h, w = raw.shape[:2]
                rows = scene['tri_page'] == i
                mask, _ = coverage(scene['lm_uv'], scene['tris'][rows], scene['tri_class'][rows], w, h)
                y, x = np.argwhere(mask)[0]
                raw[:] = 0.7
                raw[y, x] = 0
                result.written[i][y, x] = False
                holes.append((i, y, x))
        return result

    monkeypatch.setattr(module, 'run_blender', with_holes)
    out = module.bake(tags, 'level', 'room', settings, work=work, dry_run=True)
    assert holes
    for i, y, x in holes:
        np.testing.assert_allclose(out.float_pages[i][y, x], [0.7, 0.7, 0.7])


@pytest.mark.parametrize('kwargs', [{'sources': 'sun'}, {'sources': 'emit'}, {'dry_run': True}])
def test_diagnostics_write_no_tags(tiny, kwargs):
    tags, settings, work = tiny
    before = {p: p.read_bytes() for p in tags.rglob('*') if p.is_file()}
    api().bake(tags, 'level', 'room', settings, out_bsp='copy', out_scenario='copy_level', work=work, **kwargs)
    assert {p: p.read_bytes() for p in tags.rglob('*') if p.is_file()} == before


def test_blender_failure_leaves_tags_untouched(tiny, monkeypatch):
    from lightmapper.blender_job import BlenderError
    tags, settings, work = tiny
    before = {p: p.read_bytes() for p in tags.rglob('*') if p.is_file()}

    def fail(*args):
        raise BlenderError('bake failed')

    monkeypatch.setattr(api(), 'run_blender', fail)
    with pytest.raises(BlenderError, match='bake failed'):
        api().bake(tags, 'level', 'room', settings, work=work)
    assert {p: p.read_bytes() for p in tags.rglob('*') if p.is_file()} == before


def test_surface_cap_refuses(tiny, monkeypatch, caplog):
    tags, settings, work = tiny
    module = api()
    monkeypatch.setattr(module, 'MAX_BSP_SURFACES', 1)

    def no_chart(*args):
        raise AssertionError('charted before refusing stock cap')

    with monkeypatch.context() as patch:
        patch.setattr(module, 'chart_and_pack', no_chart)
        with pytest.raises(ValueError, match='surfaces'):
            module.bake(tags, 'level', 'room', settings, work=work)
    module.bake(tags, 'level', 'room', settings, work=work, dry_run=True)
    assert 'surfaces' in caplog.text
    assert not list(tags.rglob('backups/*'))


def test_charts_cache(tiny, monkeypatch):
    tags, settings, work = tiny
    module = api()
    cache = work / 'charts.npz'
    first = module.bake(tags, 'level', 'room', settings, work=work, charts=cache, dry_run=True)

    def no_chart(*args):
        raise AssertionError('recharted')

    monkeypatch.setattr(module, 'chart_and_pack', no_chart)
    second = module.bake(tags, 'level', 'room', settings, work=work, charts=cache, dry_run=True)
    np.testing.assert_array_equal(first.new_to_input, second.new_to_input)
    with pytest.raises(AssertionError, match='recharted'):
        module.bake(tags, 'level', 'room', replace(settings, texels_per_unit=.5), work=work, charts=cache, dry_run=True)
    bsp_path = tags / 'room.scenario_structure_bsp'
    bsp = read_tag(bsp_path)
    bsp.root.set('world_bounds_x', (0.0, 1.0))
    write_tag(bsp, bsp_path)
    with pytest.raises(AssertionError, match='recharted'):
        module.bake(tags, 'level', 'room', settings, work=work, charts=cache, dry_run=True)


@pytest.mark.parametrize('field, value', [('radiosity.detail_level', 3), ('radiosity.flags', 1)])
def test_charts_cache_follows_chart_shader_inputs(tiny, monkeypatch, field, value):
    tags, settings, work = tiny
    module = api()
    cache = work / 'charts.npz'
    module.bake(tags, 'level', 'room', settings, work=work, charts=cache, dry_run=True)
    shader_path = tags / 'wall.shader_environment'
    shader = read_tag(shader_path)
    shader.root.set(field, value)
    write_tag(shader, shader_path)

    def no_chart(*args):
        raise AssertionError('recharted')

    monkeypatch.setattr(module, 'chart_and_pack', no_chart)
    with pytest.raises(AssertionError, match='recharted'):
        module.bake(tags, 'level', 'room', settings, work=work, charts=cache, dry_run=True)


def test_backup_never_overwritten(tiny):
    tags, settings, work = tiny
    original = (tags / 'room.scenario_structure_bsp').read_bytes()
    api().bake(tags, 'level', 'room', settings, work=work)
    api().bake(tags, 'level', 'room', settings, work=work)
    backups = sorted((tags / 'backups').glob('*_room.scenario_structure_bsp'))
    assert len(backups) == 2
    assert all(len(p.name) == 15 + len('_room.scenario_structure_bsp') for p in backups)
    assert all(p.name[8] == '-' and p.name[15] == '_' for p in backups)
    assert backups[0].read_bytes() == original
    assert len(list((tags / 'backups').glob('*_room.bitmap'))) == 1


def test_copy_command(tiny):
    from lightmapper.cli import main
    tags, _, _ = tiny
    from lightmapper.formats.bitmap import write_bitmap
    bsp = read_tag(tags / 'room.scenario_structure_bsp')
    bsp.root.set_reference('lightmaps_bitmap', 'bitm', 'old_bitmap')
    write_tag(bsp, tags / 'room.scenario_structure_bsp')
    put(tags, 'bitm', 'old_bitmap', write_bitmap([np.zeros((8, 8), np.uint16)]))
    bitmap_bytes = (tags / 'old_bitmap.bitmap').read_bytes()
    assert main(['copy', 'level', 'room', '--tags', str(tags), '--out-bsp', 'copies\\room', '--out-scenario', 'copies\\level']) == 0
    copied = read_tag(tags / 'copies/room.scenario_structure_bsp')
    assert copied.root.reference('lightmaps_bitmap') == ('bitm', 'copies\\room')
    assert read_tag(tags / 'copies/level.scenario').root.block('structure_bsps')[0].reference('structure_bsp') == ('sbsp', 'copies\\room')
    assert (tags / 'copies/room.bitmap').read_bytes() == bitmap_bytes


def test_validate_command(tiny, capsys):
    from lightmapper.cli import main
    tags, settings, work = tiny
    api().bake(tags, 'level', 'room', settings, work=work)
    assert main(['validate', 'room', '--tags', str(tags), '--strict']) == 0
    bsp = read_tag(tags / 'room.scenario_structure_bsp')
    bsp.root.block('lightmaps')[-1].set('bitmap', 2)
    write_tag(bsp, tags / 'room.scenario_structure_bsp')
    assert main(['validate', str(tags / 'room.scenario_structure_bsp'), '--tags', str(tags)]) == 1
    assert 'last element must have bitmap -1' in capsys.readouterr().out


def test_validate_uses_tags_environment(tiny, monkeypatch):
    from lightmapper.cli import main
    tags, settings, work = tiny
    api().bake(tags, 'level', 'room', settings, work=work)
    monkeypatch.setenv('HCEEK_TAGS', str(tags))
    assert main(['validate', 'room', '--strict']) == 0


def test_validate_explicit_tags_override_environment(tiny, monkeypatch):
    from lightmapper.cli import main
    tags, settings, work = tiny
    api().bake(tags, 'level', 'room', settings, work=work)
    monkeypatch.setenv('HCEEK_TAGS', str(work / 'missing'))
    assert main(['validate', 'room', '--tags', str(tags)]) == 0


@pytest.mark.parametrize('environment', [None, ''])
def test_validate_requires_tags_without_environment(monkeypatch, capsys, environment):
    from lightmapper.cli import main
    if environment is None:
        monkeypatch.delenv('HCEEK_TAGS', raising=False)
    else:
        monkeypatch.setenv('HCEEK_TAGS', environment)
    with pytest.raises(SystemExit) as error:
        main(['validate', 'missing'])
    assert error.value.code == 2
    message = capsys.readouterr().err
    assert '--tags' in message and 'HCEEK_TAGS' in message


def test_clockwise_triangle_reaches_blender_counter_clockwise(tiny, monkeypatch):
    tags, settings, work = tiny
    # the fixture's normals are +z; (0, 2, 1) is the clockwise order Halo uses for a +z face
    room = read_tag(tags / 'room.scenario_structure_bsp')
    room.root.block('surfaces')[0].body[:] = struct.pack('>3H', 0, 2, 1)
    put(tags, 'sbsp', 'room', room)
    sent = {}
    blender = api().run_blender

    def spy(work, s):
        with np.load(work / 'scene.npz') as scene:
            sent['tris'], sent['positions'] = scene['tris'], scene['positions']
        return blender(work, s)

    monkeypatch.setattr(api(), 'run_blender', spy)
    out = api().bake(tags, 'level', 'room', settings, out_bsp='copy', work=work)
    p = sent['positions']
    t = sent['tris']
    winding = np.cross(p[t[:, 1]] - p[t[:, 0]], p[t[:, 2]] - p[t[:, 0]])
    # the second fixture triangle is already counter-clockwise, so Blender gets it flipped
    np.testing.assert_array_equal(winding[:, 2], np.array([16., -16.])[out.new_to_input[:2]])
    # the tag keeps tool's order: the clockwise surface stays clockwise in the written BSP
    written = []
    surfaces = out.bsp_tag.root.block('surfaces')
    for page in out.bsp_tag.root.block('lightmaps')[:-1]:
        for m in page.block('materials'):
            env, _ = unpack_uncompressed(m.data('uncompressed_vertices'),
                                         m.get('rendered_vertices.vertex_count'), m.get('lightmap_vertices.vertex_count'))
            first = m.get('surfaces')
            for k in range(m.get('surface_count')):
                a, b, c = struct.unpack('>3H', bytes(surfaces[first + k].body)[:6])
                written.append((int(out.new_to_input[first + k]), env['position'][[a, b, c]]))
    assert len(written) == 2
    for source, corners in written:
        z = np.cross(corners[1] - corners[0], corners[2] - corners[0])[2]
        assert (z < 0) == (source == 0)


@pytest.mark.parametrize('flagged_first', [False, True])
def test_scene_flags_ignore_normals_triangles_and_their_queries(tiny, flagged_first):
    from lightmapper.formats.bsp import OutMaterial, OutPage, read_geometry
    from lightmapper.gather import ShaderRadiosity

    tags, _, _ = tiny
    geometry = read_geometry(read_tag(tags / 'room.scenario_structure_bsp'))
    group = geometry.groups[0]
    other = replace(group, env=group.env.copy())
    other.env['position'][3] = group.env['position'][0]
    geometry.groups.append(other)
    materials = [OutMaterial(0, group.env[:3], np.zeros(3, LM), np.uint16([[0, 1, 2]]), np.int32([0])),
                 OutMaterial(1, other.env[3:], np.zeros(3, LM), np.uint16([[0, 1, 2]]), np.int32([1]))]
    flags = [flagged_first, not flagged_first]
    radiosity = [ShaderRadiosity(2 if flag else 0, 0, 0., np.zeros(3), np.ones(3)) for flag in flags]
    scene, inverse, _ = api()._scene(tags, read_tag(tags / 'level.scenario'), 0, geometry,
        [OutPage(8, 8, materials)], radiosity, np.zeros(2, np.int16))
    np.testing.assert_array_equal(scene.tri_ignore_normals, flags)
    assert inverse[0] == inverse[3]
    assert len(scene.query_pos) == 5
    assert scene.query_ignore_normals[inverse[0]]
    np.testing.assert_array_equal(scene.query_ignore_normals[inverse[[1, 2, 4, 5]]],
                                  [flagged_first] * 2 + [not flagged_first] * 2)


def test_scene_at_stock_surface_cap(tiny):
    from lightmapper.formats.bsp import BspGeometry, OutMaterial, OutPage, SourceGroup
    from lightmapper.gather import ShaderRadiosity

    tags, _, _ = tiny
    materials, groups = [], []
    total = 131072
    for start in range(0, total, 20000):
        count = min(20000, total - start)
        env = np.zeros(count * 3, ENV)
        # grid-snapped vertices exercise exact-byte query keys at the largest stock input
        env['position'][:, 0] = np.arange(start * 3, (start + count) * 3)
        env['normal'] = [0, 0, 1]
        tris = np.arange(count * 3, dtype=np.uint16).reshape(-1, 3)
        ids = np.arange(start, start + count, dtype=np.int32)[::-1].copy()
        groups.append(SourceGroup(('senv', 'wall', 0, 0, 0), None, env, tris, ids))
        materials.append(OutMaterial(len(groups) - 1, env, np.zeros(len(env), LM), tris, ids))
    final = read_tag(tags / 'room.scenario_structure_bsp').root.block('lightmaps')[-1]
    geometry = BspGeometry(groups, final, np.zeros(total, np.int16), np.array([-1], np.int16))
    radiosity = [ShaderRadiosity(2 if k % 2 == 0 else 0, 0, 2., np.array([.25, .5, 1.]), np.ones(3))
                 for k in range(len(groups))]
    classes = np.arange(total, dtype=np.int32) % 3 - 1
    scene, inverse, textures = api()._scene(tags, read_tag(tags / 'level.scenario'), 0,
        geometry, [OutPage(1024, 1024, materials)], radiosity, classes)
    assert len(scene.tris) == 131072
    assert len(scene.positions) == 393216
    assert scene.tris[-1].tolist() == [393213, 393215, 393214]
    assert scene.tri_class[:3].tolist() == [0, -1, 1]
    assert scene.query_class[:9].tolist() == [0, 0, 0, -1, -1, -1, 1, 1, 1]
    np.testing.assert_array_equal(inverse, np.arange(393216))
    np.testing.assert_array_equal(scene.query_inverse, inverse)
    expected_flags = (np.arange(total) // 20000) % 2 == 0
    np.testing.assert_array_equal(scene.tri_ignore_normals, expected_flags)
    np.testing.assert_array_equal(scene.query_ignore_normals, np.repeat(expected_flags, 3))
    np.testing.assert_array_equal(scene.tri_emission, np.tile([.5, 1., 2.], (total, 1)))
    assert scene.tri_casts_shadow.all()
    assert not scene.tri_cutout.any()
    assert (scene.tri_texture == -1).all() and textures.arrays() == []


def test_scene_textures_and_occluder_scale(tiny):
    from test_scenery import write_model, write_scenery
    from lightmapper.formats.bitmap import write_bitmap
    from lightmapper.formats.bsp import OutMaterial, OutPage, read_geometry
    from lightmapper.gather import read_radiosity

    tags, _, _ = tiny
    shader = read_tag(tags / 'wall.shader_environment')
    shader.root.set('properties.flags', 1)
    shader.root.set_reference('diffuse.base_map', 'bitm', 'base')
    put(tags, 'senv', 'wall', shader)
    put(tags, 'bitm', 'base', write_bitmap([np.full((8, 8), 0xffff, np.uint16)]))
    model_shader = new_tag('soso')
    model_shader.root.set_reference('maps.base_map', 'bitm', 'base')
    model_shader.root.set('maps.map_u_scale', .5)
    model_shader.root.set('maps.map_v_scale', 4.)
    put(tags, 'soso', 'model_shader', model_shader)
    write_model(tags, 'model', [[(0, [(0, 0, 0), (1, 0, 0), (0, 1, 0)], [0, 1, 2])]],
                shaders=(('soso', 'model_shader'),), scale=(2., 3.))
    write_scenery(tags, 'prop', 'model')
    scenario = read_tag(tags / 'level.scenario')
    palette = child(scenario.root, 'scenery_palette')
    palette.set_reference('name', 'scen', 'prop')
    scenario.root.set_block('scenery_palette', [palette])
    placements = []
    for mask, x in [(1, 100.), (2, 10.)]:
        placement = child(scenario.root, 'scenery')
        placement.set('bsp_indices', mask)
        placement.set('placement.position', (x, 0., 0.))
        placements.append(placement)
    scenario.root.set_block('scenery', placements)
    geometry = read_geometry(read_tag(tags / 'room.scenario_structure_bsp'))
    group = geometry.groups[0]
    material = OutMaterial(0, group.env, np.zeros(len(group.env), LM), group.tris, group.surface_ids)
    scene, _, textures = api()._scene(tags, scenario, 1, geometry, [OutPage(8, 8, [material])],
        [read_radiosity(tags, 'senv', 'wall')], np.array([-1, 2], np.int16))
    np.testing.assert_array_equal(scene.tex_uv, group.env['texcoord'])
    assert scene.tri_class.tolist() == [-1, 2]
    assert scene.tri_cutout.tolist() == [False, False]
    assert scene.tri_texture.tolist() == [0, 0]
    assert scene.occ_positions.tolist() == [[10., 0., 0.], [10., 1., 0.], [11., 0., 0.]]
    assert scene.occ_uv.tolist() == [[0., 6.], [.25, 6.], [.125, 6.]]
    assert scene.occ_tris.tolist() == [[0, 1, 2]]
    assert scene.occ_tri_texture.tolist() == [0]
    assert scene.occ_tri_cutout.tolist() == [True]
    assert scene.occ_tri_casts_shadow.tolist() == [True]
    assert len(textures.arrays()) == 1
    np.testing.assert_array_equal(textures.arrays()[0], np.full((8, 8, 4), 255, np.uint8))


def test_scene_carries_the_tint_of_a_see_through_group(tiny):
    from lightmapper.formats.bsp import OutMaterial, OutPage, read_geometry
    from lightmapper.gather import ShaderRadiosity

    tags, _, _ = tiny
    put(tags, 'sgla', 'glass', new_tag('sgla'))
    geometry = read_geometry(read_tag(tags / 'room.scenario_structure_bsp'))
    group = geometry.groups[0]
    group.key = ('sgla', 'glass') + tuple(group.key[2:])
    material = OutMaterial(0, group.env, np.zeros(len(group.env), LM), group.tris, group.surface_ids)
    tint = np.array([.46, .6, .66])
    scene, _, _ = api()._scene(tags, read_tag(tags / 'level.scenario'), 0, geometry,
        [OutPage(8, 8, [material])], [ShaderRadiosity(0, 0, 0., np.zeros(3), tint)], np.array([-1, -1], np.int16))
    assert not scene.tri_casts_shadow.any()
    np.testing.assert_allclose(scene.tri_tint, [tint, tint], rtol=1e-6)
    assert scene.tri_tint.dtype == np.float32


def test_scene_solid_group_has_no_tint_to_apply(tiny):
    from lightmapper.formats.bsp import OutMaterial, OutPage, read_geometry
    from lightmapper.gather import ShaderRadiosity

    tags, _, _ = tiny
    geometry = read_geometry(read_tag(tags / 'room.scenario_structure_bsp'))
    group = geometry.groups[0]
    material = OutMaterial(0, group.env, np.zeros(len(group.env), LM), group.tris, group.surface_ids)
    scene, _, _ = api()._scene(tags, read_tag(tags / 'level.scenario'), 0, geometry,
        [OutPage(8, 8, [material])], [ShaderRadiosity(0, 0, 0., np.zeros(3), np.array([.5, .5, .5]))],
        np.array([-1, -1], np.int16))
    assert scene.tri_casts_shadow.all()
    np.testing.assert_array_equal(scene.tri_tint, np.ones((2, 3)))


def put_sotr(tags, name, power, colour=(1., .5, .25)):
    header = bytearray(64)
    header[36:40] = b'sotr'
    target = resolve_reference_path(tags, 'sotr', name)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(bytes(header) + struct.pack('>Hhf3f3f', 0, 0, power, *colour, 1., 1., 1.))


def rewind_room(tags):
    bsp = read_tag(tags / 'room.scenario_structure_bsp')
    for surface, corners in zip(bsp.root.block('surfaces'), [(0, 2, 1), (3, 5, 4)]):
        surface.body[:] = struct.pack('>3H', *corners)
    write_tag(bsp, tags / 'room.scenario_structure_bsp')


def add_unlit_quad(tags, shader, z=2., group='sotr'):
    """A quad above the tiny room's first triangle, facing down, in the final group as shader. The room's own
    triangles are rewound clockwise relative to their +z normals, as Halo's are, so that they face the quad."""
    rewind_room(tags)
    bsp = read_tag(tags / 'room.scenario_structure_bsp')
    final = bsp.root.block('lightmaps')[-1]
    material = child(final, 'materials')
    material.set_reference('shader', group, shader)
    env, lm = np.zeros(4, ENV), np.zeros(0, LM)
    env['position'] = [[0, 0, z], [4, 0, z], [4, 4, z], [0, 4, z]]
    env['normal'] = [0, 0, -1]
    env['tangent'] = [1, 0, 0]
    env['binormal'] = [0, 1, 0]
    env['texcoord'] = [[0, 0], [1, 0], [1, 1], [0, 1]]
    material.set('rendered_vertices.vertex_count', 4)
    material.set('surfaces', len(bsp.root.block('surfaces')))
    material.set('surface_count', 2)
    material.set_data('uncompressed_vertices', pack_uncompressed(env, lm))
    material.set_data('compressed_vertices', compress(env, lm))
    final.set_block('materials', final.block('materials') + [material])
    surfaces = bsp.root.block('surfaces')
    for corners in [(0, 1, 2), (0, 2, 3)]:
        surface = child(bsp.root, 'surfaces')
        surface.body[:] = struct.pack('>3H', *corners)
        surfaces.append(surface)
    bsp.root.set_block('surfaces', surfaces)
    write_tag(bsp, tags / 'room.scenario_structure_bsp')


def scene_of(tags, with_textures=False):
    from lightmapper.formats.bsp import read_geometry
    from lightmapper.gather import read_radiosity, surface_classes

    scenario = read_tag(tags / 'level.scenario')
    geometry = read_geometry(read_tag(tags / 'room.scenario_structure_bsp'))
    radiosity = [read_radiosity(tags, *g.key[:2]) for g in geometry.groups]
    classes = surface_classes(geometry, 0)
    built = api()._scene(tags, scenario, 0, geometry, [], radiosity, classes)
    return (built[0], built[2]) if with_textures else built[0]


def test_scene_takes_emitters_from_the_unlit_group(tiny):
    tags, _, _ = tiny
    put_sotr(tags, 'glow', 10., (1., .5, .25))
    put_sotr(tags, 'dull', 0.)
    add_unlit_quad(tags, 'glow')
    add_unlit_quad(tags, 'dull', z=3.)
    scene = scene_of(tags)
    assert len(scene.occ_tris) == 2
    np.testing.assert_allclose(scene.occ_tri_emission, [[10., 5., 2.5]] * 2)
    assert not scene.occ_tri_casts_shadow.any()
    np.testing.assert_array_equal(scene.occ_tri_tint, np.ones((2, 3)))
    assert scene.occ_positions[:, 2].tolist() == [2.] * 6
    # Halo's clockwise winding reaches Blender counter-clockwise
    assert scene.occ_positions[:3].tolist() == [[0, 0, 2], [4, 4, 2], [4, 0, 2]]


@pytest.mark.parametrize('corrupt', [False, True])
def test_unreadable_unlit_shader_does_not_abort_a_bake(tiny, corrupt, caplog):
    tags, settings, work = tiny
    add_unlit_quad(tags, 'broken')
    if corrupt:
        (tags / 'broken.shader_transparent_generic').write_bytes(b'bad')
    with caplog.at_level('WARNING'):
        out = api().bake(tags, 'level', 'room', settings, work=work, dry_run=True)
    assert len(out.float_pages) > 0
    assert 'broken' in caplog.text and 'emission' in caplog.text


def test_unlit_group_is_written_back_unchanged(tiny):
    tags, settings, work = tiny
    put_sotr(tags, 'glow', 10.)
    add_unlit_quad(tags, 'glow')
    before = read_tag(tags / 'room.scenario_structure_bsp').root.block('lightmaps')[-1]
    out = api().bake(tags, 'level', 'room', settings, out_bsp='copy', out_scenario='copy_level', work=work)
    after = out.bsp_tag.root.block('lightmaps')[-1]
    assert after.body == before.body
    assert [m.body for m in after.block('materials')] == [m.body for m in before.block('materials')]
    for old, new in zip(before.block('materials'), after.block('materials')):
        assert old.data('uncompressed_vertices') == new.data('uncompressed_vertices')
        assert old.data('compressed_vertices') == new.data('compressed_vertices')


@pytest.mark.blender
def test_unlit_emitter_lights_the_level(tiny, monkeypatch):
    from lightmapper import blender_job
    tags, settings, work = tiny
    monkeypatch.setattr(api(), 'run_blender', blender_job.run_blender)
    settings = replace(settings, samples=16, vertex_rays=8, blender=str(LIGHTMAPPER_BLENDER))

    def lit(power, name):
        put_sotr(tags, 'glow', power)
        out = api().bake(tags, 'level', 'room', settings, out_bsp=name, out_scenario=name + '_level',
                         work=work / name, sources='emit', dry_run=True)
        return float(sum(page.sum() for page in out.float_pages))

    add_unlit_quad(tags, 'glow')
    assert lit(0., 'dark') == 0.
    assert lit(10., 'glow') > 0.


def put_scenery_over_the_room(tags, power, z=2.):
    """A 4x4 quad prop facing down at height z, in a soso shader of the given power, placed in the room's bsp."""
    from test_scenery import write_model, write_scenery

    shader = new_tag('soso')
    shader.root.set('radiosity.power', power)
    shader.root.set('radiosity.color_of_emitted_light', (1., .5, .25))
    put(tags, 'soso', 'prop_shader', shader)
    write_model(tags, 'model', [[(0, [(0, 0, 0), (4, 0, 0), (0, 4, 0), (4, 4, 0)], [0, 1, 2, 3])]],
                shaders=(('soso', 'prop_shader'),), scale=(1., 1.))
    write_scenery(tags, 'prop', 'model')
    scenario = read_tag(tags / 'level.scenario')
    palette = child(scenario.root, 'scenery_palette')
    palette.set_reference('name', 'scen', 'prop')
    scenario.root.set_block('scenery_palette', [palette])
    placement = child(scenario.root, 'scenery')
    placement.set('bsp_indices', 1)
    placement.set('placement.position', (0., 0., z))
    scenario.root.set_block('scenery', [placement])
    write_tag(scenario, tags / 'level.scenario')


def test_scenery_takes_its_shaders_emission(tiny):
    tags, _, _ = tiny
    put_scenery_over_the_room(tags, 4.)
    scene = scene_of(tags)
    assert len(scene.occ_tris) == 2
    np.testing.assert_allclose(scene.occ_tri_emission, [[4., 2., 1.]] * 2)
    assert scene.occ_tri_casts_shadow.all()


def test_non_emitting_scenery_stays_dark(tiny):
    tags, _, _ = tiny
    put_scenery_over_the_room(tags, 0.)
    scene = scene_of(tags)
    assert len(scene.occ_tris) == 2 and not scene.occ_tri_emission.any()


@pytest.mark.blender
def test_emissive_scenery_lights_the_level(tiny, monkeypatch):
    from lightmapper import blender_job
    tags, settings, work = tiny
    monkeypatch.setattr(api(), 'run_blender', blender_job.run_blender)
    settings = replace(settings, samples=16, vertex_rays=8, blender=str(LIGHTMAPPER_BLENDER))
    rewind_room(tags)

    def lit(power, name):
        put_scenery_over_the_room(tags, power)
        out = api().bake(tags, 'level', 'room', settings, out_bsp=name, out_scenario=name + '_level',
                         work=work / name, sources='emit', dry_run=True)
        return float(sum(page.sum() for page in out.float_pages))

    assert lit(0., 'dark') == 0.
    assert lit(10., 'glow') > 0.


def test_pipeline_adds_the_plane_away_sun_term_to_the_raw_page(tiny, monkeypatch):
    import lightmapper.pages as pages
    tags, settings, work = tiny
    pipeline = api()
    calls = []
    built = []
    original_prepare = pipeline.prepare_boundaries

    def prepare(*args):
        boundary = original_prepare(*args)
        built.append(boundary)
        return boundary

    compiled = vertex_cpu.run_in_child

    def with_sun_lit(*args):
        result, info = compiled(*args)
        result.sun_lit = np.ones((1, 0), bool)
        result.edge_sun_lit = np.ones((2, 0), bool)
        return result, info

    def term(scene, suns, lit, page, width, height, reach, edge_sun_lit=None, boundaries=None):
        calls.append((page, lit.shape))
        value = 0.25 if (edge_sun_lit is not None and edge_sun_lit.shape == (2, 0)
                         and len(built) == 1 and boundaries is built[0]) else 0
        return np.full((height, width, 3), value, np.float32)

    monkeypatch.setattr(vertex_cpu, 'run_in_child', with_sun_lit)
    monkeypatch.setattr(pipeline, 'prepare_boundaries', prepare)
    monkeypatch.setattr(pipeline, 'sun_term', term)
    monkeypatch.setattr(pages, '_denoise', lambda rgb, timeout, worker=None: rgb * 2)
    out = pipeline.bake(tags, 'level', 'room', replace(settings, page_max=8, padding=1, texels_per_unit=0.75,
                                                     denoise=True, detail_level_scale=(1, 1, 1, 1)), work=work)
    assert calls and calls[0][1] == (1, 0)
    assert len(out.float_pages) == 2, 'exercise sharing across pages, not just the first call'
    assert len(built) == 1
    assert all(page.max() == pytest.approx(0.5) for page in out.float_pages)


def test_pipeline_adds_flat_sun_without_rescaling_mixed_diffuse(tiny, monkeypatch):
    from lightmapper.formats.bsp import OutMaterial, OutPage
    from test_blender_bake import sun
    tags, settings, work = tiny
    pipeline = api()
    original_scene = pipeline._scene
    original_write = pipeline.write_job

    def chart(groups, shaders, s):
        lm = np.zeros(6, LM)
        lm['texcoord'] = [[0, 0], [1, 0], [0, 1], [1, 0], [1, 1], [0, 1]]
        return [OutPage(2, 2, [OutMaterial(0, groups[0].env, lm,
                    np.array([[0, 1, 2], [3, 4, 5]], np.uint16), np.arange(2))])]

    def scene(*args, **kwargs):
        value = original_scene(*args, **kwargs)
        value[0].tri_ignore_normals = np.array([True, False])
        return value

    def write(work, scene, pages, suns, *args, **kwargs):
        return original_write(work, scene, pages, [sun(strength=np.pi, classes=(-1,))], *args, **kwargs)

    def blender(work, s):
        diffuse = np.repeat(np.float32([[0, 1 / 3], [1 / 3, 1]])[:, :, None], 3, axis=2)
        return BakeResult([diffuse.copy()], np.zeros((6, 3), np.float32),
                          np.ones(6), 'test', [np.ones((2, 2), bool)], diffuse_pages=[diffuse])

    def compiled(*args):
        return vertex_gpu.VertexPassResult(np.zeros((6, 3), np.float32), np.ones(6),
            np.zeros((6, 1), bool), None, 0, np.full((6, 1, 3), 0.3, np.float32)), cpu_info()

    monkeypatch.setattr(pipeline, 'chart_and_pack', chart)
    monkeypatch.setattr(pipeline, '_scene', scene)
    monkeypatch.setattr(pipeline, 'write_job', write)
    monkeypatch.setattr(pipeline, 'run_blender', blender)
    monkeypatch.setattr(vertex_cpu, 'run_in_child', compiled)
    out = pipeline.bake(tags, 'level', 'room', replace(settings, stitch_seams=False, ignore_normals_mode='tool'), work=work, dry_run=True)
    # flagged samples are dark, unflagged samples have diffuse 1; only the independent sun term is added here
    want = np.array([[0.3, 1 / 3 + 0.3 * 2 / 3], [1 / 3 + 0.3 * 2 / 3, 1]])
    np.testing.assert_allclose(out.float_pages[0], np.repeat(want[:, :, None], 3, axis=2), rtol=1e-6)


def test_two_page_bake_preserves_page_order_and_expands_seam_query_inverse(tiny, monkeypatch):
    from lightmapper.formats.bsp import OutMaterial, OutPage
    from lightmapper.formats.bitmap_decode import decode_bitmap

    tags, settings, work = tiny
    bsp = read_tag(tags / 'room.scenario_structure_bsp')
    material = bsp.root.block('lightmaps')[0].block('materials')[0]
    env, lm = unpack_uncompressed(material.data('uncompressed_vertices'), 6, 6)
    env['position'] = [[0, 0, 0], [4, 0, 0], [0, 4, 0], [0, 0, 0], [0, 4, 0], [4, 4, 0]]
    material.set_data('uncompressed_vertices', pack_uncompressed(env, lm))
    material.set_data('compressed_vertices', compress(env, lm))
    put(tags, 'sbsp', 'room', bsp)

    def chart(groups, shaders, s):
        group = groups[0]
        pages = []
        for i in range(2):
            vertices = group.tris[i]
            uv = np.zeros(3, LM)
            uv['texcoord'] = np.float32([[1, 1], [6, 1], [1, 6]]) / 8
            pages.append(OutPage(8, 8, [OutMaterial(0, group.env[vertices], uv,
                np.array([[0, 1, 2]], np.uint16), np.array([i]))]))
        return pages

    def blender(work, s):
        with np.load(work / 'scene.npz') as scene:
            np.testing.assert_array_equal(scene['query_inverse'], [0, 1, 2, 0, 2, 3])
        return BakeResult([np.full((8, 8, 3), c, np.float32) for c in (0.2, 0.7)],
                          np.zeros((4, 3), np.float32), np.ones(4), 'test', [np.ones((8, 8), bool)] * 2)

    def compiled(*args):
        v = np.zeros((4, 3), np.float32)
        v[:, 2] = [0.1, 0.2, 0.3, 0.4]
        return vertex_gpu.VertexPassResult(v, np.ones(4), None, None, 0), cpu_info()

    monkeypatch.setattr(api(), 'chart_and_pack', chart)
    monkeypatch.setattr(api(), 'run_blender', blender)
    monkeypatch.setattr(vertex_cpu, 'run_in_child', compiled)
    out = api().bake(tags, 'level', 'room', replace(settings, dither=False, stitch_seams=False), work=work, dry_run=True)
    expected = [[0.1, 0.2, 0.3], [0.1, 0.3, 0.4]]
    for i, page in enumerate(out.bsp_tag.root.block('lightmaps')[:-1]):
        m = page.block('materials')[0]
        _, lm = unpack_uncompressed(m.data('uncompressed_vertices'), 3, 3)
        np.testing.assert_allclose(lm['incident_radiosity'][:, 2], expected[i])
        np.testing.assert_allclose(out.float_pages[i], (0.2, 0.7)[i])
        np.testing.assert_allclose(decode_bitmap(out.bitmap_tag, i)[..., :3] / 255,
                                   (0.2, 0.7)[i], atol=0.02)


def test_scenery_radiosity_is_cached_by_shader_not_uv_scale(tiny, monkeypatch):
    import lightmapper.pipeline as pipeline

    tags, _, _ = tiny
    put_scenery_over_the_room(tags, 4.)
    gather = pipeline.gather_occluders
    read = pipeline.read_radiosity
    calls = []

    def repeated(*args):
        occ = gather(*args)
        group, path, u, v = occ.shaders[0]
        occ.shaders.append((group, path, u * 2, v * 2))
        return occ

    def counted(tags, group, path):
        calls.append((group, path))
        return read(tags, group, path)

    monkeypatch.setattr(pipeline, 'gather_occluders', repeated)
    monkeypatch.setattr(pipeline, 'read_radiosity', counted)
    scene = scene_of(tags)
    np.testing.assert_allclose(scene.occ_tri_emission, [[4, 2, 1]] * 2)
    assert calls.count(('soso', 'prop_shader')) == 1


@pytest.mark.parametrize('mode', ['tool', 'enhanced'])
def test_enhanced_mode_replaces_flagged_texels_and_adds_the_glow(tiny, monkeypatch, mode):
    run_enhanced_pipeline(tiny, monkeypatch, mode)


def test_glow_pages_leave_out_the_emission_the_pages_hold(tiny, monkeypatch):
    run_enhanced_pipeline(tiny, monkeypatch, 'enhanced', emit=.5)


def test_zero_bounces_skips_the_terminal_glow(tiny, monkeypatch):
    run_enhanced_pipeline(tiny, monkeypatch, 'enhanced', bounces=0)


def run_enhanced_pipeline(tiny, monkeypatch, mode, emit=0., bounces=1):
    from lightmapper.formats.bsp import OutMaterial, OutPage
    tags, settings, work = tiny
    pipeline = api()
    original_scene = pipeline._scene

    def chart(groups, shaders, s):
        lm = np.zeros(6, LM)
        lm['texcoord'] = [[0, 0], [1, 0], [0, 1], [1, 0], [1, 1], [0, 1]]
        return [OutPage(2, 2, [OutMaterial(0, groups[0].env, lm,
                    np.array([[0, 1, 2], [3, 4, 5]], np.uint16), np.arange(2))])]

    def scene(*args, **kwargs):
        value = original_scene(*args, **kwargs)
        value[0].tri_ignore_normals = np.array([True, False])
        return value

    def blender(work, s):
        with np.load(work / 'scene.npz') as arrays:
            count = len(arrays['texel_pos'])
        assert (count > 0) == (mode == 'enhanced')
        raw = np.repeat(np.float32([[0, 1/3], [1/3, 1]])[:, :, None], 3, axis=2)
        return BakeResult([raw + emit], np.zeros((6, 3), np.float32), np.ones(6), 'test',
                          [np.ones((2, 2), bool)], diffuse_pages=[raw])

    def compiled(work, pages, timeout):
        with np.load(work / 'scene.npz') as arrays:
            count = len(arrays['texel_pos'])
        return vertex_gpu.VertexPassResult(np.zeros((6, 3), np.float32), np.ones(6), None,
            None, 0, texel_rgb=np.full((count, 3), .3, np.float32)), cpu_info()

    def glow(work, s, pages):
        assert mode == 'enhanced', 'tool mode must never start a glow run'
        assert bounces > 0, 'zero bounces must not start a reflected glow stage'
        assert pages[0].shape == (6, 6, 3)
        np.testing.assert_allclose(pages[0][0, 0], .3, rtol=1e-6)
        np.testing.assert_allclose(pages[0][2, 2], .3, rtol=1e-6)
        np.testing.assert_array_equal(pages[0][-1, -1], 0)
        unflagged = np.array([[0, 1/3], [1/3, 1]], np.float32)
        return [np.repeat((.6 * unflagged)[:, :, None], 3, axis=2)]

    monkeypatch.setattr(pipeline, 'chart_and_pack', chart)
    monkeypatch.setattr(pipeline, '_scene', scene)
    monkeypatch.setattr(pipeline, 'run_blender', blender)
    monkeypatch.setattr(pipeline, 'run_blender_glow', glow)
    monkeypatch.setattr(vertex_cpu, 'run_in_child', compiled)
    out = pipeline.bake(tags, 'level', 'room', replace(settings, ignore_normals_mode=mode, stitch_seams=False, bounces=bounces),
                        work=work, dry_run=True)
    want = (np.array([[.3, 1/3 + .3*2/3 + .6/3], [1/3 + .3*2/3 + .6/3, 1.6]])
            if mode == 'enhanced' else np.array([[0, 1/3], [1/3, 1]]))
    if mode == 'enhanced' and not bounces:
        want = np.array([[.3, 1/3 + .3*2/3], [1/3 + .3*2/3, 1.]])
    if not emit:
        np.testing.assert_allclose(out.float_pages[0][..., 0], want, rtol=1e-6)


def put_cutout_maps(tags):
    from test_textures import bitmap_tag
    put(tags, 'bitm', 'base', bitmap_tag(1, 1, 'A8R8G8B8', bytes([30, 20, 10, 255])))
    put(tags, 'bitm', 'bump', bitmap_tag(1, 1, 'A8R8G8B8', bytes([255, 128, 128, 0])))


def test_scene_senv_cuts_by_the_bump_maps_alpha(tiny):
    from test_scenery import write_model, write_scenery
    from lightmapper.formats.bitmap import write_bitmap
    from lightmapper.formats.bsp import OutMaterial, OutPage, read_geometry
    from lightmapper.gather import read_radiosity

    tags, _, _ = tiny
    shader = read_tag(tags / 'wall.shader_environment')
    shader.root.set('properties.flags', 1)
    shader.root.set_reference('diffuse.base_map', 'bitm', 'base')
    shader.root.set_reference('bump.bump_map', 'bitm', 'bump')
    put(tags, 'senv', 'wall', shader)
    put_cutout_maps(tags)
    write_model(tags, 'model', [[(0, [(0, 0, 0), (1, 0, 0), (0, 1, 0)], [0, 1, 2])]], shaders=(('senv', 'wall'),))
    write_scenery(tags, 'prop', 'model')
    scenario = read_tag(tags / 'level.scenario')
    palette = child(scenario.root, 'scenery_palette')
    palette.set_reference('name', 'scen', 'prop')
    scenario.root.set_block('scenery_palette', [palette])
    placement = child(scenario.root, 'scenery')
    placement.set('bsp_indices', 1)
    placement.set('placement.position', (10., 0., 0.))
    scenario.root.set_block('scenery', [placement])
    geometry = read_geometry(read_tag(tags / 'room.scenario_structure_bsp'))
    group = geometry.groups[0]
    material = OutMaterial(0, group.env, np.zeros(len(group.env), LM), group.tris, group.surface_ids)
    scene, _, textures = api()._scene(tags, scenario, 0, geometry, [OutPage(8, 8, [material])],
        [read_radiosity(tags, 'senv', 'wall')], np.array([-1, 2], np.int16))
    assert scene.tri_cutout.tolist() == [True, True] and scene.occ_tri_cutout.tolist() == [True]
    cut = textures.arrays()[scene.tri_texture[0]]
    assert cut[0, 0].tolist() == [10, 20, 30, 0]
    assert scene.occ_tri_texture.tolist() == [scene.tri_texture[0]]


def test_unlit_senv_emitter_takes_the_composed_texture(tiny):
    tags, _, _ = tiny
    glow = read_tag(tags / 'wall.shader_environment')
    glow.root.set('properties.flags', 1)
    glow.root.set('radiosity.power', 10.)
    glow.root.set_reference('diffuse.base_map', 'bitm', 'base')
    glow.root.set_reference('bump.bump_map', 'bitm', 'bump')
    put(tags, 'senv', 'glow_cut', glow)
    put_cutout_maps(tags)
    add_unlit_quad(tags, 'glow_cut', group='senv')
    scene, textures = scene_of(tags, with_textures=True)
    assert scene.occ_tri_cutout.tolist() == [True, True]
    assert all(textures.arrays()[k][0, 0, 3] == 0 for k in scene.occ_tri_texture)


def test_senv_bump_hole_lets_the_compiled_cpu_pass_ray_through(tiny):
    from lightmapper import vertex_cpu
    from lightmapper.blender_job import write_job
    from lightmapper.formats.bsp import OutMaterial, OutPage, read_geometry
    from lightmapper.gather import read_radiosity
    from test_vertex_cpu import load_cpu_scene

    tags, settings, work = tiny
    shader = read_tag(tags / 'wall.shader_environment')
    shader.root.set('properties.flags', 1)
    shader.root.set_reference('diffuse.base_map', 'bitm', 'base')
    shader.root.set_reference('bump.bump_map', 'bitm', 'bump')
    put(tags, 'senv', 'wall', shader)
    put_cutout_maps(tags)
    geometry = read_geometry(read_tag(tags / 'room.scenario_structure_bsp'))
    group = geometry.groups[0]
    material = OutMaterial(0, group.env, np.zeros(len(group.env), LM), group.tris, group.surface_ids)
    scene, _, textures = api()._scene(tags, read_tag(tags / 'level.scenario'), 0, geometry,
        [OutPage(8, 8, [material])], [read_radiosity(tags, 'senv', 'wall')], np.array([-1, 2], np.int16))
    write_job(work, scene, [(8, 8)], [], {}, settings, 'all', textures=textures.arrays())
    np.savez(work / 'pages.npz', page_0=np.zeros((8, 8, 3), np.float32))
    cs, bvh = load_cpu_scene(work)
    tri, _ = vertex_cpu._cast(cs.arrays(), (bvh.lo, bvh.hi, bvh.first, bvh.count, bvh.order),
                              np.float32([1, 1, 0.5]), np.float32([0, 0, -1]), False, np.ones(3))
    assert tri == -1
