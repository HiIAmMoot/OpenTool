import threading
import time
from pathlib import Path

import pytest

tk = pytest.importorskip('tkinter')

from test_pipeline import tiny
from lightmapper.blender_find import Blender, BlenderNotFound
from lightmapper.gpu import Adapter, DeviceError
from lightmapper.charts import CapError
from lightmapper.memory import SceneCounts, out_of_memory_message
from lightmapper.pipeline import SceneEstimate
from opentool.app import App
from opentool.devices import result_line
from opentool.runner import BakeOutcome
from opentool import app as app_module
from opentool import presets
from opentool import usersettings
from opentool import form as form_module
from opentool import errors, runner


ESTIMATE_FITS = SceneEstimate(SceneCounts(2, 4), [(8, 16)])


@pytest.fixture(scope='module')
def tk_root():
    root = tk.Tk()
    root.withdraw()
    yield root
    root.destroy()


RTX = Adapter('NVIDIA GeForce RTX 4060 Laptop GPU', 0x10DE, (1,), 'vulkan', True)
RADEON = Adapter('AMD Radeon(TM) Graphics', 0x1002, (2,), 'vulkan', True)


@pytest.fixture
def make_app(tmp_path, monkeypatch, request, tk_root):
    roots = []

    def create(**kwargs):
        monkeypatch.setenv('APPDATA', str(tmp_path))
        monkeypatch.setenv('LOCALAPPDATA', str(tmp_path))
        root = tk.Toplevel(tk_root)
        root.withdraw()
        roots.append(root)
        store_path = kwargs.pop('store_path', tmp_path / 's.json')
        log_path = kwargs.pop('log_path', tmp_path / 'opentool.log')
        find_blender = kwargs.pop(
            'find_blender',
            lambda remembered=None: Blender(Path('C:/b/blender.exe'), (5, 2)),
        )
        kwargs.setdefault('probe_gpu', lambda: ([RTX], None))
        kwargs.setdefault('probe_denoiser', lambda: {'CUDA': True, 'HIP': False})
        kwargs.setdefault('estimate', lambda _inputs: ESTIMATE_FITS)
        app = App(
            root,
            store_path=store_path,
            log_path=log_path,
            find_blender=find_blender,
            **kwargs,
        )
        return app

    def destroy_roots():
        for root in roots:
            try:
                root.destroy()
            except tk.TclError:
                pass

    request.addfinalizer(destroy_roots)
    return create


@pytest.fixture
def app(make_app):
    return make_app()


def pump(root, condition, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        root.update()
        if condition():
            return
        time.sleep(0.01)
    pytest.fail('condition did not become true before timeout')


def test_first_screen_widgets(app):
    for name in ('tags_entry', 'scenario_entry', 'bsp_box', 'bake_button', 'progress', 'log_box', 'open_button'):
        assert hasattr(app, name)
    assert str(app.bake_button['state']) == 'disabled' and str(app.open_button['state']) == 'disabled'
    assert app.new_names_var.get() is False and app.preset_var.get() == 'Normal'
    assert app.verbose_var.get() is False
    assert app.backups_var.get() is True
    assert app.backups_check['text'] == 'Keep backups'
    assert app.backups_check.grid_info()['row'] == app.new_names_check.grid_info()['row']
    assert app.tags_label['text'] == 'HCEK tags folder'
    assert tuple(map(int, app.advanced_entries['page_max']['values'])) == (256, 512, 1024, 2048, 4096)
    assert app.new_names_hint['text'] == "Replaces your level's tags. Backups go in a backups folder next to them."
    assert app.advanced_frame.winfo_manager() == ''
    app._toggle_advanced()
    assert app.advanced_frame.winfo_manager() == 'grid'


def test_new_names_hint_tracks_checkbox(app):
    assert app.new_names_hint['text'] == "Replaces your level's tags. Backups go in a backups folder next to them."
    app.new_names_check.invoke()
    assert app.new_names_hint['text'] == 'Saves the lit level as <name>_lit and leaves your tags as they are.'
    app.new_names_check.invoke()
    assert app.new_names_hint['text'] == "Replaces your level's tags. Backups go in a backups folder next to them."
    app.backups_check.invoke()
    assert app.new_names_hint['text'] == "Replaces your level's tags. No backups are kept."


def test_single_bsp_is_selected_and_bake_enabled(app, tiny):
    tags, *_ = tiny
    (tags / 'globals').mkdir()
    app.set_tags(str(tags))
    app.set_scenario_file(tags / 'level.scenario')
    pump(app.root, lambda: app.bsp_box.get() == 'room' and app.blender is not None)
    assert str(app.bake_button['state']) == 'normal'


def test_request_carries_the_form(app, tiny):
    tags, *_ = tiny
    (tags / 'globals').mkdir()
    app.set_tags(str(tags))
    app.set_scenario_file(tags / 'level.scenario')
    pump(app.root, lambda: app.bsp_box.get() == 'room' and app.blender is not None)
    app._select_preset('Draft')
    req = app.request()
    assert (req.scenario, req.bsp, req.new_names, req.settings.samples, req.settings.timeout_s) == ('level', 'room', False, 256, None)
    assert req.backups is True
    app.backups_var.set(False)
    assert app.request().backups is False
    app.verbose_var.set(True)
    assert app.request().verbose is True


def test_draft_fills_all_advanced_fields(app):
    app._select_preset('Draft')
    assert {key: value.get() for key, value in app.advanced_vars.items()} == dict(texels_per_unit='1', samples='256', vertex_rays='1024', sky_grid='2', bounces='1', page_max='512')


def test_advanced_edit_selects_custom_and_restores(tmp_path, make_app):
    store = tmp_path / 'settings.json'
    app = make_app(store_path=store)
    app._select_preset('Draft')
    app.advanced_vars['texels_per_unit'].set('3')
    assert app.preset_var.get() == 'Custom'
    app.remember()
    restored = make_app(store_path=store)
    assert restored.preset_var.get() == 'Custom'
    assert restored.advanced_vars['texels_per_unit'].get() == '3'


def test_equivalent_texel_value_keeps_named_preset(app):
    app.advanced_vars['texels_per_unit'].set('2')
    assert app.preset_var.get() == 'Normal'


def test_blank_custom_field_shows_label_error(app, tiny):
    tags, *_ = tiny
    (tags / 'globals').mkdir()
    app.set_tags(str(tags)); app.set_scenario_file(tags / 'level.scenario')
    pump(app.root, lambda: app.blender is not None)
    app._select_preset('Custom')
    app.advanced_vars['samples'].set('')
    with pytest.raises(ValueError, match='Rays per texel is empty'):
        app.request()
    assert app.advanced_error['text'] == 'Rays per texel is empty. Fill it in, or pick a preset.'


def test_bad_advanced_value_blocks_the_request(app, tiny):
    tags, *_ = tiny
    (tags / 'globals').mkdir()
    app.set_tags(str(tags))
    app.set_scenario_file(tags / 'level.scenario')
    pump(app.root, lambda: app.bsp_box.get() == 'room' and app.blender is not None)
    app.advanced_vars['bounces'].set('99')
    with pytest.raises(ValueError, match='Bounces'):
        app.request()


def test_settings_are_remembered(tmp_path, tiny, make_app):
    tags, *_ = tiny
    (tags / 'globals').mkdir()
    a = make_app(store_path=tmp_path / 's.json')
    a.set_tags(str(tags))
    a.set_scenario_file(tags / 'level.scenario')
    pump(a.root, lambda: a.bsp_box.get() == 'room' and a.blender is not None)
    a._select_preset('High')
    a.verbose_var.set(True)
    a.backups_var.set(False)
    a.remember()
    b = make_app(store_path=tmp_path / 's.json')
    assert b.preset_var.get() == 'High' and b.tags_entry.get() == str(tags)
    assert b.verbose_var.get() is True
    assert b.backups_var.get() is False


def test_missing_blender_shows_the_message(make_app):
    def nothing(remembered=None):
        raise BlenderNotFound('Blender 5.2 or newer was not found.')

    a = make_app(find_blender=nothing)
    pump(a.root, lambda: 'not found' in a.blender_label['text'])
    assert str(a.bake_button['state']) == 'disabled'


def test_discovery_result_comes_through_the_queue(make_app):
    threads = []

    def slow(remembered=None):
        threads.append(threading.current_thread())
        return Blender(Path('C:/b/blender.exe'), (5, 2))

    a = make_app(find_blender=slow)
    pump(a.root, lambda: a.blender is not None)
    assert threads[0] is not threading.main_thread()


def test_callback_error_goes_to_the_log_not_a_dialog(app, tmp_path):
    app.tk_root.report_callback_exception(ZeroDivisionError, ZeroDivisionError('x'), None)
    assert 'ZeroDivisionError' in (tmp_path / 'opentool.log').read_text()
    assert 'Something went wrong' in app.stage_label['text']


def test_change_rejecting_old_blender_keeps_the_working_one(app, monkeypatch, tmp_path, tiny):
    tags, *_ = tiny
    (tags / 'globals').mkdir()
    app.set_tags(str(tags))
    app.set_scenario_file(tags / 'level.scenario')
    pump(app.root, lambda: app.bsp_box.get() == 'room' and app.blender is not None)
    previous = app.blender
    picked = tmp_path / 'old-blender.exe'
    monkeypatch.setattr(app_module.filedialog, 'askopenfilename', lambda **_kwargs: str(picked))
    monkeypatch.setattr(app_module.blender_find, 'blender_version', lambda _path: (4, 1))
    app._choose_blender()
    pump(app.root, lambda: app.blender_label['text'] != 'Checking Blender…')
    assert app.blender is previous
    assert app.blender_label['text'] == 'That Blender is 4.1. OpenTool needs Blender 5.2 or newer.'
    assert str(app.bake_button['state']) == 'normal'


def test_change_rejecting_non_blender_keeps_the_working_one(app, monkeypatch, tmp_path, tiny):
    tags, *_ = tiny
    (tags / 'globals').mkdir()
    app.set_tags(str(tags))
    app.set_scenario_file(tags / 'level.scenario')
    pump(app.root, lambda: app.bsp_box.get() == 'room' and app.blender is not None)
    previous = app.blender
    picked = tmp_path / 'not-blender.exe'
    monkeypatch.setattr(app_module.filedialog, 'askopenfilename', lambda **_kwargs: str(picked))
    monkeypatch.setattr(app_module.blender_find, 'blender_version', lambda _path: None)
    app._choose_blender()
    pump(app.root, lambda: app.blender_label['text'] != 'Checking Blender…')
    assert app.blender is previous
    assert app.blender_label['text'] == "That file doesn't look like Blender. Pick blender.exe from the Blender folder."
    assert str(app.bake_button['state']) == 'normal'


def test_change_accepting_new_blender_takes_it(app, monkeypatch, tmp_path):
    pump(app.root, lambda: app.blender is not None)
    picked = tmp_path / 'new-blender.exe'
    monkeypatch.setattr(app_module.filedialog, 'askopenfilename', lambda **_kwargs: str(picked))
    monkeypatch.setattr(app_module.blender_find, 'blender_version', lambda _path: (5, 3))
    app._choose_blender()
    pump(app.root, lambda: app.blender is not None and app.blender.version == (5, 3))
    assert app.blender.path == picked
    assert 'Blender 5.3' in app.blender_label['text']


def test_empty_form_rows_are_hidden_and_tags_error_shows(app, tiny):
    for widget in (app.tags_error, app.scenario_error, app.bsp_message, app.device_label):
        assert widget.winfo_manager() == ''
    tags, *_ = tiny
    app.set_tags(str(tags / 'missing'))
    assert app.tags_error.winfo_manager() == 'grid'


def test_close_destroys_window_when_saving_fails(app, monkeypatch, tmp_path):
    def fail_save(*_args, **_kwargs):
        raise OSError('disk full')

    monkeypatch.setattr(usersettings, 'save', fail_save)
    window = app.window
    app.on_close()
    assert not window.winfo_exists()
    assert 'disk full' in (tmp_path / 'opentool.log').read_text()


def test_bad_remembered_scenario_is_logged_and_ignored(tmp_path, tiny, make_app, monkeypatch):
    tags, *_ = tiny
    (tags / 'globals').mkdir()
    invalid = tags / 'bad.scenario'
    invalid.write_bytes(b'not a scenario tag')
    usersettings.save(
        usersettings.UserSettings(tags=str(tags), scenario='bad'),
        tmp_path / 's.json',
    )
    def invalid_scenario(*_args):
        raise TypeError(f'invalid scenario tag: {invalid}')

    monkeypatch.setattr(form_module, 'list_bsps', invalid_scenario)
    app = make_app()
    assert app.tags_entry.get() == str(tags)
    assert app.scenario_var.get() == ''
    assert app.scenario_error['text'] == ''
    pump(app.root, lambda: app.blender is not None)
    assert 'bad.scenario' in (tmp_path / 'opentool.log').read_text()


def test_stale_settings_start_with_empty_tags_and_scenario(tmp_path, make_app):
    usersettings.save(
        usersettings.UserSettings(
            tags=str(tmp_path / 'removed-tags'),
            scenario='levels\\gone\\gone',
        ),
        tmp_path / 's.json',
    )
    app = make_app()
    assert app.tags_entry.get() == ''
    assert app.scenario_var.get() == ''
    assert app.tags_error['text'] == ''
    assert app.scenario_error['text'] == ''


def test_tags_entry_commits_on_return(app, tiny):
    tags, *_ = tiny
    app.tags_entry.insert(0, str(tags))
    assert app.tags_entry.bind('<Return>')
    app._commit_tags()
    assert app._tags == tags


def test_change_result_wins_over_late_discovery(make_app, monkeypatch, tmp_path):
    release_discovery = threading.Event()
    discovery_finished = threading.Event()

    def delayed_discovery(remembered=None):
        release_discovery.wait(5)
        discovery_finished.set()
        return Blender(Path('C:/b/discovered.exe'), (5, 2))

    app = make_app(find_blender=delayed_discovery)
    picked = tmp_path / 'picked.exe'
    monkeypatch.setattr(app_module.filedialog, 'askopenfilename', lambda **_kwargs: str(picked))
    monkeypatch.setattr(app_module.blender_find, 'blender_version', lambda _path: (5, 3))
    app._choose_blender()
    pump(app.root, lambda: app.blender is not None and app.blender.version == (5, 3))
    release_discovery.set()
    assert discovery_finished.wait(5)
    pump(app.root, lambda: app.events.empty())
    time.sleep(0.15)
    app.root.update()
    assert app.blender.path == picked


def test_rejected_pick_does_not_discard_discovery(make_app, monkeypatch, tmp_path, tiny):
    release = threading.Event()
    def discovery(_remembered=None):
        release.wait(5)
        return Blender(Path('C:/b/discovered.exe'), (5, 2))
    app = make_app(find_blender=discovery)
    monkeypatch.setattr(app_module.filedialog, 'askopenfilename', lambda **_kwargs: str(tmp_path / 'old.exe'))
    monkeypatch.setattr(app_module.blender_find, 'blender_version', lambda _path: (4, 1))
    app._choose_blender()
    pump(app.root, lambda: app.blender_label['text'] != 'Checking Blender…')
    release.set()
    pump(app.root, lambda: app.blender is not None)
    tags, *_ = tiny
    (tags / 'globals').mkdir()
    app.set_tags(str(tags)); app.set_scenario_file(tags / 'level.scenario')
    pump(app.root, lambda: app.memory_label['text'].startswith('Needs about'))
    assert app.blender.version == (5, 2) and str(app.bake_button['state']) == 'normal'


def test_garbage_scenario_clears_previous_bsp_and_disables_bake(app, tiny, monkeypatch):
    tags, *_ = tiny
    (tags / 'globals').mkdir()
    app.set_tags(str(tags)); app.set_scenario_file(tags / 'level.scenario')
    pump(app.root, lambda: app.blender is not None)
    garbage = tags / 'garbage.scenario'; garbage.write_bytes(b'not read by patched list_bsps')
    monkeypatch.setattr(form_module, 'list_bsps', lambda *_args: (_ for _ in ()).throw(TypeError('wrong tag group')))
    app.set_scenario_file(garbage)
    assert app._scenario == '' and tuple(app.bsp_box['values']) == () and app.bsp_var.get() == ''
    assert str(app.bake_button['state']) == 'disabled' and app.scenario_error['text']


def test_discovery_worker_does_not_capture_app(make_app, monkeypatch):
    targets = []

    class UnstartedThread:
        def __init__(self, *, target, **_kwargs):
            targets.append(target)

        def start(self):
            pass

    monkeypatch.setattr(app_module.threading, 'Thread', UnstartedThread)
    app = make_app()
    cells = [cell.cell_contents for cell in targets[0].__closure__ or ()]
    assert all(value is not app for value in cells)


def test_scenario_dialog_uses_levels_folder_then_tags(app, monkeypatch, tmp_path):
    tags = tmp_path / 'tags'
    (tags / 'globals').mkdir(parents=True)
    (tags / 'levels').mkdir()
    app.set_tags(str(tags))
    seen = []
    monkeypatch.setattr(
        app_module.filedialog, 'askopenfilename',
        lambda **kwargs: seen.append(kwargs['initialdir']) or '',
    )
    app._browse_scenario()
    assert Path(seen[-1]) == tags / 'levels'
    (tags / 'levels').rmdir()
    app._browse_scenario()
    assert Path(seen[-1]) == tags


def test_open_output_button_follows_log_box(app):
    assert app.open_button.master is app.frame
    assert int(app.open_button.grid_info()['row']) > int(app.log_box.grid_info()['row'])


OUTCOME_DONE = BakeOutcome('done', 'Done. The lit level is level.', [], Path('C:/tags/levels/test/room'), Path(), 'OPTIX')
REPLACE_TEXT = 'This replaces room and its lightmaps in place. {} Continue?'


@pytest.fixture
def ready_app(app, tiny):
    tags, *_ = tiny
    (tags / 'globals').mkdir()
    app.set_tags(str(tags))
    app.set_scenario_file(tags / 'level.scenario')
    pump(app.root, lambda: app.bsp_box.get() == 'room' and app.blender is not None
         and app.memory_label['text'].startswith('Needs about'))
    app.new_names_var.set(True)
    return app


def pump_app(app, condition, timeout=5):
    pump(app.root, condition, timeout)


def run_events(app, script):
    def fake(req, emit):
        for event in script:
            emit(event)
    app._run_bake = fake
    app.start_bake()
    pump_app(app, lambda: not app.busy)


def test_bake_runs_off_the_ui_thread(ready_app):
    seen = {}

    def fake(req, emit):
        seen['thread'] = threading.current_thread()
        emit(('done', OUTCOME_DONE))

    ready_app._run_bake = fake
    ready_app.start_bake()
    pump_app(ready_app, lambda: 'thread' in seen)
    assert seen['thread'] is not threading.main_thread()


def test_bake_button_disabled_while_running(ready_app):
    gate, calls = threading.Event(), []

    def fake(req, emit):
        calls.append(req)
        gate.wait(5)
        emit(('done', OUTCOME_DONE))

    ready_app._run_bake = fake
    ready_app.start_bake()
    assert str(ready_app.bake_button['state']) == 'disabled' and ready_app.busy
    ready_app.start_bake()
    gate.set()
    pump_app(ready_app, lambda: not ready_app.busy)
    assert len(calls) == 1
    assert str(ready_app.bake_button['state']) == 'normal'


def test_form_inputs_are_locked_during_a_bake_and_restored_after(ready_app):
    gate = threading.Event()

    def fake(req, emit):
        gate.wait(5)
        emit(('done', OUTCOME_DONE))

    ready_app._run_bake = fake
    ready_app.start_bake()
    for widget in (ready_app.tags_entry, ready_app.bsp_box, ready_app.new_names_check, ready_app.backups_check,
                   ready_app.advanced_entries['samples'], ready_app.advanced_entries['page_max']):
        assert str(widget['state']) == 'disabled'
    gate.set()
    pump_app(ready_app, lambda: not ready_app.busy)
    assert str(ready_app.tags_entry['state']) == 'normal'
    assert str(ready_app.bsp_box['state']) == 'readonly'
    assert str(ready_app.scenario_entry['state']) == 'readonly'
    assert str(ready_app.advanced_entries['page_max']['state']) == 'readonly'
    assert str(ready_app.new_names_check['state']) == 'normal'


def test_close_during_write_waits_for_the_write(ready_app, monkeypatch):
    monkeypatch.setattr(app_module.messagebox, 'askyesno', lambda *a: True)
    exits = []
    ready_app._exit_now = lambda: exits.append(1)
    gate = threading.Event()

    def fake(req, emit):
        emit(('stage', 'write'))
        gate.wait(5)
        emit(('stage', 'validate'))
        emit(('done', OUTCOME_DONE))

    ready_app._run_bake = fake
    ready_app.start_bake()
    pump_app(ready_app, lambda: ready_app.stage == 'write')
    ready_app.on_close()
    pump_app(ready_app, lambda: 'Finishing writing tags' in ready_app.stage_label['text'])
    assert exits == []
    gate.set()
    pump_app(ready_app, lambda: exits == [1])
    pump_app(ready_app, lambda: not ready_app.busy)
    assert exits == [1]


def test_close_during_blender_exits_at_once(ready_app, monkeypatch):
    asked = []
    monkeypatch.setattr(app_module.messagebox, 'askyesno', lambda *a: asked.append(a) or True)
    exits = []
    ready_app._exit_now = lambda: exits.append(1)
    gate = threading.Event()

    def fake(req, emit):
        emit(('stage', 'blender'))
        gate.wait(5)
        emit(('done', OUTCOME_DONE))

    ready_app._run_bake = fake
    ready_app.start_bake()
    pump_app(ready_app, lambda: ready_app.stage == 'blender')
    ready_app.on_close()
    assert exits == [1]
    assert asked == [('Stop the bake?', "A bake is running. Closing OpenTool stops it, and no tags will be written if it hasn't reached that step yet.")]
    gate.set()
    pump_app(ready_app, lambda: not ready_app.busy)


def test_close_during_bake_asks_first(ready_app, monkeypatch):
    monkeypatch.setattr(app_module.messagebox, 'askyesno', lambda *a: False)
    exits = []
    ready_app._exit_now = lambda: exits.append(1)
    gate = threading.Event()

    def fake(req, emit):
        gate.wait(5)
        emit(('done', OUTCOME_DONE))

    ready_app._run_bake = fake
    ready_app.start_bake()
    ready_app.on_close()
    assert exits == [] and ready_app.busy and ready_app.window.winfo_exists()
    gate.set()
    pump_app(ready_app, lambda: not ready_app.busy)


def test_close_when_idle_saves_and_destroys_without_asking(ready_app, monkeypatch):
    monkeypatch.setattr(app_module.messagebox, 'askyesno', lambda *a: pytest.fail('asked'))
    window = ready_app.window
    ready_app.on_close()
    assert not window.winfo_exists()


def test_events_drive_label_bar_and_log(ready_app):
    gate = threading.Event()

    def fake(req, emit):
        for event in [('stage', 'charts'), ('log', 'INFO: stage: charts'), ('stage', 'blender'),
                      ('device', 'HIP'), ('pages', 1, 2)]:
            emit(event)
        gate.wait(5)
        emit(('done', OUTCOME_DONE))

    ready_app._run_bake = fake
    ready_app.start_bake()
    pump_app(ready_app, lambda: 'page 1 of 2' in ready_app.stage_label['text'])
    assert ready_app.stage_label['text'] == 'Baking in Blender: page 1 of 2'
    assert ready_app.progress['value'] == pytest.approx(1000 * (0.20 + 0.60 * 1 / 2))
    assert ready_app.device_label['text'] == 'Baking on HIP'
    gate.set()
    pump_app(ready_app, lambda: not ready_app.busy)
    assert 'stage: charts' in ready_app.log_box.get('1.0', 'end')
    assert 'HIP' in ready_app.device_label['text'] and ready_app.progress['value'] == 1000
    assert ready_app.stage_label['text'] == OUTCOME_DONE.message
    assert str(ready_app.open_button['state']) == 'normal'


def test_stage_event_sets_label_and_bar(ready_app):
    gate = threading.Event()

    def fake(req, emit):
        emit(('stage', 'charts'))
        gate.wait(5)
        emit(('done', OUTCOME_DONE))

    ready_app._run_bake = fake
    ready_app.start_bake()
    pump_app(ready_app, lambda: ready_app.stage_label['text'] == 'Laying out lightmap pages')
    assert ready_app.stage == 'charts'
    assert ready_app.progress['value'] == pytest.approx(50)
    gate.set()
    pump_app(ready_app, lambda: not ready_app.busy)


def test_cpu_device_label_warns_it_is_slow(ready_app):
    gate = threading.Event()

    def fake(req, emit):
        emit(('device', 'CPU'))
        gate.wait(5)
        emit(('done', OUTCOME_DONE))

    ready_app._run_bake = fake
    ready_app.start_bake()
    pump_app(ready_app, lambda: ready_app.device_label['text'])
    assert ready_app.device_label['text'] == 'Baking on the CPU. This can take a long time.'
    gate.set()
    pump_app(ready_app, lambda: not ready_app.busy)


def test_failed_bake_shows_the_message(ready_app):
    failed = BakeOutcome('failed', 'Blender could not be started. Check the path.', [], None, Path(), None)
    run_events(ready_app, [('done', failed)])
    assert 'could not be started' in ready_app.stage_label['text']
    assert str(ready_app.stage_label.cget('foreground')) == 'red'
    assert str(ready_app.open_button['state']) == 'disabled'
    assert ready_app.progress['value'] == 1000


def test_problems_are_listed_in_the_log(ready_app):
    outcome = BakeOutcome('problems', 'Finished with problems.', ['bad uv 1', 'bad uv 2'], Path('C:/x'), Path(), None)
    run_events(ready_app, [('done', outcome)])
    log = ready_app.log_box.get('1.0', 'end')
    assert 'Problems' in log and 'bad uv 1' in log and 'bad uv 2' in log
    assert str(ready_app.stage_label.cget('foreground')) != 'red'


def test_log_keeps_the_last_5000_lines(ready_app):
    run_events(ready_app, [('log', f'line {i}') for i in range(5100)] + [('done', OUTCOME_DONE)])
    lines = ready_app.log_box.get('1.0', 'end-1c').splitlines()
    assert len(lines) == 5000 and lines[0] == 'line 100' and lines[-1] == 'line 5099'


def test_a_new_bake_clears_the_log_and_output_button(ready_app):
    run_events(ready_app, [('log', 'old line'), ('device', 'OPTIX'), ('done', OUTCOME_DONE)])
    gate = threading.Event()

    def fake(req, emit):
        gate.wait(5)
        emit(('done', OUTCOME_DONE))

    ready_app._run_bake = fake
    ready_app.start_bake()
    assert 'old line' not in ready_app.log_box.get('1.0', 'end')
    assert str(ready_app.open_button['state']) == 'disabled'
    assert ready_app.device_label['text'] == ''
    gate.set()
    pump_app(ready_app, lambda: not ready_app.busy)


def test_open_output_folder_opens_the_outcome_folder(ready_app, monkeypatch):
    opened = []
    monkeypatch.setattr(app_module.os, 'startfile', opened.append, raising=False)
    run_events(ready_app, [('done', OUTCOME_DONE)])
    ready_app.open_button.invoke()
    assert opened == [OUTCOME_DONE.output_folder]


def test_invalid_form_shows_the_error_and_starts_nothing(ready_app):
    ready_app.bsp_var.set('')
    ready_app._run_bake = lambda *a: pytest.fail('started')
    ready_app.start_bake()
    assert ready_app.stage_label['text'] == 'Pick a BSP.' and not ready_app.busy


def test_starting_a_bake_remembers_the_form(ready_app):
    run_events(ready_app, [('done', OUTCOME_DONE)])
    assert usersettings.load(ready_app.store_path).bsp == 'room'


@pytest.mark.parametrize('backups, tail', [
    (True, 'Backups go in a backups folder next to them.'),
    (False, 'No backups are kept.'),
])
def test_replacing_originals_asks_first(ready_app, monkeypatch, backups, tail):
    asked = []
    monkeypatch.setattr(app_module.messagebox, 'askyesno', lambda *a: asked.append(a) or False)
    ready_app._run_bake = lambda *a: pytest.fail('started')
    ready_app.new_names_var.set(False)
    ready_app.backups_var.set(backups)
    ready_app.start_bake()
    assert asked == [('Replace your tags?', REPLACE_TEXT.format(tail))]
    assert not ready_app.busy


def test_replacing_originals_goes_ahead_on_yes(ready_app, monkeypatch):
    monkeypatch.setattr(app_module.messagebox, 'askyesno', lambda *a: True)
    ready_app.new_names_var.set(False)
    calls = []
    ready_app._run_bake = lambda req, emit: (calls.append(req), emit(('done', OUTCOME_DONE)))
    ready_app.start_bake()
    pump_app(ready_app, lambda: not ready_app.busy)
    assert len(calls) == 1


def test_new_names_bake_does_not_ask(ready_app, monkeypatch):
    monkeypatch.setattr(app_module.messagebox, 'askyesno', lambda *a: pytest.fail('asked'))
    run_events(ready_app, [('done', OUTCOME_DONE)])


def test_close_before_the_write_starts_never_lets_it_start(ready_app, monkeypatch):
    monkeypatch.setattr(app_module.messagebox, 'askyesno', lambda *a: True)
    exits = []
    ready_app._exit_now = lambda: exits.append(1)
    go, passed = threading.Event(), []

    def fake(req, emit):
        emit(('stage', 'finish'))
        go.wait(5)
        emit(('stage', 'write'))
        passed.append(1)

    ready_app._run_bake = fake
    ready_app.start_bake()
    pump_app(ready_app, lambda: ready_app.stage == 'finish')
    ready_app.on_close()
    assert exits == [1]
    go.set()
    deadline = time.monotonic() + 0.5
    while time.monotonic() < deadline:
        ready_app.root.update()
        time.sleep(0.01)
    assert passed == [] and ready_app.stage == 'finish'


def test_close_right_after_the_write_started_waits_for_it(ready_app, monkeypatch):
    monkeypatch.setattr(app_module.messagebox, 'askyesno', lambda *a: True)
    exits = []
    ready_app._exit_now = lambda: exits.append(1)
    emitted, gate = threading.Event(), threading.Event()

    def fake(req, emit):
        emit(('stage', 'write'))
        emitted.set()
        gate.wait(5)
        emit(('stage', 'validate'))
        emit(('done', OUTCOME_DONE))

    ready_app._run_bake = fake
    ready_app.start_bake()
    assert emitted.wait(5)
    # no poll has run yet, so the window still thinks the stage is not 'write'
    assert ready_app.stage != 'write'
    ready_app.on_close()
    assert exits == []
    gate.set()
    pump_app(ready_app, lambda: exits == [1])


def test_poller_survives_a_bad_event(ready_app):
    run_events(ready_app, [('stage', 'no-such-stage'), ('done', OUTCOME_DONE)])
    assert not ready_app.busy
    assert str(ready_app.tags_entry['state']) == 'normal'
    assert str(ready_app.bake_button['state']) == 'normal'


def test_failed_settings_save_does_not_block_the_bake(ready_app, monkeypatch, tmp_path):
    def fail_save(*_args, **_kwargs):
        raise OSError('disk full')

    monkeypatch.setattr(usersettings, 'save', fail_save)
    run_events(ready_app, [('done', OUTCOME_DONE)])
    assert ready_app.stage_label['text'] == OUTCOME_DONE.message
    assert 'disk full' in (tmp_path / 'opentool.log').read_text()


def test_pages_label_drops_the_total_when_unknown(ready_app):
    gate = threading.Event()

    def fake(req, emit):
        emit(('pages', 3, 0))
        gate.wait(5)
        emit(('done', OUTCOME_DONE))

    ready_app._run_bake = fake
    ready_app.start_bake()
    pump_app(ready_app, lambda: ready_app.stage_label['text'].startswith('Baking in Blender'))
    assert ready_app.stage_label['text'] == 'Baking in Blender: page 3'
    gate.set()
    pump_app(ready_app, lambda: not ready_app.busy)


def test_failed_bake_unlocks_bake_and_form(ready_app):
    failed = BakeOutcome('failed', 'Blender could not be started.', [], None, Path(), None)
    run_events(ready_app, [('done', failed)])
    assert str(ready_app.bake_button['state']) == 'normal'
    assert str(ready_app.tags_entry['state']) == 'normal'
    assert str(ready_app.new_names_check['state']) == 'normal'


def test_finishing_label_survives_later_events(ready_app, monkeypatch):
    monkeypatch.setattr(app_module.messagebox, 'askyesno', lambda *a: True)
    ready_app._exit_now = lambda: None
    gate, end = threading.Event(), threading.Event()

    def fake(req, emit):
        emit(('stage', 'write'))
        gate.wait(5)
        emit(('pages', 1, 2))
        emit(('log', 'late line'))
        end.wait(5)
        emit(('stage', 'validate'))
        emit(('done', OUTCOME_DONE))

    ready_app._run_bake = fake
    ready_app.start_bake()
    pump_app(ready_app, lambda: ready_app.stage == 'write')
    ready_app.on_close()
    gate.set()
    pump_app(ready_app, lambda: 'late line' in ready_app.log_box.get('1.0', 'end'))
    assert ready_app.stage_label['text'] == 'Finishing writing tags…'
    end.set()
    pump_app(ready_app, lambda: not ready_app.busy)


def test_idle_close_saves_the_settings(ready_app):
    ready_app.on_close()
    assert usersettings.load(ready_app.store_path).bsp == 'room'


def probe_for(adapters, reason=None):
    return lambda: (adapters, reason)


def test_device_row_shows_the_probed_gpu(make_app):
    a = make_app(probe_gpu=probe_for([RTX, RADEON]))
    pump(a.root, lambda: a.gpu_label['text'].startswith('GPU: '))
    assert a.gpu_label['text'] == 'GPU: NVIDIA GeForce RTX 4060 Laptop GPU, AMD Radeon(TM) Graphics'
    assert a.device_hint['text'] == 'Bakes on NVIDIA GeForce RTX 4060 Laptop GPU.'
    assert tuple(a.device_box['values']) == ('Auto', 'NVIDIA', 'AMD', 'CPU only')
    assert a.device_var.get() == 'Auto' and str(a.device_box['state']) == 'readonly'
    assert a.device_box.master.grid_info()['row'] == 12 and a.stage_label.grid_info()['row'] == 14


def stage_texts(app):
    return [(name['text'], state['text'], str(state['foreground'])) for name, state, _ in app.stage_rows]


def test_stage_rows_are_green_for_gpu_and_red_for_cpu(make_app):
    a = make_app(probe_gpu=probe_for([RTX]))
    pump(a.root, lambda: a.stage_rows[0][1]['text'] == 'GPU')
    assert stage_texts(a) == [('Page bake', 'GPU', 'green'), ('Vertex pass', 'GPU', 'green'), ('Denoise', 'GPU', 'green')]
    a.device_var.set('CPU only')
    a.device_box.event_generate('<<ComboboxSelected>>')
    a.root.update()
    assert stage_texts(a) == [('Page bake', 'CPU', 'red'), ('Vertex pass', 'CPU', 'red'), ('Denoise', 'CPU', 'red')]


def test_finished_bake_shows_what_was_used_in_the_stage_rows(ready_app):
    devices = {'page_bake': 'OPTIX', 'vertex_pass': 'cpu', 'denoise': 'CPU'}
    outcome = BakeOutcome('done', 'Done. The lit level is level.', [], Path('C:/x'), Path(), 'OPTIX',
                          devices, None, [])
    run_events(ready_app, [('done', outcome)])
    assert stage_texts(ready_app) == [('Page bake', 'GPU', 'green'), ('Vertex pass', 'CPU', 'red'),
                                      ('Denoise', 'CPU', 'red')]


def cpu_outcome():
    return BakeOutcome('done', 'Done.', [], None, Path(), 'CPU',
                       {'page_bake': 'CPU', 'vertex_pass': 'cpu', 'denoise': 'CPU'}, None, [])


def test_new_bake_resets_stage_rows_to_prediction(ready_app):
    ready_app._bake_done(cpu_outcome())
    ready_app._run_bake = lambda req, emit: None
    ready_app.start_bake()
    assert [state['text'] for _, state, _ in ready_app.stage_rows] == ['GPU', 'GPU', 'GPU']


def test_failed_bake_without_devices_resets_stage_rows(ready_app):
    ready_app._bake_done(cpu_outcome())
    ready_app._bake_done(BakeOutcome('failed', 'Failed.', [], None, Path(), None))
    assert [state['text'] for _, state, _ in ready_app.stage_rows] == ['GPU', 'GPU', 'GPU']


@pytest.mark.parametrize('event', [('gpu', [RADEON], None), ('denoiser', {'CUDA': False, 'HIP': False})])
def test_late_probe_preserves_used_rows_until_input_change(ready_app, event):
    ready_app._bake_done(cpu_outcome())
    ready_app.events.put(event)
    ready_app._drain()
    assert [state['text'] for _, state, _ in ready_app.stage_rows] == ['CPU', 'CPU', 'CPU']
    ready_app.advanced_vars['page_max'].set('1024')
    assert ready_app.stage_rows[0][1]['text'] == 'GPU'


def test_denoiser_probe_waits_and_updates_on_ui_thread(make_app):
    gate, finished = threading.Event(), threading.Event()
    threads = []

    def probe():
        threads.append(threading.current_thread())
        gate.wait(5)
        finished.set()
        return {'CUDA': False, 'HIP': False}

    a = make_app(probe_denoiser=probe)
    pump(a.root, lambda: a.stage_rows[0][1]['text'] == 'GPU')
    assert a.stage_rows[2][1]['text'] == '…'
    gate.set()
    assert finished.wait(5)
    assert a.stage_rows[2][1]['text'] == '…'
    pump(a.root, lambda: a.stage_rows[2][1]['text'] == 'CPU')
    assert str(a.stage_rows[2][1]['foreground']) == 'red'
    assert threads[0] is not threading.main_thread()


def test_denoiser_probe_exception_predicts_cpu(make_app):
    def broken():
        raise OSError('no child')
    a = make_app(probe_denoiser=broken)
    pump(a.root, lambda: a.stage_rows[2][1]['text'] == 'CPU')
    assert 'could not start' in a.stage_rows[2][2]['text']


def test_device_row_says_looking_until_the_probe_answers(make_app):
    gate = threading.Event()

    def slow():
        gate.wait(5)
        return [RTX], None

    a = make_app(probe_gpu=slow)
    assert a.gpu_label['text'] == 'Looking for GPUs…' and a.device_hint['text'] == ''
    gate.set()
    pump(a.root, lambda: a.gpu_label['text'].startswith('GPU: '))


def test_probe_runs_off_the_ui_thread(make_app):
    seen = []

    def probe():
        seen.append(threading.current_thread())
        return [RTX], None

    a = make_app(probe_gpu=probe)
    pump(a.root, lambda: a.gpu_label['text'].startswith('GPU: '))
    assert seen and seen[0] is not threading.main_thread()


def test_probe_worker_does_not_capture_app(make_app, monkeypatch):
    targets = []

    class UnstartedThread:
        def __init__(self, *, target, **_kwargs):
            targets.append(target)

        def start(self):
            pass

    monkeypatch.setattr(app_module.threading, 'Thread', UnstartedThread)
    app = make_app()
    for target in targets:
        assert all(cell.cell_contents is not app for cell in target.__closure__ or ())


def test_device_choice_reaches_the_request_and_is_remembered(ready_app, tmp_path):
    ready_app.device_box.set('CPU only')
    ready_app.device_box.event_generate('<<ComboboxSelected>>')
    assert ready_app.request().settings.device == 'cpu'
    ready_app.remember()
    assert usersettings.load(tmp_path / 's.json').device == 'cpu'


def test_remembered_device_is_selected_on_start(make_app, tmp_path):
    usersettings.save(usersettings.UserSettings(device='amd'), tmp_path / 's.json')
    a = make_app(probe_gpu=probe_for([RADEON]))
    assert a.device_var.get() == 'AMD'
    pump(a.root, lambda: a.gpu_label['text'].startswith('GPU: '))
    assert a.device_hint['text'] == 'Bakes on AMD Radeon(TM) Graphics.'


def test_missing_vendor_hint_updates_on_selection(make_app):
    a = make_app(probe_gpu=probe_for([RADEON]))
    pump(a.root, lambda: a.gpu_label['text'].startswith('GPU: '))
    a.device_box.set('NVIDIA')
    a.device_box.event_generate('<<ComboboxSelected>>')
    a.root.update()
    assert a.device_hint['text'] == 'No NVIDIA GPU with ray tracing was found, so the bake would stop. Pick Auto or CPU only.'
    a.device_box.set('CPU only')
    a.device_box.event_generate('<<ComboboxSelected>>')
    a.root.update()
    assert a.device_hint['text'] == 'Bakes on the CPU, with fewer rays for Draft, Normal and High.'


def test_gpu_note_shows_for_named_presets_only(app):
    note = 'Ray counts are for a GPU. On the CPU, Draft, Normal and High use fewer rays, as in 0.1.0.'
    assert app.preset_note['text'] == note
    app._select_preset('Custom')
    assert app.preset_note['text'] == '' and app.preset_note.winfo_manager() == ''
    app._select_preset('High')
    assert app.preset_note['text'] == note


def test_device_box_is_locked_during_a_bake(ready_app):
    gate = threading.Event()

    def fake(req, emit):
        gate.wait(5)
        emit(('done', OUTCOME_DONE))

    ready_app._run_bake = fake
    ready_app.start_bake()
    assert str(ready_app.device_box['state']) == 'disabled'
    gate.set()
    pump_app(ready_app, lambda: not ready_app.busy)
    assert str(ready_app.device_box['state']) == 'readonly'


def test_finished_bake_shows_the_devices_used_and_logs_warnings(ready_app):
    devices = {'page_bake': 'CPU', 'vertex_pass': 'cpu', 'denoise': 'CPU'}
    rays = {'samples': 256, 'texel_samples': 252, 'vertex_rays': 1024}
    outcome = BakeOutcome('done', 'Done. The lit level is level.', [], Path('C:/x'), Path(), 'CPU',
                          devices, rays, ['a', 'b'])
    run_events(ready_app, [('device', 'CPU'), ('done', outcome)])
    assert ready_app.device_label['text'] == result_line(devices, rays, ['a', 'b'])
    log = ready_app.log_box.get('1.0', 'end')
    assert 'WARNING: a' in log and 'WARNING: b' in log


def test_finished_bake_without_devices_keeps_the_baking_on_text(ready_app):
    run_events(ready_app, [('device', 'OPTIX'), ('done', OUTCOME_DONE)])
    assert ready_app.device_label['text'] == 'Baking on OPTIX'


def test_late_probe_result_after_close_is_ignored(make_app):
    a = make_app()
    pump(a.root, lambda: a.blender is not None and a.gpu_label['text'].startswith('GPU: '))
    a.events.put(('gpu', [RADEON], None))
    a.window.destroy()
    a._drain()
    assert a.events.empty()


def test_a_warning_already_logged_live_is_not_logged_again_after_the_bake(ready_app):
    devices = {'page_bake': 'CPU', 'vertex_pass': 'cpu', 'denoise': 'CPU'}
    outcome = BakeOutcome('done', 'Done. The lit level is level.', [], Path('C:/x'), Path(), 'CPU',
                          devices, None, ['live one', 'twin groups'])
    run_events(ready_app, [('log', 'WARNING: live one'), ('done', outcome)])
    log = ready_app.log_box.get('1.0', 'end')
    assert log.count('WARNING: live one') == 1 and log.count('WARNING: twin groups') == 1


def test_a_live_warning_of_an_earlier_bake_does_not_hide_the_next_summary_warning(ready_app):
    devices = {'page_bake': 'CPU', 'vertex_pass': 'cpu', 'denoise': 'CPU'}
    outcome = BakeOutcome('done', 'Done. The lit level is level.', [], Path('C:/x'), Path(), 'CPU',
                          devices, None, ['same text'])
    run_events(ready_app, [('log', 'WARNING: same text'), ('done', outcome)])
    run_events(ready_app, [('done', outcome)])
    assert ready_app.log_box.get('1.0', 'end').count('WARNING: same text') == 1


def test_a_probe_that_raises_still_answers(make_app):
    def broken():
        raise OSError('no child process')

    a = make_app(probe_gpu=broken)
    pump(a.root, lambda: a.gpu_label['text'] != 'Looking for GPUs…')
    assert a.gpu_label['text'] != 'Looking for GPUs…'
    assert 'no child process' in a.gpu_label['text'] + a.device_hint['text']


def test_32bit_checkbox_reaches_the_settings_and_is_remembered(tmp_path, tiny, make_app):
    tags, *_ = tiny
    (tags / 'globals').mkdir()
    a = make_app(store_path=tmp_path / 's.json')
    a.set_tags(str(tags))
    a.set_scenario_file(tags / 'level.scenario')
    pump(a.root, lambda: a.bsp_box.get() == 'room' and a.blender is not None)
    assert a.request().settings.lightmap_32bit is False
    a.lightmap_32bit_var.set(True)
    assert a.request().settings.lightmap_32bit is True
    a.remember()
    assert make_app(store_path=tmp_path / 's.json').lightmap_32bit_var.get() is True


def test_max_resolution_hint_appears_for_large_pages(app):
    for value, shown in (('256', False), ('512', False), ('1024', False), ('2048', True), ('4096', True)):
        app.advanced_vars['page_max'].set(value)
        assert (app.page_max_hint['text'] == presets.PAGE_MAX_HINT) is shown
        assert not any(ch.isdigit() for ch in app.page_max_hint['text'])
    app._select_preset('High')
    assert app.page_max_hint['text'] == ''


def test_density_is_a_dropdown_of_the_choices(app):
    entry = app.advanced_entries['texels_per_unit']
    assert str(entry['state']) == 'readonly'
    assert tuple(entry['values']) == ('1', '2', '4', '8', '12', '16', '24', '32')


def test_presets_set_listed_densities(app):
    for name, shown in (('Draft', '1'), ('Normal', '2'), ('High', '4')):
        app._select_preset(name)
        assert app.advanced_vars['texels_per_unit'].get() == shown


def test_saved_density_outside_the_list_is_kept_as_a_custom_entry(tmp_path, tiny, make_app):
    store = tmp_path / 's.json'
    usersettings.save(
        usersettings.UserSettings(
            preset='Custom',
            advanced=dict(texels_per_unit='3.0', samples='256', vertex_rays='1024', sky_grid='2', bounces='1', page_max='512'),
        ),
        store,
    )
    app = make_app(store_path=store)
    entry = app.advanced_entries['texels_per_unit']
    assert app.advanced_vars['texels_per_unit'].get() == '3'
    assert tuple(entry['values'])[-1] == '3'
    tags, *_ = tiny
    (tags / 'globals').mkdir()
    app.set_tags(str(tags))
    app.set_scenario_file(tags / 'level.scenario')
    pump(app.root, lambda: app.bsp_box.get() == 'room' and app.blender is not None)
    assert app.request().settings.texels_per_unit == 3.0
    app.advanced_vars['texels_per_unit'].set('8')
    assert '3' not in tuple(entry['values'])


class BlockingEstimate:
    def __init__(self):
        self.calls = []
        self.gates = [threading.Event() for _ in range(10)]
        self.results = [ESTIMATE_FITS] * 10
        self.active = 0
        self.maximum_active = 0
        self.threads = []

    def __call__(self, inputs):
        index = len(self.calls)
        self.active += 1
        self.maximum_active = max(self.maximum_active, self.active)
        self.threads.append(threading.current_thread())
        self.calls.append(inputs)
        try:
            assert self.gates[index].wait(10), 'test did not release estimate'
            result = self.results[index]
            if isinstance(result, BaseException):
                raise result
            return result
        finally:
            self.active -= 1


@pytest.fixture
def app_with_fake(make_app):
    fake = BlockingEstimate()
    app = make_app(estimate=fake)
    yield app, fake
    for gate in fake.gates:
        gate.set()
    for thread in fake.threads:
        thread.join(5)
    app._drain()


@pytest.fixture
def ready_app_with_fake(app_with_fake, tiny):
    app, fake = app_with_fake
    tags, *_ = tiny
    (tags / 'globals').mkdir()
    app.set_tags(str(tags))
    app.set_scenario_file(tags / 'level.scenario')
    pump(app.root, lambda: len(fake.calls) == 1)
    fake.gates[0].set()
    pump(app.root, lambda: app.memory_label['text'].startswith('Needs about') and app.blender is not None)
    return app, fake


def finish_estimate(app, fake, index=1):
    pump(app.root, lambda: len(fake.calls) > index)
    fake.gates[index].set()
    pump(app.root, lambda: app.estimate_state != 'calculating')


def test_selection_starts_an_estimate_and_greys_out_bake(ready_app_with_fake):
    app, fake = ready_app_with_fake
    app.advanced_vars['page_max'].set('1024')
    assert app.memory_label['text'] == 'Calculating…'
    assert str(app.bake_button['state']) == 'disabled'
    finish_estimate(app, fake)
    assert fake.calls[1].page_max == 1024
    assert fake.threads[1] is not threading.main_thread()
    assert app.memory_label['text'].startswith('Needs about')
    assert str(app.bake_button['state']) == 'normal'


def test_two_quick_selections_show_only_the_last_result(ready_app_with_fake, monkeypatch):
    app, fake = ready_app_with_fake
    monkeypatch.setattr(app_module, 'free_physical_bytes', lambda: None)
    fake.results[1] = SceneEstimate(SceneCounts(2, 4), [(1024, 1024)])
    fake.results[2] = SceneEstimate(SceneCounts(2, 4), [(2048, 2048)])
    app.advanced_vars['page_max'].set('1024')
    pump(app.root, lambda: len(fake.calls) == 2)
    app.advanced_vars['page_max'].set('2048')
    assert len(fake.calls) == 2
    fake.gates[1].set()
    pump(app.root, lambda: len(fake.calls) == 3)
    assert app.memory_label['text'] == 'Calculating…'
    fake.gates[2].set()
    pump(app.root, lambda: app.estimate_state == 'fits')
    assert app.memory_label['text'] == 'Needs about 12 GB'
    assert [c.page_max for c in fake.calls] == [512, 1024, 2048]
    assert fake.maximum_active == 1


def test_stale_estimate_is_ignored(ready_app_with_fake):
    app, fake = ready_app_with_fake
    previous = (app.memory_label['text'], app.estimate_state, str(app.bake_button['state']))
    app.events.put(('estimate', app.generation - 1, fake.calls[0], CapError('old')))
    app._drain()
    assert (app.memory_label['text'], app.estimate_state, str(app.bake_button['state'])) == previous


def test_non_chart_field_starts_no_estimate(ready_app_with_fake):
    app, fake = ready_app_with_fake
    previous = app.memory_label['text'], app.estimate_state
    app.advanced_vars['samples'].set('512')
    app.advanced_vars['vertex_rays'].set('8192')
    app.device_var.set('CPU only')
    app._update_device_hint()
    app.root.update()
    assert len(fake.calls) == 1
    assert (app.memory_label['text'], app.estimate_state) == previous


def test_incomplete_form_starts_no_estimate(app_with_fake, tiny, monkeypatch):
    app, fake = app_with_fake
    tags, *_ = tiny
    (tags / 'globals').mkdir()
    monkeypatch.setattr(form_module, 'list_bsps', lambda *_: ['room', 'other'])
    app.set_tags(str(tags))
    app.set_scenario_file(tags / 'level.scenario')
    app.root.update()
    assert fake.calls == []
    assert app.estimate_inputs() is None
    assert app.memory_label['text'] == '' and app.estimate_state == 'idle'
    app.bsp_var.set('other')
    app._inputs_changed()
    pump(app.root, lambda: len(fake.calls) == 1)
    assert fake.calls[0].bsp == 'other'


def test_bake_enabled_with_warning_when_the_level_may_not_fit(ready_app_with_fake, monkeypatch):
    app, fake = ready_app_with_fake
    monkeypatch.setattr(app_module, 'free_physical_bytes', lambda: 1)
    app.advanced_vars['page_max'].set('1024')
    finish_estimate(app, fake)
    assert app.memory_label['text'] == (out_of_memory_message(ESTIMATE_FITS.counts, ESTIMATE_FITS.pages, 1024, 1)
                                        + ' You can still bake, at your own risk.')
    assert app.estimate_state == 'warning'
    assert str(app.bake_button['state']) == 'normal'


@pytest.mark.parametrize('answer', [False, True])
@pytest.mark.parametrize('new_names', [False, True])
def test_memory_warning_asks_before_baking(ready_app, monkeypatch, answer, new_names):
    app = ready_app
    monkeypatch.setattr(app_module, 'free_physical_bytes', lambda: 1)
    inputs = app.estimate_inputs()
    app.events.put(('estimate', app.generation, inputs, ESTIMATE_FITS))
    app._drain()
    app.new_names_var.set(new_names)
    message = out_of_memory_message(ESTIMATE_FITS.counts, ESTIMATE_FITS.pages, inputs.page_max, 1)
    asked, started = [], []

    def ask(*args):
        asked.append(args)
        return answer

    def bake(req, emit):
        started.append(req)
        emit(('done', OUTCOME_DONE))

    monkeypatch.setattr(app_module.messagebox, 'askyesno', ask)
    app._run_bake = bake
    app.start_bake()
    assert asked[:1] == [('Not enough memory', message + '\n\nBake anyway? Blender may run out of memory and stop.')]
    if answer:
        pump_app(app, lambda: not app.busy)
        assert len(started) == 1
        assert [q[0] for q in asked] == (['Not enough memory'] if new_names else
                                       ['Not enough memory', 'Replace your tags?'])
    else:
        assert not app.busy and started == []
        assert len(asked) == 1


def test_fitting_bake_does_not_ask_memory_question(ready_app, monkeypatch):
    monkeypatch.setattr(app_module.messagebox, 'askyesno', lambda *a: pytest.fail('asked'))
    started = []

    def bake(req, emit):
        started.append(req)
        emit(('done', OUTCOME_DONE))

    ready_app._run_bake = bake
    ready_app.start_bake()
    pump_app(ready_app, lambda: not ready_app.busy)
    assert len(started) == 1


def check_estimate_error(ready_app_with_fake, exc, state, button):
    app, fake = ready_app_with_fake
    fake.results[1] = exc
    app.advanced_vars['page_max'].set('1024')
    finish_estimate(app, fake)
    assert app.memory_label['text'] == errors.explain(exc)
    assert app.estimate_state == state
    assert str(app.bake_button['state']) == button
    assert str(exc) in app.log_path.read_text()


def test_cap_error_disables_bake(ready_app_with_fake):
    check_estimate_error(ready_app_with_fake, CapError('cap'), 'refused', 'disabled')


def test_estimate_error_leaves_bake_enabled(ready_app_with_fake):
    check_estimate_error(ready_app_with_fake, RuntimeError('broken'), 'error', 'normal')


def test_free_memory_is_read_when_the_result_is_shown(ready_app_with_fake, monkeypatch):
    app, fake = ready_app_with_fake
    reads = []
    monkeypatch.setattr(app_module, 'free_physical_bytes', lambda: reads.append(1) or None)
    app.advanced_vars['page_max'].set('1024')
    pump(app.root, lambda: len(fake.calls) == 2)
    app.advanced_vars['page_max'].set('2048')
    assert reads == []
    fake.gates[1].set()
    pump(app.root, lambda: len(fake.calls) == 3)
    assert reads == []
    finish_estimate(app, fake, 2)
    assert reads == [1]
    assert app.memory_label['text'] == 'Needs about 0.0 GB'
    assert str(app.bake_button['state']) == 'normal'


def test_bake_request_uses_the_estimates_chart_cache(ready_app):
    req = ready_app.request()
    assert req.charts == runner.chart_cache_path(req.tags, req.scenario, req.bsp)


def test_unchanged_estimate_inputs_do_not_restart(ready_app_with_fake):
    app, fake = ready_app_with_fake
    generation = app.generation
    app._inputs_changed()
    app.advanced_vars['texels_per_unit'].set('2.0')
    app.set_scenario_file(app._scenario_file)
    app.set_tags(str(app._tags))
    assert app.generation == generation
    assert len(fake.calls) == 1 and app.estimate_state == 'fits'


def test_density_selection_and_preset_trigger_estimates(ready_app_with_fake):
    app, fake = ready_app_with_fake
    app.advanced_vars['texels_per_unit'].set('8')
    finish_estimate(app, fake)
    assert fake.calls[1].texels_per_unit == 8
    app._select_preset('Draft')
    finish_estimate(app, fake, 2)
    assert fake.calls[2].texels_per_unit == 1


def test_preset_switch_estimates_once_on_the_final_inputs(ready_app_with_fake):
    app, fake = ready_app_with_fake
    app._select_preset('Normal')
    finish_estimate(app, fake, len(fake.calls) - 1)
    before = len(fake.calls)
    app._select_preset('High')
    pump(app.root, lambda: len(fake.calls) > before)
    app.root.update()
    assert [(c.texels_per_unit, c.page_max) for c in fake.calls[before:]] == [(4.0, 1024)]


@pytest.mark.parametrize('field, value', [('texels_per_unit', ''), ('texels_per_unit', 'bad'),
                                         ('page_max', ''), ('page_max', 'bad')])
def test_invalid_chart_inputs_clear_estimate(ready_app_with_fake, field, value):
    app, fake = ready_app_with_fake
    app.advanced_vars[field].set(value)
    assert app.estimate_inputs() is None
    assert app.memory_label['text'] == '' and app.estimate_state == 'idle'
    assert len(fake.calls) == 1


def test_cleared_scenario_discards_running_and_waiting_estimates(ready_app_with_fake):
    app, fake = ready_app_with_fake
    app.advanced_vars['page_max'].set('1024')
    pump(app.root, lambda: len(fake.calls) == 2)
    app.advanced_vars['page_max'].set('2048')
    app._clear_scenario()
    fake.gates[1].set()
    pump(app.root, lambda: app._estimate_running is None)
    assert app.memory_label['text'] == '' and app.estimate_state == 'idle'
    assert len(fake.calls) == 2
    assert str(app.bake_button['state']) == 'disabled'


def test_replacing_tags_invalidates_running_estimate(ready_app_with_fake, tmp_path):
    app, fake = ready_app_with_fake
    app.advanced_vars['page_max'].set('1024')
    pump(app.root, lambda: len(fake.calls) == 2)
    tags = tmp_path / 'other_tags'
    (tags / 'globals').mkdir(parents=True)
    app.set_tags(str(tags))
    fake.gates[1].set()
    pump(app.root, lambda: app._estimate_running is None)
    assert app.memory_label['text'] == '' and app.estimate_state == 'idle'
    assert len(fake.calls) == 2


def test_only_newest_waiting_selection_runs(ready_app_with_fake):
    app, fake = ready_app_with_fake
    app.advanced_vars['page_max'].set('1024')
    pump(app.root, lambda: len(fake.calls) == 2)
    app.advanced_vars['page_max'].set('2048')
    app.advanced_vars['page_max'].set('4096')
    fake.gates[1].set()
    pump(app.root, lambda: len(fake.calls) == 3)
    assert [c.page_max for c in fake.calls] == [512, 1024, 4096]
    finish_estimate(app, fake, 2)
    assert fake.maximum_active == 1


def test_no_estimate_during_or_after_bake(ready_app_with_fake):
    app, fake = ready_app_with_fake
    app.busy = True
    generation, text = app.generation, app.memory_label['text']
    app.advanced_vars['page_max'].set('1024')
    assert app.generation == generation and len(fake.calls) == 1
    app._bake_done(OUTCOME_DONE)
    app._drain()
    assert len(fake.calls) == 1 and app.memory_label['text'] == text


def test_estimate_worker_does_not_capture_app(ready_app_with_fake, monkeypatch):
    app, fake = ready_app_with_fake
    targets = []

    class UnstartedThread:
        def __init__(self, *, target, **_kwargs):
            targets.append(target)

        def start(self):
            pass

    monkeypatch.setattr(app_module.threading, 'Thread', UnstartedThread)
    app.advanced_vars['page_max'].set('1024')
    assert len(targets) == 1
    assert all(cell.cell_contents is not app for cell in targets[0].__closure__ or ())


def test_empty_pages_estimate_ends_calculating_and_enables_bake(ready_app_with_fake, monkeypatch):
    app, fake = ready_app_with_fake
    monkeypatch.setattr(app_module, 'free_physical_bytes', lambda: None)
    fake.results[1] = SceneEstimate(SceneCounts(2, 4), [])
    app.advanced_vars['page_max'].set('1024')
    pump(app.root, lambda: len(fake.calls) == 2)
    assert app.estimate_state == 'calculating'
    fake.gates[1].set()
    pump(app.root, lambda: app._estimate_running is None)
    assert app.estimate_state == 'fits'
    assert app.memory_label['text'] == 'Needs about 0.0 GB'
    assert str(app.bake_button['state']) == 'normal'


def test_describe_failure_ends_calculating_and_enables_bake(ready_app_with_fake, monkeypatch):
    app, fake = ready_app_with_fake

    def broken_describe(*_args):
        raise RuntimeError('description failed')

    monkeypatch.setattr(app_module.estimate_module, 'describe', broken_describe)
    app.advanced_vars['page_max'].set('1024')
    pump(app.root, lambda: len(fake.calls) == 2)
    assert app.estimate_state == 'calculating'
    fake.gates[1].set()
    pump(app.root, lambda: app._estimate_running is None)
    assert app.estimate_state == 'error'
    assert app.memory_label['text'] == 'The bake stopped: description failed. The full error is in the log.'
    assert str(app.bake_button['state']) == 'normal'
    assert 'RuntimeError: description failed' in app.log_path.read_text()
