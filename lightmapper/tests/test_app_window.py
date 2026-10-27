import threading
import time
from pathlib import Path

import pytest

tk = pytest.importorskip('tkinter')

from test_pipeline import tiny
from lightmapper.blender_find import Blender, BlenderNotFound
from opentool.app import App
from opentool.runner import BakeOutcome
from opentool import app as app_module
from opentool import usersettings
from opentool import form as form_module


@pytest.fixture(scope='module')
def tk_root():
    root = tk.Tk()
    root.withdraw()
    yield root
    root.destroy()


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
    assert (req.scenario, req.bsp, req.new_names, req.settings.samples, req.settings.timeout_s) == ('level', 'room', False, 64, None)
    assert req.backups is True
    app.backups_var.set(False)
    assert app.request().backups is False
    app.verbose_var.set(True)
    assert app.request().verbose is True


def test_draft_fills_all_advanced_fields(app):
    app._select_preset('Draft')
    assert {key: value.get() for key, value in app.advanced_vars.items()} == dict(texels_per_unit='1.0', samples='64', vertex_rays='256', sky_grid='2', bounces='1', page_max='512')


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
    pump(app.root, lambda: app.bsp_box.get() == 'room' and app.blender is not None)
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
