import threading
import time
from pathlib import Path

import pytest

tk = pytest.importorskip('tkinter')

from test_pipeline import tiny
from lightmapper.blender_find import Blender, BlenderNotFound
from opentool.app import App
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
    app._close()
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
