import threading
import time
from pathlib import Path

import pytest

tk = pytest.importorskip('tkinter')

from test_pipeline import tiny
from lightmapper.blender_find import Blender, BlenderNotFound
from opentool.app import App


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
        app.root = root
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
    assert app.new_names_var.get() is True and app.preset_var.get() == 'Normal'
    assert not app.advanced_frame.winfo_ismapped()


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
    app.preset_var.set('Draft')
    req = app.request()
    assert (req.scenario, req.bsp, req.new_names, req.settings.samples, req.settings.timeout_s) == ('level', 'room', True, 64, None)


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
    a.preset_var.set('High')
    a.remember()
    b = make_app(store_path=tmp_path / 's.json')
    assert b.preset_var.get() == 'High' and b.tags_entry.get() == str(tags)


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
