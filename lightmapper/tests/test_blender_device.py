import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

BAKE_SCRIPT = Path(__file__).resolve().parents[1] / 'blender' / 'bake.py'


def load_bake(monkeypatch, available=()):
    supported = {'OPTIX', 'CUDA', *available}

    class Preferences:
        devices = [SimpleNamespace(type=kind, use=False) for kind in available]

        @property
        def compute_device_type(self):
            return self._compute_device_type

        @compute_device_type.setter
        def compute_device_type(self, value):
            if value not in supported:
                raise TypeError(value)
            self._compute_device_type = value

        def refresh_devices(self):
            pass

    prefs = Preferences()
    scene = SimpleNamespace(cycles=SimpleNamespace())
    bpy = ModuleType('bpy')
    bpy.context = SimpleNamespace(preferences=SimpleNamespace(addons={'cycles': SimpleNamespace(preferences=prefs)}), scene=scene)
    mathutils = ModuleType('mathutils')
    mathutils.bvhtree = ModuleType('mathutils.bvhtree')
    mathutils.bvhtree.BVHTree = object
    monkeypatch.setitem(sys.modules, 'bpy', bpy)
    monkeypatch.setitem(sys.modules, 'mathutils', mathutils)
    monkeypatch.setitem(sys.modules, 'mathutils.bvhtree', mathutils.bvhtree)
    spec = importlib.util.spec_from_file_location('bake_under_test', BAKE_SCRIPT)
    bake = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bake)
    return bake, scene


def test_device_order(monkeypatch):
    assert load_bake(monkeypatch)[0].DEVICE_TYPES == ('OPTIX', 'CUDA', 'HIP', 'ONEAPI')


@pytest.mark.parametrize('available, expected', [(['HIP'], 'HIP'), (['ONEAPI'], 'ONEAPI'),
                                                  (['CUDA', 'HIP'], 'CUDA'), ([], 'CPU')])
def test_pick_device_walks_the_order(available, expected, monkeypatch):
    bake, scene = load_bake(monkeypatch, available)
    assert bake.pick_device(['OPTIX', 'CUDA', 'HIP', 'ONEAPI'], False, False) == expected
    assert scene.cycles.device == ('CPU' if expected == 'CPU' else 'GPU')


def test_cpu_wanted_stays_on_cpu(monkeypatch):
    bake, _ = load_bake(monkeypatch, ['HIP'])
    assert bake.pick_device([], False, False) == 'CPU'
