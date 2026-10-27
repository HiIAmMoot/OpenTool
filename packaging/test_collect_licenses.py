import ast
import importlib.util
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    'collect_licenses', Path(__file__).with_name('collect_licenses.py'))
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)

BUILD_PYTHON = Path(__file__).resolve().parents[1] / 'build/venv/Scripts/python.exe'


def test_slangpy_collection_excludes_unused_backends_and_test_data():
    tree = ast.parse(Path(__file__).with_name('opentool.spec').read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == 'include_slangpy')
    namespace = {'Path': Path}
    exec(compile(ast.Module(body=[function], type_ignores=[]), 'opentool.spec', 'exec'), namespace)
    include = namespace['include_slangpy']
    candidates = [
        'slangpy/optix/8_0/optix.h', 'slangpy/optix/8_1/optix.h',
        'slangpy/optix/9_0/optix.h', 'slangpy/D3D12/D3D12Core.dll',
        'D3D12Core.dll', 'd3d12SDKLayers.dll', 'slangpy/dxcompiler.dll',
        'slangpy/dxil.dll', 'slangpy/WinPixEventRuntime.dll', 'slangpy/slang-llvm.dll',
        'slangpy/tests/test_device.py', 'slangpy/benchmarks/bench.py',
        'slangpy/data/test_images/test.png',
        r'C:\wheel\slangpy\optix\9_0\optix.h',
        'slangpy/sgl.dll', 'slangpy/slang-compiler.dll', 'slangpy/slang-glslang.dll',
        'slangpy/slang-glsl-module.dll', 'slangpy/shaders/slang-rhi/slang-rhi-device.h',
        'slangpy/slang-standard-module-2026.12/slang/neural.slang-module',
        'slangpy.shaders', 'slangpy.torchintegration',
    ]
    assert [name for name in candidates if include(name)] == candidates[-8:]


def test_slangpy_and_its_components_are_collected(tmp_path, monkeypatch):
    real_files = collector.metadata.files
    base = tmp_path / 'base'
    for name in ('LICENSE.txt', 'tcl/tcl8.6/license.terms', 'tcl/tk8.6/license.terms'):
        path = base / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('test runtime license')
    dist = tmp_path / 'stub.dist-info'
    dist.mkdir()
    (dist / 'LICENSE').write_text('test distribution license')
    (dist / 'RECORD').write_text('stub.dist-info/LICENSE,,\n')
    stub = collector.metadata.Distribution.at(dist).files
    monkeypatch.setattr(collector.metadata, 'files', lambda name: real_files(name) if name == 'slangpy' else stub)
    monkeypatch.setattr(sys, 'base_prefix', str(base))
    collector.collect(tmp_path)
    assert (tmp_path / 'slangpy' / 'licenses' / 'LICENSE').is_file()
    assert sorted(p.name for p in (tmp_path / 'slangpy-third-party').iterdir()) == sorted(
        p.name for p in (Path(collector.__file__).parent / 'licenses' / 'slangpy').iterdir())


class LicenseTests(unittest.TestCase):
    def test_missing_distribution_licenses_are_all_reported(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / 'licenses'
            with patch.object(collector.metadata, 'files', return_value=[]):
                with self.assertRaises(RuntimeError) as raised:
                    collector.collect(destination)
            for name in collector.DISTRIBUTIONS:
                self.assertIn(name, str(raised.exception))
            self.assertFalse(destination.exists())

    @unittest.skipUnless(BUILD_PYTHON.is_file(),
                         'needs the build venv (build/venv) with the packaged distributions installed')
    def test_real_build_environment_licenses_are_copied(self):
        if Path(sys.executable).resolve() != BUILD_PYTHON:
            done = subprocess.run([str(BUILD_PYTHON), str(Path(__file__).resolve()),
                                   'LicenseTests.test_real_build_environment_licenses_are_copied'],
                                  capture_output=True, text=True)
            self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
            return
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / 'licenses'
            collector.collect(destination)
            for name in collector.DISTRIBUTIONS:
                self.assertTrue(any((destination / name).rglob('*')), name)
            # These transitive imports are present in the frozen PYZ, not just the build venv.
            for name in ('packaging',):
                self.assertTrue(any((destination / name).rglob('*')), name)
            for name in ('python/LICENSE.txt', 'tcltk/tcl-license.terms',
                         'tcltk/tk-license.terms', 'oidn/LICENSE.txt'):
                self.assertTrue((destination / name).is_file(), name)
            self.assertEqual(len(list((destination / 'oidn').glob('third-party-programs*.txt'))), 3)


if __name__ == '__main__':
    unittest.main()
