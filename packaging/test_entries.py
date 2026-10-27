import ctypes
import os
import tempfile
from pathlib import Path
import runpy
import sys
import types
import unittest
from unittest.mock import patch


class EntryTests(unittest.TestCase):
    def test_frozen_entry_resets_dll_directory_before_app_import(self):
        for script, module, function in (
            ('opentool_gui.py', 'opentool.app', 'main'),
            ('opentool_cli.py', 'lightmapper.entry', 'cli_entry'),
        ):
            with self.subTest(script=script):
                calls = []
                kernel32 = types.SimpleNamespace(SetDllDirectoryW=lambda value: calls.append(value))
                windll = types.SimpleNamespace(kernel32=kernel32)
                fake_module = types.ModuleType(module)
                def get_entry(name):
                    if name != function:
                        raise AttributeError(name)
                    self.assertEqual(calls, [None], 'app imported before inherited DLL directory was cleared')
                    return lambda: 0
                fake_module.__getattr__ = get_entry
                added = []
                loaded = []
                with tempfile.TemporaryDirectory() as bundle:
                    (Path(bundle) / 'vcruntime140.dll').write_bytes(b'')
                    with patch.object(sys, 'frozen', True, create=True), \
                         patch.object(sys, '_MEIPASS', bundle, create=True), \
                         patch.object(os, 'add_dll_directory',
                                      lambda path: added.append((path, list(calls)))), \
                         patch.object(ctypes, 'WinDLL', loaded.append, create=True), \
                         patch.object(ctypes, 'windll', windll, create=True), \
                         patch.dict(sys.modules, {module: fake_module}):
                        runpy.run_path(str(Path(__file__).with_name(script)), run_name='entry_test')
                    # the bundle stays this process's DLL directory, added after the reset
                    self.assertEqual(added, [(bundle, [None])])
                    self.assertEqual(loaded, [str(Path(bundle) / 'vcruntime140.dll')])
                self.assertEqual(calls, [None])


if __name__ == '__main__':
    unittest.main()
