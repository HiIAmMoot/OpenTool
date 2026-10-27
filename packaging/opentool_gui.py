import sys

if getattr(sys, 'frozen', False):
    import ctypes
    import os
    # Children (Blender) inherit the process DLL directory, and PyInstaller sets it to
    # _internal, where its old or mismatched copies of system DLLs break them. Clear it.
    ctypes.windll.kernel32.SetDllDirectoryW(None)
    # This process still must run on the bundled 14.4x C++ runtime, not System32's, and
    # PyInstaller registers no other search path. AddDllDirectory and loaded modules are
    # per process, so they are not inherited. The handles are kept so they stay valid.
    _dll_directory = os.add_dll_directory(sys._MEIPASS)
    _runtime = []
    for _name in ('vcruntime140', 'vcruntime140_1', 'msvcp140', 'msvcp140_1',
                  'msvcp140_2', 'vcomp140'):
        _path = os.path.join(sys._MEIPASS, _name + '.dll')
        if os.path.isfile(_path):
            _runtime.append(ctypes.WinDLL(_path))

from opentool.app import main


if __name__ == '__main__':
    raise SystemExit(main())
