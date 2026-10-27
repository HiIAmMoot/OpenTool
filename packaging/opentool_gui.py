import sys

if getattr(sys, 'frozen', False):
    import ctypes
    # Blender must use its own/system DLLs, not inherit PyInstaller's bundle directory.
    ctypes.windll.kernel32.SetDllDirectoryW(None)

from opentool.app import main


if __name__ == '__main__':
    raise SystemExit(main())
