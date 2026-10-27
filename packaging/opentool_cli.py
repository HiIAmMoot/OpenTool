import sys

if getattr(sys, 'frozen', False):
    import ctypes
    # Blender must use its own/system DLLs, not inherit PyInstaller's bundle directory.
    ctypes.windll.kernel32.SetDllDirectoryW(None)

from lightmapper.entry import cli_entry


if __name__ == '__main__':
    raise SystemExit(cli_entry())
