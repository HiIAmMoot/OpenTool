from importlib import metadata, util
from pathlib import Path
import sys

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules


root = Path(SPECPATH).parent
sys.path.insert(0, str(root))
from lightmapper.resources import REQUIRED

gui_script = root / 'packaging' / 'opentool_gui.py'
cli_script = root / 'packaging' / 'opentool_cli.py'
datas = [(str(root / name), str(Path(name).parent)) for name in REQUIRED]
oidn_bin = Path(metadata.distribution('pyoidn').locate_file('pyoidn/oidn/bin'))
binaries = [(str(root / 'lightmapper/vendor/xatlas/build/xatlas.dll'),
             'lightmapper/vendor/xatlas/build')]
# OIDN loads devices at runtime beside its package; the three exe tools are not needed.
binaries += [(str(path), 'pyoidn/oidn/bin') for path in sorted(oidn_bin.glob('*.dll'))]


def include_slangpy(name):
    parts = name.replace('\\', '/').replace('.', '/').lower().split('/')
    return (not {'d3d12', 'optix', 'tests', 'benchmarks'} & set(parts)
            and 'data/test_images' not in '/'.join(parts)
            and Path(name).name.lower() not in {
                'd3d12core.dll', 'd3d12sdklayers.dll', 'dxcompiler.dll', 'dxil.dll',
                'winpixeventruntime.dll', 'slang-llvm.dll'})


binaries += [entry for entry in collect_dynamic_libs('slangpy') if include_slangpy(entry[0])]
datas += [entry for entry in collect_data_files('slangpy') if include_slangpy(entry[0])]
slangpy_modules = collect_submodules('slangpy', filter=include_slangpy)
# setuptools and pkg_resources are not used at run time; they bring unlicensed vendored packages.
excludes = ['torch', 'pytest', 'IPython', 'matplotlib', 'numpy.tests', 'setuptools', 'pkg_resources']
# SlangPy imports torchintegration unconditionally, even without PyTorch.
excludes += ['slangpy.tests', 'slangpy.benchmarks']
scipy_root = Path(util.find_spec('scipy').origin).parent
excludes += ['scipy.' + '.'.join(path.relative_to(scipy_root).parts)
             for path in sorted(scipy_root.rglob('tests')) if path.is_dir()]

a = Analysis(
    [str(gui_script), str(cli_script)],
    pathex=[str(root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=['_cffi_backend', *slangpy_modules],
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)

# Dependency scanning found Oculus debug DLLs and a Java UCRT on PATH. Windows
# provides these; bundling them can break Blender with mismatched system DLLs.
system_dlls = {'dbghelp.dll', 'dbgcore.dll', 'ucrtbase.dll'}
# api-ms-win-* are Windows' own API-set stubs (a Java runtime on PATH supplied about 40).
a.binaries = [entry for entry in a.binaries
              if Path(entry[0]).name.lower() not in system_dlls
              and not Path(entry[0]).name.lower().startswith('api-ms-win-')
              and include_slangpy(entry[0])]
a.datas = [entry for entry in a.datas if include_slangpy(entry[0])]

# Each exe keeps all runtime hooks, but must not run the other application's entry.
gui_scripts = [entry for entry in a.scripts if entry[0] != 'opentool_cli']
cli_scripts = [entry for entry in a.scripts if entry[0] != 'opentool_gui']
gui_exe = EXE(pyz, gui_scripts, [], exclude_binaries=True, name='OpenTool',
              console=False, upx=False)
cli_exe = EXE(pyz, cli_scripts, [], exclude_binaries=True, name='OpenTool-cli',
              console=True, upx=False)
coll = COLLECT(gui_exe, cli_exe, a.binaries, a.datas, name='OpenTool', upx=False)
