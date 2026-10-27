"""Collect the license texts needed by the Windows download."""

from importlib import metadata
from pathlib import Path
import re
import shutil
import sys


DISTRIBUTIONS = (
    'numpy', 'scipy', 'numba', 'llvmlite', 'pyoidn', 'cffi', 'pycparser',
    'pillow', 'pyinstaller', 'packaging',
)
LICENSE_NAME = re.compile(r'^(LICEN[CS]E|COPYING|NOTICE)', re.IGNORECASE)
OIDN_LICENSES = Path(__file__).resolve().parent / 'licenses' / 'oidn'


def collect(destination: Path) -> None:
    copies = []
    missing = []
    for name in DISTRIBUTIONS:
        try:
            files = metadata.files(name) or ()
        except metadata.PackageNotFoundError:
            files = ()
        licenses = [file for file in files
                    if any(part.endswith('.dist-info') for part in file.parts)
                    and LICENSE_NAME.match(file.name) and Path(file.locate()).is_file()]
        if not licenses:
            missing.append(name)
        for file in licenses:
            # Preserve nested license folders, including wheel-vendored notices.
            index = next(i for i, part in enumerate(file.parts) if part.endswith('.dist-info'))
            relative = Path(*file.parts[index + 1:])
            copies.append((Path(file.locate()), destination / name / relative))

    base = Path(sys.base_prefix)
    tcl_license = base / 'tcl' / 'tcl8.6' / 'license.terms'
    if not tcl_license.is_file():
        import tkinter
        # python.org 3.11.9 omits this text; do not substitute it for an unknown Tcl version.
        version = tkinter.Tcl().eval('info patchlevel')
        if version == '8.6.12':
            tcl_license = Path(__file__).resolve().parent / 'licenses' / 'tcl' / 'license.terms'
    for source, target in (
        (base / 'LICENSE.txt', destination / 'python' / 'LICENSE.txt'),
        (tcl_license, destination / 'tcltk' / 'tcl-license.terms'),
        (base / 'tcl' / 'tk8.6' / 'license.terms', destination / 'tcltk' / 'tk-license.terms'),
    ):
        if not source.is_file():
            missing.append(str(source))
        copies.append((source, target))

    oidn_files = [OIDN_LICENSES / 'LICENSE.txt',
                  *sorted(OIDN_LICENSES.glob('third-party-programs*.txt'))]
    if len(oidn_files) < 2:
        missing.append('OIDN third-party-programs*.txt')
    for source in oidn_files:
        if not source.is_file():
            missing.append(str(source))
        copies.append((source, destination / 'oidn' / source.name))

    if missing:
        raise RuntimeError('Missing license files:\n' + '\n'.join(f'  {name}' for name in missing))
    for source, target in copies:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    print(f'Collected {len(copies)} license files into {destination}')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('Usage: collect_licenses.py <destination>')
    collect(Path(sys.argv[1]))
