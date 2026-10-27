# Third-party notices

OpenTool is licensed under version 3 of the GNU General Public License (see [LICENSE](LICENSE)). It includes or uses the following third-party work. Each part keeps its own license, and each of those licenses (MIT, BSD, HPND, Apache 2.0) allows use in a GPLv3 program.

## Included in this repository

### xatlas

`lightmapper/vendor/xatlas/` holds upstream source as vendored from [xatlas](https://github.com/jpcy/xatlas), compiled into a shared library on first use. xatlas is under the MIT license, copyright (c) 2018-2020 Jonathan Young. The full text is in `lightmapper/vendor/xatlas/LICENSE` and `xatlas.cpp:1-23`.

The same `lightmapper/vendor/xatlas/xatlas.cpp` also includes these components:

- [thekla_atlas](https://github.com/Thekla/thekla_atlas), MIT license. Its notice at lines 24-29 names copyright (c) 2013 Thekla, Inc. and copyright NVIDIA Corporation 2006, Ignacio Castano.
- [Fast-BVH](https://github.com/brandonpelfrey/Fast-BVH), MIT license, copyright (c) 2012 Brandon Pelfrey. Its notice is at lines 31-34.
- OpenNL, BSD 3-clause license, copyright (c) 2004-2010 Bruno Levy. Its full notice begins at line 3886 and is reproduced below for binary distributions as well.

### OpenNL license

Copyright (c) 2004-2010, Bruno Levy
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice,
  this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.
* Neither the name of the ALICE Project-Team nor the names of its
  contributors may be used to endorse or promote products derived from this
  software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

If you modify this software, you should include a notice giving the
name of the person performing the modification, the date of modification,
and the reason for such modification.

### Tag definitions

`lightmapper/tagio/layouts/` holds XML tag layouts from General-101's [Halo Asset Blender Development Toolset](https://github.com/General-101/Halo-Asset-Blender-Development-Toolset), under its MIT license, copyright (c) 2020 Steven Garcia. They describe the H1A editing kit's tag definitions. The full license text is in `lightmapper/tagio/layouts/LICENSE`. The upstream commit and any local edits were not recorded, so these files are not claimed to be an exact upstream snapshot.

## Included in the app download

The Windows app includes the libraries below. Full license texts are in `licenses/` beside the executables, except for the Microsoft runtime terms linked below and the xatlas notices above.

Python 3.11 is included under the PSF License. Its license also covers the bundled `python311.dll`.

Tcl and Tk provide the window toolkit. They use BSD-style licenses; see `licenses/tcltk/`.

The PyInstaller bootloader uses GPLv2 or later with an exception that permits distributing bundled applications under their own licenses. See `licenses/pyinstaller/`.

NumPy and SciPy use BSD licenses. Their wheels include OpenBLAS and Fortran runtime libraries; the wheel license texts include the BSD terms and the GCC Runtime Library Exception.

Numba and llvmlite use BSD licenses. llvmlite includes LLVM under Apache 2.0 with the LLVM exception. See their license folders for the bundled components' notices.

cffi and pycparser use MIT and BSD licenses, respectively.

The packaging library is included because Pillow imports it. It offers Apache 2.0 or BSD terms.

Pillow is included under the HPND license.

pyoidn, the Python binding for Open Image Denoise, uses the MIT license.

Open Image Denoise, oneTBB and the oneAPI/Level Zero loaders use Apache 2.0 and MIT licenses. Their license texts and component notices are in `licenses/oidn/`.

The Microsoft Visual C++ runtime DLLs are redistributed as Distributable Code under the [Visual Studio license terms](https://visualstudio.microsoft.com/license-terms/vs2022-ga-community/).

xatlas is built from the vendored source and included under the MIT license. Its component notices are reproduced above.

## Used by a source checkout

A source checkout uses the programs below as separate installs. The app download bundles everything here except Blender, which the user installs.

- [Blender](https://www.blender.org) (GNU GPL) runs as a separate program. `lightmapper/blender/bake.py` imports Blender's Python API, `bpy` and `mathutils`, so it must be under a GPL-compatible license. It is, as part of OpenTool under GPLv3.
- [Intel Open Image Denoise](https://www.openimagedenoise.org) (Apache 2.0), through the separate [pyoidn](https://github.com/Hyiker/pyoidn) binding (MIT license).
- [NumPy](https://numpy.org), [SciPy](https://scipy.org) and [Numba](https://numba.pydata.org) (BSD licenses).
- [Pillow](https://python-pillow.org) (HPND license), used to decode bitmap images.

Halo is a trademark of Microsoft. OpenTool is not affiliated with or endorsed by Microsoft, Halo Studios or Bungie.
