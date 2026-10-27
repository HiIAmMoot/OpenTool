# SlangPy license sources

These notices cover the Windows x64 SlangPy 0.43.1 wheel and the components
in its native libraries, shader includes and data. They are copied to
`licenses/slangpy-third-party/` in the app. The wheel's own `LICENSE` is also
collected under `licenses/slangpy/licenses/`.

sgl.dll is built with NVIDIA OptiX support compiled in, as distributed in the SlangPy wheel; the OptiX SDK headers are not included.

SlangPy's pins are recorded in its [v0.43.1 source tree](https://github.com/shader-slang/slangpy/tree/v0.43.1),
especially `external/CMakeLists.txt`, the gitlinks under `external/`, and
`CMakePresets.json`. Its slang-rhi gitlink is
`1a9768741246b1f356116d5dfa305933034ce049`. Slang is pinned to 2026.12;
its dependency gitlinks are in the [v2026.12 tree](https://github.com/shader-slang/slang/tree/v2026.12).
The static image-library versions below come from SlangPy's vcpkg gitlink
`120deac3062162151622ca4860575a33844ba10b`, under `ports/<name>/`.
SlangPy's `REUSE.toml` also credits NVIDIA CORPORATION & AFFILIATES and The
Khronos Group (2024), Wenzel Jakob (2017) for the Apache-2.0 bitmap code, and
Martin Moene (2020) for the Boost-licensed bit header. Slang's `REUSE.toml`
credits Carnegie Mellon University and NVIDIA Corporation (2016), The Khronos
Group (2024), and Alexander Peslyak (2001) for its Apache-2.0 crypto code.

Each file contains the upstream license text, or the license section of the
vendored header. Where upstream separates a notice, license and exception,
the texts are concatenated in that order.

| File | Component and pinned source | License |
| --- | --- | --- |
| `slangpy.txt` | SlangPy 0.43.1: `LICENSE`, `LICENSES/Apache-2.0.txt`, `LICENSES/LLVM-exception.txt` | Apache-2.0 WITH LLVM-exception |
| `slang.txt` | Slang 2026.12: same three paths; covers `slang-compiler.dll`, `slang-glsl-module.dll` and `slang-standard-module-2026.12/` | Apache-2.0 WITH LLVM-exception |
| `slang-rhi.txt` | slang-rhi `1a9768741246b1f356116d5dfa305933034ce049`: same three paths; native runtime and `shaders/slang-rhi/` | Apache-2.0 WITH LLVM-exception |
| `fmt.txt` | [fmt](https://github.com/fmtlib/fmt) `40626af88bd7df9a5fb80be7b25ac85b122d6c21`, `LICENSE` | MIT |
| `glfw.txt` | [GLFW](https://github.com/glfw/glfw) `dc46d3f8129712e42856c20e99a604a3b08ad581`, `LICENSE.md` | Zlib |
| `nanobind.txt` | [nanobind](https://github.com/skallweitNV/nanobind) `0b719c12ad2d82fc601c317ffd57a07ba91ac36a`, `LICENSE` | BSD-3-Clause |
| `robin-map.txt` | [robin-map](https://github.com/Tessil/robin-map) `4ec1bf19c6a96125ea22062f38c2cf5b958e448e`, `LICENSE`; nanobind's gitlink | MIT |
| `nanothread.txt` | [nanothread](https://github.com/mitsuba-renderer/nanothread) `dd8d5f23a13c38d4d2a72f276f862c82b4ad35cc`, `LICENSE` | BSD-3-Clause |
| `tevclient.txt` | [tevclient](https://github.com/westlicht/tevclient) `db443d5cba56b60aec2079465726a29b9078bea0`, `LICENSE` | BSD-3-Clause |
| `lmdb.txt` | SlangPy 0.43.1 `external/lmdb/COPYRIGHT` and `LICENSE` | OpenLDAP Public License 2.8 |
| `imgui.txt` | [Dear ImGui](https://github.com/ocornut/imgui) `v1.92.7-docking`, `LICENSE.txt`; its embedded stb notices are in `stb.txt` | MIT |
| `stb.txt` | SlangPy 0.43.1 `external/include/stb_image.h` v2.30 license section, identical to `stb_image_write.h` v1.16 and ImGui v1.92.7-docking's `imstb_rectpack.h`, `imstb_textedit.h`, `imstb_truetype.h` license sections | MIT or public domain |
| `tinyexr.txt` | SlangPy 0.43.1 `external/include/tinyexr.h` license section, including its OpenEXR notice | BSD-3-Clause |
| `renderdoc.txt` | SlangPy 0.43.1 `external/include/renderdoc_app.h` license section (API header only, no RenderDoc runtime) | MIT |
| `offset-allocator.txt` | slang-rhi pinned above, `src/core/offset-allocator.cpp`: Sebastian Aaltonen 2023 notice and MIT terms | MIT |
| `nvapi.txt` | [NVAPI](https://github.com/NVIDIA/nvapi) `cd6918f60b3c9a0476fdfe7e89bb32330602049d` (R610), `License.txt`; native import library and `shaders/nvapi/` headers | MIT |
| `fonts.txt` | [SlangPy data](https://github.com/shader-slang/slangpy-data) `6ecdf7c1dffd3c9ea1eb40c00eeb968fd5e804c1`, `fonts/LICENSE.txt`; Inconsolata and Montserrat copyrights retained from the bundled fonts' name tables | SIL OFL 1.1 |
| `libjpeg-turbo.txt` | [libjpeg-turbo](https://github.com/libjpeg-turbo/libjpeg-turbo) 3.1.1, `LICENSE.md` and `README.ijg` | IJG, BSD-3-Clause, Zlib |
| `libpng.txt` | [libpng](https://github.com/pnggroup/libpng) v1.6.50, `LICENSE` | libpng-2.0 |
| `openexr.txt` | [OpenEXR](https://github.com/AcademySoftwareFoundation/openexr) v3.3.5, `LICENSE.md` | BSD-3-Clause |
| `imath.txt` | [Imath](https://github.com/AcademySoftwareFoundation/Imath) v3.2.0, `LICENSE.md` | BSD-3-Clause |
| `libdeflate.txt` | [libdeflate](https://github.com/ebiggers/libdeflate) v1.24, `COPYING` | MIT |
| `zlib.txt` | [zlib](https://github.com/madler/zlib) v1.3.1, `LICENSE` | Zlib |
| `asmjit.txt` | [AsmJit](https://github.com/asmjit/asmjit) `e1b20711cc40c29ec4918b54e328ace96470f6e5`, `LICENSE.md` | Zlib |
| `bit.txt` | SlangPy 0.43.1 `src/sgl/stl/bit.h`, Martin Moene notice from `REUSE.toml` and `LICENSES/BSL-1.0.txt` | Boost Software License 1.0 |
| `dxc-api.txt` | Slang 2026.12 `external/dxc/dxcapi.h`, assigned `LicenseRef-UOI-NCSA` by its `REUSE.toml`; full text from `LICENSES/LicenseRef-UOI-NCSA.txt` (API declarations, not the excluded DXC DLL) | University of Illinois/NCSA |
| `glslang.txt` | [glslang](https://github.com/KhronosGroup/glslang) `d1f52c8993a501bd52d4fbd044bfeb9ecdceb9f4`, `LICENSE.txt`; `slang-glslang.dll` | BSD and MIT notices |
| `spirv-tools.txt` | [SPIRV-Tools](https://github.com/KhronosGroup/SPIRV-Tools) `12f9d942a7b19c039bbf2a18b84a36bb7b12664a`, `LICENSE` | Apache-2.0 |
| `spirv-headers.txt` | [SPIRV-Headers](https://github.com/KhronosGroup/SPIRV-Headers) `c63848ecf2200425511319fd8bf2c17b751e501e`, `LICENSE` | MIT |
| `vulkan-headers.txt` | [Vulkan-Headers](https://github.com/KhronosGroup/Vulkan-Headers) v1.4.347 (slang-rhi) and `39f924b810e561fd86b2558b6711ca68d4363f68` (Slang), identical `LICENSE.md` and `LICENSES/{Apache-2.0,MIT}.txt` | Apache-2.0 or MIT |
| `miniz.txt` | [miniz](https://github.com/richgel999/miniz) `6ef6c68f4fcbb8287aa8edf9c6670804932f41c6`, `LICENSE` | MIT |
| `lz4.txt` | [LZ4](https://github.com/lz4/lz4) `7f71be01f2c6f6b2c3cddb67eef8c38eab12ffd4`, `lib/LICENSE` (library only) | BSD-2-Clause |
| `unordered-dense.txt` | [unordered_dense](https://github.com/martinus/unordered_dense) `73f3cbb237e84d483afafc743f1f14ec53e12314`, `LICENSE` | MIT |
| `cmark.txt` | [swift-cmark](https://github.com/swiftlang/swift-cmark) `924936d0427cb25a61169739a7660230bffa6ea6`, `COPYING`, including vendored-component notices | BSD-2-Clause and MIT |
| `fast-float.txt` | [fast_float](https://github.com/fastfloat/fast_float) `e0b53eaf63c6d00e0725788ef1dbb759aa321d79`, `LICENSE-MIT` (MIT option selected) | MIT |
| `mimalloc.txt` | [mimalloc](https://github.com/microsoft/mimalloc) v2.1.7, `LICENSE`; Slang's optional SPIRV-Tools static allocator | MIT |

The D3D12 SDK DLLs, DXC, DXIL, PIX, `slang-llvm.dll`, OptiX SDK headers,
test images, tests and benchmarks are not bundled. SlangPy imports its small
`torchintegration` package unconditionally, so that package is retained under
SlangPy's license; PyTorch itself is not bundled.
No license for an excluded SDK is represented as permission to redistribute it.

This software is based in part on the work of the Independent JPEG Group.
