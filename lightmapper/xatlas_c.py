"""ctypes binding of the vendored xatlas C API, built on first use.

The PyPI wheels of xatlas 0.0.10 and 0.0.11 were observed to segfault in add_mesh
on Windows, so the upstream source as vendored under vendor/xatlas is compiled
with clang into a shared library instead. The C API reports uvs in texels of
the atlas, not normalised as the Python package did.
"""

import ctypes as C
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

from . import frozen


VENDOR = Path(__file__).resolve().parent / "vendor" / "xatlas"
BUILD = VENDOR / "build"
CLANG_FALLBACK = Path(r"C:\Program Files\LLVM\bin\clang++.exe")
VERTEX = np.dtype([("atlas", "<i4"), ("chart", "<i4"), ("uv", "<f4", 2), ("xref", "<u4")])


class _Chart(C.Structure):
    _fields_ = [("faceArray", C.POINTER(C.c_uint32)), ("atlasIndex", C.c_uint32), ("faceCount", C.c_uint32),
                ("type", C.c_int), ("material", C.c_uint32)]


class _Vertex(C.Structure):
    _fields_ = [("atlasIndex", C.c_int32), ("chartIndex", C.c_int32), ("uv", C.c_float * 2), ("xref", C.c_uint32)]


class _Mesh(C.Structure):
    _fields_ = [("chartArray", C.POINTER(_Chart)), ("indexArray", C.POINTER(C.c_uint32)),
                ("vertexArray", C.POINTER(_Vertex)), ("chartCount", C.c_uint32), ("indexCount", C.c_uint32),
                ("vertexCount", C.c_uint32)]


class _Atlas(C.Structure):
    _fields_ = [("image", C.POINTER(C.c_uint32)), ("meshes", C.POINTER(_Mesh)),
                ("utilization", C.POINTER(C.c_float)), ("width", C.c_uint32), ("height", C.c_uint32),
                ("atlasCount", C.c_uint32), ("chartCount", C.c_uint32), ("meshCount", C.c_uint32),
                ("texelsPerUnit", C.c_float)]


class _MeshDecl(C.Structure):
    _fields_ = [("vertexPositionData", C.c_void_p), ("vertexNormalData", C.c_void_p),
                ("vertexUvData", C.c_void_p), ("indexData", C.c_void_p), ("faceIgnoreData", C.c_void_p),
                ("faceMaterialData", C.c_void_p), ("faceVertexCount", C.c_void_p), ("vertexCount", C.c_uint32),
                ("vertexPositionStride", C.c_uint32), ("vertexNormalStride", C.c_uint32),
                ("vertexUvStride", C.c_uint32), ("indexCount", C.c_uint32), ("indexOffset", C.c_int32),
                ("faceCount", C.c_uint32), ("indexFormat", C.c_int), ("epsilon", C.c_float)]


class _UvMeshDecl(C.Structure):
    _fields_ = [("vertexUvData", C.c_void_p), ("indexData", C.c_void_p), ("faceMaterialData", C.c_void_p),
                ("vertexCount", C.c_uint32), ("vertexStride", C.c_uint32), ("indexCount", C.c_uint32),
                ("indexOffset", C.c_int32), ("indexFormat", C.c_int)]


class _ChartOptions(C.Structure):
    _fields_ = [("paramFunc", C.c_void_p), ("maxChartArea", C.c_float), ("maxBoundaryLength", C.c_float),
                ("normalDeviationWeight", C.c_float), ("roundnessWeight", C.c_float),
                ("straightnessWeight", C.c_float), ("normalSeamWeight", C.c_float),
                ("textureSeamWeight", C.c_float), ("maxCost", C.c_float), ("maxIterations", C.c_uint32),
                ("useInputMeshUvs", C.c_bool), ("fixWinding", C.c_bool)]


class _PackOptions(C.Structure):
    _fields_ = [("maxChartSize", C.c_uint32), ("padding", C.c_uint32), ("texelsPerUnit", C.c_float),
                ("resolution", C.c_uint32), ("bilinear", C.c_bool), ("blockAlign", C.c_bool),
                ("bruteForce", C.c_bool), ("createImage", C.c_bool), ("rotateChartsToAxis", C.c_bool),
                ("rotateCharts", C.c_bool)]


@dataclass
class PackSettings:
    resolution: int = 1024
    texels_per_unit: float = 2.0
    padding: int = 4
    bilinear: bool = True
    rotate_charts: bool = True
    brute_force: bool = False
    max_chart_area: float = 0.0
    max_boundary_length: float = 0.0
    normal_seam_weight: float = 4.0


_dll = None


def _find_clang() -> str:
    found = shutil.which("clang++")
    if found:
        return found
    if CLANG_FALLBACK.is_file():
        return str(CLANG_FALLBACK)
    raise RuntimeError("clang++ not found: install LLVM (https://releases.llvm.org) and put it on PATH "
                       "to build the vendored xatlas library")


def _library_path() -> Path:
    return BUILD / ("xatlas.dll" if sys.platform == "win32" else "libxatlas.so")


def build() -> Path:
    library = _library_path()
    source = VENDOR / "xatlas.cpp"
    newest = max(source.stat().st_mtime, (VENDOR / "xatlas.h").stat().st_mtime,
                 (VENDOR / "xatlas_c.h").stat().st_mtime)
    if library.is_file() and library.stat().st_mtime >= newest:
        return library
    BUILD.mkdir(exist_ok=True)
    # compiled to a temp name so a half-written library is never loaded
    temp = library.with_name(f"{os.getpid()}-{library.name}")
    command = [_find_clang(), "-O2", "-std=c++17", "-shared", "-DXATLAS_C_API=1", "-DXATLAS_EXPORT_API=1",
               "-o", str(temp), str(source)]
    if sys.platform != "win32":
        command.insert(4, "-fPIC")
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        temp.unlink(missing_ok=True)
        raise RuntimeError(f"building xatlas failed:\n{result.stderr}")
    os.replace(temp, library)
    return library


def library() -> Path:
    if frozen.is_frozen():
        path = _library_path()
        if not path.is_file():
            raise RuntimeError('xatlas.dll is missing from the OpenTool folder; unzip the download again')
        return path
    return build()


def _load():
    global _dll
    if _dll is None:
        dll = C.CDLL(str(library()))
        dll.xatlasCreate.restype = C.POINTER(_Atlas)
        dll.xatlasDestroy.argtypes = [C.POINTER(_Atlas)]
        dll.xatlasAddMesh.argtypes = [C.POINTER(_Atlas), C.POINTER(_MeshDecl), C.c_uint32]
        dll.xatlasAddMesh.restype = C.c_int
        dll.xatlasAddUvMesh.argtypes = [C.POINTER(_Atlas), C.POINTER(_UvMeshDecl)]
        dll.xatlasAddUvMesh.restype = C.c_int
        dll.xatlasGenerate.argtypes = [C.POINTER(_Atlas), C.POINTER(_ChartOptions), C.POINTER(_PackOptions)]
        dll.xatlasMeshDeclInit.argtypes = [C.POINTER(_MeshDecl)]
        dll.xatlasUvMeshDeclInit.argtypes = [C.POINTER(_UvMeshDecl)]
        dll.xatlasChartOptionsInit.argtypes = [C.POINTER(_ChartOptions)]
        dll.xatlasPackOptionsInit.argtypes = [C.POINTER(_PackOptions)]
        dll.xatlasAddMeshErrorString.restype = C.c_char_p
        dll.xatlasAddMeshErrorString.argtypes = [C.c_int]
        _dll = dll
    return _dll


class Atlas:
    def __init__(self):
        self._dll = _load()
        self._handle = self._dll.xatlasCreate()
        self._keep = []
        self._kind = None

    def __del__(self):
        handle = getattr(self, "_handle", None)
        if handle:
            self._dll.xatlasDestroy(handle)
            self._handle = None

    def _claim(self, kind: str):
        # xatlas refuses a mesh and a uv mesh in one atlas (xatlas.cpp:9267)
        if self._kind not in (None, kind):
            raise ValueError("meshes and uv meshes cannot be added to the same atlas")
        self._kind = kind

    def _check(self, error: int):
        if error:
            raise ValueError("xatlas refused the mesh: " + self._dll.xatlasAddMeshErrorString(error).decode())

    def add_mesh(self, positions, tris, normals=None):
        self._claim("mesh")
        pos = np.ascontiguousarray(positions, np.float32)
        idx = np.ascontiguousarray(tris, np.uint32).ravel()
        decl = _MeshDecl()
        self._dll.xatlasMeshDeclInit(C.byref(decl))
        decl.vertexPositionData = pos.ctypes.data
        decl.vertexCount = len(pos)
        decl.vertexPositionStride = 12
        keep = [pos, idx]
        if normals is not None:
            nor = np.ascontiguousarray(normals, np.float32)
            decl.vertexNormalData = nor.ctypes.data
            decl.vertexNormalStride = 12
            keep.append(nor)
        decl.indexData = idx.ctypes.data
        decl.indexCount = len(idx)
        decl.indexFormat = 1
        # (xatlas keeps pointers into the caller's arrays until generate returns)
        self._keep.append(keep)
        self._check(self._dll.xatlasAddMesh(self._handle, C.byref(decl), 0))

    def add_uv_mesh(self, uvs, tris):
        self._claim("uv")
        uv = np.ascontiguousarray(uvs, np.float32)
        idx = np.ascontiguousarray(tris, np.uint32).ravel()
        decl = _UvMeshDecl()
        self._dll.xatlasUvMeshDeclInit(C.byref(decl))
        decl.vertexUvData = uv.ctypes.data
        decl.vertexCount = len(uv)
        decl.vertexStride = 8
        decl.indexData = idx.ctypes.data
        decl.indexCount = len(idx)
        decl.indexFormat = 1
        self._keep.append([uv, idx])
        self._check(self._dll.xatlasAddUvMesh(self._handle, C.byref(decl)))

    def generate(self, pack: PackSettings):
        chart = _ChartOptions()
        self._dll.xatlasChartOptionsInit(C.byref(chart))
        chart.maxChartArea = pack.max_chart_area
        chart.maxBoundaryLength = pack.max_boundary_length
        chart.normalSeamWeight = pack.normal_seam_weight
        options = _PackOptions()
        self._dll.xatlasPackOptionsInit(C.byref(options))
        options.resolution = pack.resolution
        options.texelsPerUnit = pack.texels_per_unit
        options.padding = pack.padding
        options.bilinear = pack.bilinear
        options.rotateCharts = pack.rotate_charts
        options.bruteForce = pack.brute_force
        self._dll.xatlasGenerate(self._handle, C.byref(chart), C.byref(options))
        self._keep.clear()

    @property
    def page_count(self) -> int:
        return self._handle.contents.atlasCount

    @property
    def width(self) -> int:
        return self._handle.contents.width

    @property
    def height(self) -> int:
        return self._handle.contents.height

    def vertices(self, mesh: int) -> np.ndarray:
        m = self._handle.contents.meshes[mesh]
        if not m.vertexCount:
            return np.empty(0, VERTEX)
        raw = (C.c_char * (m.vertexCount * C.sizeof(_Vertex))).from_address(C.addressof(m.vertexArray.contents))
        return np.frombuffer(raw, VERTEX).copy()

    def indices(self, mesh: int) -> np.ndarray:
        m = self._handle.contents.meshes[mesh]
        raw = (C.c_uint32 * m.indexCount).from_address(C.addressof(m.indexArray.contents))
        return np.frombuffer(raw, np.uint32).reshape(-1, 3).astype(np.int64)
