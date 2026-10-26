"""Encode stock H1A vertex buffers and ordered-dithered r5g6b5 pages.

Vertex products deliberately round in float32 before floor, matching the
engine rather than a higher-precision approximation of its quantisation.
"""

import numpy as np


ENV = np.dtype([("position", "<f4", 3), ("normal", "<f4", 3),
                ("binormal", "<f4", 3), ("tangent", "<f4", 3), ("texcoord", "<f4", 2)])
LM = np.dtype([("incident_radiosity", "<f4", 3), ("texcoord", "<f4", 2)])

_ENV_COMPRESSED = np.dtype([("position", "<f4", 3), ("normal", "<u4"),
                            ("binormal", "<u4"), ("tangent", "<u4"), ("texcoord", "<f4", 2)])
_LM_COMPRESSED = np.dtype([("incident_radiosity", "<u4"), ("texcoord", "<i2", 2)])
_BAYER = np.array([[0, 8, 2, 10], [12, 4, 14, 6],
                   [3, 11, 1, 9], [15, 7, 13, 5]])


def unpack_uncompressed(data: bytes, count: int, lightmap_count: int) -> tuple[np.ndarray, np.ndarray]:
    env = np.frombuffer(data, dtype=ENV, count=count).copy()
    lm = np.frombuffer(data, dtype=LM, count=lightmap_count, offset=count * ENV.itemsize).copy()
    return env, lm


def pack_uncompressed(env, lm) -> bytes:
    return np.asarray(env, dtype=ENV).tobytes() + np.asarray(lm, dtype=LM).tobytes()


def compress_vector(v: np.ndarray) -> np.ndarray:
    clamped = np.clip(np.asarray(v, dtype=np.float32), np.float32(-1), np.float32(1))
    scale = np.array([1023.5, 1023.5, 511.5], dtype=np.float32)
    packed = np.floor(clamped * scale).astype(np.int32)
    packed &= np.array([0x7ff, 0x7ff, 0x3ff], dtype=np.int32)
    packed = packed.astype(np.uint32)
    return ((packed[..., 2] << 11) | packed[..., 1]) << 11 | packed[..., 0]


def compress(env, lm) -> bytes:
    result_env = np.empty(len(env), dtype=_ENV_COMPRESSED)
    result_env["position"] = env["position"]
    for field in ("normal", "binormal", "tangent"):
        result_env[field] = compress_vector(env[field])
    result_env["texcoord"] = env["texcoord"]
    result_lm = np.empty(len(lm), dtype=_LM_COMPRESSED)
    result_lm["incident_radiosity"] = compress_vector(lm["incident_radiosity"])
    uv = np.clip(np.asarray(lm["texcoord"], dtype=np.float32), np.float32(-1), np.float32(1))
    result_lm["texcoord"] = np.floor(uv * np.float32(32767.5)).astype("<i2")
    return result_env.tobytes() + result_lm.tobytes()


def encode_r5g6b5(rgb: np.ndarray, dither: bool) -> np.ndarray:
    rgb = np.asarray(rgb)
    threshold = 0.5
    if dither:
        height, width = rgb.shape[:2]
        threshold = ((_BAYER[np.arange(height)[:, None] % 4,
                              np.arange(width)[None, :] % 4] + 0.5) / 16)[..., None]
    levels = np.array([31, 63, 31])
    quantised = np.clip(np.floor(rgb * levels + threshold), 0, levels).astype(np.uint16)
    return quantised[..., 0] << 11 | quantised[..., 1] << 5 | quantised[..., 2]


def decode_r5g6b5(px: np.ndarray) -> np.ndarray:
    px = np.asarray(px, dtype=np.uint16)
    return np.stack(((px >> 11) / 31, ((px >> 5) & 63) / 63, (px & 31) / 31), axis=-1)
