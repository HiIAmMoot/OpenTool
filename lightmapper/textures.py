"""Pick each shader's albedo map, its UV scale and whether it is a cut-out that casts shadows.

Only the base colour is used for now: detail, bump, reflection and multipurpose
maps, tints, animation and shader colours are ignored. A shader without an
albedo map gets the constant fallback colour from the caller.
"""

from dataclasses import dataclass
import logging
from pathlib import Path
import struct

import numpy as np

from .formats.bitmap_decode import decode_bitmap
from .tagio.tagfile import read_tag, resolve_reference_path


logger = logging.getLogger(__name__)
ALPHA_TESTED = 0x1
NOT_ALPHA_TESTED = 0x4


@dataclass(frozen=True)
class ShaderTexture:
    albedo: str | None
    uv_scale: tuple[float, float]
    cutout: bool
    casts_shadow: bool


NO_TEXTURE = ShaderTexture(None, (1.0, 1.0), False, False)


def _scale(shader: str, model: tuple[float, float], u: float, v: float) -> tuple[float, float]:
    if not (model[0] and model[1] and u and v):
        # a zero factor would collapse the texture to a point, so each reads as 1
        logger.info('%s: a zero map scale reads as 1', shader)
    return (model[0] or 1.0) * (u or 1.0), (model[1] or 1.0) * (v or 1.0)


def shader_texture(tags_root: Path, group: str | None, path: str,
                   model_uv_scale: tuple[float, float] = (1.0, 1.0)) -> ShaderTexture:
    if group not in ('senv', 'sgla', 'soso', 'schi', 'scex') or not path:
        return NO_TEXTURE
    root = read_tag(resolve_reference_path(tags_root, group, path)).root
    if group == 'senv':
        albedo = root.reference('diffuse.base_map')[1]
        cutout = bool(albedo) and bool(root.get('properties.flags') & ALPHA_TESTED)
        return ShaderTexture(albedo or None, (1.0, 1.0), cutout, True)
    if group == 'sgla':
        albedo = root.reference('background_tint.background_tint_map')[1]
        return ShaderTexture(albedo or None, (1.0, 1.0), False, False)
    if group == 'soso':
        albedo = root.reference('maps.base_map')[1]
        scale = _scale(path, model_uv_scale, root.get('maps.map_u_scale'), root.get('maps.map_v_scale'))
        cutout = bool(albedo) and not root.get('properties.flags') & NOT_ALPHA_TESTED
        return ShaderTexture(albedo or None, scale, cutout, True)
    stages = root.block('maps' if group == 'schi' else '_4_stage_maps')
    albedo = stages[0].reference('parameters.map')[1] if stages else ''
    if not albedo:
        return NO_TEXTURE
    tested = bool(root.get('properties.flags') & ALPHA_TESTED)
    scale = _scale(path, model_uv_scale, stages[0].get('parameters.map_u_scale'),
                   stages[0].get('parameters.map_v_scale'))
    return ShaderTexture(albedo, scale, tested, tested)


class TextureSet:
    def __init__(self, tags_root: Path):
        self.tags_root = Path(tags_root)
        self._index: dict[str, int] = {}
        self._arrays: list[np.ndarray] = []

    def index(self, bitmap_path: str) -> int:
        if bitmap_path not in self._index:
            self._index[bitmap_path] = self._load(bitmap_path)
        return self._index[bitmap_path]

    def _load(self, bitmap_path: str) -> int:
        file = resolve_reference_path(self.tags_root, 'bitm', bitmap_path)
        try:
            image = decode_bitmap(read_tag(file))
        except (OSError, ValueError, KeyError, struct.error) as error:
            logger.warning('bitmap %s is not usable: %s', bitmap_path, error)
            return -1
        self._arrays.append(image)
        return len(self._arrays) - 1

    def arrays(self) -> list[np.ndarray]:
        return self._arrays
