"""Pick each shader's albedo map, its UV scale and whether it is a cut-out that casts shadows.

Only the base colour is used for now: detail, reflection and multipurpose maps, tints,
animation and shader colours are ignored. The exception is the senv cut-out, whose alpha
comes from the bump map (the stage the game alpha-tests), not from the base map. A shader
without an albedo map gets the constant fallback colour from the caller.
"""

from dataclasses import dataclass
import logging
from pathlib import Path
import struct

import numpy as np

from .formats.bitmap_decode import decode_bitmap, decode_bitmap_alpha
from .tagio.tagfile import read_tag, resolve_reference_path


logger = logging.getLogger(__name__)
ALPHA_TESTED = 0x1
BUMP_IS_SPECULAR_MASK = 0x2
NOT_ALPHA_TESTED = 0x4
MAX_COMPOSED = 4096


@dataclass(frozen=True)
class ShaderTexture:
    albedo: str | None
    uv_scale: tuple[float, float]
    cutout: bool
    casts_shadow: bool
    cutout_alpha: str | None = None
    cutout_alpha_scale: float = 1.0


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
        bump = root.reference('bump.bump_map')[1]
        flags = root.get('properties.flags')
        # the lightmap pass (rasterizer_xbox_environment.c:749-764) alpha-tests the bump;
        # 0x2 or a missing bump uses the opaque default bump texture, so it stays solid
        cutout = bool(albedo) and bool(bump) and bool(flags & ALPHA_TESTED) and not flags & BUMP_IS_SPECULAR_MASK
        alpha = bump if cutout else None
        alpha_scale = (root.get('bump.bump_map_scale') or 1.0) if cutout else 1.0
        return ShaderTexture(albedo or None, (1.0, 1.0), cutout, True, alpha, alpha_scale)
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


def _axis(size: int, out: int, scale: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # texel centres at half-integers, wrapping, as bake.sample_bilinear(..., repeat=True)
    x = (np.arange(out) + 0.5) / out * scale * size - 0.5
    x0 = np.floor(x).astype(np.int64)
    return x0 % size, (x0 + 1) % size, x - x0


def _resample(image: np.ndarray, width: int, height: int, scale: int) -> np.ndarray:
    h, w = image.shape[:2]
    x0, x1, fx = _axis(w, width, scale)
    y0, y1, fy = _axis(h, height, scale)
    out = np.empty((height, width, image.shape[2]), np.float64)
    for y in range(height):
        row = image[y0[y]] * (1 - fy[y]) + image[y1[y]] * fy[y]
        out[y] = row[x0] * (1 - fx)[:, None] + row[x1] * fx[:, None]
    return out


def compose_cutout(colour: np.ndarray, alpha: np.ndarray, scale: int) -> np.ndarray:
    h, w = colour.shape[:2]
    ha, wa = alpha.shape
    # (read at call time so a test can patch it)
    width, height = min(MAX_COMPOSED, max(w, wa * scale)), min(MAX_COMPOSED, max(h, ha * scale))
    out = np.empty((height, width, 4), np.uint8)
    # the colour is resampled again by each pass when this is larger than the base map; that softening of
    # the bounce albedo is accepted to keep the bump's holes at full detail
    out[..., :3] = np.rint(_resample(colour[..., :3].astype(np.float64), width, height, 1))
    out[..., 3] = np.rint(_resample(alpha[..., None].astype(np.float64), width, height, scale))[..., 0]
    return out


class TextureSet:
    def __init__(self, tags_root: Path):
        self.tags_root = Path(tags_root)
        self._index: dict[str, int] = {}
        self._arrays: list[np.ndarray] = []
        self._colours: dict[str, np.ndarray | None] = {}
        self._alphas: dict[str, np.ndarray | None] = {}
        self._composed: dict[tuple[str, str, float], int] = {}

    def index(self, bitmap_path: str) -> int:
        if bitmap_path not in self._index:
            image = self._colour(bitmap_path)
            if image is None:
                self._index[bitmap_path] = -1
            else:
                self._arrays.append(image)
                self._index[bitmap_path] = len(self._arrays) - 1
        return self._index[bitmap_path]

    def texture_of(self, shader: ShaderTexture) -> int:
        if not shader.albedo:
            return -1
        if shader.cutout_alpha is None:
            return self.index(shader.albedo)
        key = (shader.albedo, shader.cutout_alpha, shader.cutout_alpha_scale)
        if key not in self._composed:
            self._composed[key] = self._compose(*key)
        return self._composed[key]

    def _compose(self, albedo: str, bump: str, scale: float) -> int:
        colour = self._colour(albedo)
        if colour is None:
            return -1
        alpha = self._alpha(bump)
        if alpha is None or scale < 1 or scale != int(scale):
            # the game reads an unusable bump as its opaque default, and only a whole-number scale tiles
            # with the base map, so the shader is solid
            logger.warning('%s: bump map %s is not usable as a cut-out (scale %s); the shader stays solid',
                           albedo, bump, scale)
            alpha, scale = np.full((1, 1), 255, np.uint8), 1
        image = compose_cutout(colour, alpha, int(scale))
        ideal = (max(colour.shape[0], alpha.shape[0] * int(scale)), max(colour.shape[1], alpha.shape[1] * int(scale)))
        if image.shape[:2] != ideal:
            logger.warning('%s with bump map %s: the cut-out texture is capped at %d, not %dx%d',
                           albedo, bump, MAX_COMPOSED, ideal[1], ideal[0])
        self._arrays.append(image)
        return len(self._arrays) - 1

    def _colour(self, bitmap_path: str) -> np.ndarray | None:
        if bitmap_path not in self._colours:
            file = resolve_reference_path(self.tags_root, 'bitm', bitmap_path)
            try:
                self._colours[bitmap_path] = decode_bitmap(read_tag(file))
            except (OSError, ValueError, KeyError, struct.error) as error:
                logger.warning('bitmap %s is not usable: %s', bitmap_path, error)
                self._colours[bitmap_path] = None
        return self._colours[bitmap_path]

    def _alpha(self, bitmap_path: str) -> np.ndarray | None:
        if bitmap_path not in self._alphas:
            file = resolve_reference_path(self.tags_root, 'bitm', bitmap_path)
            try:
                self._alphas[bitmap_path] = decode_bitmap_alpha(read_tag(file))
            except (OSError, ValueError, KeyError, struct.error):
                self._alphas[bitmap_path] = None
        return self._alphas[bitmap_path]

    def arrays(self) -> list[np.ndarray]:
        return self._arrays
