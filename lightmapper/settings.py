"""Lightmapper settings: chart fields, bake fields and the calibrated weights.

The weight fields (light-source weights, tone gamma, vector mode) default to
calibration.json; a settings file may still override any of them. Keys are the
field names and an unknown key is an error.
"""

from dataclasses import asdict, dataclass, field, fields
import json
import os
from pathlib import Path

from .charts import ChartSettings


CALIBRATION_PATH = Path(__file__).with_name('calibration.json')
DEFAULT_BLENDER = r'C:\Program Files\Blender Foundation\Blender 5.2\blender.exe'
RADIOSITY_MODES = ('coherence', 'scaled')


def _calibrated(key):
    # the shipped weights are tool's analytic model, not a fit: a Cycles sun supplies irradiance,
    # and a diffuse bake returns irradiance/pi, so tool's sun sample needs a factor pi.
    # emission strength is radiance: a receiver surrounded by radiance L bakes to L, matching
    # tool's power x colour radiosity with form factor at most 1, so emission needs weight 1.
    # ambient needs 1 and tool applies no gamma; a draft fit learns its unshot remainder as ambient
    def read():
        return json.loads(CALIBRATION_PATH.read_text())[key]
    return field(default_factory=read)


@dataclass
class Settings(ChartSettings):
    samples: int = 256
    # edge texels own few of their 9 sub-texels, so they are baked again until each has about this many samples in
    # all; None follows the interior's count, so the edges keep up when samples changes; 0 bakes every texel once
    texel_samples: int | None = None
    # one bounce matches tool's shadow brightness on bloodgulch (1.05x tool; three bounces give 1.12x)
    bounces: int = 1
    albedo_scale: float = 1.0
    albedo_fallback: float = 0.5
    denoise: bool = True
    stitch_seams: bool = True
    dither: bool = True
    # 64 let one ray that hit a bright emitter carry most of a corner's light, so the incident vectors went 15-39
    # degrees off and the bump shading showed triangles on flat walls (Chiron); 4096 brings them within 2-5 degrees
    vertex_rays: int = 4096
    # tool's final-quality grid of suns per sky light (its draft uses 2)
    sky_grid: int = 4
    # tool never lets object shaders emit; off leaves only the BSP's own emitters, for an in-game A/B
    scenery_emission: bool = True
    # how far a plane-away face's sun reaches in from its lit boundary: tool's final-quality floor for the mesh cells
    # its radiosity refines to near a shadow edge (non-emissive, detail level 0)
    reach: float = 0.225
    device: str = 'OPTIX'
    blender: str = field(default_factory=lambda: os.environ.get('LIGHTMAPPER_BLENDER', DEFAULT_BLENDER))
    timeout_s: int | None = 7200  # None means no limit, which the app uses because a CPU bake can take hours
    w_sun: float = _calibrated('w_sun')
    w_emit: float = _calibrated('w_emit')
    w_amb: float = _calibrated('w_amb')
    gamma: float = _calibrated('gamma')
    radiosity_mode: str = _calibrated('radiosity_mode')
    radiosity_scale: float = _calibrated('radiosity_scale')

    def __post_init__(self):
        super().__post_init__()
        if self.sky_grid < 1:
            raise ValueError(f'sky_grid must be at least 1, not {self.sky_grid}')
        if not self.reach > 0:
            raise ValueError(f'reach must be positive, not {self.reach}')
        if self.radiosity_mode not in RADIOSITY_MODES:
            raise ValueError(f'radiosity_mode must be one of {RADIOSITY_MODES}, not {self.radiosity_mode!r}')

    def resolved_texel_samples(self) -> int:
        # the interior bakes 9 sub-texels at round(samples / 9) each, as the Blender side rounds it
        if self.texel_samples is None:
            return 9 * max(1, round(self.samples / 9))
        return self.texel_samples

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def load_settings(path: Path | None) -> Settings:
    if path is None:
        return Settings()
    data = json.loads(Path(path).read_text())
    unknown = sorted(set(data) - {f.name for f in fields(Settings)})
    if unknown:
        raise ValueError(f'unknown settings key(s): {", ".join(unknown)}')
    return Settings(**data)
