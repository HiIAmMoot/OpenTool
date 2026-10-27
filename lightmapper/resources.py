from pathlib import Path

from .frozen import bundle_dir


REQUIRED: tuple[str, ...] = (
    'lightmapper/calibration.json',
    'lightmapper/blender/bake.py',
    'lightmapper/shaders/vertex_pass.slang',
    'lightmapper/tagio/layouts/LICENSE',
    'lightmapper/tagio/layouts/bitmap.xml',
    'lightmapper/tagio/layouts/gbxmodel.xml',
    'lightmapper/tagio/layouts/scenario_structure_bsp.xml',
    'lightmapper/tagio/layouts/scenario.xml',
    'lightmapper/tagio/layouts/scenery.xml',
    'lightmapper/tagio/layouts/shader_environment.xml',
    'lightmapper/tagio/layouts/shader_model.xml',
    'lightmapper/tagio/layouts/shader_transparent_chicago_extended.xml',
    'lightmapper/tagio/layouts/shader_transparent_chicago.xml',
    'lightmapper/tagio/layouts/shader_transparent_glass.xml',
    'lightmapper/tagio/layouts/sky.xml',
    'lightmapper/vendor/xatlas/LICENSE',
)


def missing(root: Path | None = None) -> list[str]:
    base = bundle_dir() if root is None else Path(root)
    return [name for name in REQUIRED if not (base / name).is_file()]
