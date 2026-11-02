from pathlib import Path

from lightmapper import frozen

# one image per Settings field, named after it, showing the compared values side by side: the
# generator writes opentool/examples/<name>.png for each name here
EXAMPLE_ROWS = ('texels_per_unit', 'samples', 'vertex_rays', 'bounces', 'page_max',
                'adaptive_sampling', 'ignore_normals_mode', 'lightmap_32bit', 'render_only_shadows')
EXAMPLE_DIR = frozen.bundle_dir() / 'opentool' / 'examples'

EXAMPLE_TEXT = {
    'texels_per_unit': 'Lightmap texels per world unit. Higher is sharper and slower to bake.',
    'samples': 'Rays per lightmap texel. More rays give less noise.',
    'vertex_rays': 'Rays per vertex, for the light direction bump maps use. More give smoother bump shading.',
    'bounces': 'How often light bounces between surfaces. More light reaches the shade.',
    'page_max': 'The largest lightmap page. Bigger pages fit more of the level on each.',
    'adaptive_sampling': 'Stops casting rays at a texel once it is clean enough. Faster; off casts every ray.',
    'ignore_normals_mode': "How surfaces that ignore normals, such as foliage, are lit. Traditional copies tool; Enhanced lights them per texel.",
    'lightmap_32bit': '32-bit pages show no banding but take twice the space.',
    'render_only_shadows': "How surfaces with no collision under them, such as a sea plane, shadow what lies beneath. By bump alpha tests them by their shader's bump map, Solid blocks all light, None lets it all through as tool does.",
}


def example_image(name: str, folder: Path | None = None) -> Path | None:
    path = (EXAMPLE_DIR if folder is None else Path(folder)) / f'{name}.png'
    return path if path.is_file() else None
