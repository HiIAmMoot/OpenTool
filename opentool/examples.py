from pathlib import Path

from lightmapper import frozen

# one full frame per compared value of each Settings field, named <name>_0.png, <name>_1.png and so on
# in opentool/examples/ (a lone <name>.png is one still frame); the app loops through them
EXAMPLE_ROWS = ('texels_per_unit', 'samples', 'vertex_rays', 'bounces', 'page_max',
                'adaptive_sampling', 'lightmap_32bit', 'render_only_shadows')
EXAMPLE_DIR = frozen.bundle_dir() / 'opentool' / 'examples'

EXAMPLE_TEXT = {
    'texels_per_unit': 'Lightmap texels per world unit. Higher is sharper and slower to bake.',
    'samples': 'Rays per lightmap texel. More rays give less noise.',
    'vertex_rays': 'Rays per vertex, for the light direction bump maps use. More give smoother bump shading.',
    'bounces': 'How often light bounces between surfaces. More light reaches the shade.',
    'page_max': 'The largest lightmap page. Bigger pages fit more of the level on each.',
    'adaptive_sampling': 'Stops casting rays at a texel once it is clean enough. Faster; off casts every ray.',
    'lightmap_32bit': '32-bit pages show no banding but take twice the space.',
    'render_only_shadows': "How surfaces with no collision under them, such as a sea plane, shadow what lies beneath. By bump alpha tests them by their shader's bump map, Solid blocks all light, None lets it all through as tool does.",
}


def example_frames(name: str, folder: Path | None = None) -> list[Path]:
    base = EXAMPLE_DIR if folder is None else Path(folder)
    frames = []
    # (a gap ends the list: a frame renamed or missed would otherwise skip silently past the rest)
    while (base / f'{name}_{len(frames)}.png').is_file():
        frames.append(base / f'{name}_{len(frames)}.png')
    if not frames and (base / f'{name}.png').is_file():
        frames.append(base / f'{name}.png')
    return frames
