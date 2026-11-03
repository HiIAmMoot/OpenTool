from pathlib import Path

from lightmapper import frozen

# one full frame per compared value of each Settings field, named <name>_0.png, <name>_1.png and so on
# in opentool/examples/ (a lone <name>.png is one still frame); the app loops through them
EXAMPLE_ROWS = ('texels_per_unit', 'samples', 'vertex_rays', 'bounces', 'page_max',
                'render_only_shadows', 'adaptive_sampling', 'lightmap_32bit')
EXAMPLE_DIR = frozen.bundle_dir() / 'opentool' / 'examples'

# the grey line beside each row while nothing is hovered or focused
IDLE_HINTS = {
    'texels_per_unit': 'Lightmap detail per world unit. Higher is sharper and slower.',
    'samples': 'Rays per lightmap texel. More is less noise.',
    'vertex_rays': 'Rays per vertex for bump shading. More is smoother.',
    'bounces': 'How often light bounces between surfaces.',
    'page_max': 'The largest lightmap page. Bigger fits more of the level.',
    'render_only_shadows': 'Shadows of no-collision surfaces, like a sea plane.',
    'adaptive_sampling': 'Stops casting rays at a clean texel. Faster.',
    'lightmap_32bit': 'No banding, but twice the file size and texture memory.',
}

EXAMPLE_TEXT = {
    'texels_per_unit': 'Lightmap texels per world unit. Higher is sharper and slower to bake.',
    'samples': 'Rays per lightmap texel. More rays give less noise.',
    'vertex_rays': 'Rays per vertex, for the light direction bump maps use. More give smoother bump shading.',
    'bounces': 'How often light bounces between surfaces. More light reaches the shade.',
    'page_max': 'The largest lightmap page, for the bake and the written map. Auto picks the smallest size (256 to 4096) that fits the biggest surface at your texel density, so a surface that does not fit raises it. If the level would need more than 10 pages, Auto picks a larger size.',
    'adaptive_sampling': 'Stops casting rays at a texel once it is clean enough. Faster; off casts every ray.',
    'lightmap_32bit': '32-bit pages show no banding but take twice the file size and texture memory.',
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
