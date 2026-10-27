from lightmapper.settings import Settings


PRESETS: dict[str, dict] = {
    'Draft': {'texels_per_unit': 1.0, 'samples': 64, 'vertex_rays': 256, 'sky_grid': 2, 'bounces': 1, 'page_max': 512},
    'Normal': {'texels_per_unit': 2.0, 'samples': 256, 'vertex_rays': 1024, 'sky_grid': 4, 'bounces': 1, 'page_max': 512},
    'High': {'texels_per_unit': 4.0, 'samples': 512, 'vertex_rays': 4096, 'sky_grid': 4, 'bounces': 1, 'page_max': 1024},
}
PRESET_NAMES = ('Draft', 'Normal', 'High')
CUSTOM = 'Custom'
PRESET_HINTS: dict[str, str] = {
    'Draft': 'Quicker, noisier. For checking a level.',
    'Normal': "The default. Close to tool's look.",
    'High': 'Sharper shadows. About four times the lightmap texels, so slower.',
    'Custom': 'Your own values from Advanced.',
}
ADVANCED = ('texels_per_unit', 'samples', 'vertex_rays', 'sky_grid', 'bounces', 'page_max')
PAGE_MAX_CHOICES = (256, 512, 1024, 2048, 4096)
LABELS = {'texels_per_unit': 'Texel density', 'samples': 'Rays per texel', 'vertex_rays': 'Rays per vertex', 'sky_grid': 'Suns per sky light', 'bounces': 'Bounces', 'page_max': 'Max resolution'}


def parse_advanced(texts: dict[str, str]) -> dict:
    values = {}
    ranges = {'samples': (1, 4096), 'vertex_rays': (1, 65536), 'sky_grid': (1, 8), 'bounces': (0, 8)}
    for field in ADVANCED:
        text = texts.get(field, '').strip()
        if not text:
            continue
        try:
            if field == 'texels_per_unit':
                value = float(text)
                valid = 0.25 <= value <= 16
            elif field == 'page_max':
                value = int(text)
                valid = value in PAGE_MAX_CHOICES
            else:
                value = int(text)
                low, high = ranges[field]
                valid = low <= value <= high
            if not valid:
                raise ValueError
        except ValueError:
            allowed = {'texels_per_unit': '0.25 to 16', 'samples': '1 to 4096', 'vertex_rays': '1 to 65536', 'sky_grid': '1 to 8', 'bounces': '0 to 8', 'page_max': '256, 512, 1024, 2048, or 4096'}[field]
            raise ValueError(f'{LABELS[field]} must be a number from {allowed}.') from None
        values[field] = value
    return values


def build_settings(preset: str, advanced: dict, blender: str) -> Settings:
    if preset == CUSTOM:
        if set(advanced) != set(ADVANCED):
            raise ValueError('Custom needs all six Advanced values.')
        values = dict(advanced)
    else:
        values = {**PRESETS[preset], **advanced}
    return Settings(**values, blender=blender, timeout_s=None)
