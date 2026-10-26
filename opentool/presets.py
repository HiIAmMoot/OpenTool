from lightmapper.settings import Settings


PRESETS: dict[str, dict] = {
    'Draft': {'samples': 64, 'vertex_rays': 1024, 'sky_grid': 2},
    'Normal': {},
    'High': {'texels_per_unit': 4.0},
}
PRESET_NAMES = ('Draft', 'Normal', 'High')
PRESET_HINTS: dict[str, str] = {
    'Draft': 'Quicker, noisier. For checking a level.',
    'Normal': "The default. Close to tool's look.",
    'High': 'Sharper shadows. About four times the lightmap texels, so slower.',
}
ADVANCED = ('texels_per_unit', 'bounces', 'page_max')
PAGE_MAX_CHOICES = (256, 512, 1024)


def parse_advanced(texts: dict[str, str]) -> dict:
    values = {}
    for field in ADVANCED:
        text = texts.get(field, '').strip()
        if not text:
            continue
        try:
            if field == 'texels_per_unit':
                value = float(text)
                if not 0.25 <= value <= 16:
                    raise ValueError
            elif field == 'bounces':
                value = int(text)
                if not 0 <= value <= 8:
                    raise ValueError
            else:
                value = int(text)
                if value not in PAGE_MAX_CHOICES:
                    raise ValueError
        except ValueError:
            allowed = {'texels_per_unit': '0.25 to 16', 'bounces': '0 to 8', 'page_max': '256, 512, or 1024'}[field]
            label = {'texels_per_unit': 'Texel density', 'bounces': 'Bounces', 'page_max': 'Page size'}[field]
            raise ValueError(f'{label} must be a number from {allowed}.' if field != 'page_max'
                             else f'{label} must be 256, 512, or 1024.') from None
        values[field] = value
    return values


def build_settings(preset: str, advanced: dict, blender: str) -> Settings:
    values = {**PRESETS[preset], **advanced}
    return Settings(**values, blender=blender, timeout_s=None)
