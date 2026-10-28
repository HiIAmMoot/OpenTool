"""Build stock lightmap bitmap pages and their source colour plate."""

import numpy as np
import struct
import zlib
import logging

from ..tagio.tagfile import Tag, new_element, new_tag


logger = logging.getLogger(__name__)
# the source plate dimensions are unsigned ShortInteger fields, not the runtime page limits
PLATE_DIMENSION_MAX = np.iinfo(np.uint16).max


def _plate_layout(sizes):
    if not sizes:
        return 0, 0, []
    origins, y = [], 1
    for width, height in sizes:
        origins.append((1, y))
        y += height + 2
    return max(w for w, h in sizes) + 2, y + 1, origins


def _color_plate(pages):
    width, height, origins = _plate_layout([(p.shape[1], p.shape[0]) for p in pages])
    if not pages:
        return width, height, b''
    if width > PLATE_DIMENSION_MAX or height > PLATE_DIMENSION_MAX:
        logger.warning('editor colour plate would need %dx%d; omitting it because its dimensions '
                       'exceed the tag field limit', width, height)
        return 0, 0, b''
    # tool stacks pages in bitmap order at x=1, y=1 then h+2 apart; width is max(w)+2
    # and height is sum(h+2)+2. Blue BGRA (255,0,0,0) separates sequences; opaque
    # red/yellow checkers are unwritten texels inside tool's pages, not separators.
    # our final pages are all written, so expand 565 to opaque BGRA without a second quantisation
    plate = np.full((height, width, 4), [255, 0, 0, 0], np.uint8)
    for page, (x, y) in zip(pages, origins):
        h, w = page.shape
        v = page.astype(np.uint32)
        if page.dtype == np.uint32:
            plate[y:y+h, x:x+w] = np.stack([v & 255, v >> 8 & 255, v >> 16 & 255,
                                            np.full_like(v, 255)], axis=-1)
            continue
        plate[y:y+h, x:x+w] = np.stack([(v & 31) * 255 // 31,
            (v >> 5 & 63) * 255 // 63, (v >> 11 & 31) * 255 // 31,
            np.full_like(v, 255)], axis=-1)
    raw = plate.tobytes()
    return width, height, struct.pack('>I', len(raw)) + zlib.compress(raw)


def write_bitmap(pages: list[np.ndarray], template: Tag | None = None) -> Tag:
    tag = (Tag(template.header, template.group, template.root.copy())
           if template is not None else new_tag('bitm'))
    root = tag.root
    if template is None:
        for key, value in [('type', 0), ('encoding_format', 3), ('usage', 4)]:
            root.set(key, value)
    wide = any(page.dtype == np.uint32 for page in pages)
    if wide:
        root.set('encoding_format', 4)
    width, height, plate = _color_plate(pages)
    root.set('color_plate.color_plate_width', width)
    root.set('color_plate.color_plate_height', height)
    root.set_data('color_plate.compressed_color_plate_data', plate)
    layouts = {f.key: f.layout for f in root.layout.fields if f.kind == 'block'}
    bitmaps, sequences, pixels = [], [], []
    offset = 0
    for i, page in enumerate(pages):
        height, width = page.shape
        bitmap = new_element(layouts['bitmaps'])
        for key, value in [('signature', b'bitm'), ('width', width), ('height', height),
                           ('depth', 1), ('format', 10 if wide else 6), ('flags', 1),
                           ('registration_point', (width // 2, height // 2)),
                           ('pixel_data_offset', offset)]:
            bitmap.set(key, value)
        bitmaps.append(bitmap)
        raw = np.asarray(page, dtype='<u4' if wide else '<u2').tobytes()
        pixels.append(raw)
        offset += len(raw)
        sequence = new_element(layouts['sequences'])
        sequence.set('first_bitmap_index', i)
        sequence.set('bitmap_count', 1)
        sequences.append(sequence)
    final = new_element(layouts['sequences'])
    final.set('first_bitmap_index', -1)
    sequences.append(final)
    root.set_block('sequences', sequences)
    root.set_block('bitmaps', bitmaps)
    root.set_data('processed_pixel_data', b''.join(pixels))
    return tag
