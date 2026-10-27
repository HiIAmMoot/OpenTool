"""Read and edit stock H1A tags without normalising their opaque bytes.

Block bodies precede their children on disk, rather than each element being
stored beside its children. Raw bodies retain padding and unused header words.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import os
import shutil
import struct
import tempfile
import time
import zlib

from .layout import Layout, load_group, group_version

# tag paths are bytes in the writer's Windows code page, not ASCII; latin-1 keeps every byte through a read and
# write, and names bytes from 0xa0 up as cp1252 does, so such paths also find their files
PATH_ENCODING = "latin-1"

_GROUP_EXTENSIONS = {
    "sbsp": "scenario_structure_bsp", "bitm": "bitmap", "scnr": "scenario",
    "sky ": "sky", "senv": "shader_environment", "sgla": "shader_transparent_glass",
    "soso": "shader_model", "schi": "shader_transparent_chicago",
    "scex": "shader_transparent_chicago_extended", "scen": "scenery", "mod2": "gbxmodel",
    "swat": "shader_transparent_water", "smet": "shader_transparent_meter",
    "spla": "shader_transparent_plasma", "sotr": "shader_transparent_generic",
}


def resolve_reference_path(tags_root: Path, group: str, path: str) -> Path:
    base = Path(tags_root) / path.replace("\\", "/")
    if group == "mode":
        model = base.with_name(base.name + ".model")
        # pc object tags retain mode references even when the file is a gbxmodel
        if model.is_file():
            return model
        return base.with_name(base.name + ".gbxmodel")
    return base.with_name(base.name + "." + _GROUP_EXTENSIONS[group])


class Element:
    def __init__(self, layout: Layout, body: bytes | bytearray):
        self.layout = layout
        self.body = bytearray(body)
        self._fields = {f.key: f for f in layout.fields if f.kind != "pad"}
        self._children = {}

    def get(self, key):
        field = self._fields[key]
        if field.kind != "value":
            raise TypeError(f"{key} is not a value field")
        values = struct.unpack_from(field.fmt, self.body, field.offset)
        return values[0] if len(values) == 1 else values

    def set(self, key, value):
        field = self._fields[key]
        if field.kind != "value":
            raise TypeError(f"{key} is not a value field")
        values = value if isinstance(value, tuple) else (value,)
        struct.pack_into(field.fmt, self.body, field.offset, *values)

    def block(self, key) -> list[Element]:
        self._check_kind(key, "block")
        return self._children[key]

    def set_block(self, key, elements: list[Element]):
        self._check_kind(key, "block")
        self._children[key] = list(elements)

    def data(self, key) -> bytes:
        self._check_kind(key, "data")
        return self._children[key]

    def set_data(self, key, b: bytes):
        self._check_kind(key, "data")
        self._children[key] = bytes(b)

    def reference(self, key) -> tuple[str | None, str]:
        field = self._check_kind(key, "reference")
        group = bytes(self.body[field.offset:field.offset + 4])
        path = self._children[key]
        return (None if group == b"\xff" * 4 else group.decode("ascii"),
                path[:-1].decode(PATH_ENCODING) if path else "")

    def set_reference(self, key, group: str | None, path: str):
        field = self._check_kind(key, "reference")
        fourcc = b"\xff" * 4 if group is None else group.encode("ascii")
        if len(fourcc) != 4:
            raise ValueError("reference group must have four characters")
        self.body[field.offset:field.offset + 4] = fourcc
        self._children[key] = path.encode(PATH_ENCODING) + b"\0" if path else b""

    def _check_kind(self, key, kind):
        field = self._fields[key]
        if field.kind != kind:
            raise TypeError(f"{key} is not a {kind} field")
        return field

    def copy(self) -> Element:
        result = Element(self.layout, self.body)
        result._children = {
            key: [element.copy() for element in child] if isinstance(child, list) else child
            for key, child in self._children.items()
        }
        return result


@dataclass
class Tag:
    header: bytes
    group: str
    root: Element


def new_element(layout: Layout) -> Element:
    element = Element(layout, bytes(layout.size))
    for field in layout.fields:
        if field.kind == "block":
            element._children[field.key] = []
        elif field.kind == "data":
            element._children[field.key] = b""
        elif field.kind == "reference":
            struct.pack_into(">4siii", element.body, field.offset, b"\xff" * 4, 0, 0, -1)
            element._children[field.key] = b""
    return element


def new_tag(group: str) -> Tag:
    header = bytearray(64)
    header[36:40] = group.encode("ascii")
    struct.pack_into(">I", header, 44, 64)
    struct.pack_into(">H", header, 56, group_version(group))
    header[58:64] = b"\x00\xffblam"
    return Tag(bytes(header), group, new_element(load_group(group)))


class _Reader:
    def __init__(self, raw: bytes):
        self.raw = raw
        self.offset = 64

    def take(self, size: int) -> bytes:
        if size < 0 or self.offset + size > len(self.raw):
            raise ValueError(f"tag payload out of bounds at {self.offset}: {size} bytes")
        start = self.offset
        self.offset += size
        return self.raw[start:self.offset]

    def children(self, element: Element):
        for field in element.layout.fields:
            if field.kind == "data":
                size = struct.unpack_from(">i", element.body, field.offset)[0]
                element._children[field.key] = self.take(size)
            elif field.kind == "reference":
                length = struct.unpack_from(">i", element.body, field.offset + 8)[0]
                if length < 0:
                    raise ValueError("negative reference length")
                element._children[field.key] = self.take(length + 1 if length else 0)
            elif field.kind == "block":
                count = struct.unpack_from(">i", element.body, field.offset)[0]
                if count < 0:
                    raise ValueError("negative block count")
                bodies = self.take(count * field.layout.size)
                size = field.layout.size
                elements = [Element(field.layout, bodies[i * size:(i + 1) * size])
                            for i in range(count)]
                element._children[field.key] = elements
                for child in elements:
                    self.children(child)


def read_tag(path: Path) -> Tag:
    raw = Path(path).read_bytes()
    if len(raw) < 64:
        raise ValueError("tag header is shorter than 64 bytes")
    group = raw[36:40].decode("ascii")
    layout = load_group(group)
    reader = _Reader(raw)
    root = Element(layout, reader.take(layout.size))
    reader.children(root)
    if reader.offset != len(raw):
        raise ValueError(f"unconsumed tag bytes at {reader.offset} of {len(raw)}")
    return Tag(raw[:64], group, root)


def _body(element: Element) -> bytes:
    body = bytearray(element.body)
    for field in element.layout.fields:
        if field.kind in {"block", "data"}:
            struct.pack_into(">i", body, field.offset, len(element._children[field.key]))
        elif field.kind == "reference":
            path = element._children[field.key]
            struct.pack_into(">i", body, field.offset + 8, len(path) - 1 if path else 0)
    return bytes(body)


def _children(element: Element, chunks: list[bytes]):
    for field in element.layout.fields:
        if field.kind in {"data", "reference"}:
            chunks.append(element._children[field.key])
        elif field.kind == "block":
            elements = element._children[field.key]
            chunks.extend(_body(child) for child in elements)
            for child in elements:
                _children(child, chunks)


def tag_bytes(tag: Tag) -> bytes:
    chunks = [_body(tag.root)]
    _children(tag.root, chunks)
    payload = b"".join(chunks)
    header = bytearray(tag.header)
    struct.pack_into(">I", header, 40, zlib.crc32(payload) ^ 0xFFFFFFFF)
    return bytes(header) + payload


def write_tag(tag: Tag, path: Path, backup: bool = True) -> None:
    path = Path(path)
    raw = tag_bytes(tag)
    if path.exists() and backup:
        backup_dir = path.parent / "backups"
        backup_dir.mkdir(exist_ok=True)
        # a failed recovery copy must stop before the atomic replacement can destroy the old tag.
        while True:
            backup_path = backup_dir / (datetime.now().strftime("%Y%m%d-%H%M%S_") + path.name)
            try:
                destination = backup_path.open("xb")
                break
            except FileExistsError:
                # a timestamp collision must never overwrite an earlier recovery copy
                time.sleep(0.05)
        with destination, path.open("rb") as source:
            shutil.copyfileobj(source, destination)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(raw)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
