"""Read H1A field definitions from the vendored toolset XML layouts.

Inline structs occupy their parent's bytes, while block layouts describe
separate elements. Offsets therefore reset only at a block boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import struct
import xml.etree.ElementTree as ET


@dataclass
class FieldDef:
    kind: str
    key: str
    size: int
    fmt: str | None
    layout: Layout | None
    offset: int


@dataclass
class Layout:
    name: str
    size: int
    fields: list[FieldDef]


_LAYOUTS_DIR = Path(__file__).with_name("layouts")
_GROUP_FILES = {
    "sbsp": "scenario_structure_bsp", "bitm": "bitmap", "scnr": "scenario",
    "sky ": "sky", "senv": "shader_environment", "sgla": "shader_transparent_glass",
    "soso": "shader_model", "schi": "shader_transparent_chicago",
    "scex": "shader_transparent_chicago_extended", "scen": "scenery", "mod2": "gbxmodel",
    "ligh": "light",
}
_FORMATS = {
    "Angle": "f", "AngleBounds": "2f", "ArgbColor": "4b",
    "ByteFlags": "b", "CharInteger": "b", "CharEnum": "b", "CharBlockIndex": "b",
    "ShortInteger": "h", "ShortEnum": "h", "ShortBlockIndex": "h",
    "CustomShortBlockIndex": "h", "WordFlags": "h", "WordBlockFlags": "h",
    "LongInteger": "i", "LongEnum": "i", "LongBlockIndex": "i",
    "CustomLongBlockIndex": "i", "LongFlags": "i",
    "Real": "f", "RealFraction": "f", "RealBounds": "2f", "RealFractionBounds": "2f",
    "RealPoint2D": "2f", "RealPoint3D": "3f", "RealVector2D": "2f",
    "RealVector3D": "3f", "RealEulerAngles2D": "2f", "RealEulerAngles3D": "3f",
    "RealPlane2D": "3f", "RealPlane3D": "4f", "RealQuaternion": "4f",
    "RealRgbColor": "3f", "RealArgbColor": "4f", "Matrix3x3": "9f",
    "Point2D": "2h", "Rectangle2D": "4h", "ShortBounds": "2h", "RgbColor": "4B",
    "String": "32s", "LongString": "256s", "Tag": "4s",
}
_HEADERS = {"Block": ("block", "iii"), "Data": ("data", "iiiii"),
            "TagReference": ("reference", "4siii")}


def _group_xml(group: str) -> ET.Element:
    return ET.parse(_LAYOUTS_DIR / f"{_GROUP_FILES[group]}.xml").getroot()


def group_version(group: str) -> int:
    return int(_group_xml(group).attrib["version"])


def load_group(group: str) -> Layout:
    return _parse_layout(_group_xml(group).find("Layout"))


def _parse_layout(node: ET.Element) -> Layout:
    latest = [fs for fs in node.findall("FieldSet") if fs.get("isLatest") == "true"]
    if len(latest) != 1:
        raise ValueError(f"{node.get('name')}: expected exactly one latest field set")
    field_set = latest[0]
    fields = []
    offset = 0
    for element in field_set:
        tag = element.tag
        if tag == "Explanation":
            continue
        key = element.get("CStyleName", "")
        if tag == "Struct":
            nested = _parse_layout(element.find("Layout"))
            fields.extend(replace(f, key=f"{key}.{f.key}", offset=offset + f.offset)
                          for f in nested.fields)
            offset += nested.size
            continue
        nested = None
        fmt = None
        kind = "value"
        if tag in {"Pad", "Skip", "UselessPad"}:
            kind = "pad"
            size = int(element.attrib["length"])
        else:
            if tag in _HEADERS:
                kind, fmt = _HEADERS[tag]
                if tag == "Block":
                    nested = _parse_layout(element.find("Layout"))
            else:
                fmt = _FORMATS[tag]
            if element.get("unsigned") == "true":
                fmt = fmt.translate(str.maketrans("bhi", "BHI"))
            fmt = element.get("endianOverride", ">") + fmt
            size = struct.calcsize(fmt)
        fields.append(FieldDef(kind, key, size, fmt, nested, offset))
        offset += size

    counts = {}
    used = set()
    for field in fields:
        if field.kind == "pad":
            continue
        base = field.key
        count = counts.get(base, 0)
        key = f"{base}_{count}" if count else base
        counts[base] = count + 1
        if key in used:
            raise ValueError(f"{node.get('name')}: duplicate key after suffixing: {key}")
        used.add(key)
        field.key = key

    size = int(field_set.attrib["sizeofValue"])
    if offset != size:
        raise ValueError(f"{node.get('name')}: fields occupy {offset} bytes, expected {size}")
    return Layout(node.attrib["name"], size, fields)
