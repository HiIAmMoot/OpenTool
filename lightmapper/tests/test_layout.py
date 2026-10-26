"""Check sizes, keys and binary decoding against the vendored tag definitions."""

import struct

import pytest

from lightmapper.tagio import layout as layout_module
from lightmapper.tagio.layout import group_version, load_group


GROUPS = ["sbsp", "bitm", "scnr", "sky ", "senv", "sgla", "soso", "schi", "scex", "scen", "mod2"]


def all_layouts(root):
    yield root
    for field in root.fields:
        if field.layout is not None:
            yield from all_layouts(field.layout)


def block(root, key):
    return next(field.layout for field in root.fields if field.key == key)


@pytest.mark.parametrize("group", GROUPS)
def test_every_field_set_sums_to_its_size(group):
    try:
        root = load_group(group)
    except ValueError as exc:
        assert False, f"layout failed size validation: {exc}"
    for layout in all_layouts(root):
        assert sum(field.size for field in layout.fields) == layout.size
        offset = 0
        for field in layout.fields:
            assert field.offset == offset
            offset += field.size


def test_sbsp_root():
    root = load_group("sbsp")
    assert root.size == 648 and group_version("sbsp") == 5
    assert [field.kind for field in root.fields if field.key == "lightmaps"] == ["block"]


def test_duplicate_keys_suffixed():
    leaf = block(load_group("sbsp"), "leaves")
    assert [field.key for field in leaf.fields[:3]] == ["vertices", "vertices_1", "vertices_2"]


def test_keys_unique_per_layout():
    for group in GROUPS:
        for layout in all_layouts(load_group(group)):
            keys = [field.key for field in layout.fields if field.kind != "pad"]
            assert len(keys) == len(set(keys))


def test_sizes():
    seen = set()
    for group in GROUPS:
        for layout in all_layouts(load_group(group)):
            for field in layout.fields:
                if field.kind in {"block", "data", "reference"}:
                    assert field.size == {"block": 12, "data": 20, "reference": 16}[field.kind]
                    seen.add(field.kind)
                if field.fmt is not None:
                    assert struct.calcsize(field.fmt) == field.size
    assert seen == {"block", "data", "reference"}


@pytest.mark.parametrize("group, version, size", [
    ("bitm", 7, 108), ("scnr", 2, 1456), ("sky ", 1, 208),
    ("senv", 2, 836), ("sgla", 1, 480), ("scen", 1, 508), ("mod2", 5, 232),
])
def test_group_metadata(group, version, size):
    assert group_version(group) == version
    assert load_group(group).size == size


def test_inline_structs_flatten_with_absolute_offsets():
    fields = {field.key: field for field in load_group("bitm").fields}
    assert fields["processing.detail_fade_factor"].offset == 8
    assert fields["color_plate.color_plate_width"].offset == 24
    assert fields["color_plate.color_plate_width"].fmt == ">H"
    assert all(field.kind != "struct" for field in fields.values())


def test_signed_unsigned_and_float_formats_decode_raw_values():
    root = load_group("sbsp")
    leaf = block(root, "leaves")
    assert struct.unpack(leaf.fields[0].fmt, b"\xff\xff") == (65535,)
    cluster = next(field for field in leaf.fields if field.key == "cluster")
    assert struct.unpack(cluster.fmt, b"\xff\xff") == (-1,)
    floor = next(field for field in root.fields if field.key == "vehicle_floor")
    assert struct.unpack(floor.fmt, b"\x3f\x80\x00\x00") == (1.0,)


def write_group(tmp_path, monkeypatch, fields, size):
    xml = f'''<TagGroup group="sbsp" name="fixture" version="5">
      <Layout name="fixture">
        <FieldSet sizeofValue="99" isLatest="false"><Pad length="99"/></FieldSet>
        <FieldSet sizeofValue="{size}" isLatest="true">{fields}</FieldSet>
      </Layout></TagGroup>'''
    (tmp_path / "scenario_structure_bsp.xml").write_text(xml, encoding="utf-8")
    monkeypatch.setattr(layout_module, "_LAYOUTS_DIR", tmp_path)


def test_latest_selection_padding_and_endian_override(tmp_path, monkeypatch):
    write_group(tmp_path, monkeypatch, '''
      <Pad length="2"/><Skip length="3"/><UselessPad length="4"/>
      <Explanation name="description"/>
      <ShortInteger CStyleName="number" endianOverride="&lt;" unsigned="true"/>
      <Angle CStyleName="angle"/>''', 15)
    root = load_group("sbsp")
    assert [(f.kind, f.size, f.offset) for f in root.fields[:3]] == [
        ("pad", 2, 0), ("pad", 3, 2), ("pad", 4, 5),
    ]
    assert len(root.fields) == 5
    assert root.fields[3].key == "number"
    assert struct.unpack(root.fields[3].fmt, b"\x34\x12") == (4660,)
    assert root.fields[4].fmt == ">f"


def test_nested_struct_offsets_and_block_offsets(tmp_path, monkeypatch):
    write_group(tmp_path, monkeypatch, '''<Pad length="4"/>
      <Struct CStyleName="outer"><Layout name="outer">
        <FieldSet sizeofValue="18" isLatest="true"><ShortInteger CStyleName="a"/>
          <Struct CStyleName="inner"><Layout name="inner">
            <FieldSet sizeofValue="16" isLatest="true"><Real CStyleName="b"/>
              <Block CStyleName="items"><Layout name="item">
                <FieldSet sizeofValue="2" isLatest="true"><ShortInteger CStyleName="c"/></FieldSet>
              </Layout></Block>
            </FieldSet></Layout></Struct>
        </FieldSet></Layout></Struct>''', 22)
    fields = load_group("sbsp").fields
    assert [(f.key, f.offset) for f in fields[1:]] == [
        ("outer.a", 4), ("outer.inner.b", 6), ("outer.inner.items", 10),
    ]
    assert fields[-1].layout.fields[0].offset == 0


def test_remaining_suffix_collision_is_rejected(tmp_path, monkeypatch):
    write_group(tmp_path, monkeypatch, '''<ShortInteger CStyleName="x"/>
      <ShortInteger CStyleName="x"/><ShortInteger CStyleName="x_1"/>''', 6)
    with pytest.raises(ValueError):
        load_group("sbsp")


def test_size_mismatch_is_rejected(tmp_path, monkeypatch):
    write_group(tmp_path, monkeypatch, '<ShortInteger CStyleName="x"/>', 4)
    with pytest.raises(ValueError):
        load_group("sbsp")


def test_largest_vendored_group_and_long_duplicate_run():
    scenario = load_group("scnr")
    assert len(list(all_layouts(scenario))) == 64
    parts = block(block(load_group("mod2"), "geometries"), "parts")
    indices = [field.key for field in parts.fields if field.key.startswith("local_node_indices")]
    assert indices == ["local_node_indices"] + [f"local_node_indices_{i}" for i in range(1, 22)]
