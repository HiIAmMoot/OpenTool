"""Exercise stock tag traversal, edits and atomic output on real EK tags."""

import struct
import zlib

import pytest

from conftest import HCEEK_TAGS as TAGS
from lightmapper.tagio import tagfile
from lightmapper.tagio.layout import load_group


EK_TAGS = [
    "levels/test/bloodgulch/bloodgulch.scenario",
    "levels/test/bloodgulch/bloodgulch.scenario_structure_bsp",
]


@pytest.mark.ek
@pytest.mark.parametrize("rel", EK_TAGS)
def test_round_trip_is_byte_exact(rel):
    assert tagfile.tag_bytes(tagfile.read_tag(TAGS / rel)) == (TAGS / rel).read_bytes()


def reference_path(reference):
    return tagfile.resolve_reference_path(TAGS, *reference)


def test_mode_reference_falls_back_but_prefers_existing_model(tmp_path):
    folder = tmp_path / "scenery" / "rocks"
    folder.mkdir(parents=True)
    gbxmodel = folder / "boulder.gbxmodel"
    gbxmodel.write_bytes(b"gbxmodel")
    assert tagfile.resolve_reference_path(tmp_path, "mode", r"scenery\rocks\boulder") == gbxmodel
    model = folder / "boulder.model"
    model.write_bytes(b"model")
    assert tagfile.resolve_reference_path(tmp_path, "mode", r"scenery\rocks\boulder") == model


@pytest.mark.parametrize("group, extension", [
    ("sbsp", "scenario_structure_bsp"), ("bitm", "bitmap"), ("scnr", "scenario"),
    ("sky ", "sky"), ("senv", "shader_environment"), ("sgla", "shader_transparent_glass"),
    ("soso", "shader_model"), ("schi", "shader_transparent_chicago"),
    ("scex", "shader_transparent_chicago_extended"), ("scen", "scenery"), ("mod2", "gbxmodel"),
])
def test_reference_extensions(tmp_path, group, extension):
    assert tagfile.resolve_reference_path(tmp_path, group, r"folder\asset") == (
        tmp_path / "folder" / ("asset." + extension))


@pytest.mark.ek
def test_round_trip_bloodgulch_referenced_tags():
    scenario = tagfile.read_tag(TAGS / EK_TAGS[0])
    bsp = tagfile.read_tag(TAGS / EK_TAGS[1])
    shader = next(m.reference("shader") for lm in bsp.root.block("lightmaps")
                  for m in lm.block("materials") if m.reference("shader")[0] == "senv")
    scenery = scenario.root.block("scenery_palette")[0].reference("name")
    scenery_tag = tagfile.read_tag(reference_path(scenery))
    for reference in [shader, scenery, scenery_tag.root.reference("model")]:
        path = reference_path(reference)
        assert tagfile.tag_bytes(tagfile.read_tag(path)) == path.read_bytes()


def test_new_tag_header():
    tag = tagfile.new_tag("bitm")
    h = tag.header
    assert h[:36] == bytes(36)
    assert h[0x24:0x28] == b"bitm"
    assert h[0x2C:0x38] == b"\x00\x00\x00\x40" + bytes(8)
    assert h[0x38:0x3A] == b"\x00\x07"
    assert h[0x3A:0x40] == b"\x00\xffblam"
    assert tag.root.get("processing.detail_fade_factor") == 0.0


def test_values_data_references_and_deep_copy(tmp_path):
    tag = tagfile.new_tag("sbsp")
    root = tag.root
    root.set("vehicle_floor", 1.25)
    root.set("world_bounds_x", (-2.0, 3.0))
    root.set_data("cluster_data", b"\x00\xffpayload")
    root.set_reference("lightmaps_bitmap", None, "")
    lm_layout = next(f.layout for f in load_group("sbsp").fields if f.key == "lightmaps")
    lm = tagfile.new_element(lm_layout)
    lm.set("bitmap", -1)
    root.set_block("lightmaps", [lm])
    copied = root.copy()
    copied.block("lightmaps")[0].set("bitmap", 9)
    copied.set_data("cluster_data", b"changed")
    copied.set_reference("lightmaps_bitmap", "bitm", "other")
    assert root.block("lightmaps")[0].get("bitmap") == -1
    assert root.data("cluster_data") == b"\x00\xffpayload"
    assert root.reference("lightmaps_bitmap") == (None, "")
    path = tmp_path / "new"
    tagfile.write_tag(tag, path)
    back = tagfile.read_tag(path).root
    assert back.get("vehicle_floor") == 1.25
    assert back.get("world_bounds_x") == (-2.0, 3.0)
    assert back.data("cluster_data") == b"\x00\xffpayload"
    assert back.reference("lightmaps_bitmap") == (None, "")
    assert back.block("lightmaps")[0].get("bitmap") == -1


def test_non_ascii_reference_path_round_trips(tmp_path):
    from lightmapper.textures import shader_texture
    tag = tagfile.new_tag("senv")
    tag.root.set_reference("diffuse.base_map", "bitm", "levels\\test\\·rock\\base")
    path = tmp_path / "levels" / "test" / "rock.shader_environment"
    path.parent.mkdir(parents=True)
    tagfile.write_tag(tag, path)
    assert b"\\\xb7rock\\base\x00" in path.read_bytes()
    back = tagfile.read_tag(path)
    assert back.root.reference("diffuse.base_map") == ("bitm", "levels\\test\\·rock\\base")
    assert tagfile.tag_bytes(back) == path.read_bytes()
    assert shader_texture(tmp_path, "senv", "levels\\test\\rock").albedo == "levels\\test\\·rock\\base"


def test_preserves_raw_bodies_and_header_words(tmp_path):
    root = bytearray(108)
    struct.pack_into(">iiiii", root, 28, 3, 11, 12, 13, 14)
    struct.pack_into(">iii", root, 84, 0, 21, 22)
    struct.pack_into(">iii", root, 96, 0, 31, 32)
    header = bytearray(64)
    header[36:40] = b"bitm"
    struct.pack_into(">I", header, 40, zlib.crc32(root + b"abc") ^ 0xFFFFFFFF)
    raw = bytes(header + root + b"abc")
    path = tmp_path / "raw"
    path.write_bytes(raw)
    tag = tagfile.read_tag(path)
    assert tagfile.tag_bytes(tag) == raw
    tag.root.set_data("color_plate.compressed_color_plate_data", b"longer")
    out = tagfile.tag_bytes(tag)
    assert struct.unpack_from(">iiiii", out, 64 + 28) == (6, 11, 12, 13, 14)


def test_in_place_backup_and_replace_failure(tmp_path, monkeypatch):
    path = tmp_path / "tag"
    old = b"original"
    path.write_bytes(old)
    tag = tagfile.new_tag("bitm")
    tagfile.write_tag(tag, path)
    tagfile.write_tag(tag, path)
    backups = list(tmp_path.glob("tag.lm-backup-*"))
    assert len(backups) == 2
    assert old in [p.read_bytes() for p in backups]
    before = path.read_bytes()
    tag.root.set("processing.detail_fade_factor", 0.5)

    def refuse_replace(source, destination):
        raise OSError("replace refused")

    monkeypatch.setattr(tagfile.os, "replace", refuse_replace)
    with pytest.raises(OSError, match="replace refused"):
        tagfile.write_tag(tag, path)
    assert path.read_bytes() == before
    assert not list(tmp_path.glob("*.tmp"))


def test_largest_supported_surface_block(tmp_path):
    tag = tagfile.new_tag("sbsp")
    layout = next(f.layout for f in load_group("sbsp").fields if f.key == "surfaces")
    tag.root.set_block("surfaces", [tagfile.new_element(layout) for _ in range(131072)])
    path = tmp_path / "large"
    tagfile.write_tag(tag, path)
    back = tagfile.read_tag(path)
    assert len(back.root.block("surfaces")) == 131072
    assert tagfile.tag_bytes(back) == path.read_bytes()
