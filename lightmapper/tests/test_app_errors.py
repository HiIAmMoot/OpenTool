import os
import stat
from pathlib import Path

import pytest

from lightmapper.blender_find import BlenderNotFound
from lightmapper.blender_job import BlenderError
from lightmapper.charts import CapError
from opentool.errors import explain


@pytest.mark.parametrize('exc, needle', [
    (BlenderNotFound('Blender 5.2 or newer was not found.'), '5.2 or newer was not found'),
    (BlenderError('blender could not start: [WinError 2]'), 'pick blender.exe again'),
    (BlenderError('blender exited with 1\nTraceback ...'), 'blender.log'),
    (CapError('127 lightmap pages needed'), 'texel density'),
    (ValueError('140000 BSP surfaces exceed stock cap 131072'), "can't be lightmapped"),
    (ValueError("expected one structure BSP named 'x', found 0"), 'no BSP with that name'),
    (MemoryError(), 'out of memory'),
    (RuntimeError('OIDN denoise failed (1): ...'), 'unzip the download again'),
    (RuntimeError('xatlas.dll is missing from the OpenTool folder'), 'unzip the download again'),
    (KeyError('weird'), 'The bake stopped')])
def test_explain(exc, needle):
    assert needle in explain(exc, Path('W'))


def test_permission_error_names_the_file():
    assert 'C:\\tags\\a.scenario' in explain(PermissionError(13, 'denied', 'C:\\tags\\a.scenario'))


@pytest.mark.skipif(os.name != 'nt', reason='Windows refuses to replace a read-only file')
def test_replace_onto_read_only_tag_names_the_tag(tmp_path):
    target = tmp_path / 'room_lit.bitmap'
    target.write_bytes(b'old')
    temp = tmp_path / 'abc123.tmp'
    temp.write_bytes(b'new')
    os.chmod(target, stat.S_IREAD)
    try:
        with pytest.raises(PermissionError) as caught:
            os.replace(temp, target)
    finally:
        os.chmod(target, stat.S_IREAD | stat.S_IWRITE)
    text = explain(caught.value)
    assert 'room_lit.bitmap' in text and '.tmp' not in text


def test_missing_tag_names_the_file():
    assert 'x.shader' in explain(FileNotFoundError(2, 'nope', 'x.shader'))


def test_cap_error_is_not_the_generic_value_error():
    assert explain(CapError('a chart exceeds page_max including padding')) == (
        'This level needs more lightmap space than the stock format allows at these settings. '
        'Lower the texel density, or raise Max resolution under Advanced.')


@pytest.mark.parametrize('exc', [RuntimeError(), AssertionError(), KeyError()])
def test_empty_exception_message_has_generic_explanation(exc):
    assert 'The bake stopped' in explain(exc)


def test_xatlas_message_uses_plain_sentence():
    assert explain(RuntimeError('xatlas failed')) == 'Part of OpenTool is missing (xatlas), so unzip the download again.'


def test_oserror_without_filename_uses_plain_fallback():
    assert 'a file' in explain(PermissionError(13, 'denied'))


def test_blender_error_without_work_names_work_folder():
    assert "blender.log in the bake's work folder" in explain(BlenderError('blender exited with 1'))
