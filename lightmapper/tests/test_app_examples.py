from opentool import examples
from opentool import presets


def test_every_example_row_has_text_and_an_image():
    assert examples.EXAMPLE_ROWS == (*presets.ADVANCED, 'adaptive_sampling', 'ignore_normals_mode', 'lightmap_32bit')
    for name in examples.EXAMPLE_ROWS:
        assert examples.EXAMPLE_TEXT[name]
        assert examples.example_image(name).is_file()


def test_missing_image_is_none(tmp_path):
    assert examples.example_image('samples', tmp_path) is None
