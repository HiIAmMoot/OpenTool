from opentool import examples
from opentool import presets


def test_every_example_row_has_text_and_an_image():
    assert examples.EXAMPLE_ROWS == (*presets.ADVANCED, 'adaptive_sampling', 'lightmap_32bit',
                                     'render_only_shadows')
    for name in examples.EXAMPLE_ROWS:
        assert examples.EXAMPLE_TEXT[name]
        if name != IMAGE_PENDING:
            assert examples.example_image(name).is_file()


# the release task makes its image
IMAGE_PENDING = 'render_only_shadows'


def test_missing_image_is_none(tmp_path):
    assert examples.example_image('samples', tmp_path) is None


def test_there_is_no_example_for_the_hidden_suns_row():
    assert 'sky_grid' not in examples.EXAMPLE_ROWS
    assert examples.example_image('sky_grid') is None
