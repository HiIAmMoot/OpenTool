from opentool import examples
from opentool import presets


def test_every_example_row_has_text_and_an_image():
    assert examples.EXAMPLE_ROWS == (*presets.ADVANCED, 'render_only_shadows',
                                     'adaptive_sampling', 'lightmap_32bit')
    for name in examples.EXAMPLE_ROWS:
        assert examples.EXAMPLE_TEXT[name]
        if name not in IMAGE_PENDING:
            assert examples.example_frames(name)


# the release task makes its image
IMAGE_PENDING = ('render_only_shadows', 'adaptive_sampling', 'lightmap_32bit')


def test_missing_frames_are_empty(tmp_path):
    assert examples.example_frames('samples', tmp_path) == []


def test_there_is_no_example_for_the_hidden_suns_row():
    assert 'sky_grid' not in examples.EXAMPLE_ROWS
    assert examples.example_frames('sky_grid') == []


def test_frames_come_in_index_order(tmp_path):
    for index in (2, 0, 1, 10):
        (tmp_path / f'samples_{index}.png').write_bytes(b'x')
    assert examples.example_frames('samples', tmp_path) == [tmp_path / f'samples_{i}.png' for i in range(3)]


def test_a_gap_ends_the_frames(tmp_path):
    for index in (0, 1, 3):
        (tmp_path / f'samples_{index}.png').write_bytes(b'x')
    assert len(examples.example_frames('samples', tmp_path)) == 2


def test_a_lone_png_is_one_frame(tmp_path):
    (tmp_path / 'samples.png').write_bytes(b'x')
    assert examples.example_frames('samples', tmp_path) == [tmp_path / 'samples.png']


def test_numbered_frames_win_over_the_lone_png(tmp_path):
    (tmp_path / 'samples.png').write_bytes(b'x')
    (tmp_path / 'samples_0.png').write_bytes(b'x')
    assert examples.example_frames('samples', tmp_path) == [tmp_path / 'samples_0.png']
