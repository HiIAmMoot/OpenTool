from lightmapper import report

FULL = {
    'seconds': {'probe': 1.0, 'gather': 2.0, 'charts': 3.0, 'scene': 4.0, 'blender': 100.0, 'blender_lamp_batches': 20.0,
                'vertex_pass': 30.0, 'glow': 40.0, 'finish': 5.0, 'denoise': 6.0, 'write': 7.0, 'total': 4320.0},
    'pages': 2, 'sizes': [[1024, 512], [256, 256]], 'written': {'bitmap': 12_900_000, 'bsp': 1_048_576},
    'devices': {'page_bake': 'OPTIX', 'vertex_pass': 'cpu', 'denoise': 'CPU', 'glow_pass': None},
}


def test_format_duration():
    assert [report.format_duration(s) for s in (0.4, 45.2, 59.6, 125, 4320)] == [
        '0 s', '45 s', '1 min 0 s', '2 min 5 s', '1 h 12 min']


def test_format_size():
    assert report.format_size(12_900_000) == '12.3 MB'


def test_full_summary_gives_the_twelve_lines():
    assert report.bake_report(FULL) == [
        'Total time: 1 h 12 min', 'Preparing: 10 s', 'Main bake: 2 min 0 s', 'Vertex pass: 30 s', 'Glow pass: 40 s',
        'Finishing: 5 s', 'Denoising: 6 s', 'Writing: 7 s', 'Pages: 2 (1024x512, 256x256)',
        'Bitmap written: 12.3 MB', 'BSP written: 1.0 MB',
        'Devices: page bake OPTIX, vertex pass cpu, glow pass skipped, denoise CPU']


def test_missing_keys_leave_their_lines_out():
    summary = {**FULL, 'seconds': {k: v for k, v in FULL['seconds'].items()
                                   if k not in ('glow', 'blender', 'blender_lamp_batches')}, 'written': {}}
    lines = report.bake_report(summary)
    assert len(lines) == 8
    assert not any(line.startswith(('Glow pass', 'Main bake', 'Bitmap', 'BSP')) for line in lines)


def test_main_bake_counts_a_run_without_lamp_batches():
    assert 'Main bake: 1 min 40 s' in report.bake_report({'seconds': {'blender': 100.0}})


def test_read_summary_of_a_missing_file_is_none(tmp_path):
    assert report.read_summary(tmp_path / 'summary.json') is None


def test_a_glow_pass_recorded_as_off_reads_skipped():
    lines = report.bake_report({'devices': {**FULL['devices'], 'glow_pass': 'off'}})
    assert lines == ['Devices: page bake OPTIX, vertex pass cpu, glow pass skipped, denoise CPU']
