"""Keep OpenTool importable without installing it."""

import os
from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

HCEEK_TAGS = Path(os.environ['HCEEK_TAGS']) if os.environ.get('HCEEK_TAGS') else None
LIGHTMAPPER_BLENDER = Path(os.environ.get(
    "LIGHTMAPPER_BLENDER", r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe"
))


@pytest.fixture
def blender_exe():
    return LIGHTMAPPER_BLENDER


@pytest.fixture(autouse=True)
def neutral_calibration(tmp_path_factory, monkeypatch):
    # tests expect unit weights rather than the shipped analytic source-unit conversions
    path = tmp_path_factory.mktemp("calibration") / "calibration.json"
    path.write_text('{"w_sun": 1, "w_emit": 1, "w_amb": 1, "gamma": 1, '
                    '"radiosity_mode": "coherence", "radiosity_scale": 1}\n')
    from lightmapper import settings
    monkeypatch.setattr(settings, "CALIBRATION_PATH", path)


def pytest_configure(config):
    config.addinivalue_line("markers", "ek: requires H1A editing kit tags")
    config.addinivalue_line("markers", "blender: requires Blender 5.2")


def pytest_collection_modifyitems(items):
    for item in items:
        if "ek" in item.keywords and (HCEEK_TAGS is None or not HCEEK_TAGS.is_dir()):
            item.add_marker(pytest.mark.skip(reason="HCEEK_TAGS is unavailable"))
        if "blender" in item.keywords and not LIGHTMAPPER_BLENDER.is_file():
            item.add_marker(pytest.mark.skip(reason="LIGHTMAPPER_BLENDER is unavailable"))
