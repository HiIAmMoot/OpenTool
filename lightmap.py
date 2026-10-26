"""Run the lightmapper directly from the OpenTool checkout."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lightmapper.entry import cli_entry


if __name__ == '__main__':
    raise SystemExit(cli_entry())
