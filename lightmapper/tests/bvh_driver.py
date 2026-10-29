"""Runs inside Blender to compare float32 soup rays with BVHTree."""

import importlib.util
from pathlib import Path
import sys


def main(work):
    oracle_path = Path(__file__).resolve().parents[1] / 'blender' / 'bvh_oracle.py'
    spec = importlib.util.spec_from_file_location('lm_bvh_oracle', oracle_path)
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    oracle.main(work)


if __name__ == '__main__':
    main(Path(sys.argv[sys.argv.index('--') + 1]))
