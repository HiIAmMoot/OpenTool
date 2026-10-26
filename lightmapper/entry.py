"""Lazy command entry so the worker can preload its runtime before numpy."""

import sys

from .denoise_worker import WORKER_FLAG


def cli_entry(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == [WORKER_FLAG]:
        from . import denoise_worker
        return denoise_worker.main()
    from . import cli
    return cli.main(argv)
