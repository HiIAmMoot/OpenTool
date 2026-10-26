from pathlib import Path

from lightmapper.blender_find import BlenderNotFound
from lightmapper.blender_job import BlenderError
from lightmapper.charts import CapError


def explain(exc: BaseException, work: Path | None = None) -> str:
    if isinstance(exc, BlenderNotFound):
        return str(exc)
    if isinstance(exc, BlenderError):
        if str(exc).startswith('blender could not start'):
            return 'Blender could not be started. Check the Blender path, or pick blender.exe again.'
        log_location = f'blender.log in {work}' if work is not None else "blender.log in the bake's work folder"
        return f"Blender stopped during the bake. The end of its log is below, and the whole log is {log_location}.\n{exc}"
    if isinstance(exc, CapError):
        return 'This level needs more lightmap space than the stock format allows at these settings. Lower the texel density under Advanced, or use a smaller page size.'
    if isinstance(exc, ValueError):
        text = str(exc)
        if 'exceed stock cap' in text:
            return "This BSP has more surfaces than the stock format allows (131,072), so it can't be lightmapped."
        if text.startswith('expected one structure BSP named'):
            return 'The scenario has no BSP with that name. Pick the scenario again.'
    if isinstance(exc, PermissionError):
        file = getattr(exc, 'filename2', None) or exc.filename or 'a file'
        return f'OpenTool could not write {file}. Close any program that has it open, or move the tags folder somewhere you can write to (not Program Files).'
    if isinstance(exc, FileNotFoundError):
        file = getattr(exc, 'filename2', None) or exc.filename or 'a file'
        return f"A tag the level needs is missing: {file}. Check that you picked the editing kit's tags folder and that the level's tags are there."
    if isinstance(exc, MemoryError):
        return 'The computer ran out of memory. Close other programs, or lower the texel density under Advanced.'
    if isinstance(exc, RuntimeError) and 'OIDN denoise' in str(exc):
        return 'The denoiser failed. Your OpenTool folder may be incomplete; unzip the download again.'
    if isinstance(exc, RuntimeError) and 'xatlas' in str(exc):
        return 'Part of OpenTool is missing (xatlas), so unzip the download again.'
    lines = str(exc).splitlines()
    first_line = lines[0] if lines else type(exc).__name__
    return f'The bake stopped: {first_line}. The full error is in the log.'
