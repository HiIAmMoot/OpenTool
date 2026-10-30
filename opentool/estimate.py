from dataclasses import dataclass
from pathlib import Path

from lightmapper import memory, pipeline
from lightmapper.charts import CapError
from lightmapper.pipeline import SceneEstimate
from lightmapper.settings import Settings
from . import errors, runner


RISK_SUFFIX = ' You can still bake, at your own risk.'


@dataclass(frozen=True)
class EstimateInputs:
    tags: Path
    scenario: str
    bsp: str
    texels_per_unit: float
    page_max: int


def run_estimate(inputs: EstimateInputs) -> SceneEstimate:
    return pipeline.estimate_scene(
        inputs.tags, inputs.scenario, inputs.bsp,
        Settings(texels_per_unit=inputs.texels_per_unit, page_max=inputs.page_max),
        runner.chart_cache_path(inputs.tags, inputs.scenario, inputs.bsp),
    )


def describe(result: SceneEstimate | BaseException, page_max: int,
             free: int | None) -> tuple[str, str]:
    if isinstance(result, BaseException):
        return errors.explain(result), 'refused' if isinstance(result, CapError) else 'error'
    need = memory.bake_bytes(result.counts, result.pages)
    if free is not None and need > free:
        return memory.out_of_memory_message(result.counts, result.pages, page_max, free) + RISK_SUFFIX, 'warning'
    text = f'Needs about {memory.format_gb(need)} GB'
    if free is not None:
        text += f'; {memory.format_gb(free)} GB free'
    return text, 'fits'
