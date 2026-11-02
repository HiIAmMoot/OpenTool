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
    # (only the glow stage's estimate depends on it, but a change of mode is a change of estimate)
    ignore_normals_mode: str = 'tool'
    device: str = 'auto'
    bounces: int = 1


def run_estimate(inputs: EstimateInputs) -> SceneEstimate:
    return pipeline.estimate_scene(
        inputs.tags, inputs.scenario, inputs.bsp,
        Settings(texels_per_unit=inputs.texels_per_unit, page_max=inputs.page_max,
                 ignore_normals_mode=inputs.ignore_normals_mode, device=inputs.device,
                 bounces=inputs.bounces),
        runner.chart_cache_path(inputs.tags, inputs.scenario, inputs.bsp),
    )


STATES = {'fits': 'fits', 'page_file': 'warning', 'out_of_memory': 'risk'}


def describe(result: SceneEstimate | BaseException, page_max: int, machine: memory.Machine | None,
             ignore_normals_mode: str = 'tool', bounces: int = 1) -> tuple[str, str]:
    if isinstance(result, BaseException):
        return errors.explain(result), 'refused' if isinstance(result, CapError) else 'error'
    inputs = pipeline.stage_inputs(result, ignore_normals_mode, machine, bounces)
    check = memory.check_bake_memory(inputs, page_max, machine)
    if not check.fits:
        return check.message + RISK_SUFFIX, STATES[check.level]
    text = f'Needs about {memory.format_gb(check.estimate)} GB'
    if machine is not None:
        text += f'; this computer has {memory.format_gb(machine.ram)} GB'
    return text, 'fits'
