"""Fit the light-source weights, tone gamma and vector form to tool's lightmaps.

Sun-only and emission-only bakes at unit weight give the two irradiance fields;
ambient is analytic per class. The weights and gamma are fitted on those, then one
full bake at the fitted weights supplies the per-vertex sums for choosing between
the two vector forms.
"""

from dataclasses import dataclass, replace
import json
import logging
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from .diff import (LUMA, Calibration, Samples, by_input_order, emitter_shares, load_pages,
                   paired_samples, pearson, render_markdown, report, surface_table)
from .gather import gather_lighting, surface_classes
from .formats.bsp import read_geometry
from .formats.vertices import ENV, unpack_uncompressed
from .pages import expose
from .pipeline import bake, find_bsp
from .radiosity import finalize, query_points
from .settings import Settings
from .tagio.tagfile import read_tag, resolve_reference_path


logger = logging.getLogger(__name__)
SATURATED = 0.98
DARK = 0.01
SCALES = np.logspace(-3, 3, 61)


@dataclass
class VectorChoice:
    mode: str
    scale: float
    excluded: int
    errors: dict


def _total(w_sun, w_emit, w_amb, e_sun, e_emit, ambient) -> np.ndarray:
    return w_sun * np.asarray(e_sun) + w_emit * np.asarray(e_emit) + w_amb * np.asarray(ambient)


def predict(w_sun, w_emit, w_amb, gamma, e_sun, e_emit, ambient) -> np.ndarray:
    total = _total(w_sun, w_emit, w_amb, e_sun, e_emit, ambient)
    return expose(np.maximum(total, 0) ** gamma)


def fit(samples_sun: Samples, samples_emit: Samples, ambient_per_sample, tool_rgb) -> Calibration:
    tool = np.asarray(tool_rgb, np.float64)
    sun, emit = np.asarray(samples_sun.ours_rgb, np.float64), np.asarray(samples_emit.ours_rgb, np.float64)
    ambient = np.asarray(ambient_per_sample, np.float64)
    # a black tool texel carries no information about the weights
    used = tool @ LUMA > DARK
    if used.sum() < 2:
        raise ValueError(f'only {int(used.sum())} usable samples to fit on')
    sun, emit, ambient, tool = sun[used], emit[used], ambient[used], tool[used]
    # a clipped tool channel only says ours must reach the clip: brighter costs nothing
    clipped = tool >= SATURATED
    free = ~clipped.any(axis=1)
    if not free.any():
        raise ValueError('no unclipped samples to judge the fit')

    def residual(p):
        predicted = predict(p[0], p[1], p[2], p[3], sun, emit, ambient)
        return np.where(clipped, np.maximum(0, SATURATED - predicted), predicted - tool).ravel()

    solution = least_squares(residual, [1, 1, 1, 1],
                             bounds=([0, 0, 0, 0.25], [np.inf, np.inf, np.inf, 4]))
    w_sun, w_emit, w_amb, gamma = (float(v) for v in solution.x)
    predicted = predict(w_sun, w_emit, w_amb, gamma, sun, emit, ambient)
    # the reported error and correlation stay over the unclipped samples, as before the
    # clipped ones joined the fit, so they remain comparable with earlier calibrations
    return Calibration(w_sun, w_emit, w_amb, gamma,
                       median_abs_error=float(np.median(np.abs(predicted[free] - tool[free]))),
                       pearson_r=pearson(predicted[free], tool[free]), samples_used=int(used.sum()))


def choose_vector_mode(tool_vec, tool_normal, V, E, N) -> VectorChoice:
    tool_vec = np.asarray(tool_vec)
    # a vector equal to its normal can be an ambient-dominated measurement, not a placeholder
    wanted = tool_vec.astype(np.float64)

    def error(mode, scale):
        ours = finalize(V, E, N, mode, scale).astype(np.float64)
        return float(np.median(np.linalg.norm(wanted - ours, axis=1)))

    scaled = [error('scaled', c) for c in SCALES]
    best = int(np.argmin(scaled))
    errors = {'coherence': error('coherence', 1.0), 'scaled': scaled[best]}
    if errors['scaled'] < errors['coherence']:
        return VectorChoice('scaled', float(SCALES[best]), 0, errors)
    return VectorChoice('coherence', 1.0, 0, errors)


def _vertex_sums(our_bsp, work: Path):
    # the vertex pass's raw result is the only place its sums survive; the shared
    # queries are recomputed from the tag's own vertices, in the pipeline's order
    with np.load(work / 'out.npz') as data:
        V, E = np.array(data['vertex_vector'], np.float64), np.array(data['vertex_irradiance'], np.float64)
    envs = [m for p in our_bsp.root.block('lightmaps') if p.get('bitmap') >= 0 for m in p.block('materials')]
    env = np.concatenate([unpack_uncompressed(m.data('uncompressed_vertices'),
                                              m.get('rendered_vertices.vertex_count'),
                                              m.get('lightmap_vertices.vertex_count'))[0]
                          for m in envs]).astype(ENV, copy=False)
    _, normal, _, inverse = query_points(env, np.empty((0, 3), np.int32), np.empty(0, np.int16))
    return V, E, normal, inverse


def calibrate(tags: Path, scenario: str, bsp: str, work: Path, s: Settings | None = None) -> dict:
    tags, work = Path(tags), Path(work)
    work.mkdir(parents=True, exist_ok=True)
    # the fit compares against the tool's bake, regardless of the user's preview mode
    unit = replace(s or Settings(), w_sun=1.0, w_emit=1.0, w_amb=1.0,
                   radiosity_mode='coherence', radiosity_scale=1.0, ignore_normals_mode='tool')
    charts = work / 'charts.npz'
    sun = bake(tags, scenario, bsp, unit, work=work / 'sun', sources='sun', charts=charts)
    emit = bake(tags, scenario, bsp, unit, work=work / 'emit', sources='emit', charts=charts)

    scenario_tag, _, path = find_bsp(tags, scenario, bsp)
    tool_bsp = read_tag(resolve_reference_path(tags, 'sbsp', path))
    tool_bitmap = load_pages(tags, tool_bsp)
    paired_sun = paired_samples(tool_bsp, tool_bitmap, sun.float_pages, sun.bsp_tag, sun.new_to_input)
    paired_emit = paired_samples(tool_bsp, tool_bitmap, emit.float_pages, emit.bsp_tag, emit.new_to_input)

    geometry = read_geometry(tool_bsp)
    lighting = gather_lighting(scenario_tag, tags)
    classes = surface_classes(geometry, len(scenario_tag.root.block('skies')))
    if not np.array_equal(paired_sun.surface, paired_emit.surface):
        raise ValueError('the sun and emission bakes do not cover the same surfaces')
    for missing in sorted({int(classes[i]) for i in paired_sun.surface} - set(lighting.ambient)):
        logger.warning('no ambient for class %d: its surfaces get none in the fit', missing)
    ambient = np.array([lighting.ambient.get(int(classes[i]), np.zeros(3)) for i in paired_sun.surface])
    fitted = fit(paired_sun, paired_emit, ambient, paired_sun.tool_rgb)

    final = replace(unit, w_sun=fitted.w_sun, w_emit=fitted.w_emit, w_amb=fitted.w_amb)
    everything = bake(tags, scenario, bsp, final, work=work / 'all', sources='all',
                      charts=charts, dry_run=True)
    V, E, normal, inverse = _vertex_sums(everything.bsp_tag, work / 'all')
    mine = by_input_order(surface_table(everything.bsp_tag), everything.new_to_input)
    corner = inverse[mine.vertex[paired_sun.surface].reshape(-1)]
    choice = choose_vector_mode(paired_sun.tool_vec, paired_sun.tool_normal,
                                V[corner], E[corner], paired_sun.our_normal)
    fitted = replace(fitted, radiosity_mode=choice.mode, radiosity_scale=choice.scale)

    weights = (fitted.w_sun, fitted.w_emit, fitted.w_amb)
    modelled = replace(paired_sun, ours_rgb=predict(
        *weights, fitted.gamma, paired_sun.ours_rgb, paired_emit.ours_rgb, ambient),
        ours_peak=_total(*weights, paired_sun.ours_rgb, paired_emit.ours_rgb, ambient).max(axis=1),
        our_vec=finalize(V[corner], E[corner], paired_sun.our_normal, choice.mode, choice.scale))
    emission = fitted.w_emit * np.asarray(paired_emit.ours_rgb)
    near_emitters = emitter_shares(paired_sun.material, emission, _total(*weights, paired_sun.ours_rgb,
                                                                         paired_emit.ours_rgb, ambient))
    data = report(modelled, fitted, near_emitters)
    data['vector'].update(excluded=choice.excluded, errors=choice.errors, mode=choice.mode,
                          scale=choice.scale)
    (work / 'calibration.md').write_text(render_markdown(data))
    (work / 'calibration_report.json').write_text(json.dumps(data, indent=2))
    accepted = all(data['acceptance'].values())
    # the package's weights are tool's analytic model; a fit is for validating them and stays in the work folder
    (work / 'calibration.json').write_text(json.dumps({
        'w_sun': fitted.w_sun, 'w_emit': fitted.w_emit, 'w_amb': fitted.w_amb,
        'gamma': fitted.gamma, 'radiosity_mode': fitted.radiosity_mode,
        'radiosity_scale': fitted.radiosity_scale}, indent=2) + '\n')
    return data
