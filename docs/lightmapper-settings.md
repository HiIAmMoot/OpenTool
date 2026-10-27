# Lightmapper settings

Pass settings to `bake` or `calibrate` as a JSON file with `--settings file.json`; the [lightmapper guide](lightmapper.md#settings) has an example. Keys are the setting names below, and an unknown key is an error. Anything you leave out keeps its default. The settings a bake used are saved in its `summary.json`.

The defaults aim for tool's final-quality lighting, plus light from scenery shaders, which tool does not have. Changing settings trades bake time and memory for sharper or cleaner lighting.

## Charts and pages

| Setting | Default | What it does |
|---|---|---|
| `texels_per_unit` | 2.0 | Lightmap texels per world unit, for shaders at detail level 1. Doubling it doubles the sharpness and roughly quadruples the texels. |
| `detail_level_scale` | (2.667, 1.0, 0.6, 0.03) | Multiplier on `texels_per_unit` for shader detail levels 0 to 3, taken from tool's final-quality ratios. Detail level 3 surfaces get almost no texels. |
| `padding` | 4 | Empty texels around each chart, so filtering does not bleed between charts. |
| `page_max` | 1024 | Largest page side in texels. Stock maps use at most 512 by 1024. Pages up to 1024 by 1024 have been checked in game; larger pages are untested. |
| `page_min` | 8 | Smallest page side in texels. |
| `max_pages` | 127 | Most pages a level may use. The stock format allows 127 plus an empty last entry. |
| `max_materials_per_page` | 2048 | Stock limit on materials per page. |
| `max_vertices` | 64000 | Stock limit on vertices per material. A larger material is split. |
| `max_surfaces` | 20000 | Stock limit on surfaces per material. A larger material is split. |
| `respect_normal_seams` | true | Cut charts at hard creases, as tool does. Shaders that ask for simple parameterization stay joined across creases either way. |

A level that cannot fit the configured limits is refused. Errors can name a chart that exceeds `page_max` including padding, or several possible packing limits; they do not always identify one remedy. Lowering density or increasing `page_max` within tested sizes may help. The `max_*` settings are configurable, so do not raise them above their stock defaults. Later writer or validation checks can reject values above stock.

## The bake

| Setting | Default | What it does |
|---|---|---|
| `seed` | 0 | Noise seed for the bake, 0 to 999. Another seed gives different noise with the same lighting, which helps when you want to see how much of a result is noise. |
| `samples` | 256 | Cycles samples per texel when a GPU bakes the pages. Each texel is baked as a 3 by 3 grid, so each of the nine gets a ninth. |
| `cpu_samples` | null | Samples per texel when Cycles runs on the CPU, including when `auto` finds no GPU or Cycles falls back mid-bake. `null` follows `samples`; a value must be at least 1. When `texel_samples` is `null`, the edge target follows this count on the CPU. |
| `texel_samples` | null | Texels at a chart's edge that own few of their nine samples are baked again until they reach about this many. `null` follows `samples`, `0` turns the extra pass off, a number sets the target. |
| `bounces` | 1 | Diffuse bounces. One bounce matches tool's shadow brightness. More bounces give brighter, more physically correct shadows. |
| `vertex_rays` | 4096 | Rays per unique position and normal query for incident light vectors, plus separate sun-shadow rays, when a GPU runs the vertex pass. Fewer rays bake faster but can show triangle patterns in bump-mapped surfaces. |
| `cpu_vertex_rays` | null | Rays per query when the vertex pass runs on the CPU, including the fallback after a GPU pass fails. `null` follows `vertex_rays`; a value must be at least 1. |
| `sky_grid` | 4 | Each sky light becomes a `sky_grid` by `sky_grid` grid of sun lamps, as in tool's final quality (tool's draft uses 2). |
| `scenery_emission` | true | Let scenery shaders give off light. This adds a source beyond tool's lighting model. Turn it off to compare with tool. |
| `reach` | 0.225 | On faces turned away from the sun, how far in world units the sun's light reaches in from the lit corners. Tool's value. |
| `albedo_scale` | 1.0 | Multiplier on texture colour for bounced light. |
| `albedo_fallback` | 0.5 | Grey used as the bounce colour when a shader has no readable base map. |
| `device` | "auto" | Where the bake runs: `auto`, `nvidia`, `amd` or `cpu`. `auto` uses a GPU with Vulkan ray queries and prefers NVIDIA over AMD. With no such GPU, the vertex pass runs on the CPU with a warning; pages still try Cycles' GPU backends (OptiX, CUDA, HIP and oneAPI) before falling back to the CPU. `nvidia` and `amd` are required: the bake stops with an error if there is no GPU of that vendor with ray queries, and also if that GPU's vertex pass fails. `cpu` runs every stage on the CPU. The older names still load: `OPTIX` and `CUDA` mean `nvidia`, `HIP` means `amd` and `ONEAPI` means `auto`, and any other value is an error. |
| `blender` | `LIGHTMAPPER_BLENDER` if set, otherwise Blender 5.2's standard install path | Path to `blender.exe`. A JSON setting overrides the environment default. The standard path is `C:\Program Files\Blender Foundation\Blender 5.2\blender.exe`. |
| `timeout_s` | 7200 | Timeout in seconds for Blender, for the GPU vertex pass and separately for each OIDN denoise page. This is not a total bake deadline. `null` means no limit; the denoiser's start-up is still cut off after 300 s then. The probe that looks for GPUs at the start of a bake has its own fixed limit of 60 s, and a bake continues without a GPU if the probe hangs. |

## Finishing

| Setting | Default | What it does |
|---|---|---|
| `denoise` | true | Run Intel Open Image Denoise on each page. |
| `stitch_seams` | true | Make the texels on both sides of a chart seam agree. |
| `dither` | true | Dither when encoding to 16 bits, to hide banding. |

## Light-source weights

These six default to `lightmapper/calibration.json`. The shipped values use an analytic model of tool's lighting. `calibrate` can fit them to a tool bake for experiments.

| Setting | Default | What it does |
|---|---|---|
| `w_sun` | π | Weight on sky lights. Cycles bakes irradiance divided by π, so π brings sky lights back to tool's units. |
| `w_emit` | 1.0 | Weight on emissive shaders. |
| `w_amb` | 1.0 | Weight on the sky's ambient light. |
| `gamma` | 1.0 | Tone curve exponent. Tool applies none. |
| `radiosity_mode` | "coherence" | How the per-vertex vector is formed. `coherence` uses a unit vector along the light's main direction, scaled by the share arriving from that direction (`|V| / E`, at most 1), and zero where no light arrives. `scaled` multiplies the summed direction by `radiosity_scale`, caps it at length 1 and falls back to the surface normal where no light arrives. |
| `radiosity_scale` | 1.0 | Scale for `scaled` mode. |

## summary.json

Every successful bake writes `summary.json` into its work folder. Besides the settings and page sizes, it records what ran where:

| Key | What it holds |
|---|---|
| `devices` | The `device` setting, the vendor and adapter chosen, the Cycles devices tried, and what each stage ran on: `page_bake` (a Cycles device such as `OPTIX` or `HIP`, or `CPU`), `vertex_pass` (the GPU and its backend, or `cpu`) and `denoise` (`CUDA`, `CPU` or `off`). |
| `warnings` | Text lines for anything that fell back or was left unpaired: no GPU with ray queries found, Cycles falling back to the CPU, a failed GPU vertex pass, the denoiser running on the CPU, coincident triangle groups of more than two that could not be paired. An empty list means nothing fell back. |
| `rays` | The counts the bake used: `samples` per texel, `texel_samples` for chart edges and `vertex_rays` per vertex query. They are the CPU counts for a stage that ran on the CPU. |
| `vertex_rays_cast` | How many rays the vertex pass cast in all. |
| `seconds` | Seconds per stage: `probe`, `gather`, `charts`, `scene`, `blender` and inside it `blender_start`, `blender_scene`, `page_bake` and `blender_vertex`, then `vertex_pass`, `denoise`, `finish`, `write` and `total`. |
