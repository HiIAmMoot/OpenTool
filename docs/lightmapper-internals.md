# How the lightmapper works

This page follows one bake in the order `lightmapper/pipeline.py` runs it. Most stages leave their output in the work folder, so you can inspect it after the bake. For how to run a bake, see the [lightmapper guide](lightmapper.md).

## What tool's lightmaps are

A Halo CE structure BSP keeps its static lighting in two places:

- Lightmap pages are stored in a `bitmap` tag: 16-bit `r5g6b5` images, no mipmaps, power-of-two sides. Each page is drawn with a set of "materials" in the BSP, one per shader and page.
- Every lightmapped vertex has a second, 20-byte vertex next to its normal one, holding its lightmap UV and an `incident_radiosity` vector. That vector is the direction and strength of the light arriving at the vertex. The game uses it to light objects standing on the surface (which way the Chief's shadow falls, how bright he is) and for bump mapping.

The lightmapper fills both in the stock layout. So far its output has been tested with the H1A editing kit and in game in OpenCE.

## 1. Reading the level

The bake starts from a scenario and the name of one of its BSPs (`find_bsp`). From the tags folder it reads:

- the `scenario_structure_bsp`, through `tagio/`, which reads and writes tag files using the XML layouts in `tagio/layouts/`;
- the BSP's surfaces, vertices and lightmap materials (`formats/bsp.py`), grouped by shader so each group can be charted and lit as one;
- the scenario's skies and their lights (`gather.py`). Each sky light becomes a grid of sun lamps, `sky_grid` by `sky_grid`, as tool's final quality does, and the sky's indoor and outdoor ambient colours are kept for later;
- each shader's radiosity fields: emitted light power and colour, the detail level that sets how many texels its surfaces get, and whether it asks for "simple parameterization" (lightmap UVs taken from the texture UVs);
- which sky each part of the level belongs to (`surface_classes`), so a sky's light only reaches the clusters that see that sky.

The BSP may hold at most 131,072 surfaces, the stock limit. A larger BSP is refused. A dry run warns and continues for timing only, without writing tags.

## 2. Charting and packing

Every lightmapped surface needs its own place on a page (`charts.py`). For each shader group:

- Shaders that ask for simple parameterization keep their texture UVs as the lightmap layout, scaled to the right density. If those texture UVs overlap, which would make two faces share texels, the group is charted normally and a warning says so.
- Every other group goes to [xatlas](https://github.com/jpcy/xatlas), which cuts the surfaces into charts and unfolds them. Charts are cut at hard creases (where the vertex normals on either side differ), because a crease shares no lighting. Rock and terrain shaders that ask for simple parameterization stay joined across their creases, as they do in tool.
- Texel density is `texels_per_unit`, scaled per shader detail level by `detail_level_scale`.

All charts of the level are then packed onto as few pages as possible, and among layouts with the same page count, the one with the least area wins. Pages are at most `page_max` texels on a side. All pieces of one shader group on a page become one material, split at the configured vertex and surface caps (64,000 vertices and 20,000 surfaces by default). The default limits refuse over 127 pages or 2,048 materials on a page. Keep these configurable caps at or below stock values.

The layout is saved in `charts.npz`, together with a key made from the input BSP file, the chart settings and the shaders' radiosity fields. A later bake reuses it only when it finds the file (same `--work` folder, or `--charts`) and the key still matches. A normal bake rewrites its input BSP, so the next bake of that level charts again. Dry runs, the diagnostic `--sources` passes and bakes to a different `--out-bsp` leave the input alone, so their layout can be reused.

## 3. Building the scene

`_scene` in `pipeline.py` turns the level into the arrays the renderer needs:

- Receivers are the lightmapped BSP triangles with their lightmap UVs, texture UVs, normals, emission and page assignments.
- Scenery placed in the scenario for this BSP (`formats/scenery.py`) casts shadows but has no lightmap of its own. It uses the highest-detail parts of its render model for shadows. Tool uses the collision model.
- BSP surfaces without a lightmap can still emit shader light. These emitters block nothing.
- Shader base maps are decoded from bitmap tags (`textures.py`, `formats/bitmap_decode.py`). They give the colour that light picks up when it bounces, the alpha for cut-outs such as grates, and the tint of glass. A `shader_environment` takes its cut-out from the bump map's alpha (the stage the game alpha-tests), and only when "alpha tested" is set and "bump map is specular mask" is clear. A bump map that cannot be read, or a bump scale that is not a whole number of at least 1, leaves the surface solid, with a warning.
- Sun lamps carry the list of receivers each may light. BSP shader emission and, by default, scenery shader emission also light the scene. BSP default-lighting fields are not read.

Each placement's model is first moved by the scenery tag's origin offset, then turned and placed, as the game does. A physics tag's centre-of-mass shift is not applied; none of the 251 scenery tags of the editing kit references a physics tag (custom tag sets checked showed none either).

The bake does not read these scenery flags and fields:

| Flag or field | Why the bake ignores it |
|---|---|
| does not cast shadow | the engine reads it only for dynamic shadow maps; tool's bake ignores it, and scenery shadows bake regardless (Reclaimers Library) |
| transparent self occlusion | an extra render pass only |
| brighter than it should be | brightens the object's own lighting sample in game, not the lightmap |
| not a pathfinding obstacle | AI pathfinding only |
| extension of parent | unused |
| cast shadow by default (H1A) | forces dynamic shadows on static objects; does not stop lightmapped shadows |
| does not have anniversary geometry (H1A) | anniversary graphics only; the bake uses the classic geometry |
| bounding offset | bounds only; never moves the model |

Skipping shadows for "does not cast shadow" would make the bake diverge from tool.

Lights attached to scenery (`formats/lights.py`, `formats/scenery.py`) bake too, always (there is no setting for it). A light bakes when its tag is not dynamic, its radiosity intensity is above zero, its radiosity colour is not black and its position lies inside a cluster of the BSP being baked (tool drops the others). The summary's `lights` counts what was attached, left out and baked. Tool reads only these fields of a light, whatever the light looks like in the game:

| Record field | Where it comes from |
|---|---|
| position, axis | the placement, the model's marker (node 0 when it is missing) and the placement's transform |
| strength | radiosity colour times radiosity intensity |
| kind | a point light when the cutoff angle is not positive or both angles reach pi, otherwise a spot |
| reach | the distance where intensity over d squared drops under 1/255: sqrt(255 x intensity); the light adds nothing beyond it |
| cutoff, falloff | cosines of the shape's cutoff and falloff angles; a spot ramps linearly in the cosine of the angle to its axis between them |

A lamp lights every class of receiver, not only those a sun reaches.

The bake differs from tool in these ways:

- Receivers take the cosine of the angle between their normal and the lamp. Tool has none for a point light (only 1/d squared). This is a deliberate choice (owner), so pages and vertices agree with Cycles' physical lamps.
- A lamp's own placement never shadows it (decision 8), so a lamp inside its fixture's model is not blocked by it.
- Ignore-normals receivers keep the spot cone but take no cosine, as tool does.
- In tool mode the light on ignore-normals triangles comes from the vertex pass (`light_flat`, added by `flat_light_term`), interpolated from the triangle's vertices like the flat sun term. A lamp over the middle of a large flagged triangle therefore reads coarse. Enhanced mode gathers it per texel.
- Occlusion is a shadow ray, not tool's PVS.
- There is no lamp term at shadow terminators.
- Animated scenery sits at its default node transforms.

Lamps are Cycles lights, and Cycles limits how many distinct light-linking sets one bake holds: one blocker set per lamp-owning placement allows 62 owners per run. Past that the lamps bake in extra runs (batches), each added to the main pages, which is exact because light adds linearly and the sun and other sources run in the first batch only. The vertex passes start after every batch. The summary's `lights.batches` says how many ran.

Cycles' indirect sample clamp is off for every bake (owner). With the default of 10 a bright emitter or lamp bounce would be clipped, and summed batches would no longer equal one bake.

These are written into the work folder as `job.json`, `scene.npz` and `textures.npz` (`blender_job.py`).

## 4. Rendering in Blender

Blender runs as a separate program on that job (`run_blender`). The script it runs, `blender/bake.py`, exchanges data through those files and `out.npz`. It imports Blender's Python API (`bpy` and `mathutils`), so it must be under a GPL-compatible license; it is, as part of OpenTool under GPLv3. Blender's console output is saved as `blender.log` in the work folder.

In Blender the script:

1. Builds the receivers and occluders as meshes, with simple materials: the base texture for colour, a cut-out where the alpha says so, emission for glowing shaders, and a tint switch for glass that colours the shadows passing through it.
2. Adds the sun lamps and links each one to the receivers it may light.
3. Bakes each page with Cycles at three times its resolution, as a 3 by 3 grid of samples per texel, and averages the samples that land on a surface. It bakes incoming light: direct, indirect and emission. `bounces` controls indirect light and defaults to one bounce. The game multiplies in the surface's own colour when it draws.
4. Bakes again where a texel at a chart's edge owns only a few of its nine samples, until it has about as many samples as a texel in the middle of a chart. Without this, small charts come out noisy.
5. Casts `vertex_rays` rays (4,096 by default) from every lightmapped vertex to measure where its light comes from. Vertices at the same position with the same normal are measured once. When a GPU with ray queries is in use, this vertex pass does not run in Blender: after Blender finishes, the lightmapper runs it in a separate process on the GPU through SlangPy and Vulkan. Without a usable GPU, Blender runs it on the CPU with its own ray tracer (`cpu_vertex_rays` rays), and that is the slow case.

A short probe at the start of each bake finds the GPUs that support ray queries. `device` then picks the vendor: NVIDIA pages bake on OptiX, falling back to CUDA, and AMD pages on HIP. `auto` takes NVIDIA before AMD. When the probe finds no GPU with ray queries, the vertex pass runs on the CPU, but pages still try Cycles' GPU backends: OptiX, CUDA, HIP and oneAPI. If Cycles cannot open a GPU device, the pages bake on the CPU with a warning. Stages that run on the CPU use `cpu_samples` and `cpu_vertex_rays`, so the CPU gets fewer rays. A bake that asked for NVIDIA or AMD stops instead of falling back, if the vertex pass cannot run on that GPU.

Triangles that sit exactly on top of each other facing opposite ways, such as the two faces of a two-sided sheet, are paired first, so the two faces no longer shadow each other in Cycles or in the vertex pass.

## 5. Finishing the pages

Back in Python, `pages.py` and `terminator.py` turn the raw renders into final pages:

- Tool lights a face from the sun only if the face's plane faces the sun, then blends light in from the lit corners of the mesh. The lightmapper copies that behaviour so shadow edges look like tool's (`reach` sets how far the light reaches).
- Texels just outside each chart get the colour of the nearest texel inside, so filtering at chart edges does not pull in black.
- Intel Open Image Denoise runs through `pyoidn` in its own process.
- The sky's flat ambient light is added, as tool does.
- Where one mesh edge appears in two charts or on two pages, texels on both sides are solved together to reduce visible seams. Real shadow edges are left alone.
- Colours that go over 1 are scaled down keeping their hue, then encoded once to `r5g6b5`, with dithering.

The per-vertex results become the `incident_radiosity` vectors (`radiosity.py`).

## 6. Writing the tags

`formats/bsp.py` rewrites the BSP's lightmap materials, lightmap vertices and surface order for the new layout. `formats/bitmap.py` writes the pages into the lightmap bitmap, reusing the old bitmap tag's settings. Only when everything has succeeded are the tags written:

- each tag is written to a temporary file and moved into place, so a failed write never leaves half a tag;
- the old file is kept in a `backups` folder next to the tag, named `<YYYYMMDD-HHMMSS>_<tag file name>`; backups are never overwritten;
- backups are on by default and can be disabled with `--no-backups`;
- `--dry-run` and diagnostic bakes (`--sources sun` or `emit`) write no tags at all.

After a successful bake the work folder also holds `summary.json` (settings, page sizes, how full each page is, and stage timings), the pages before encoding in `float_pages.npz`, Blender's raw output in `out.npz`, and `new_to_input.npy`, which `diff` uses to pair the old and new surface order.

## The other commands

- `copy` writes a renamed copy of a scenario, its BSP and its lightmap bitmap. Both output options are required; `--out-bsp` names the bitmap too.
- `validate` checks stock limits and layout: page count, dimensions and format, material and vertex counts, surface ranges, vertex buffer sizes, lightmap UVs in [0, 1], finite normal, binormal, tangent and incident-vector components within ±1.005, cluster and leaf surface references, bitmap sequence count and pixel offsets, and the final `-1` lightmap entry. No output means it passed. `--strict` also checks byte-exact vertex recompression and the lightmap buffer type; some retail tags fail these extra checks.
- `diff` compares two lightmaps of the same level surface by surface, with one bilinear sample at each surface's UV centre, and compares the light vectors corner by corner. Surfaces smaller than one texel in either version are left out. The [guide](lightmapper.md#how-close-it-gets-to-tool) explains its numbers.
- `calibrate` fits the light-source weights to a bake made by tool. The shipped `calibration.json` holds tool's own lighting model, so fitting is only needed for experiments.
