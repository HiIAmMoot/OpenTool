# Lightmapper

The lightmapper replaces the editing kit's `tool lightmaps`. It lights a level's structure BSP and writes the result into the BSP and its lightmap bitmap, in the same stock format tool writes.

For the stages it goes through, see [lightmapper-internals.md](lightmapper-internals.md). For every setting, see [lightmapper-settings.md](lightmapper-settings.md).

On Windows you can use the app instead of Python: see [Download the app](../README.md#download-the-app). Its `OpenTool-cli.exe` takes the commands on this page in place of `python lightmap.py`, plus `check`, which tests every part of the download.

## What it does differently from tool

Tool stores a level's lighting in two tags: the lightmap pages in a `bitmap` tag, and per-vertex lighting data in the `scenario_structure_bsp`. The lightmapper fills the same two tags, but builds them its own way:

- It lays out the lightmap UVs with [xatlas](https://github.com/jpcy/xatlas). Charts split at hard creases, and the texels on either side of a chart seam are stitched so the seam is hard to see.
- It renders the light with Blender's Cycles path tracer, on a supported GPU if you have one.
- It denoises the pages with Intel Open Image Denoise and keeps the light in floating point until the final 16-bit encode.
- It uses the light sources tool uses: the sky's lights and ambient, and emissive shaders, including emitters on surfaces that have no lightmap. Scenery shaders can also give off light, which tool never allows; this is on by default and `scenery_emission` turns it off.

## How close it gets to tool

These numbers come from earlier bakes of three stock multiplayer maps, compared with tool's own lightmaps using `lightmap.py diff`:

| Map | Correlation with tool | Median brightness difference | Light direction difference |
|---|---|---|---|
| Blood Gulch | 0.981 | 0.016 | 1.7° |
| Rat Race | 0.895 | 0.036 | 5.6° |
| Chiron TL-34 | 0.877 | 0.056 | 5.7° |

`diff` samples each surface once at the middle of its lightmap UVs in both versions. Correlation is Pearson's r over those samples, where 1 is a perfect match. The brightness difference is in lightmap units from 0 to 1. The direction difference is the median angle between the two versions' per-vertex light vectors.

Outdoor lighting matches tool closely. On indoor maps lit mostly by lamps, some metal surfaces come out 10 to 20 percent brighter than tool's, and the cause is still being researched.

One difference is on purpose. Some shaders take their lightmap layout from their texture coordinates, and where those overlap, tool lets unrelated faces share lightmap texels, which paints false shadows. The lightmapper gives those faces their own space instead.

## Requirements

On top of the general requirements in the [README](../README.md#requirements):

- [Blender 5.2](https://www.blender.org/download/), which the lightmapper runs as a separate program for the bake.
- A GPU can speed up bakes. Cycles tries OptiX, CUDA, HIP (AMD), then oneAPI (Intel), and falls back to the CPU.
- NVIDIA is the only GPU vendor tested.
- [LLVM](https://releases.llvm.org) and the Visual Studio Build Tools with the "Desktop development with C++" workload. The first run compiles xatlas from the bundled source, and clang needs the C++ standard library headers and Windows SDK from the Build Tools. LLVM in its default folder, `C:\Program Files\LLVM`, is found even when it is not on `PATH`.
- The Microsoft Visual C++ x64 runtime, version 14.4 or newer, for the denoiser.

Blender is expected at `C:\Program Files\Blender Foundation\Blender 5.2\blender.exe`. If it is somewhere else, set `LIGHTMAPPER_BLENDER` to its path, or put a `"blender"` entry in a settings file.

## Baking a level

Every command needs the editing kit's tags folder, either as `--tags` or through the `HCEEK_TAGS` environment variable. `--tags` wins when both are given. Tag paths are relative to that folder and have no extension.

In PowerShell, with your own paths:

```powershell
$env:HCEEK_TAGS = "C:\HCEEK\tags"
$env:LIGHTMAPPER_BLENDER = "C:\Apps\Blender 5.2\blender.exe"   # only if Blender is not in its standard folder
```

The examples below pass `--tags` anyway, so each one works on its own.

### 1. Copy the level

Work on a copy so the original stays as it is:

```powershell
python lightmap.py copy --tags "C:\HCEEK\tags" levels\test\bloodgulch\bloodgulch bloodgulch --out-bsp levels\test\bloodgulch\bloodgulch_ot --out-scenario levels\test\bloodgulch\bloodgulch_ot
```

The arguments are the scenario and then the name of the BSP inside it. `--out-bsp` names both the copied BSP and its lightmap bitmap, and `--out-scenario` names the copied scenario. If a tag with that name already exists it is backed up first, unless you pass `--no-backups`.

### 2. Bake

```powershell
python lightmap.py bake --tags "C:\HCEEK\tags" levels\test\bloodgulch\bloodgulch_ot bloodgulch_ot --work bake\bloodgulch -v
```

The bake reads the scenario and BSP, then writes the lit BSP and bitmap after every stage finishes. Before replacing a tag, it saves a copy in a `backups` folder next to the tag. Backups are on by default; pass `--no-backups` to turn them off. Backups stay until you delete them: the tool never removes them.

Add `-v` to see progress lines.

`--work` is a folder for the bake's files. It ends up holding the Blender job (`job.json`, `scene.npz`, `textures.npz`), the chart layout (`charts.npz`), Blender's log (`blender.log`), the raw and finished pages (`out.npz`, `float_pages.npz`), the surface mapping `diff` needs (`new_to_input.npy`), and `summary.json` with the settings, page sizes and timings. Without `--work`, each bake gets a new folder in your temp directory.

Other options:

- `--settings file.json` changes settings; see below.
- `--out-bsp` and `--out-scenario` write the result under new names and leave the input tags alone.
- `--no-backups` keeps no copy of a tag that is replaced. Without it, the old tag goes in a `backups` folder next to it.
- `--dry-run` runs the whole bake but writes no tags.
- `--charts file.npz` points at a saved chart layout. It is reused only when the input BSP, the chart settings and the shaders' radiosity fields are unchanged. A normal bake rewrites its BSP, so the next bake of that level charts it again.
- `--sources sun` bakes the sky lights alone and `--sources emit` the shader emission alone, for testing. Both leave out ambient and write no tags. The default is `--sources all`.

### Settings

A settings file is a JSON object with the settings you want to change; everything else keeps its default. Backslashes in Windows paths have to be doubled in JSON:

```json
{
  "texels_per_unit": 8,
  "page_max": 1024,
  "blender": "C:\\Apps\\Blender 5.2\\blender.exe"
}
```

```powershell
python lightmap.py bake --tags "C:\HCEEK\tags" levels\test\bloodgulch\bloodgulch_ot bloodgulch_ot --settings settings.json --work bake\bloodgulch
```

This example quadruples the default texel density for sharper shadows. [lightmapper-settings.md](lightmapper-settings.md) lists every setting.

### 3. Check the result

```powershell
python lightmap.py validate --tags "C:\HCEEK\tags" levels\test\bloodgulch\bloodgulch_ot
```

`validate` checks the BSP and its bitmap against the stock format's limits and layout. It prints nothing when they pass. With `--strict` it also checks that the compressed vertices match the uncompressed ones exactly and that the lightmap buffer types are right; some retail tags fail those extra checks, so use it on the lightmapper's own output.

To compare with tool's lightmaps for the same level:

```powershell
python lightmap.py diff --tags "C:\HCEEK\tags" --work bake\bloodgulch levels\test\bloodgulch\bloodgulch levels\test\bloodgulch\bloodgulch_ot
```

`diff` writes `diff.md` into the work folder, with the correlation, brightness per material and light-direction differences. Give it the bake's own work folder, because that holds the mapping between the two BSPs' surfaces.

`validate` and `diff` also accept a full path to a `.scenario_structure_bsp` file. They still need the tags folder to find the bitmap.

### 4. Build the map

Build the map as usual, for example with the editing kit's `tool.exe`, run from the kit's folder:

```powershell
Set-Location "C:\HCEEK"
.\tool.exe build-cache-file levels\test\bloodgulch\bloodgulch_ot
```

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Success. `diff` returns 0 whatever its numbers are. |
| 1 | `validate` found problems, or `calibrate` failed its acceptance checks. |
| 2 | Wrong arguments, including a missing tags folder. |

Other failures stop the command with an error message and a traceback.

## Bake times

On a laptop with an RTX 4060, earlier bakes of multiplayer maps took two to seven minutes each. The GPU part, baking the pages, took 5 to 12 seconds per map. Most of the time went into the per-vertex pass on the CPU, which works out the direction of the light at every vertex. `summary.json` records how long each stage took on your machine.

Tool's time grows much faster with the size of a level. On one campaign level, tool took 2 hours 20 minutes against about two minutes for an earlier version of the lightmapper. That version cast fewer rays per vertex, so current bakes of such a level take longer.
