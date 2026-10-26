# OpenTool

OpenTool is a set of command-line tools for building Halo: Combat Evolved levels, written for OpenCE. It reimplements parts of the editing kit's `tool.exe` in Python so each step can be read, tested and changed.

[OpenCE](https://github.com/OpenCommunityEdition/OpenCE) is a community port of Halo: Combat Evolved to Linux, Windows, Android and the web. OpenTool targets the Halo: Combat Evolved Anniversary (H1A) editing kit in MCC. This is a different kit from the original Halo Custom Edition editing kit.

This is a research and learning project. The code is under active development, its output is checked against tool's own results, and it may change without notice. Keep backups of any tags you point it at.

## Scope

OpenTool covers two of tool's jobs:

| Area | Replaces | Status |
|---|---|---|
| Lightmaps | `tool lightmaps` | Working. Bakes lightmaps for H1A structure BSPs. |
| Structure BSPs | `tool structure` | Planned. Not in this repository yet. |

Everything else in the editing kit (bitmaps, models, sounds, scripts, map building) is out of scope. You use the H1A editing kit or Invader for those.

## Working with Invader

[Invader](https://github.com/SnowyMouse/invader) is an open-source toolkit for Halo CE that handles building cache files, bitmaps, models, sounds, strings, tag editing and checks. OpenTool is meant to sit alongside it. As of Invader 0.55, `invader-lightmap` lays out lightmap UVs but does not compute lighting or write the lightmap bitmap, and Invader has no structure BSP builder.

OpenTool reads and writes ordinary tags in the stock layout, so tags it has written can go back into an Invader workflow. The output has been checked with the H1A editing kit and in OpenCE in game. Compatibility with each Invader tool has not been checked.

## What the lightmapper does

`tool lightmaps` lights a level and stores the result in two tags: the lightmap pages in a `bitmap` tag, and per-vertex lighting data in the `scenario_structure_bsp`. The lightmapper writes those two tags in the stock format tested with the H1A editing kit and OpenCE.

It differs from tool in how it gets there:

- It lays out the lightmap UVs itself with [xatlas](https://github.com/jpcy/xatlas) instead of reusing tool's layout. Charts split at hard creases, and stitching reduces visible seams between charts.
- It renders the light with Blender's Cycles path tracer, on an NVIDIA GPU if you have one.
- It denoises the pages with Intel Open Image Denoise and keeps everything in floating point until the final 16-bit encode.
- It reads the sky's lights and ambient and the BSP's emissive shaders, including emitters on surfaces without lightmaps. Scenery shaders also emit light by default; turn off `scenery_emission` for comparisons with tool.

[docs/how-it-works.md](docs/how-it-works.md) walks through every stage. [docs/settings.md](docs/settings.md) lists the settings.

### How close it gets to tool

Previously recorded results on three stock multiplayer maps, compared with tool's own lightmaps (`lightmap.py diff`):

| Map | Correlation with tool | Median absolute RGB difference | Light direction difference |
|---|---|---|---|
| Blood Gulch | 0.981 | 0.016 | 1.7° |
| Rat Race | 0.895 | 0.036 | 5.6° |
| Chiron TL-34 | 0.877 | 0.056 | 5.7° |

`diff` bilinearly samples each surface once at its UV centre in each layout, drops surfaces smaller than one texel in either layout, and compares incident-light vectors per corner. Correlation is Pearson's r over the sampled RGB channels; 1 means perfect linear correlation. Median absolute RGB difference uses decoded lightmap units from 0 to 1. Light direction difference is the median angle in degrees between nonzero paired corner vectors. Per-material brightness is mean luminance, with a ratio of OpenTool to tool. The repository has no saved artifacts or script that reproduces this historical table.

Outdoor lighting matches tool closely. Indoor maps lit mostly by lamps are 10 to 20 percent brighter than tool on some metal surfaces, and the cause is still being researched. One difference is deliberate: where a shader's texture coordinates overlap, tool shares lightmap texels between unrelated faces and paints false shadows, and OpenTool gives those faces their own space instead.

## Requirements

- Windows 10 or 11. The code has only been run on Windows so far.
- [Git for Windows](https://git-scm.com/downloads/win).
- Python 3.11 exactly for this setup, tested with 3.11.7. Python 3.13 and newer have no wheels for these pinned NumPy and Numba versions. Use `py -3.11` to select the right interpreter; a bare `python` on a fresh Windows install may open the Microsoft Store stub.
- [Blender 5.2](https://www.blender.org/download/), used as a separate program for the light bake.
- An NVIDIA GPU for fast bakes. Cycles tries OptiX where Blender supports it, then CUDA, and falls back to the CPU.
- [LLVM](https://releases.llvm.org) and Visual Studio Build Tools with the Desktop development with C++ workload, including the C++ standard library headers and Windows SDK. LLVM 23.1.2 is installed in the tested environment. The first run compiles xatlas from bundled source. `C:\Program Files\LLVM\bin\clang++.exe` is found without adding LLVM to `PATH`.
- The Microsoft Visual C++ x64 runtime 14.4x for the Open Image Denoise library used by `pyoidn`.
- The H1A editing kit with its tags extracted, and the kit's `tool.exe` to build maps. Steam installs it under `<Steam library>\steamapps\common\HCEEK`, often `C:\Program Files (x86)\Steam\steamapps\common\HCEEK`. Extract the kit's supplied tag archive before using its `tags` folder. The shorter `C:\HCEEK` path below is an example; substitute your install path.

## Installing

Run commands in PowerShell. Create a virtual environment with Python 3.11:

```powershell
git clone https://github.com/HiIAmMoot/OpenTool.git
cd OpenTool
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

There is no OpenTool package to install. `lightmap.py` runs from the checkout. If PowerShell blocks activation, use `.\.venv\Scripts\python.exe` in place of `python` in the commands below.

The `blender` setting defaults to `LIGHTMAPPER_BLENDER` when that environment variable is set, otherwise to `C:\Program Files\Blender Foundation\Blender 5.2\blender.exe`. A `"blender"` value in a settings file overrides either default.

## Using it

Every command needs the editing kit's tags folder, supplied by `--tags` or the `HCEEK_TAGS` environment variable. `--tags` takes precedence. If neither is supplied, the command asks you to pass `--tags`. Tag paths are relative to that folder and have no extension.

To set the variables in PowerShell, replace these example paths with your own. Omit `LIGHTMAPPER_BLENDER` if Blender is in its standard location:

```powershell
$env:HCEEK_TAGS = "C:\HCEEK\tags"
$env:LIGHTMAPPER_BLENDER = "C:\Apps\Blender 5.2\blender.exe"
```

### 1. Copy the level so the original stays untouched

```powershell
python lightmap.py copy --tags "C:\HCEEK\tags" levels\test\bloodgulch\bloodgulch bloodgulch --out-bsp levels\test\bloodgulch\bloodgulch_ot --out-scenario levels\test\bloodgulch\bloodgulch_ot
```

The arguments are the scenario, then the name of the BSP inside it. Both output options are required for `copy`: `--out-bsp` is the full tag path for both the copied BSP and its lightmap bitmap, and `--out-scenario` names the copied scenario.

### 2. Bake

```powershell
python lightmap.py bake --tags "C:\HCEEK\tags" levels\test\bloodgulch\bloodgulch_ot bloodgulch_ot --work bake\bloodgulch
```

The bake reads the scenario and the BSP, lights the level and writes the new BSP and lightmap bitmap back into the tags folder. Before it replaces a tag it saves the old file next to it as `<name>.lm-backup-<date>-<time>`, and it only writes once every stage has finished.

Backups accumulate beside the tags and are not cleaned up automatically. You can delete them when you no longer need to restore those versions.

`--work` keeps `job.json`, `scene.npz`, `textures.npz`, `charts.npz`, Blender's console log `blender.log`, raw output `out.npz`, finished float pages `float_pages.npz`, the surface mapping `new_to_input.npy` and `summary.json` with settings, page sizes and timings. Some products appear only after a successful stage or bake. Without `--work`, each bake gets a timestamped folder in your temp directory.

Useful options:

- `--settings file.json` overrides any setting (see [docs/settings.md](docs/settings.md)).
- `--out-bsp` and `--out-scenario` write the result under new names and leave the input alone.
- `--dry-run` does all the work but writes no tags.
- `--charts file.npz` selects a chart cache. Reuse needs the same work folder or an explicit `--charts` file, unchanged input BSP bytes, unchanged chart settings and unchanged shader radiosity flags and detail levels. An in-place bake changes the BSP bytes, so its next bake normally charts again. Dry runs, diagnostic source passes and bakes to a different `--out-bsp` leave the input BSP unchanged.
- `--sources all` is the normal bake. `--sources sun` isolates sky lights and `--sources emit` isolates shader emission, including scenery when enabled. Both diagnostic modes omit ambient and write work products only, with no tag writes.

A settings file is a JSON object. Save this complete example as `settings.json` and replace the Blender path with your install path; omitted settings retain their defaults. Windows path backslashes must be escaped in JSON:

```json
{
  "texels_per_unit": 8,
  "page_max": 1024,
  "samples": 256,
  "blender": "C:\\Apps\\Blender 5.2\\blender.exe"
}
```

```powershell
python lightmap.py bake --tags "C:\HCEEK\tags" levels\test\bloodgulch\bloodgulch_ot bloodgulch_ot --settings settings.json --work bake\bloodgulch
```

### 3. Check the result

```powershell
python lightmap.py validate --tags "C:\HCEEK\tags" levels\test\bloodgulch\bloodgulch_ot
```

This checks the written BSP and bitmap against the stock format's limits and layout, and prints any problem.

No output from `validate` means it passed. Add `--strict` to check byte-exact vertex recompression and lightmap buffer types for OpenTool output; some retail tags fail those extra checks. Both `validate` and `diff` also accept existing BSP file paths with the extension, while still needing the tags folder to resolve bitmap references.

```powershell
python lightmap.py validate --tags "C:\HCEEK\tags" "C:\HCEEK\tags\levels\test\bloodgulch\bloodgulch_ot.scenario_structure_bsp" --strict
```

To compare with tool's lightmaps for the same level:

```powershell
python lightmap.py diff --tags "C:\HCEEK\tags" --work bake\bloodgulch levels\test\bloodgulch\bloodgulch levels\test\bloodgulch\bloodgulch_ot
```

`diff` writes `diff.md` into the work folder, with the correlation, per-material brightness and light-direction differences. Pass the bake's own work folder, since it holds the mapping between the two BSPs' surfaces.

Successful commands return exit code 0. `validate` returns 1 when it finds problems; `calibrate` returns 1 when its acceptance checks fail. Argument errors, including missing `--tags` and `HCEEK_TAGS`, return 2. Runtime exceptions also fail the command and may print a traceback. `diff` reports comparison metrics without using its acceptance results as an exit-code gate.

### 4. Build a map

Build the map as you normally would, for example with the editing kit's `tool.exe`, run from the editing kit's folder:

```powershell
Set-Location "C:\HCEEK"
.\tool.exe build-cache-file levels\test\bloodgulch\bloodgulch_ot remastered
```

### Bake times

Previous runs on a laptop with an RTX 4060 took two to seven minutes per multiplayer map, with 5 to 12 seconds in GPU page baking. Most time went to the CPU per-vertex pass. These are observations, not a performance guarantee; use `summary.json` to measure your settings and level.

A separate historical campaign run took tool 2 hours 20 minutes and an earlier lightmapper about two minutes. That version cast fewer rays per vertex. This timing does not predict current campaign bake time, and no timing artifacts are included here.

## Running the tests

```powershell
python -m pip install pytest
python -m pytest -q lightmapper/tests
```

Two groups of tests need outside files and are skipped when those are missing:

- tests marked `ek` read stock tags from the editing kit. Point `HCEEK_TAGS` at your tags folder.
- tests marked `blender` run Blender. Point `LIGHTMAPPER_BLENDER` at `blender.exe` if it is not in the default place.

Runtime depends on the available tags, Blender device and machine. The tests that need neither external resource can run on their own.

## Repository layout

```
lightmap.py               command-line entry point
lightmapper/
  cli.py                  commands: bake, copy, validate, diff, calibrate
  pipeline.py             the bake from start to finish
  tagio/                  reading and writing tag files, with the tag layouts
  formats/                BSP geometry, vertices, bitmaps and scenery
  gather.py               sky lights, ambient and shader emission
  charts.py, seams.py     UV charting, packing and seam stitching
  blender/bake.py         the script Blender runs to render the light
  pages.py                denoising, ambient, tone and 16-bit encoding
  radiosity.py            per-vertex incident light vectors
  validate.py, diff.py    format checks and comparison with tool
  vendor/xatlas/          xatlas source, compiled on first use
  tests/                  the test suite
docs/                     how it works and the settings reference
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Contact

For questions and discussion, find Moot on the [OpenCE Discord server](https://discord.gg/9gqcHyr5km). Report bugs as GitHub issues.

## License

OpenTool is owned by Moot and licensed under version 3 of the GNU General Public License, as published by the Free Software Foundation on 29 June 2007. See [LICENSE](LICENSE) for the exact text. It is not licensed under any earlier or later version of the GNU General Public License, or under any other license, except at Moot's discretion.

In short: you may use, study, change and share OpenTool. If you share it, changed or not, you share its source code under the same license. The lightmaps and tags OpenTool produces for your levels are yours and are not covered by this license.

Third-party parts keep their own licenses; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Halo is a trademark of Microsoft. OpenTool is a fan project with no affiliation to Microsoft, Halo Studios or Bungie.
