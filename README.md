# OpenTool

OpenTool is a set of command-line tools that rebuild parts of the Halo: Combat Evolved editing kit's `tool.exe` in Python, so each step can be read, tested and changed. It is a research and learning project: it changes often, and its results are checked against tool's own. Keep backups of any tags you point it at.

OpenTool is developed alongside [OpenCE](https://github.com/OpenCommunityEdition/OpenCE), the open-source port of Halo: Combat Evolved to Linux, Windows, Android and the web. Everything it writes today is ordinary tags in the stock format of the Halo: Combat Evolved Anniversary (H1A) editing kit in MCC, so the results also work in MCC and with the editing kit's own tools. Nothing in it needs OpenCE yet. Features that only OpenCE can use are planned for later and will sit behind a switch.

## Tools

| Tool | Replaces | Status | Documentation |
|---|---|---|---|
| Lightmapper | `tool lightmaps` | Usable | [docs/lightmapper.md](docs/lightmapper.md) |
| Structure BSP builder | `tool structure` | Not started | See the roadmap below |

The rest of the editing kit's work (bitmaps, models, sounds, scripts, building maps) is out of scope. Use the H1A editing kit or Invader for those.

## Roadmap

- Done: the lightmapper, with lightmaps that closely match tool's.
- Next: better lightmaps in the stock format, with higher resolution, better bounced light, better shadows and faster bakes.
- Next: the structure BSP builder, a replacement for `tool structure` with clearer portal errors and room for larger levels.
- Later, OpenCE only: instanced geometry and per-texel directional lightmaps, behind a switch that keeps stock output the default.

## Working with Invader

[Invader](https://github.com/SnowyMouse/invader) is an open-source Halo CE toolkit for building cache files, bitmaps, models, sounds, strings, tag editing and checks. OpenTool is meant to be used next to it and covers jobs Invader leaves open. As of Invader 0.55, `invader-lightmap` lays out lightmap UVs but does not compute any lighting or write the lightmap bitmap, and Invader has no structure BSP builder.

OpenTool reads and writes ordinary tags, so its output can go back into an Invader workflow. So far it has been checked with the H1A editing kit and in game in OpenCE, not yet with each Invader tool.

## Requirements

- Windows 10 or 11. OpenTool has only been run on Windows so far.
- [Git for Windows](https://git-scm.com/downloads/win).
- Python 3.11, tested with 3.11.7. The pinned NumPy and Numba versions have no packages for Python 3.13 or newer. On a fresh Windows install, a bare `python` command may open the Microsoft Store, so use the `py -3.11` launcher.
- The H1A editing kit with its tags extracted. Steam installs it in `<Steam library>\steamapps\common\HCEEK`, often `C:\Program Files (x86)\Steam\steamapps\common\HCEEK`. Extract the kit's tag archive before you point OpenTool at its `tags` folder. The examples in these docs use `C:\HCEEK` as a short stand-in for your install folder.

Each tool may need more. The lightmapper also needs Blender, a C++ compiler and the Visual C++ runtime; see [its requirements](docs/lightmapper.md#requirements).

## Installing

In PowerShell:

```powershell
git clone https://github.com/HiIAmMoot/OpenTool.git
cd OpenTool
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The tools run straight from the checkout; there is no package to install. If PowerShell refuses to run the activation script, call `.\.venv\Scripts\python.exe` wherever the docs say `python`.

## How to use

### Lightmapper

You don't need to set anything up in Blender. Install Blender 5.2 and the lightmapper starts it in the background, builds the scene from your tags, bakes, and closes it again. If Blender is not in its standard folder, set `LIGHTMAPPER_BLENDER` to the path of `blender.exe`.

This bakes Blood Gulch into new tags, so the original level stays as it is:

```powershell
python lightmap.py bake --tags "C:\HCEEK\tags" levels\test\bloodgulch\bloodgulch bloodgulch --out-bsp levels\test\bloodgulch\bloodgulch_lit --out-scenario levels\test\bloodgulch\bloodgulch_lit --work bake\bloodgulch
```

| Part | What to put there |
|---|---|
| `--tags "C:\HCEEK\tags"` | The editing kit's `tags` folder. You can set `HCEEK_TAGS` once instead. |
| `levels\test\bloodgulch\bloodgulch` | The scenario to light, as a tag path relative to the tags folder, without the extension. |
| `bloodgulch` | The name of the BSP inside that scenario. Most multiplayer levels have one, named like the level. |
| `--out-bsp levels\...\bloodgulch_lit` | Where to write the lit BSP and its lightmap bitmap. Leave it out to overwrite the original BSP; the old file is kept as a backup. |
| `--out-scenario levels\...\bloodgulch_lit` | Where to write a scenario that uses the new BSP. Build this one to play the result. |
| `--work bake\bloodgulch` | A folder for the bake's working files and its `summary.json`. Optional. |

Two more options you will often want:

- `--settings settings.json` changes how the bake looks, for example `{"texels_per_unit": 8}` for sharper shadows. See the [settings](docs/lightmapper-settings.md).
- `--dry-run` runs the whole bake but writes no tags.

Then build the map as usual, for example with `.\tool.exe build-cache-file levels\test\bloodgulch\bloodgulch_lit remastered` from the editing kit's folder. The [lightmapper guide](docs/lightmapper.md) covers the other commands: `copy`, `validate` to check the result against the stock format, and `diff` to compare it with tool's lightmaps.

## Running the tests

```powershell
python -m pip install pytest
python -m pytest -q lightmapper/tests
```

Two groups of tests need files from outside the repository and are skipped without them. Tests marked `ek` read stock tags, so set `HCEEK_TAGS` to your tags folder. Tests marked `blender` run Blender, so set `LIGHTMAPPER_BLENDER` if Blender is not in its standard folder.

## Repository layout

```
lightmap.py               the lightmapper's command-line entry point
lightmapper/              the lightmapper
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
docs/                     a guide, internals and settings for each tool
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Contact

For questions and discussion, find Moot on the [OpenCE Discord server](https://discord.gg/9gqcHyr5km). Report bugs as GitHub issues.

## License

OpenTool is owned by Moot and licensed under version 3 of the GNU General Public License, as published by the Free Software Foundation on 29 June 2007. See [LICENSE](LICENSE) for the exact text. It is not licensed under any earlier or later version of the GNU General Public License, or under any other license, except at Moot's discretion.

In short, you may use, study, change and share OpenTool. If you share it, changed or not, you share its source code under the same license. The lightmaps and tags OpenTool makes for your levels are yours, and this license does not cover them.

Third-party parts keep their own licenses; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Halo is a trademark of Microsoft. OpenTool is a fan project with no affiliation to Microsoft, Halo Studios or Bungie.
