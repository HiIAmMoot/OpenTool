# Contributing

OpenTool is a research and learning project. Questions, experiments and bug reports are welcome.

## Before you start

- For a bug, use the bug report template. Include OS, Python, Blender and GPU versions, the command and settings, `summary.json`, console output and `blender.log` if available. Redact personal paths. Do not upload editing-kit tags or game assets. For a visual bug, include same-view screenshots of OpenTool and tool output if you can share them.
- For a new feature or a change in how the lighting looks, open an issue first so we can agree on the approach. The scope is lightmaps and structure BSPs only (see the README).
- Default output must stay in the stock tag format. Compatibility has been tested with the H1A editing kit and OpenCE in game. Improvements that need new tag fields or renderer changes need a separate switch and a discussion first.

## Setting up

Follow the install steps in the README, then add pytest:

```powershell
python -m pip install pytest
```

Point `HCEEK_TAGS` at an editing-kit tags folder and, if needed, `LIGHTMAPPER_BLENDER` at `blender.exe`, so the tests that need them run instead of being skipped.

## How we work

- Write the test first. Add a test that fails without your change, watch it fail, then make it pass. For a bug fix, the test reproduces the bug.
- Make sure the test can catch the bug: undo the key line of your fix in a scratch copy and check that the new test fails.
- Run the whole suite before you open a pull request: `python -m pytest -q lightmapper/tests`.
- When a change affects the output, check it on a real level. Bake a stock level such as Blood Gulch into a copy (`lightmap.py copy`), run `validate`, and run `diff` against tool's lightmaps. Put the numbers from before and after in the pull request.
- If input goes over a stock limit, stop with a clear message. Never cut data silently.
- Keep the light in floating point until the final 16-bit encode.
- `lightmapper/blender/bake.py` runs inside Blender and must not import anything else from this repository. It talks to the rest only through the job files.

## Code style

- Match the code around you: plain functions, numpy arrays, dataclasses for settings.
- Comments explain hidden constraints, workarounds or the reason for copying a behaviour from tool. Leave out comments that repeat the code.
- New settings get a default that keeps the current output, an entry in `docs/lightmapper-settings.md`, and a test.

## Commits and pull requests

- Keep commit messages short: one line that says what changed.
- Work on a branch and open a pull request against `main`. Every pull request gets a review before it is merged.
- Describe what you changed, why, how you tested it, and any change in the output.

## Releases

To build the app zip on your own machine you need Python 3.11 from python.org (not Anaconda), [LLVM](https://releases.llvm.org) and Visual Studio 2022's C++ runtime files (the "Desktop development with C++" workload). Then, in PowerShell:

```powershell
packaging\build.ps1 -Python <path to python.org python.exe>
```

The zip lands in `dist\`. The script stops if the Python is Anaconda or not 3.11, or if a bundled runtime file is too old.

Only a maintainer cuts a release. Update `main`, tag it with `vX.Y.Z` and push the tag:

```powershell
git switch main
git pull
git tag vX.Y.Z
git push origin vX.Y.Z
```

The release workflow then builds the Windows app with `packaging/build.ps1`, checks that the tagged commit is on `main`, and creates a GitHub release named after the tag with the zip attached. A tag that is not on `main` is not released. That check only catches a mistake: a pushed tag runs the workflow file of the tagged commit, so it does not stop someone who can push tags. The repository owner should add a tag ruleset that restricts who may create `v*` tags. The build skips the Blender self-test, because the runner has no Blender.

The release body is `packaging/release-notes/vX.Y.Z.md` if that file exists, otherwise GitHub's generated notes. Commit the notes file before you tag.

A push to `main` or a manual run of the workflow builds the zip and keeps it as a workflow artifact, with no release.

## Questions

For questions that are not bugs, find Moot on the [OpenCE Discord server](https://discord.gg/9gqcHyr5km).

## License of contributions

OpenTool is licensed under version 3 of the GNU General Public License only (see [LICENSE](LICENSE)). By contributing, you agree that your contribution is published under the same license.
