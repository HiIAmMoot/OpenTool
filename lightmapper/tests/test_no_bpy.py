"""Keep Blender-only imports on the subprocess side of the GPL boundary."""

from pathlib import Path
import re


def test_only_blender_scripts_import_bpy():
    root = Path(__file__).resolve().parents[1]
    forbidden = re.compile(r"^\s*(import|from)\s+(bpy|mathutils)\b", re.MULTILINE)
    violations = []
    for path in root.rglob("*.py"):
        relative = path.relative_to(root)
        # (a test driver runs inside Blender as the subprocess, like blender/)
        if relative.parts[0] != "blender" and not (relative.parts[0] == "tests" and path.name.endswith("_driver.py")):
            if forbidden.search(path.read_text(encoding="utf-8")):
                violations.append(str(path.relative_to(root)))
    assert violations == []
