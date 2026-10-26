"""Keep Blender-only imports on the subprocess side of the GPL boundary."""

from pathlib import Path
import re


def test_only_blender_scripts_import_bpy():
    root = Path(__file__).resolve().parents[1]
    forbidden = re.compile(r"^\s*(import|from)\s+(bpy|mathutils)\b", re.MULTILINE)
    violations = []
    for path in root.rglob("*.py"):
        if path.relative_to(root).parts[0] != "blender":
            if forbidden.search(path.read_text(encoding="utf-8")):
                violations.append(str(path.relative_to(root)))
    assert violations == []
