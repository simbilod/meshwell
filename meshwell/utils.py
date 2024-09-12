from difflib import unified_diff
from pathlib import Path

def compare_meshes(meshfile: Path):

    meshfile1 = meshfile
    meshfile2 = meshfile.parent / "references" / (str(meshfile.with_suffix("")) + '.reference.msh')

    with open(str(meshfile1), "r") as f:
        expected_lines = f.readlines()
    with open(str(meshfile2), "r") as f:
        actual_lines = f.readlines()

    diff = list(unified_diff(expected_lines, actual_lines))
    assert diff == [], f"Mesh {str(meshfile)} differs from its reference!" 