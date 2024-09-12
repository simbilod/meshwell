from difflib import unified_diff
from pathlib import Path
from meshwell.config import PATH


def compare_meshes(meshfile: Path):
    meshfile1 = meshfile
    meshfile2 = PATH.references / (str(meshfile.with_suffix("")) + ".reference.msh")

    with open(str(meshfile1)) as f:
        expected_lines = f.readlines()
    with open(str(meshfile2)) as f:
        actual_lines = f.readlines()

    diff = list(unified_diff(expected_lines, actual_lines))
    assert diff == [], f"Mesh {str(meshfile)} differs from its reference!"
