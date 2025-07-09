from difflib import unified_diff
from pathlib import Path
from meshwell.config import PATH


def compare_gmsh_files(meshfile: Path, other_meshfile: Path | None = None):
    meshfile2 = meshfile
    if other_meshfile is None:
        meshfile1 = PATH.references / str(meshfile)
    else:
        meshfile1 = other_meshfile

    with open(str(meshfile1)) as f:
        expected_lines = f.readlines()
    with open(str(meshfile2)) as f:
        actual_lines = f.readlines()

    diff = list(unified_diff(expected_lines, actual_lines))
    assert diff == [], f"Mesh {str(meshfile)} differs from its reference!"


def compare_mesh_headers(meshfile: Path, other_meshfile: Path | None = None):
    meshfile2 = meshfile
    if other_meshfile is None:
        meshfile1 = PATH.references / (str(meshfile.with_suffix("")) + ".reference.msh")
    else:
        meshfile1 = other_meshfile

    with open(str(meshfile1)) as f:
        expected_lines = []
        for line in f:
            expected_lines.append(line)
            if line.startswith("$Entities"):
                # Get one more line after $Entities
                expected_lines.append(next(f))
                break

    with open(str(meshfile2)) as f:
        actual_lines = []
        for line in f:
            actual_lines.append(line)
            if line.startswith("$Entities"):
                # Get one more line after $Entities
                actual_lines.append(next(f))
                break

    diff = list(unified_diff(expected_lines, actual_lines))
    assert diff == [], f"Mesh headers in {str(meshfile)} differ from reference!"
