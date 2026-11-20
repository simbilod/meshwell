"""The GMSH 4.x parse must stay correct while being linear in size."""
from __future__ import annotations

from pathlib import Path

import gmsh
import numpy as np
import pytest

from meshwell.quality import MeshQualityAnalyzer


@pytest.fixture(scope="module")
def msh(tmp_path_factory) -> Path:
    """A small two-region 2D mesh in ASCII MSH 4.1, written fresh for the test."""
    path = tmp_path_factory.mktemp("quality") / "two_regions.msh"
    owns_gmsh = not gmsh.isInitialized()
    if owns_gmsh:
        gmsh.initialize()
    try:
        gmsh.model.add("quality_parse")
        outer = gmsh.model.occ.addRectangle(0, 0, 0, 10, 10)
        inner = gmsh.model.occ.addRectangle(3, 3, 0, 4, 4)
        gmsh.model.occ.fragment([(2, outer)], [(2, inner)])
        gmsh.model.occ.synchronize()
        surfaces = [tag for _, tag in gmsh.model.getEntities(2)]
        ring = max(surfaces, key=lambda tag: gmsh.model.occ.getMass(2, tag))
        core = next(tag for tag in surfaces if tag != ring)
        gmsh.model.addPhysicalGroup(2, [ring], 1, "outer")
        gmsh.model.addPhysicalGroup(2, [core], 2, "inner")
        gmsh.option.setNumber("Mesh.MeshSizeMax", 0.5)
        gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
        gmsh.option.setNumber("Mesh.Binary", 0)
        gmsh.model.mesh.generate(2)
        gmsh.write(str(path))
    finally:
        gmsh.model.remove()
        if owns_gmsh:
            gmsh.finalize()
    return path


def _reference_nodes(path: Path) -> dict[int, tuple[float, float, float]]:
    """Independent, deliberately naive reader of the MSH 4.x ``$Nodes`` block."""
    lines = path.read_text().splitlines()
    i = lines.index("$Nodes") + 1
    n_blocks = int(lines[i].split()[0])
    i += 1
    nodes: dict[int, tuple[float, float, float]] = {}
    for _ in range(n_blocks):
        _dim, _tag, _parametric, count = (int(p) for p in lines[i].split())
        i += 1
        ids = [int(lines[i + k]) for k in range(count)]
        i += count
        for node_id in ids:
            x, y, z = (float(p) for p in lines[i].split())
            nodes[node_id] = (x, y, z)
            i += 1
    return nodes


def _analyzer(msh: Path) -> MeshQualityAnalyzer:
    analyzer = MeshQualityAnalyzer(msh)
    assert analyzer.parse_gmsh_mesh()
    return analyzer


def test_nodes_match_an_independent_reader(msh):
    analyzer = _analyzer(msh)
    expected = _reference_nodes(msh)

    assert set(analyzer.nodes) == set(expected)
    for node_id, coords in expected.items():
        assert np.allclose(analyzer.nodes[node_id], coords)


def test_physical_tag_lookup_matches_the_linear_scan(msh):
    analyzer = _analyzer(msh)
    for dim in (0, 1, 2, 3):
        expected = next(
            (tag for tag, (d, _) in analyzer.physical_names.items() if d == dim), None
        )
        assert analyzer._find_physical_tag_for_entity(0, dim) == expected


def test_elements_get_a_physical_tag_for_their_dimension(msh):
    analyzer = _analyzer(msh)
    assert analyzer.triangle_elements
    tagged = [e["physical_tag"] for e in analyzer.triangle_elements]
    assert all(tag is not None for tag in tagged)
    assert set(tagged) <= set(analyzer.physical_names)
