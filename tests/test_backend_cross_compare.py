"""Cross-backend equivalence tests.

Each test in this file exercises both ``cad_gmsh`` and ``cad_occ`` on the
same input and asserts the output meshes are equivalent at the level of
physical groups and per-group geometric mass.

Two patterns are used:

1. Parametrized tests via the ``cad_pipeline`` fixture (in conftest.py):
   pytest runs the test twice -- once per backend -- and the same
   assertions hit both. Useful for tests where the assertion is
   self-contained and backend-agnostic.

2. Side-by-side tests that call BOTH backends explicitly in one test
   body and compare summaries. Useful for catching subtle divergence
   that wouldn't trigger an isolated assertion failure.
"""
from __future__ import annotations

from pathlib import Path

import meshio
import numpy as np
import shapely

from meshwell.cad_gmsh import cad_gmsh
from meshwell.cad_occ import cad_occ
from meshwell.mesh import mesh
from meshwell.occ_xao_writer import write_xao
from meshwell.polyprism import PolyPrism

# ----- Geometric mass helpers -----------------------------------------------


def _tet_volume(p0, p1, p2, p3) -> float:
    """Signed tetrahedron volume."""
    return abs(np.dot(p1 - p0, np.cross(p2 - p0, p3 - p0))) / 6.0


def _tri_area(p0, p1, p2) -> float:
    """Triangle area via cross product."""
    return 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0))


def _line_length(p0, p1) -> float:
    return float(np.linalg.norm(p1 - p0))


def _cells_mass(points: np.ndarray, cells: np.ndarray, cell_type: str) -> float:
    """Sum geometric mass (volume / area / length) over a block of cells."""
    total = 0.0
    if cell_type == "tetra":
        for c in cells:
            p0, p1, p2, p3 = points[c]
            total += _tet_volume(p0, p1, p2, p3)
    elif cell_type == "triangle":
        for c in cells:
            p0, p1, p2 = points[c]
            total += _tri_area(p0, p1, p2)
    elif cell_type == "line":
        for c in cells:
            p0, p1 = points[c]
            total += _line_length(p0, p1)
    return total


# ----- Side-by-side comparator ----------------------------------------------


def _concat_blocks(m: meshio.Mesh, cell_type: str) -> np.ndarray:
    """Concatenate all cell blocks of ``cell_type`` into one array.

    meshio stores multiple blocks of the same type when the mesh was read
    from a file with per-entity tags.  ``cell_sets_dict`` indices are
    global into this concatenated array.
    """
    blocks = [c.data for c in m.cells if c.type == cell_type]
    if not blocks:
        return np.empty((0, 0), dtype=int)
    return np.concatenate(blocks, axis=0)


def _mesh_summary(m: meshio.Mesh) -> dict[str, dict[str, tuple[int, float]]]:
    """Per-(physical_group, element_type) -> (count, total_mass).

    Mass is geometric: tetra volume, triangle area, line length.
    """
    summary: dict[str, dict[str, tuple[int, float]]] = {}
    for name, cell_arrays in m.cell_sets_dict.items():
        for cell_type, indices in cell_arrays.items():
            if cell_type not in ("tetra", "triangle", "line"):
                continue
            all_cells = _concat_blocks(m, cell_type)
            # Filter out sentinel values (-1, very large) used by some meshio readers
            valid = indices[(indices >= 0) & (indices < len(all_cells))]
            cells = all_cells[valid]
            count = len(valid)
            mass = _cells_mass(m.points, cells, cell_type)
            summary.setdefault(name, {})[cell_type] = (count, mass)
    return summary


def _assert_summaries_equivalent(
    s_gmsh: dict[str, dict[str, tuple[int, float]]],
    s_occ: dict[str, dict[str, tuple[int, float]]],
    rel_tol: float = 1e-3,
    ignore_groups: set[str] = frozenset(),
) -> None:
    """Assert two per-backend mesh summaries are equivalent.

    Both summaries must have the same (group, type) keys and per-key
    masses within ``rel_tol`` (relative). Element COUNTS are NOT
    required to match -- mesher non-determinism is fine; only the
    integrated mass per group matters.

    Groups whose name starts with ``"gmsh:"`` (meshio internal markers
    such as ``gmsh:bounding_entities``) are always skipped -- they are
    not user-defined physical groups and differ across backends.
    """

    def _skip(g: str) -> bool:
        return g in ignore_groups or g.startswith("gmsh:")

    g_keys = {(g, t) for g, types in s_gmsh.items() if not _skip(g) for t in types}
    o_keys = {(g, t) for g, types in s_occ.items() if not _skip(g) for t in types}
    assert g_keys == o_keys, (
        f"Group/type sets differ.\n"
        f"  gmsh-only: {g_keys - o_keys}\n"
        f"  occ-only:  {o_keys - g_keys}"
    )
    for g, t in g_keys:
        gc, gm = s_gmsh[g][t]
        oc, om = s_occ[g][t]
        if gm == 0.0 and om == 0.0:
            continue
        denom = max(abs(gm), abs(om))
        rel = abs(gm - om) / denom if denom > 0 else 0.0
        assert rel < rel_tol, (
            f"Mass mismatch on ({g!r}, {t!r}): gmsh={gm:.6g} (n={gc}) "
            f"vs occ={om:.6g} (n={oc}); rel={rel:.3e} > rel_tol={rel_tol:.3e}"
        )


def _run_both(
    entities_factory, tmp_path: Path, dim: int = 3
) -> tuple[meshio.Mesh, meshio.Mesh]:
    """Run the same scene through gmsh and occ backends.

    ``entities_factory`` is a zero-arg callable returning a fresh entities
    list -- needed because the pre-pass mutates entities in place.
    """
    # gmsh path
    gmsh_msh = tmp_path / "gmsh.msh"
    _, mm = cad_gmsh(entities_factory())
    m_gmsh = mesh(
        dim=dim,
        default_characteristic_length=1.0,
        model=mm,
        output_file=str(gmsh_msh),
        n_threads=1,
    )
    mm.finalize()

    # occ path
    occ_msh = tmp_path / "occ.msh"
    occ_xao = tmp_path / "occ.xao"
    labeled = cad_occ(entities_factory())
    write_xao(labeled, str(occ_xao))
    m_occ = mesh(
        dim=dim,
        default_characteristic_length=1.0,
        input_file=str(occ_xao),
        output_file=str(occ_msh),
        n_threads=1,
    )
    return m_gmsh, m_occ


# ----- Smoke test -----------------------------------------------------------


def test_smoke_two_unit_prisms_match(tmp_path):
    """Sanity check that the comparator works on a trivial scene."""

    def make():
        A = shapely.Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
        B = shapely.Polygon([(5, 0), (10, 0), (10, 5), (5, 5)])
        buffers = {0.0: 0.0, 2.0: 0.0}
        return [
            PolyPrism(polygons=A, buffers=buffers, physical_name="A", mesh_order=1),
            PolyPrism(polygons=B, buffers=buffers, physical_name="B", mesh_order=2),
        ]

    m_gmsh, m_occ = _run_both(make, tmp_path)
    s_gmsh = _mesh_summary(m_gmsh)
    s_occ = _mesh_summary(m_occ)
    _assert_summaries_equivalent(s_gmsh, s_occ)
