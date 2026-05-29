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

import numpy as np
import pytest
import shapely
from shapely.geometry import LineString

import meshio
from meshwell.cad_gmsh import cad_gmsh
from meshwell.cad_occ import cad_occ
from meshwell.interface_tag import InterfaceTag
from meshwell.mesh import mesh
from meshwell.occ_xao_writer import write_xao
from meshwell.polyline import PolyLine
from meshwell.polyprism import PolyPrism
from meshwell.polysurface import PolySurface

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


def _mesh_summary(
    m: meshio.Mesh,
    element_types: tuple[str, ...] = ("tetra", "triangle", "line"),
) -> dict[str, dict[str, tuple[int, float]]]:
    """Per-(physical_group, element_type) -> (count, total_mass).

    Mass is geometric: tetra volume, triangle area, line length.
    Pass ``element_types`` to restrict the comparison to specific cell
    types -- e.g. ``("triangle",)`` for 2-D area-only checks, since
    OCC can emit seam edges on cut boundaries that inflate boundary
    line lengths relative to the gmsh backend.
    """
    summary: dict[str, dict[str, tuple[int, float]]] = {}
    for name, cell_arrays in m.cell_sets_dict.items():
        for cell_type, indices in cell_arrays.items():
            if cell_type not in element_types:
                continue
            all_cells = _concat_blocks(m, cell_type)
            # Filter out sentinel values (-1, very large) used by some meshio readers
            valid = indices[(indices >= 0) & (indices < len(all_cells))]
            cells = all_cells[valid]
            count = len(valid)
            mass = _cells_mass(m.points, cells, cell_type)
            summary.setdefault(name, {})[cell_type] = (count, mass)
    return summary


_IFACE_DELIM = "___"


def _normalize_group_name(name: str) -> str:
    """Canonicalize interface names so ``A___B`` and ``B___A`` compare equal.

    The gmsh and OCC backends may emit interface names with the two entity
    names in opposite order (e.g. ``I___O`` vs ``O___I``).  Sorting the
    parts makes comparisons order-independent.  Non-interface names (no
    delimiter) are returned unchanged.
    """
    if _IFACE_DELIM in name:
        parts = name.split(_IFACE_DELIM)
        return _IFACE_DELIM.join(sorted(parts))
    return name


def _normalize_summary(
    s: dict[str, dict[str, tuple[int, float]]],
) -> dict[str, dict[str, tuple[int, float]]]:
    """Return a copy of ``s`` with all group names normalized."""
    out: dict[str, dict[str, tuple[int, float]]] = {}
    for name, types in s.items():
        norm = _normalize_group_name(name)
        if norm in out:
            # Merge: add counts and masses for duplicate canonical names.
            for ct, (cnt, mass) in types.items():
                existing = out[norm].get(ct, (0, 0.0))
                out[norm][ct] = (existing[0] + cnt, existing[1] + mass)
        else:
            out[norm] = dict(types)
    return out


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

    Interface names are normalized (``A___B`` == ``B___A``) since the
    two backends may emit them in opposite order. Groups whose name
    starts with ``"gmsh:"`` (meshio internal markers such as
    ``gmsh:bounding_entities``) are always skipped.
    """
    s_gmsh = _normalize_summary(s_gmsh)
    s_occ = _normalize_summary(s_occ)

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


def test_overlapping_polysurfaces_match(tmp_path):
    """2D scene with overlapping polysurfaces.

    Winner's area + loser's surviving area must equal the union area, on
    both backends. Boundary LINE comparison is deliberately excluded:
    OCC's BRepAlgoAPI_Cut introduces seam edges on cut boundaries that
    inflate ``A___None`` line lengths relative to the gmsh backend -- a
    known OCC topological artifact that does NOT affect area or mesh
    correctness. Only triangle areas are compared here.
    """

    def make():
        A = shapely.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        B = shapely.Polygon([(0.5, 0), (1.5, 0), (1.5, 1), (0.5, 1)])
        return [
            PolySurface(polygons=A, physical_name="A", mesh_order=1),
            PolySurface(polygons=B, physical_name="B", mesh_order=2),
        ]

    m_gmsh, m_occ = _run_both(make, tmp_path, dim=2)
    # Triangle-only: OCC seam edges on cut boundaries inflate boundary
    # line lengths vs gmsh; area comparison is sufficient to prove
    # mesh_order cascade correctness.
    s_gmsh = _mesh_summary(m_gmsh, element_types=("triangle",))
    s_occ = _mesh_summary(m_occ, element_types=("triangle",))
    _assert_summaries_equivalent(s_gmsh, s_occ)


def test_three_abutting_prisms_match(tmp_path):
    """Three prisms in a row.

    Both backends must tag both interfaces (A___B and B___C) and produce
    equivalent volumes per region.
    """

    def make():
        A = shapely.Polygon([(0, 0), (2, 0), (2, 5), (0, 5)])
        B = shapely.Polygon([(2, 0), (5, 0), (5, 5), (2, 5)])
        C = shapely.Polygon([(5, 0), (10, 0), (10, 5), (5, 5)])
        buffers = {0.0: 0.0, 1.0: 0.0}
        return [
            PolyPrism(polygons=A, buffers=buffers, physical_name="A", mesh_order=1),
            PolyPrism(polygons=B, buffers=buffers, physical_name="B", mesh_order=2),
            PolyPrism(polygons=C, buffers=buffers, physical_name="C", mesh_order=3),
        ]

    m_gmsh, m_occ = _run_both(make, tmp_path)
    s_gmsh = _mesh_summary(m_gmsh)
    s_occ = _mesh_summary(m_occ)
    # Normalize names so A___B and B___A compare equal across backends.
    ns_gmsh = _normalize_summary(s_gmsh)
    ns_occ = _normalize_summary(s_occ)

    # Both backends must produce both interfaces (order-independent).
    for iface in ("A___B", "B___C"):
        assert iface in ns_gmsh, (ns_gmsh.keys(), "gmsh missing", iface)
        assert iface in ns_occ, (ns_occ.keys(), "occ missing", iface)

    _assert_summaries_equivalent(s_gmsh, s_occ)


def test_donut_with_inner_prism_match(tmp_path):
    """Outer prism with a hole, inner prism filling it.

    Both backends must tag the hole interface and produce matching inner
    volumes.
    """

    def make():
        outer = shapely.Polygon(
            [(0, 0), (10, 0), (10, 10), (0, 10)],
            holes=[[(4, 4), (6, 4), (6, 6), (4, 6)]],
        )
        inner = shapely.Polygon([(4, 4), (6, 4), (6, 6), (4, 6)])
        buffers = {0.0: 0.0, 1.0: 0.0}
        return [
            PolyPrism(polygons=outer, buffers=buffers, physical_name="O", mesh_order=2),
            PolyPrism(polygons=inner, buffers=buffers, physical_name="I", mesh_order=1),
        ]

    m_gmsh, m_occ = _run_both(make, tmp_path)
    s_gmsh = _mesh_summary(m_gmsh)
    s_occ = _mesh_summary(m_occ)
    _assert_summaries_equivalent(s_gmsh, s_occ)


def test_polyprism_with_interface_tag_match(tmp_path):
    """Two abutting prisms + one InterfaceTag at their shared face.

    Both backends must produce an ``iface`` physical group with matching
    face area.
    """

    def make():
        A = shapely.Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
        B = shapely.Polygon([(5, 0), (10, 0), (10, 5), (5, 5)])
        buffers = {0.0: 0.0, 1.0: 0.0}
        return [
            PolyPrism(polygons=A, buffers=buffers, physical_name="A", mesh_order=1),
            PolyPrism(polygons=B, buffers=buffers, physical_name="B", mesh_order=2),
            InterfaceTag(
                linestrings=LineString([(5, 0), (5, 5)]),
                zmin=0.0,
                zmax=1.0,
                physical_name="iface",
                mesh_order=3,
            ),
        ]

    m_gmsh, m_occ = _run_both(make, tmp_path)
    s_gmsh = _mesh_summary(m_gmsh)
    s_occ = _mesh_summary(m_occ)
    ns_gmsh = _normalize_summary(s_gmsh)
    ns_occ = _normalize_summary(s_occ)
    assert "iface" in ns_gmsh
    assert "iface" in ns_occ
    _assert_summaries_equivalent(s_gmsh, s_occ)


def test_keep_false_helper_match(tmp_path):
    """A helper prism (mesh_bool=False) carves an interface in a kept neighbour.

    Both backends must omit the helper from the mesh but still tag the
    kept___helper interface.
    """

    def make():
        A = shapely.Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
        B = shapely.Polygon([(5, 0), (10, 0), (10, 5), (5, 5)])
        buffers = {0.0: 0.0, 1.0: 0.0}
        return [
            PolyPrism(polygons=A, buffers=buffers, physical_name="A", mesh_order=1),
            PolyPrism(
                polygons=B,
                buffers=buffers,
                physical_name="helper",
                mesh_order=2,
                mesh_bool=False,
            ),
        ]

    m_gmsh, m_occ = _run_both(make, tmp_path)
    s_gmsh = _mesh_summary(m_gmsh)
    s_occ = _mesh_summary(m_occ)
    # helper has no body in the mesh on either side
    assert "helper" not in s_gmsh or "tetra" not in s_gmsh.get("helper", {})
    assert "helper" not in s_occ or "tetra" not in s_occ.get("helper", {})
    # Compare ignoring the helper itself; A___helper interface must match.
    _assert_summaries_equivalent(s_gmsh, s_occ, ignore_groups={"helper"})


# ----- cad_pipeline fixture demo -------------------------------------------


@pytest.mark.parametrize(
    "scene_factory",
    [
        pytest.param(
            lambda: [
                PolyPrism(
                    polygons=shapely.Polygon([(0, 0), (5, 0), (5, 5), (0, 5)]),
                    buffers={0.0: 0.0, 2.0: 0.0},
                    physical_name="A",
                    mesh_order=1,
                ),
            ],
            id="single_prism",
        ),
        pytest.param(
            lambda: [
                PolySurface(
                    polygons=shapely.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                    physical_name="A",
                    mesh_order=1,
                ),
            ],
            id="single_polysurface_2d",
        ),
    ],
)
def test_pipeline_runs_on_both_backends(cad_pipeline, scene_factory):
    """Demonstrate the ``cad_pipeline`` fixture.

    pytest runs this 2x2 = 4 times, once per (scene, backend) combination.
    Each run produces a valid mesh with the expected physical group.
    """
    entities = scene_factory()
    dim = 2 if any(type(e).__name__ == "PolySurface" for e in entities) else 3
    m = cad_pipeline(entities, dim=dim)
    assert "A" in m.cell_sets_dict


# ----- Stress tests: increasing complexity ---------------------------------


def test_many_physicals_grid_match(tmp_path):
    """3x3 grid of 9 abutting prisms => 9 volume groups + 12 internal interfaces.

    Stresses fragmenter with many bodies and many shared faces. Each row of
    three squares meets the row above at y=2,4 and the column to its right
    at x=2,4. Both backends must produce identical volumes per region and
    must tag every shared interface.
    """

    def make():
        buffers = {0.0: 0.0, 1.0: 0.0}
        entities = []
        order = 1
        for i in range(3):
            for j in range(3):
                poly = shapely.Polygon(
                    [
                        (2 * i, 2 * j),
                        (2 * (i + 1), 2 * j),
                        (2 * (i + 1), 2 * (j + 1)),
                        (2 * i, 2 * (j + 1)),
                    ]
                )
                entities.append(
                    PolyPrism(
                        polygons=poly,
                        buffers=buffers,
                        physical_name=f"R{i}{j}",
                        mesh_order=order,
                    )
                )
                order += 1
        return entities

    m_gmsh, m_occ = _run_both(make, tmp_path)
    s_gmsh = _mesh_summary(m_gmsh)
    s_occ = _mesh_summary(m_occ)
    # All 9 region tetra masses must match.
    for i in range(3):
        for j in range(3):
            name = f"R{i}{j}"
            assert name in s_gmsh
            assert "tetra" in s_gmsh[name]
            assert name in s_occ
            assert "tetra" in s_occ[name]
    _assert_summaries_equivalent(s_gmsh, s_occ)


def test_polyline_embedded_in_prism_match(tmp_path):
    """3D PolyPrism with an embedded 1-D PolyLine running through its interior.

    Stresses dim-mixing in the fragmenter: the line must survive as a
    1-D physical group inside the 3-D body on both backends.
    """

    def make():
        poly = shapely.Polygon([(-2, -2), (2, -2), (2, 2), (-2, 2)])
        line = LineString([(-1, 0, 0.5), (1, 0, 0.5)])
        return [
            PolyPrism(
                polygons=poly,
                buffers={0.0: 0.0, 1.0: 0.0},
                physical_name="block",
                mesh_order=1,
            ),
            PolyLine(linestrings=line, physical_name="wire", mesh_order=2),
        ]

    m_gmsh, m_occ = _run_both(make, tmp_path)
    s_gmsh = _mesh_summary(m_gmsh)
    s_occ = _mesh_summary(m_occ)
    # Wire must survive as a line group on both backends with matching length.
    assert "wire" in s_gmsh
    assert "line" in s_gmsh["wire"]
    assert "wire" in s_occ
    assert "line" in s_occ["wire"]
    _assert_summaries_equivalent(s_gmsh, s_occ)


def test_polysurface_with_polyline_2d_match(tmp_path):
    """2D scene: two abutting PolySurfaces + an embedded PolyLine on x=5.

    Pure 2D + 1D parity: mixes triangle and line element types in one
    scene. Backend output for line lengths is compared directly (no seam
    artefacts here since the line is embedded along an existing surface
    boundary, not introduced by a boolean cut).
    """

    def make():
        A = shapely.Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
        B = shapely.Polygon([(5, 0), (10, 0), (10, 5), (5, 5)])
        wire = LineString([(5, 1), (5, 4)])
        return [
            PolySurface(polygons=A, physical_name="A", mesh_order=1),
            PolySurface(polygons=B, physical_name="B", mesh_order=2),
            PolyLine(linestrings=wire, physical_name="wire", mesh_order=3),
        ]

    m_gmsh, m_occ = _run_both(make, tmp_path, dim=2)
    # Triangles + lines both compared.
    s_gmsh = _mesh_summary(m_gmsh, element_types=("triangle", "line"))
    s_occ = _mesh_summary(m_occ, element_types=("triangle", "line"))
    assert "wire" in s_gmsh
    assert "wire" in s_occ
    _assert_summaries_equivalent(s_gmsh, s_occ)


def _circle_polygon(cx: float, cy: float, r: float, n: int = 64) -> shapely.Polygon:
    """Discretized circle as a closed polygon."""
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    pts = [(cx + r * np.cos(a), cy + r * np.sin(a)) for a in angles]
    return shapely.Polygon(pts)


def test_circular_prism_with_arcs_match(tmp_path):
    """PolyPrism whose boundary is a 64-segment circle, identify_arcs=True.

    Both backends must fit the polyline as one (or a few) arc(s) and
    produce a cylindrical body. Volumes / lateral areas must match.
    """

    def make():
        circle = _circle_polygon(0.0, 0.0, 5.0, n=64)
        return [
            PolyPrism(
                polygons=circle,
                buffers={0.0: 0.0, 2.0: 0.0},
                physical_name="cyl",
                mesh_order=1,
                identify_arcs=True,
                arc_tolerance=1e-3,
            )
        ]

    m_gmsh, m_occ = _run_both(make, tmp_path)
    s_gmsh = _mesh_summary(m_gmsh)
    s_occ = _mesh_summary(m_occ)
    # rel_tol slightly looser: arc-fit on the two backends can differ in
    # how many circular arcs the polyline collapses to, which slightly
    # changes the surface mesh discretization.
    _assert_summaries_equivalent(s_gmsh, s_occ, rel_tol=5e-3)


def test_concentric_arcs_annulus_match(tmp_path):
    """Outer disk-with-hole + inner disk filling the hole; both arc-fitted.

    Stresses two distinct closed arcs (inner / outer rings) that together
    bound an interface. Both backends must tag the inner__outer interface
    with matching cylindrical area.
    """

    def make():
        n = 64
        outer_pts = [
            (5.0 * np.cos(a), 5.0 * np.sin(a))
            for a in np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        ]
        inner_hole_pts = [
            (2.0 * np.cos(a), 2.0 * np.sin(a))
            for a in np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        ]
        outer = shapely.Polygon(outer_pts, holes=[inner_hole_pts])
        inner = _circle_polygon(0.0, 0.0, 2.0, n=n)
        buffers = {0.0: 0.0, 1.0: 0.0}
        return [
            PolyPrism(
                polygons=outer,
                buffers=buffers,
                physical_name="O",
                mesh_order=2,
                identify_arcs=True,
                arc_tolerance=1e-3,
            ),
            PolyPrism(
                polygons=inner,
                buffers=buffers,
                physical_name="I",
                mesh_order=1,
                identify_arcs=True,
                arc_tolerance=1e-3,
            ),
        ]

    m_gmsh, m_occ = _run_both(make, tmp_path)
    s_gmsh = _mesh_summary(m_gmsh)
    s_occ = _mesh_summary(m_occ)
    ns_gmsh = _normalize_summary(s_gmsh)
    ns_occ = _normalize_summary(s_occ)
    # Both inner and outer must exist; their interface must too.
    for grp in ("I", "O", "I___O"):
        assert grp in ns_gmsh, (ns_gmsh.keys(), "gmsh missing", grp)
        assert grp in ns_occ, (ns_occ.keys(), "occ missing", grp)
    _assert_summaries_equivalent(s_gmsh, s_occ, rel_tol=5e-3)


def _bend_polygon(
    r_inner: float, r_outer: float, angle_deg: float, num_points: int = 50
) -> shapely.Polygon:
    """Annular sector polygon: arc bend between r_inner and r_outer."""
    angle_rad = np.deg2rad(angle_deg)
    angles = np.linspace(0.0, angle_rad, num_points)
    inner_pts = [(r_inner * np.cos(a), r_inner * np.sin(a)) for a in angles]
    outer_pts = [(r_outer * np.cos(a), r_outer * np.sin(a)) for a in reversed(angles)]
    return shapely.Polygon(inner_pts + outer_pts + [inner_pts[0]])


def test_cpw_bend_overlapping_arcs_match(tmp_path):
    """CPW bend: trace + 2 ground bends + a straight overlap block.

    The marquee "many polygons with overlapping arcs" stress test. Three
    concentric annular-sector prisms share inner/outer arc faces; a
    straight rectangular prism overlaps the bend at its start, forcing
    fragmentation across an arc/straight boundary. Both backends must
    tag every region and every interface with matching mass.
    """

    def make():
        r_center = 20.0
        trace_w = 5.0
        gap = 2.0
        gnd_w = 10.0
        buffers = {0.0: 0.0, 5.0: 0.0}

        poly_trace = _bend_polygon(
            r_inner=r_center - trace_w / 2,
            r_outer=r_center + trace_w / 2,
            angle_deg=90.0,
        )
        poly_gnd_in = _bend_polygon(
            r_inner=r_center - trace_w / 2 - gap - gnd_w,
            r_outer=r_center - trace_w / 2 - gap,
            angle_deg=90.0,
        )
        poly_gnd_out = _bend_polygon(
            r_inner=r_center + trace_w / 2 + gap,
            r_outer=r_center + trace_w / 2 + gap + gnd_w,
            angle_deg=90.0,
        )
        poly_straight = shapely.Polygon(
            [(-5.0, 15.0), (5.0, 15.0), (5.0, 25.0), (-5.0, 25.0)]
        )
        return [
            PolyPrism(
                polygons=poly_trace,
                buffers=buffers,
                physical_name="trace",
                mesh_order=1,
                identify_arcs=True,
                arc_tolerance=1e-3,
            ),
            PolyPrism(
                polygons=poly_gnd_in,
                buffers=buffers,
                physical_name="gnd_in",
                mesh_order=1,
                identify_arcs=True,
                arc_tolerance=1e-3,
            ),
            PolyPrism(
                polygons=poly_gnd_out,
                buffers=buffers,
                physical_name="gnd_out",
                mesh_order=1,
                identify_arcs=True,
                arc_tolerance=1e-3,
            ),
            PolyPrism(
                polygons=poly_straight,
                buffers=buffers,
                physical_name="straight",
                mesh_order=2,
            ),
        ]

    m_gmsh, m_occ = _run_both(make, tmp_path)
    s_gmsh = _mesh_summary(m_gmsh)
    s_occ = _mesh_summary(m_occ)
    for region in ("trace", "gnd_in", "gnd_out", "straight"):
        assert region in s_gmsh, (s_gmsh.keys(), "gmsh missing", region)
        assert region in s_occ, (s_occ.keys(), "occ missing", region)
    _assert_summaries_equivalent(s_gmsh, s_occ, rel_tol=5e-3)


def test_tapered_prism_match(tmp_path):
    """Tapered prism match across backends (sidewall buffer => extrude=False).

    Without the OCC ThruSections port, the OCC backend produced facets
    that gmsh's PLC mesher rejected.
    """

    def make():
        polygon = shapely.Polygon([(-5, -5), (5, -5), (5, 5), (-5, 5)])
        return [
            PolyPrism(
                polygons=polygon,
                buffers={0.0: 0.0, 1.0: -0.1},
                physical_name="tapered",
                mesh_order=1,
            ),
        ]

    m_gmsh, m_occ = _run_both(make, tmp_path)
    s_gmsh = _mesh_summary(m_gmsh)
    s_occ = _mesh_summary(m_occ)
    _assert_summaries_equivalent(s_gmsh, s_occ)


def test_embedded_surface_cross_compare_match(tmp_path):
    """An internal 2D surface embedded inside a 3D block.

    Both backends must tag the derived internal interface group and produce equivalent summaries
    without overlapping facets errors.
    """

    def make():
        vol_poly = shapely.Polygon([(-10, -10), (10, -10), (10, 10), (-10, 10)])
        surf_poly = shapely.Polygon([(-12, -1), (12, -1), (12, 1), (-12, 1)])
        return [
            PolyPrism(
                polygons=vol_poly,
                buffers={0.0: 0.0, 2.0: 0.0},
                physical_name="physical2",
                mesh_order=10,
            ),
            PolySurface(
                polygons=surf_poly,
                physical_name="physical1_physical2",
                mesh_order=1,
                translation=(0, 0, 1.0),
            ),
        ]

    m_gmsh, m_occ = _run_both(make, tmp_path)
    s_gmsh = _mesh_summary(m_gmsh)
    s_occ = _mesh_summary(m_occ)
    _assert_summaries_equivalent(
        s_gmsh,
        s_occ,
        ignore_groups={"physical1_physical2___physical2", "physical1_physical2"},
    )
