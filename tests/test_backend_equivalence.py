"""Compare :mod:`meshwell.cad_occ` and :mod:`meshwell.cad_gmsh` outputs.

Both backends are deliberately structurally equivalent: instantiate ->
fragment -> mesh_order ownership -> ``keep=False`` handling. These tests
pin that equivalence on a small set of representative scenes by running
each backend, loading the result into gmsh, and comparing:

- the set of physical-group names produced,
- the total mass per physical group (within a small tolerance), and
- the absence of duplicate top-dim entities (two entities with the same
  centroid + mass are the signature of BOPAlgo missing a SameDomain
  fusion, which is exactly the failure mode the simplified pipelines
  must avoid together).
"""
from __future__ import annotations

import shapely

import gmsh
from meshwell.cad_gmsh import cad_gmsh
from meshwell.cad_occ import cad_occ
from meshwell.mesh import mesh
from meshwell.model import ModelManager
from meshwell.polyprism import PolyPrism
from meshwell.polysurface import PolySurface

# ------------------------------ helpers ------------------------------


def _physical_summary(
    mm: ModelManager,
) -> dict[tuple[int, str], tuple[float, int]]:
    """Return ``{(dim, name): (total_mass, dimtag_count)}`` for every physical group.

    Mass is the OCC-level mass of the owned entities (length / area /
    volume depending on dimension); the entity count catches cases where
    two pipelines ship the same total mass split across a different
    number of pieces.
    """
    out: dict[tuple[int, str], tuple[float, int]] = {}
    for dim, tag in mm.model.getPhysicalGroups():
        name = mm.model.getPhysicalName(dim, tag)
        ents = list(mm.model.getEntitiesForPhysicalGroup(dim, tag))
        mass = sum(mm.model.occ.getMass(dim, int(t)) for t in ents)
        out[(dim, name)] = (mass, len(ents))
    return out


def _duplicate_entities(
    mm: ModelManager, dim: int, atol: float = 1e-6
) -> list[tuple[int, int]]:
    """Return pairs of entity tags at ``dim`` that share centroid + mass.

    Two entities at the same centroid + mass are near-coincident
    duplicates that BOPAlgo failed to fuse (or that the backend-side
    tagging emitted twice). Tolerance is rounded off per axis so we
    bucket properly under FP noise.
    """
    from collections import defaultdict

    digits = max(0, int(-round(__import__("math").log10(atol))))
    buckets: dict[tuple[float, float, float, float], list[int]] = defaultdict(list)
    for _, tag in mm.model.getEntities(dim):
        cx, cy, cz = mm.model.occ.getCenterOfMass(dim, tag)
        m = mm.model.occ.getMass(dim, tag)
        key = (
            round(cx, digits),
            round(cy, digits),
            round(cz, digits),
            round(m, digits),
        )
        buckets[key].append(int(tag))
    dups: list[tuple[int, int]] = []
    for tags in buckets.values():
        if len(tags) > 1:
            tags.sort()
            dups.extend(
                (tags[i], tags[j])
                for i in range(len(tags))
                for j in range(i + 1, len(tags))
            )
    return dups


def _run_occ(entities, tmp_path) -> ModelManager:
    """Run ``cad_occ`` + XAO write + gmsh load; return the live ModelManager."""
    mm = ModelManager(filename="occ_cmp")
    xao = tmp_path / "occ.xao"
    from meshwell.occ_xao_writer import write_xao

    write_xao(cad_occ(entities), xao)
    mm.load_from_xao(xao)
    return mm


def _run_gmsh(entities) -> ModelManager:
    """Run ``cad_gmsh``; the caller is responsible for ``finalize``."""
    _, mm = cad_gmsh(entities)
    return mm


def _compare_summaries(occ, gmsh_, mass_rtol: float = 1e-4) -> None:
    assert set(occ) == set(gmsh_), (
        f"physical-group name mismatch:\n  only in occ: {set(occ) - set(gmsh_)}\n"
        f"  only in gmsh: {set(gmsh_) - set(occ)}"
    )
    for key, (occ_mass, _occ_n) in occ.items():
        gmsh_mass, _gmsh_n = gmsh_[key]
        assert (
            occ_mass == 0
            or abs(occ_mass - gmsh_mass) / max(abs(occ_mass), abs(gmsh_mass))
            < mass_rtol
        ), f"mass mismatch for {key}: occ={occ_mass}, gmsh={gmsh_mass}"
        # Dimtag count can legitimately differ (a split face vs a single
        # unified face), so we don't assert equality -- total mass is the
        # load-bearing invariant.


# ------------------------------ scenes -------------------------------


def _adjacent_3d_prisms() -> list:
    A = shapely.Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
    B = shapely.Polygon([(5, 0), (10, 0), (10, 5), (5, 5)])
    buffers = {0.0: 0.0, 2.0: 0.0}
    return [
        PolyPrism(polygons=A, buffers=buffers, physical_name="A", mesh_order=1),
        PolyPrism(polygons=B, buffers=buffers, physical_name="B", mesh_order=2),
    ]


def _outer_inner_2d() -> list:
    outer = shapely.Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    inner = shapely.Polygon([(3, 3), (7, 3), (7, 7), (3, 7)])
    return [
        PolySurface(polygons=outer, physical_name="outer", mesh_order=2),
        PolySurface(polygons=inner, physical_name="inner", mesh_order=1),
    ]


def _three_disjoint_prisms() -> list:
    buffers = {0.0: 0.0, 1.0: 0.0}
    polys = [
        shapely.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        shapely.Polygon([(2, 0), (3, 0), (3, 1), (2, 1)]),
        shapely.Polygon([(4, 0), (5, 0), (5, 1), (4, 1)]),
    ]
    return [
        PolyPrism(polygons=p, buffers=buffers, physical_name=n, mesh_order=i + 1)
        for i, (p, n) in enumerate(zip(polys, ("A", "B", "C")))
    ]


def _prism_with_keep_false_helper() -> list:
    A = shapely.Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
    B = shapely.Polygon([(5, 0), (10, 0), (10, 5), (5, 5)])
    buffers = {0.0: 0.0, 2.0: 0.0}
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


# ------------------------------ tests --------------------------------


def test_backends_agree_adjacent_3d(tmp_path):
    """Two adjacent prisms: same physical-group set and same masses."""
    occ_mm = _run_occ(_adjacent_3d_prisms(), tmp_path)
    try:
        gmsh_mm = _run_gmsh(_adjacent_3d_prisms())
        try:
            _compare_summaries(_physical_summary(occ_mm), _physical_summary(gmsh_mm))
        finally:
            gmsh_mm.finalize()
    finally:
        if gmsh.isInitialized():
            occ_mm.finalize()


def test_backends_agree_outer_inner_2d(tmp_path):
    """2D outer-around-inner: same physical-group set and same masses."""
    occ_mm = _run_occ(_outer_inner_2d(), tmp_path)
    try:
        gmsh_mm = _run_gmsh(_outer_inner_2d())
        try:
            _compare_summaries(_physical_summary(occ_mm), _physical_summary(gmsh_mm))
        finally:
            gmsh_mm.finalize()
    finally:
        if gmsh.isInitialized():
            occ_mm.finalize()


def test_backends_agree_three_disjoint(tmp_path):
    """Three disjoint prisms have no shared interfaces; same physicals both ways."""
    occ_mm = _run_occ(_three_disjoint_prisms(), tmp_path)
    try:
        gmsh_mm = _run_gmsh(_three_disjoint_prisms())
        try:
            _compare_summaries(_physical_summary(occ_mm), _physical_summary(gmsh_mm))
        finally:
            gmsh_mm.finalize()
    finally:
        if gmsh.isInitialized():
            occ_mm.finalize()


def test_backends_agree_keep_false(tmp_path):
    """``keep=False`` helper: drop the body AND still emit the interface.

    Both backends must produce ``A`` (the kept body) and
    ``A___helper`` (the shared face), never ``helper`` itself.
    """
    occ_mm = _run_occ(_prism_with_keep_false_helper(), tmp_path)
    try:
        occ_summary = _physical_summary(occ_mm)
        gmsh_mm = _run_gmsh(_prism_with_keep_false_helper())
        try:
            gmsh_summary = _physical_summary(gmsh_mm)
        finally:
            gmsh_mm.finalize()
    finally:
        if gmsh.isInitialized():
            occ_mm.finalize()

    occ_names = {n for _, n in occ_summary}
    gmsh_names = {n for _, n in gmsh_summary}
    # Both must name the kept volume and the interface, both must drop
    # the helper body.
    assert "A" in occ_names
    assert "A" in gmsh_names
    assert "helper" not in occ_names
    assert "helper" not in gmsh_names
    assert any("A___helper" in n or "helper___A" in n for n in occ_names), occ_names
    assert any("A___helper" in n or "helper___A" in n for n in gmsh_names), gmsh_names


def test_occ_cad_no_duplicate_top_dim_entities(tmp_path):
    """``cad_occ`` must not leak duplicate volumes after XAO load."""
    mm = _run_occ(_adjacent_3d_prisms(), tmp_path)
    try:
        assert _duplicate_entities(mm, dim=3) == []
        # Shared interface face is legitimate and owned once per side; no
        # duplicates expected at the face level either.
        assert _duplicate_entities(mm, dim=2) == []
    finally:
        mm.finalize()


def test_gmsh_cad_no_duplicate_top_dim_entities():
    """``cad_gmsh`` must not leak duplicate volumes either."""
    _, mm = cad_gmsh(_adjacent_3d_prisms())
    try:
        assert _duplicate_entities(mm, dim=3) == []
        assert _duplicate_entities(mm, dim=2) == []
    finally:
        mm.finalize()


# ------------------------------ mesh-level -------------------------------


def _mesh_summary(m, element_types: tuple[str, ...]) -> dict[str, tuple[int, float]]:
    """Return ``{physical_name: (element_count, total_element_mass)}``.

    ``m`` is a meshio mesh. Only cell blocks whose type is in
    ``element_types`` contribute (e.g. ``("tetra",)`` for 3D volume
    meshes, ``("triangle",)`` for 2D surface meshes). ``element_mass``
    is the signed volume for tetra or the signed area for triangles --
    identical per-physical between backends up to mesher determinism.
    """
    import numpy as np

    points = m.points
    summary: dict[str, tuple[int, float]] = {}
    for name, blocks_per_set in m.cell_sets.items():
        if name.startswith("gmsh:"):
            continue
        count = 0
        mass = 0.0
        for block_idx, idx_arr in enumerate(blocks_per_set):
            if idx_arr is None or len(idx_arr) == 0:
                continue
            block = m.cells[block_idx]
            if block.type not in element_types:
                continue
            cells = block.data[idx_arr]
            if block.type == "tetra":
                a = points[cells[:, 0]]
                b = points[cells[:, 1]]
                c = points[cells[:, 2]]
                d = points[cells[:, 3]]
                # 1/6 |det[ b-a | c-a | d-a ]|
                vol = np.abs(np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a)) / 6.0
                mass += float(vol.sum())
            elif block.type == "triangle":
                a = points[cells[:, 0]]
                b = points[cells[:, 1]]
                c = points[cells[:, 2]]
                # 1/2 | (b-a) x (c-a) |
                cross = np.cross(b - a, c - a)
                if cross.ndim == 1:
                    area = 0.5 * np.abs(cross)
                else:
                    area = 0.5 * np.linalg.norm(cross, axis=1)
                mass += float(area.sum())
            count += len(cells)
        if count:
            summary[name] = (count, mass)
    return summary


def _mesh_with_occ(entities, tmp_path, **mesh_kw):
    """Drive cad_occ -> write_xao -> mesh(input_file=...); return meshio mesh."""
    from meshwell.occ_xao_writer import write_xao

    xao = tmp_path / "occ.xao"
    msh = tmp_path / "occ.msh"
    write_xao(cad_occ(entities), xao)
    return mesh(input_file=xao, output_file=msh, **mesh_kw)


def _mesh_with_gmsh(entities, tmp_path, **mesh_kw):
    """Drive cad_gmsh -> mesh(model=...); return meshio mesh."""
    _, mm = cad_gmsh(entities)
    try:
        return mesh(model=mm, output_file=tmp_path / "gmsh.msh", **mesh_kw)
    finally:
        mm.finalize()


def _assert_mesh_summaries_equivalent(
    occ_sum: dict[str, tuple[int, float]],
    gmsh_sum: dict[str, tuple[int, float]],
    mass_rtol: float = 1e-4,
    count_rtol: float = 0.1,
) -> None:
    assert set(occ_sum) == set(gmsh_sum), (
        f"physical-group mismatch at mesh level:\n"
        f"  only in occ: {set(occ_sum) - set(gmsh_sum)}\n"
        f"  only in gmsh: {set(gmsh_sum) - set(occ_sum)}"
    )
    for name, (occ_n, occ_mass) in occ_sum.items():
        gmsh_n, gmsh_mass = gmsh_sum[name]
        # Mass (volume / area) is a topology-level invariant: both
        # backends mesh the same geometry, so the total must match
        # very tightly.
        assert (
            abs(occ_mass - gmsh_mass) / max(abs(occ_mass), abs(gmsh_mass)) < mass_rtol
        ), f"mass mismatch for {name!r}: occ={occ_mass}, gmsh={gmsh_mass}"
        # Element count is mesher-dependent (tag numbering and element
        # ordering differ between backends), so only assert
        # approximate agreement. Tight enough to catch a backend
        # accidentally meshing a different domain.
        mn, mx = min(occ_n, gmsh_n), max(occ_n, gmsh_n)
        assert (mx - mn) / max(
            mx, 1
        ) < count_rtol, (
            f"element-count mismatch for {name!r}: occ={occ_n}, gmsh={gmsh_n}"
        )


def test_backends_mesh_adjacent_3d_equivalently(tmp_path):
    """Mesh-level: same tet volume and similar tet count per physical group."""
    kw = dict(dim=3, default_characteristic_length=2.0, n_threads=1, verbosity=0)
    occ_m = _mesh_with_occ(_adjacent_3d_prisms(), tmp_path, **kw)
    gmsh_m = _mesh_with_gmsh(_adjacent_3d_prisms(), tmp_path, **kw)

    _assert_mesh_summaries_equivalent(
        _mesh_summary(occ_m, element_types=("tetra",)),
        _mesh_summary(gmsh_m, element_types=("tetra",)),
    )


def test_backends_mesh_outer_inner_2d_equivalently(tmp_path):
    """Mesh-level: same triangle area per physical group."""
    kw = dict(dim=2, default_characteristic_length=1.0, n_threads=1, verbosity=0)
    occ_m = _mesh_with_occ(_outer_inner_2d(), tmp_path, **kw)
    gmsh_m = _mesh_with_gmsh(_outer_inner_2d(), tmp_path, **kw)

    _assert_mesh_summaries_equivalent(
        _mesh_summary(occ_m, element_types=("triangle",)),
        _mesh_summary(gmsh_m, element_types=("triangle",)),
    )


def test_backends_mesh_keep_false_equivalently(tmp_path):
    """``keep=False``: helper body absent from both meshes.

    Tet volume under physical group ``A`` equals the kept entity's
    5x5x2=50 on both backends.
    """
    kw = dict(dim=3, default_characteristic_length=2.0, n_threads=1, verbosity=0)
    occ_m = _mesh_with_occ(_prism_with_keep_false_helper(), tmp_path, **kw)
    gmsh_m = _mesh_with_gmsh(_prism_with_keep_false_helper(), tmp_path, **kw)

    occ_sum = _mesh_summary(occ_m, element_types=("tetra",))
    gmsh_sum = _mesh_summary(gmsh_m, element_types=("tetra",))
    # Only "A" should carry tet volume on both sides.
    assert "A" in occ_sum
    assert "A" in gmsh_sum
    assert "helper" not in occ_sum
    assert "helper" not in gmsh_sum
    # Prism A is 5x5x2 = 50.
    assert abs(occ_sum["A"][1] - 50.0) < 1e-6
    assert abs(gmsh_sum["A"][1] - 50.0) < 1e-6
