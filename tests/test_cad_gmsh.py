"""Tests for the gmsh-native CAD backend.

:mod:`meshwell.cad_gmsh` is the pivot away from the OCP + XAO pipeline:
we drive gmsh's built-in OCC kernel via its Python API, fragment with
``gmsh.model.occ.fragment``, and assign pieces to entities by the same
``mesh_order`` ladder as :mod:`meshwell.cad_occ`. These tests pin the
shared ownership semantics, the interface / domain-boundary tagging,
and end-to-end meshing for a handful of representative scenes.
"""
from __future__ import annotations

import shapely

import gmsh
from meshwell.cad_gmsh import cad_gmsh
from meshwell.mesh import mesh
from meshwell.polyprism import PolyPrism
from meshwell.polysurface import PolySurface


def _physical_names() -> list[tuple[int, str]]:
    return [
        (d, gmsh.model.getPhysicalName(d, t)) for d, t in gmsh.model.getPhysicalGroups()
    ]


def test_cad_gmsh_2d_inner_outer_tags_interface_and_boundary():
    """Outer-around-inner polysurface scene produces ``___`` interface + ``___None`` domain boundary."""
    outer_sq = shapely.Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    inner_sq = shapely.Polygon([(3, 3), (7, 3), (7, 7), (3, 7)])
    entities = [
        PolySurface(polygons=outer_sq, physical_name="outer", mesh_order=2),
        PolySurface(polygons=inner_sq, physical_name="inner", mesh_order=1),
    ]
    try:
        _labeled, mm = cad_gmsh(entities)
        names = {n for _, n in _physical_names()}
        assert "outer" in names
        assert "inner" in names
        # interface is sorted alphabetically by physical name
        assert "inner___outer" in names
        assert "outer___None" in names
        # inner is a hole in the domain -- no exterior boundary of its own
        assert "inner___None" not in names

        m = mesh(dim=2, model=mm, default_characteristic_length=1.0, verbosity=0)
        assert any(c.type == "triangle" and len(c.data) for c in m.cells)
    finally:
        mm.finalize()


def test_cad_gmsh_3d_adjacent_prisms_share_interface():
    """Two prisms stitched along x=5 must produce an ``A___B`` interface."""
    A = shapely.Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
    B = shapely.Polygon([(5, 0), (10, 0), (10, 5), (5, 5)])
    buffers = {0.0: 0.0, 2.0: 0.0}
    try:
        _, mm = cad_gmsh(
            [
                PolyPrism(polygons=A, buffers=buffers, physical_name="A", mesh_order=1),
                PolyPrism(polygons=B, buffers=buffers, physical_name="B", mesh_order=2),
            ]
        )
        names = {n for _, n in _physical_names()}
        assert names == {"A", "B", "A___B", "A___None", "B___None"}
        m = mesh(dim=3, model=mm, default_characteristic_length=2.0, verbosity=0)
        assert any(c.type == "tetra" and len(c.data) for c in m.cells)
    finally:
        mm.finalize()


def test_cad_gmsh_mesh_order_lower_wins_in_overlap():
    """Overlapping scene: lower ``mesh_order`` claims the overlap piece.

    Two unit-square polysurfaces overlap on a 1x0.5 strip. Winner owns the
    overlap and so has larger area; loser only owns the non-overlapping
    half.
    """
    A = shapely.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    B = shapely.Polygon([(0.5, 0), (1.5, 0), (1.5, 1), (0.5, 1)])

    try:
        labeled, mm = cad_gmsh(
            [
                PolySurface(polygons=A, physical_name="A", mesh_order=1),
                PolySurface(polygons=B, physical_name="B", mesh_order=2),
            ]
        )
        # Compute per-entity area via gmsh mass properties.
        areas: dict[str, float] = {}
        for ent in labeled:
            total = 0.0
            for dim, tag in ent.dimtags:
                total += gmsh.model.occ.getMass(dim, tag)
            areas[ent.physical_name[0]] = total
        assert areas["A"] > areas["B"], areas
        # A (winner) is the full unit square; B is the non-overlap half.
        assert abs(areas["A"] - 1.0) < 1e-6, areas
        assert abs(areas["B"] - 0.5) < 1e-6, areas
    finally:
        mm.finalize()


def test_cad_gmsh_same_mesh_order_ties_to_insertion_order():
    """At identical ``mesh_order``, the first-inserted entity owns ties."""
    A = shapely.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    B = shapely.Polygon([(0.5, 0), (1.5, 0), (1.5, 1), (0.5, 1)])

    try:
        labeled, mm = cad_gmsh(
            [
                PolySurface(polygons=A, physical_name="A", mesh_order=1),
                PolySurface(polygons=B, physical_name="B", mesh_order=1),
            ]
        )
        areas = {
            ent.physical_name[0]: sum(
                gmsh.model.occ.getMass(d, t) for d, t in ent.dimtags
            )
            for ent in labeled
        }
        assert areas["A"] > areas["B"], areas
    finally:
        mm.finalize()


def test_cad_gmsh_keep_false_top_dim_removed_but_interface_named():
    """A ``keep=False`` neighbour must not be meshed, but its shared face with a kept neighbour must still be tagged as ``kept___helper``."""
    A = shapely.Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
    B = shapely.Polygon([(5, 0), (10, 0), (10, 5), (5, 5)])
    buffers = {0.0: 0.0, 2.0: 0.0}
    try:
        labeled, mm = cad_gmsh(
            [
                PolyPrism(polygons=A, buffers=buffers, physical_name="A", mesh_order=1),
                PolyPrism(
                    polygons=B,
                    buffers=buffers,
                    physical_name="helper",
                    mesh_order=2,
                    mesh_bool=False,
                ),
            ]
        )
        names = {n for _, n in _physical_names()}
        assert "A" in names
        assert "helper" not in names
        # helper's boundary with A must still carry the interface name.
        assert "A___helper" in names
        # helper has been removed -- no volume for it.
        helper = next(e for e in labeled if e.physical_name == ("helper",))
        assert helper.dimtags == []

        m = mesh(dim=3, model=mm, default_characteristic_length=2.0, verbosity=0)
        # Every tet must belong to A (the only kept volume).
        tets_by_group: dict[str, int] = {}
        for block_idx, arr in enumerate(m.cell_sets.get("A", [])):
            if arr is None:
                continue
            if m.cells[block_idx].type == "tetra":
                tets_by_group["A"] = tets_by_group.get("A", 0) + len(arr)
        assert tets_by_group.get("A", 0) > 0
        total_tets = sum(len(c.data) for c in m.cells if c.type == "tetra")
        assert tets_by_group["A"] == total_tets
    finally:
        mm.finalize()


def test_cad_gmsh_keep_false_lower_dim_cut_is_tagged():
    """An embedded ``keep=False`` lower-dim cut surface keeps its own physical group (it's the useful artefact of the helper)."""
    outer_poly = shapely.Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
    cut_poly = shapely.Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
    try:
        _, mm = cad_gmsh(
            [
                PolyPrism(
                    polygons=outer_poly,
                    buffers={0.0: 0.0, 2.0: 0.0},
                    physical_name="body",
                    mesh_order=2,
                ),
                PolySurface(
                    polygons=cut_poly,
                    physical_name="cut",
                    mesh_order=1,
                    translation=(0.0, 0.0, 1.0),
                    mesh_bool=False,
                ),
            ]
        )
        names = {n for _, n in _physical_names()}
        assert "body" in names
        # The embedded cut keeps its own physical group even though
        # mesh_bool=False -- it's the interior face the user wants to
        # name and to mesh against.
        assert "cut" in names
    finally:
        mm.finalize()


def test_cad_gmsh_single_entity_no_fragment():
    """A single-entity scene short-circuits fragmentation."""
    poly = shapely.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    try:
        labeled, mm = cad_gmsh(
            [PolySurface(polygons=poly, physical_name="only", mesh_order=1)]
        )
        assert len(labeled) == 1
        assert labeled[0].dimtags
        names = {n for _, n in _physical_names()}
        assert "only" in names
        # No neighbour -> no interface groups, only the domain-boundary.
        assert "only___None" in names
    finally:
        mm.finalize()
