"""Stress tests: stacked structured layers with different XY patterns per z-interval.

Multiple z-intervals are stacked vertically. Each z-interval is tiled with a
different pattern of structured polyprisms (some kept, some marked as
``mesh_bool=False`` voids). At each interface plane, the cut lines from the
layer above differ from those of the layer below, so the planner must produce
a face_partition that honours the union of all-layer cut lines for the mesh
to remain conformal.

Failure modes these tests aim to expose:
- Slabs in layer N whose top face lacks the cut lines induced by layer N+1's
  internal interfaces (orphan triangles at the interface).
- Slabs in layer N+1 whose bottom face lacks the cut lines induced by layer N's
  internal interfaces (mirror of the above).
- Pieces produced by the face partition that do not match across the
  interface, leading to non-conformal wedge stacks.
"""
from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import Polygon


def _box(x0: float, y0: float, x1: float, y1: float) -> Polygon:
    return Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


def _frame_box(zlo, zhi, name, x0=0.0, y0=0.0, x1=4.0, y1=4.0, n_layers=1):
    """Low-priority wrapping rectangle so the cohort footprint stays constant."""
    from meshwell.polyprism import PolyPrism
    from meshwell.structured import StructuredExtrusionResolutionSpec

    return PolyPrism(
        polygons=_box(x0, y0, x1, y1),
        buffers={float(zlo): 0.0, float(zhi): 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[n_layers])],
        physical_name=name,
        mesh_order=10.0,
    )


def _frame_disc(zlo, zhi, name, half_side=1.1, n_layers=1):
    """Low-priority wrapping square for disc-based cohorts."""
    import shapely

    from meshwell.polyprism import PolyPrism
    from meshwell.structured import StructuredExtrusionResolutionSpec

    return PolyPrism(
        polygons=shapely.box(-half_side, -half_side, half_side, half_side),
        buffers={float(zlo): 0.0, float(zhi): 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[n_layers])],
        physical_name=name,
        mesh_order=10.0,
    )


def _wedge_boundary_keys(mesh):
    """Return (wedge_tris, wedge_quads) frozenset sets from a meshio mesh."""
    wedge_blocks = [cb.data for cb in mesh.cells if cb.type == "wedge"]
    if not wedge_blocks:
        return set(), set()
    wedges = np.concatenate(wedge_blocks, axis=0)
    tris = {frozenset((int(w[0]), int(w[1]), int(w[2]))) for w in wedges}
    tris |= {frozenset((int(w[3]), int(w[4]), int(w[5]))) for w in wedges}
    quads: set[frozenset] = set()
    for w in wedges:
        for i, j in [(0, 1), (1, 2), (2, 0)]:
            quads.add(frozenset((int(w[i]), int(w[j]), int(w[j + 3]), int(w[i + 3]))))
    return tris, quads


def _count_orphan_triangles(mesh):
    """Boundary triangles that are not wedge bot/top tris or sub-tris of wedge laterals."""
    wedge_tris, wedge_quads = _wedge_boundary_keys(mesh)
    orphans = 0
    for cb in mesh.cells:
        if cb.type != "triangle":
            continue
        for t in cb.data:
            ts = frozenset((int(t[0]), int(t[1]), int(t[2])))
            if ts in wedge_tris:
                continue
            if any(ts.issubset(q) for q in wedge_quads):
                continue
            orphans += 1
    return orphans


def test_plan_three_stacked_layers_propagates_interface_cuts():
    """Plan-only: each slab's face_partition reflects the neighbour's cut lines.

    Layer 1 (z=[0,1]): two vertical strips meeting at x=2  -> seam parallel to y at x=2.
    Layer 2 (z=[1,2]): two horizontal strips meeting at y=2 -> seam parallel to x at y=2.
    Layer 3 (z=[2,3]): four quadrants meeting at (x=2, y=2).

    At z=1, layer 1's slabs each face a horizontal seam from layer 2 above;
    at z=2, layer 2's slabs each face a vertical seam from layer 3 above.
    For a clean mesh, each slab's face_partition must contain >=2 pieces in
    those layers where the opposite face is cut by a neighbour.
    """
    from meshwell.polyprism import PolyPrism
    from meshwell.structured import StructuredExtrusionResolutionSpec, build_plan

    layer1 = [
        PolyPrism(
            polygons=_box(0, 0, 2, 4),
            buffers={0.0: 0.0, 1.0: 0.0},
            structured=True,
            resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
            physical_name="L1L",
        ),
        PolyPrism(
            polygons=_box(2, 0, 4, 4),
            buffers={0.0: 0.0, 1.0: 0.0},
            structured=True,
            resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
            physical_name="L1R",
        ),
    ]
    layer2 = [
        PolyPrism(
            polygons=_box(0, 0, 4, 2),
            buffers={1.0: 0.0, 2.0: 0.0},
            structured=True,
            resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
            physical_name="L2B",
        ),
        PolyPrism(
            polygons=_box(0, 2, 4, 4),
            buffers={1.0: 0.0, 2.0: 0.0},
            structured=True,
            resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
            physical_name="L2T",
        ),
    ]
    layer3 = [
        PolyPrism(
            polygons=_box(0, 0, 2, 2),
            buffers={2.0: 0.0, 3.0: 0.0},
            structured=True,
            resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
            physical_name="L3BL",
        ),
        PolyPrism(
            polygons=_box(2, 0, 4, 2),
            buffers={2.0: 0.0, 3.0: 0.0},
            structured=True,
            resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
            physical_name="L3BR",
        ),
        PolyPrism(
            polygons=_box(0, 2, 2, 4),
            buffers={2.0: 0.0, 3.0: 0.0},
            structured=True,
            resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
            physical_name="L3TL",
        ),
        PolyPrism(
            polygons=_box(2, 2, 4, 4),
            buffers={2.0: 0.0, 3.0: 0.0},
            structured=True,
            resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
            physical_name="L3TR",
        ),
    ]

    plan = build_plan(layer1 + layer2 + layer3)
    # 2 + 2 + 4 = 8 slabs.
    assert len(plan.slabs) == 8, f"expected 8 slabs, got {len(plan.slabs)}"

    by_name = {}
    for slab in plan.slabs:
        names = [n for n in (slab.physical_name or ()) if isinstance(n, str)]
        key = names[0] if names else None
        by_name.setdefault(key, []).append(slab)

    # Each layer-1 strip should be partitioned by layer 2's horizontal seam at y=2.
    for name in ("L1L", "L1R"):
        slabs = by_name.get(name, [])
        assert len(slabs) == 1, f"{name}: expected 1 slab, got {len(slabs)}"
        assert len(slabs[0].face_partition) >= 2, (
            f"{name}: face_partition has {len(slabs[0].face_partition)} pieces; "
            "expected >=2 from the layer-2 horizontal seam at y=2"
        )

    # Each layer-2 strip should be partitioned by layer 1's vertical seam at x=2
    # AND by layer 3's seams. At minimum: >=2 pieces from layer 1 below, and
    # also pieces matching layer 3's quadrant cuts above.
    for name in ("L2B", "L2T"):
        slabs = by_name.get(name, [])
        assert len(slabs) == 1, f"{name}: expected 1 slab, got {len(slabs)}"
        # Both interfaces should contribute cuts: lower face gets a vertical
        # cut at x=2 (from L1L/L1R interface), upper face gets a vertical cut
        # at x=2 (from L3's quadrant interface). Net: at least 2 pieces.
        assert len(slabs[0].face_partition) >= 2, (
            f"{name}: face_partition has {len(slabs[0].face_partition)} pieces; "
            "expected >=2 from cross-layer cuts at z=1 and z=2"
        )


def test_three_stacked_layers_different_xy_tilings_mesh_clean_wedges(tmp_path):
    """Three stacked structured z-intervals with different XY tilings produce a clean wedge mesh.

    Stress: at each interface, the slabs above and below have different
    internal seams. A conformal mesh requires the union of those seams.
    """
    pytest.importorskip("meshio")
    pytest.importorskip("gmsh")
    import meshio
    from meshwell.orchestrator import generate_mesh
    from meshwell.polyprism import PolyPrism
    from meshwell.structured import StructuredExtrusionResolutionSpec

    def _strip(x0, y0, x1, y1, zlo, zhi, n, name):
        return PolyPrism(
            polygons=_box(x0, y0, x1, y1),
            buffers={zlo: 0.0, zhi: 0.0},
            structured=True,
            resolutions=[StructuredExtrusionResolutionSpec(n_layers=[n])],
            physical_name=name,
        )

    entities = [
        # z=[0,1]: vertical bisection
        _strip(0, 0, 2, 4, 0.0, 1.0, 2, "L1L"),
        _strip(2, 0, 4, 4, 0.0, 1.0, 2, "L1R"),
        # z=[1,2]: horizontal bisection
        _strip(0, 0, 4, 2, 1.0, 2.0, 2, "L2B"),
        _strip(0, 2, 4, 4, 1.0, 2.0, 2, "L2T"),
        # z=[2,3]: four quadrants
        _strip(0, 0, 2, 2, 2.0, 3.0, 2, "L3BL"),
        _strip(2, 0, 4, 2, 2.0, 3.0, 2, "L3BR"),
        _strip(0, 2, 2, 4, 2.0, 3.0, 2, "L3TL"),
        _strip(2, 2, 4, 4, 2.0, 3.0, 2, "L3TR"),
    ]

    out = tmp_path / "stacked_patterns.msh"
    generate_mesh(entities, dim=3, output_mesh=out, default_characteristic_length=1.0)

    m = meshio.read(out)
    cell_types = {cb.type for cb in m.cells}
    assert any(
        ct in cell_types for ct in ("wedge", "wedge6", "wedge15")
    ), f"expected wedge cells, got {cell_types}"

    for name in ("L1L", "L1R", "L2B", "L2T", "L3BL", "L3BR", "L3TL", "L3TR"):
        assert name in m.field_data, f"missing physical {name}"

    orphans = _count_orphan_triangles(m)
    assert orphans == 0, (
        f"{orphans} non-conformal boundary triangles across stacked-pattern "
        "interfaces; the mesh did not honour the union of all-layer cut lines"
    )


def test_three_stacked_layers_with_void_keep_patterns_mesh_clean(tmp_path):
    """Stacked structured layers where each z-range has a different keep / void pattern.

    "Void" regions are simply absent (no PolyPrism covers them). The set of
    voids varies per layer, so the union of kept footprints differs at each
    z-interface. Frame slabs fill the voids at each z-interval to satisfy the
    planner's cohort footprint constancy invariant; the inner slabs retain
    higher priority (mesh_order=0) and win where they exist.

    Pattern (4x4 footprint):
      Layer 1 (z=[0,1]): three 2x2 cells kept, top-right corner empty
        - L1_BL (0..2, 0..2), L1_BR (2..4, 0..2), L1_TL (0..2, 2..4); void at (2..4, 2..4)
      Layer 2 (z=[1,2]): three 2x2 cells kept, bottom-left corner empty
        - L2_BR (2..4, 0..2), L2_TL (0..2, 2..4), L2_TR (2..4, 2..4); void at (0..2, 0..2)
      Layer 3 (z=[2,3]): two diagonally opposite 2x2 cells kept, the other two empty
        - L3_BL (0..2, 0..2), L3_TR (2..4, 2..4); voids at (2..4, 0..2) and (0..2, 2..4)

    At z=1 the void shifts from TR (layer 1) to BL (layer 2): only the BR and
    TL columns share material on both sides, the others see a void on one side.
    At z=2 the kept set shrinks again to a diagonal, exercising different
    cross-layer interface patterns at both interfaces.
    """
    pytest.importorskip("meshio")
    pytest.importorskip("gmsh")
    import meshio
    from meshwell.orchestrator import generate_mesh
    from meshwell.polyprism import PolyPrism
    from meshwell.structured import StructuredExtrusionResolutionSpec

    def _cell(x0, y0, x1, y1, zlo, zhi, name):
        return PolyPrism(
            polygons=_box(x0, y0, x1, y1),
            buffers={zlo: 0.0, zhi: 0.0},
            structured=True,
            resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
            physical_name=name,
            mesh_order=1.0,
        )

    entities = [
        # Layer 1: TR void
        _cell(0, 0, 2, 2, 0.0, 1.0, "L1_BL"),
        _cell(2, 0, 4, 2, 0.0, 1.0, "L1_BR"),
        _cell(0, 2, 2, 4, 0.0, 1.0, "L1_TL"),
        # Layer 2: BL void
        _cell(2, 0, 4, 2, 1.0, 2.0, "L2_BR"),
        _cell(0, 2, 2, 4, 1.0, 2.0, "L2_TL"),
        _cell(2, 2, 4, 4, 1.0, 2.0, "L2_TR"),
        # Layer 3: diagonal
        _cell(0, 0, 2, 2, 2.0, 3.0, "L3_BL"),
        _cell(2, 2, 4, 4, 2.0, 3.0, "L3_TR"),
        # Frame slabs: fill each layer's void so union footprint = [0,4]x[0,4].
        _frame_box(0.0, 1.0, "Frame_z0_TR"),  # fills TR void
        _frame_box(1.0, 2.0, "Frame_z1_BL"),  # fills BL void
        _frame_box(2.0, 3.0, "Frame_z2"),  # fills both diagonal voids
    ]

    out = tmp_path / "stacked_voids.msh"
    generate_mesh(entities, dim=3, output_mesh=out, default_characteristic_length=1.0)

    m = meshio.read(out)
    cell_types = {cb.type for cb in m.cells}
    assert any(
        ct in cell_types for ct in ("wedge", "wedge6", "wedge15")
    ), f"expected wedge cells, got {cell_types}"
    for name in (
        "L1_BL",
        "L1_BR",
        "L1_TL",
        "L2_BR",
        "L2_TL",
        "L2_TR",
        "L3_BL",
        "L3_TR",
    ):
        assert name in m.field_data, f"missing physical {name}"

    orphans = _count_orphan_triangles(m)
    assert orphans == 0, (
        f"{orphans} non-conformal boundary triangles across void-pattern "
        "stacked interfaces; the mesh did not honour the union of all-layer "
        "cut lines around voids"
    )


def test_four_stacked_layers_misaligned_seams_mesh_clean(tmp_path):
    """Hardest case: 4 stacked z-intervals with misaligned XY seams per layer.

    Each layer's internal interface sits at a different x-coordinate, so no
    pair of adjacent layers shares a seam. For a conformal mesh, every
    slab's top and bottom faces must include the seam(s) of the neighbour(s)
    on that face -- the planner must compute the union of cut lines from
    above AND below for every interior z-plane.

    Seams (each layer is a 4x2 footprint sliced into two strips):
      Layer 1 (z=[0,1]): seam at x=1.0  -> A1 [0,1]x[0,2], B1 [1,4]x[0,2]
      Layer 2 (z=[1,2]): seam at x=1.7  -> A2 [0,1.7]x[0,2], B2 [1.7,4]x[0,2]
      Layer 3 (z=[2,3]): seam at x=2.5  -> A3 [0,2.5]x[0,2], B3 [2.5,4]x[0,2]
      Layer 4 (z=[3,4]): seam at x=3.2  -> A4 [0,3.2]x[0,2], B4 [3.2,4]x[0,2]

    At each interior z-plane (z=1, z=2, z=3) the face on one side has a
    different seam than the other, so each interface plane must end up with
    two cut lines (one from below, one from above) to remain conformal.
    """
    pytest.importorskip("meshio")
    pytest.importorskip("gmsh")
    import meshio
    from meshwell.orchestrator import generate_mesh
    from meshwell.polyprism import PolyPrism
    from meshwell.structured import StructuredExtrusionResolutionSpec

    def _cell(x0, y0, x1, y1, zlo, zhi, name):
        return PolyPrism(
            polygons=_box(x0, y0, x1, y1),
            buffers={zlo: 0.0, zhi: 0.0},
            structured=True,
            resolutions=[
                StructuredExtrusionResolutionSpec(
                    n_layers=[1], recombine_lateral_faces=True
                )
            ],
            physical_name=name,
        )

    seams = [1.0, 1.7, 2.5, 3.2]
    entities = []
    for i, sx in enumerate(seams):
        zlo, zhi = float(i), float(i + 1)
        entities.append(_cell(0.0, 0.0, sx, 2.0, zlo, zhi, f"L{i + 1}_A"))
        entities.append(_cell(sx, 0.0, 4.0, 2.0, zlo, zhi, f"L{i + 1}_B"))

    out = tmp_path / "stacked_misaligned.msh"
    generate_mesh(entities, dim=3, output_mesh=out, default_characteristic_length=0.5)

    m = meshio.read(out)
    cell_types = {cb.type for cb in m.cells}
    assert any(
        ct in cell_types for ct in ("wedge", "wedge6", "wedge15")
    ), f"expected wedge cells, got {cell_types}"

    for i in range(1, 5):
        for side in ("A", "B"):
            name = f"L{i}_{side}"
            assert name in m.field_data, f"missing physical {name}"

    orphans = _count_orphan_triangles(m)
    assert orphans == 0, (
        f"{orphans} non-conformal boundary triangles at misaligned-seam "
        "interfaces; the planner failed to compute the union of all-layer "
        "cut lines across the 4-layer stack"
    )


# --- Arc-bearing stress tests --------------------------------------------


def _disc(cx: float, cy: float, r: float, n: int = 32) -> Polygon:
    """N-vertex polygon inscribed in a circle of radius r centred at (cx, cy)."""
    import math

    return Polygon(
        [
            (
                cx + r * math.cos(2 * math.pi * i / n),
                cy + r * math.sin(2 * math.pi * i / n),
            )
            for i in range(n)
        ]
    )


def test_plan_stacked_concentric_discs_propagates_arc_provenance():
    """Plan-only: three concentric arc-bearing discs of decreasing radius.

    Layer 1 (z=[0,1]): disc R=1.0, identify_arcs=True.
    Layer 2 (z=[1,2]): disc R=0.7, identify_arcs=True (concentric, smaller).
    Layer 3 (z=[2,3]): disc R=0.5, identify_arcs=True (smaller still).

    At z=1: layer 1's top is cut by layer 2's circle at R=0.7 → 2 pieces.
    At z=2: layer 2's top is cut by layer 3's circle at R=0.5 → 2 pieces total
            (layer 2 also has the R=1 outer boundary inherited from layer 1
            below into its arc_index, but R=1 is the slab's *own* footprint
            arc — so layer 2's face_partition should have 2 pieces driven by
            the R=0.5 cut from above).

    Verifies that:
      - Layer 1's face_partition has an inherited R=0.7 PieceArcEdge.
      - Layer 2's face_partition has both an inherited R=1 arc (from below)
        and an inherited R=0.5 arc (from above) — or at least one inherited
        arc edge somewhere in provenance.
      - Layer 3 (terminal layer, no z-touching arc neighbour above) has at
        most its own footprint arc.
    """
    from meshwell.polyprism import PolyPrism
    from meshwell.structured import StructuredExtrusionResolutionSpec, build_plan

    l1 = PolyPrism(
        polygons=_disc(0, 0, 1.0),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        identify_arcs=True,
        min_arc_points=4,
        arc_tolerance=1e-3,
        physical_name="L1",
        mesh_order=1.0,
    )
    l2 = PolyPrism(
        polygons=_disc(0, 0, 0.7),
        buffers={1.0: 0.0, 2.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        identify_arcs=True,
        min_arc_points=4,
        arc_tolerance=1e-3,
        physical_name="L2",
        mesh_order=1.0,
    )
    l3 = PolyPrism(
        polygons=_disc(0, 0, 0.5),
        buffers={2.0: 0.0, 3.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        identify_arcs=True,
        min_arc_points=4,
        arc_tolerance=1e-3,
        physical_name="L3",
        mesh_order=1.0,
    )

    plan = build_plan(
        [
            l1,
            l2,
            l3,
            _frame_disc(0.0, 1.0, "Frame_z0"),
            _frame_disc(1.0, 2.0, "Frame_z1"),
            _frame_disc(2.0, 3.0, "Frame_z2"),
        ]
    )
    by_name = {s.physical_name[0]: s for s in plan.slabs}

    # Layer 1 must be split by layer 2's smaller circle on its top face.
    assert len(by_name["L1"].face_partition) >= 2, (
        f"L1: expected >=2 pieces from R=0.7 top cut; "
        f"got {len(by_name['L1'].face_partition)}"
    )
    # Layer 2 must be split by layer 3's smaller circle on its top face.
    assert len(by_name["L2"].face_partition) >= 2, (
        f"L2: expected >=2 pieces from R=0.5 top cut; "
        f"got {len(by_name['L2'].face_partition)}"
    )

    import math as _math

    def _has_boundary_pts_at_radius(slab, r_target, tol=0.05):
        """Return True if any face_partition piece has boundary points at r_target."""
        for piece in slab.face_partition:
            for x, y in piece.exterior.coords:
                if abs(_math.hypot(x, y) - r_target) < tol:
                    return True
        return False

    # Layer 1's partition must include boundary points at R≈0.7 (from L2's circle).
    assert _has_boundary_pts_at_radius(by_name["L1"], 0.7), (
        "L1: no face_partition boundary points at R≈0.7; "
        "arc inheritance from L2 above did not propagate geometrically"
    )
    # Layer 2's partition must include boundary points at R≈0.5 (from L3's circle).
    assert _has_boundary_pts_at_radius(by_name["L2"], 0.5), (
        "L2: no face_partition boundary points at R≈0.5; "
        "arc inheritance from L3 above did not propagate geometrically"
    )


@pytest.mark.xfail(
    raises=Exception,
    reason="Annular face_partition pieces from stacked concentric discs break "
    "gmsh transfinite meshing ('1D mesh not forming a closed loop'). "
    "See MEMORY.md: project_structured_annular_arc_transfinite.md",
)
def test_stacked_concentric_arc_discs_mesh_clean(tmp_path):
    """End-to-end mesh: three concentric arc-bearing discs produce a clean wedge mesh.

    Same scene as the plan-only test. Wrapping frame slabs satisfy the cohort
    footprint constancy invariant while the inner discs retain high priority.
    Verifies:
      - Wedge cells are produced (no tets fall back due to mid-height cuts).
      - Zero orphan boundary triangles (all boundary tris are either wedge
        caps or sub-triangles of wedge lateral quads).
      - All three physicals (L1, L2, L3) appear in the mesh.
    """
    pytest.importorskip("meshio")
    pytest.importorskip("gmsh")
    import meshio
    from meshwell.orchestrator import generate_mesh
    from meshwell.polyprism import PolyPrism
    from meshwell.structured import StructuredExtrusionResolutionSpec

    def _arc_slab(r, zlo, zhi, name):
        return PolyPrism(
            polygons=_disc(0, 0, r),
            buffers={zlo: 0.0, zhi: 0.0},
            structured=True,
            resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
            identify_arcs=True,
            min_arc_points=4,
            arc_tolerance=1e-3,
            physical_name=name,
            mesh_order=1.0,
        )

    entities = [
        _arc_slab(1.0, 0.0, 1.0, "L1"),
        _arc_slab(0.7, 1.0, 2.0, "L2"),
        _arc_slab(0.5, 2.0, 3.0, "L3"),
        _frame_disc(0.0, 1.0, "Frame_z0"),
        _frame_disc(1.0, 2.0, "Frame_z1"),
        _frame_disc(2.0, 3.0, "Frame_z2"),
    ]

    out = tmp_path / "stacked_arc_discs.msh"
    generate_mesh(entities, dim=3, output_mesh=out, default_characteristic_length=0.5)

    m = meshio.read(out)
    cell_types = {cb.type for cb in m.cells}
    assert any(
        ct in cell_types for ct in ("wedge", "wedge6", "wedge15")
    ), f"expected wedge cells, got {cell_types}"
    for name in ("L1", "L2", "L3"):
        assert name in m.field_data, f"missing physical {name}"

    orphans = _count_orphan_triangles(m)
    assert (
        orphans == 0
    ), f"{orphans} non-conformal boundary triangles in concentric-disc stack"


def _ring_segment(
    cx: float,
    cy: float,
    r_inner: float,
    r_outer: float,
    theta_start: float,
    theta_end: float,
    n_circle: int = 24,
) -> Polygon:
    """Thick-ring segment polygon with vertices on a global angular grid.

    Boundary (single loop): outer arc CCW from ``theta_start`` to ``theta_end``
    along radius ``r_outer``, radial line at ``theta_end`` down to ``r_inner``,
    inner arc CW back from ``theta_end`` to ``theta_start`` along radius
    ``r_inner``, radial line at ``theta_start`` back up to ``r_outer``.

    Topologically a disk — no interior holes.

    Vertices on each arc fall on a **global angular grid** (multiples of
    ``2*pi/n_circle``). When two ring segments share an angular range
    (e.g., L1 spans 0..pi and L2 spans pi/2..3*pi/2 share pi/2..pi), their
    polygon vertices on the shared portion coincide exactly — no polyline-
    approximation mismatch, no spurious sliver pieces in polygonize output.
    The caller should pick ``theta_start`` / ``theta_end`` that themselves
    fall on the grid (i.e., multiples of ``2*pi/n_circle``).
    """
    import math

    step = 2 * math.pi / n_circle
    eps = 1e-9
    interior_angles = [
        k * step
        for k in range(math.ceil(theta_start / step), math.floor(theta_end / step) + 1)
        if theta_start + eps < k * step < theta_end - eps
    ]
    angles = [theta_start, *interior_angles, theta_end]

    outer = [(cx + r_outer * math.cos(a), cy + r_outer * math.sin(a)) for a in angles]
    inner = [
        (cx + r_inner * math.cos(a), cy + r_inner * math.sin(a))
        for a in reversed(angles)
    ]
    return Polygon(outer + inner)


def test_plan_stacked_overlapping_ring_segments_propagates_radial_cuts():
    """Plan-only: three stacked half-rings rotated by 90 degrees per layer.

    Each layer is a half-annulus (180-degree thick ring segment), R_inner=0.5,
    R_outer=1.0, rotated 90 degrees per layer so consecutive layers overlap
    in a quarter-ring:

      L1 (z=[0,1]): theta in [0, pi]    (upper half-annulus)
      L2 (z=[1,2]): theta in [pi/2, 3*pi/2]  (left half-annulus)
      L3 (z=[2,3]): theta in [pi, 2*pi]  (lower half-annulus)

    Overlaps:
      L1 cap L2 = quarter-ring theta in [pi/2, pi]
      L2 cap L3 = quarter-ring theta in [pi, 3*pi/2]
      L1 cap L3 = empty (touch at theta=pi only, zero area)

    At z=1: L1's top face is cut by L2's radial-at-pi/2 boundary (lies inside
    L1's footprint). Similarly L2's bottom face is cut by L1's radial-at-pi.
    Each slab's face_partition should reach >= 2 pieces from radial cuts that
    cross its interior. All sub-pieces are single-loop ring-segment polygons —
    transfinite-compatible without the annular-split spec.
    """
    import math

    from meshwell.polyprism import PolyPrism
    from meshwell.structured import StructuredExtrusionResolutionSpec, build_plan

    l1 = PolyPrism(
        polygons=_ring_segment(0, 0, 0.5, 1.0, 0.0, math.pi),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        identify_arcs=True,
        min_arc_points=4,
        arc_tolerance=1e-3,
        physical_name="L1",
        mesh_order=1.0,
    )
    l2 = PolyPrism(
        polygons=_ring_segment(0, 0, 0.5, 1.0, math.pi / 2, 3 * math.pi / 2),
        buffers={1.0: 0.0, 2.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        identify_arcs=True,
        min_arc_points=4,
        arc_tolerance=1e-3,
        physical_name="L2",
        mesh_order=1.0,
    )
    l3 = PolyPrism(
        polygons=_ring_segment(0, 0, 0.5, 1.0, math.pi, 2 * math.pi),
        buffers={2.0: 0.0, 3.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        identify_arcs=True,
        min_arc_points=4,
        arc_tolerance=1e-3,
        physical_name="L3",
        mesh_order=1.0,
    )

    plan = build_plan(
        [
            l1,
            l2,
            l3,
            _frame_disc(0.0, 1.0, "Frame_z0"),
            _frame_disc(1.0, 2.0, "Frame_z1"),
            _frame_disc(2.0, 3.0, "Frame_z2"),
        ]
    )
    by_name = {s.physical_name[0]: s for s in plan.slabs}

    # L1 must be split by L2's radial-at-pi/2 cut (inside L1's 0..pi range).
    assert len(by_name["L1"].face_partition) >= 2, (
        f"L1: expected >=2 pieces from L2 radial-at-pi/2 top cut; "
        f"got {len(by_name['L1'].face_partition)}"
    )
    # L2 sees cuts from both sides: L1's radial-at-pi from below, L3's
    # radial-at-pi from above (same line! so net 1 internal cut). Plus L2's
    # own footprint spans pi/2..3pi/2, so the cuts at pi do split it.
    assert len(by_name["L2"].face_partition) >= 2, (
        f"L2: expected >=2 pieces from interface cuts; "
        f"got {len(by_name['L2'].face_partition)}"
    )
    # L3 must be split by L2's radial-at-3pi/2 cut (inside L3's pi..2pi range).
    assert len(by_name["L3"].face_partition) >= 2, (
        f"L3: expected >=2 pieces from L2 radial-at-3pi/2 bottom cut; "
        f"got {len(by_name['L3'].face_partition)}"
    )

    # No piece should have interior rings — ring-segment topology is a disk,
    # and rotated overlaps produce only single-loop sub-pieces.
    for name, slab in by_name.items():
        for i, piece in enumerate(slab.face_partition):
            assert len(piece.interiors) == 0, (
                f"{name}.face_partition[{i}] has {len(piece.interiors)} interior "
                "ring(s); expected single-loop (no holes) for ring-segment overlap"
            )


def test_stacked_overlapping_ring_segments_mesh_clean(tmp_path):
    """End-to-end mesh: three rotated half-rings produce a clean wedge mesh.

    Same scene as the plan-only test. Wrapping frame slabs satisfy the cohort
    footprint constancy invariant. Verifies:
      - Wedges are produced (transfinite succeeds on every ring-segment piece).
      - Zero orphan boundary triangles.
      - All three physicals (L1, L2, L3) appear in the mesh.
    """
    pytest.importorskip("meshio")
    pytest.importorskip("gmsh")
    import math

    import meshio
    from meshwell.orchestrator import generate_mesh
    from meshwell.polyprism import PolyPrism
    from meshwell.structured import StructuredExtrusionResolutionSpec

    def _ring_slab(theta_start, theta_end, zlo, zhi, name):
        return PolyPrism(
            polygons=_ring_segment(0, 0, 0.5, 1.0, theta_start, theta_end),
            buffers={zlo: 0.0, zhi: 0.0},
            structured=True,
            resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
            identify_arcs=True,
            min_arc_points=4,
            arc_tolerance=1e-3,
            physical_name=name,
            mesh_order=1.0,
        )

    entities = [
        _ring_slab(0.0, math.pi, 0.0, 1.0, "L1"),
        _ring_slab(math.pi / 2, 3 * math.pi / 2, 1.0, 2.0, "L2"),
        _ring_slab(math.pi, 2 * math.pi, 2.0, 3.0, "L3"),
        _frame_disc(0.0, 1.0, "Frame_z0"),
        _frame_disc(1.0, 2.0, "Frame_z1"),
        _frame_disc(2.0, 3.0, "Frame_z2"),
    ]

    out = tmp_path / "stacked_ring_segments.msh"
    generate_mesh(entities, dim=3, output_mesh=out, default_characteristic_length=0.3)

    m = meshio.read(out)
    cell_types = {cb.type for cb in m.cells}
    assert any(
        ct in cell_types for ct in ("wedge", "wedge6", "wedge15")
    ), f"expected wedge cells, got {cell_types}"
    for name in ("L1", "L2", "L3"):
        assert name in m.field_data, f"missing physical {name}"

    orphans = _count_orphan_triangles(m)
    assert (
        orphans == 0
    ), f"{orphans} non-conformal boundary triangles in ring-segment stack"


def test_stacked_overlapping_ring_segments_with_lower_priority_planes_mesh_clean(
    tmp_path,
):
    """Reinforced: three rotated half-rings + three same-z-extent lower-priority planes.

    Adds, at each ring's z-extent, a large square ``P_i`` PolyPrism with a
    higher ``mesh_order`` (lower priority). Policy B carves each plane by
    the ring at that z-level, so the plane gets ``resolved_footprint`` =
    big_square minus ring_footprint and the ring keeps its full half-
    annulus footprint. The plane has the ring's footprint as an interior
    hole, exercising the annular-face split path AND the ring↔plane
    lateral interface (curved + straight portions).

    Verifies the full source-of-truth pipeline end-to-end:
      - Wedges produced (transfinite succeeds on every piece including
        the annular-split sub-pieces).
      - All six physicals (L1, L2, L3, P1, P2, P3) appear.
      - Zero duplicate mesh nodes (shared interfaces collapse to one node).
      - Zero orphan boundary triangles.
      - Every L_i↔P_i pair has an interface physical group (e.g.
        "L1___P1"), confirming the same-z ring↔plane lateral interface
        is shared at the OCC/mesh level.
      - Every triangle face is referenced by 1 (boundary) or 2 (interior)
        cells; no pathological 3+ sharings.
    """
    pytest.importorskip("meshio")
    pytest.importorskip("gmsh")
    import math

    import numpy as np

    import meshio
    from meshwell.orchestrator import generate_mesh
    from meshwell.polyprism import PolyPrism
    from meshwell.structured import StructuredExtrusionResolutionSpec

    def _ring_slab(theta_start, theta_end, zlo, zhi, name):
        return PolyPrism(
            polygons=_ring_segment(0, 0, 0.5, 1.0, theta_start, theta_end),
            buffers={zlo: 0.0, zhi: 0.0},
            structured=True,
            resolutions=[
                StructuredExtrusionResolutionSpec(
                    n_layers=[3], recombine_lateral_faces=True
                )
            ],
            identify_arcs=True,
            min_arc_points=4,
            arc_tolerance=1e-3,
            physical_name=name,
            mesh_order=1.0,
        )

    def _plane_slab(zlo, zhi, name):
        return PolyPrism(
            polygons=Polygon([(-1.5, -1.5), (1.5, -1.5), (1.5, 1.5), (-1.5, 1.5)]),
            buffers={zlo: 0.0, zhi: 0.0},
            structured=True,
            resolutions=[
                StructuredExtrusionResolutionSpec(
                    n_layers=[3], recombine_lateral_faces=True
                )
            ],
            physical_name=name,
            mesh_order=2.0,
        )

    entities = [
        _ring_slab(0.0, math.pi, 0.0, 1.0, "L1"),
        _ring_slab(math.pi / 2, 3 * math.pi / 2, 1.0, 2.0, "L2"),
        _ring_slab(math.pi, 2 * math.pi, 2.0, 3.0, "L3"),
        _plane_slab(0.0, 1.0, "P1"),
        _plane_slab(1.0, 2.0, "P2"),
        _plane_slab(2.0, 3.0, "P3"),
    ]

    out = tmp_path / "stacked_rings_with_planes.msh"
    generate_mesh(entities, dim=3, output_mesh=out, default_characteristic_length=0.3)

    m = meshio.read(out)

    cell_types = {cb.type for cb in m.cells}
    assert any(
        ct in cell_types for ct in ("wedge", "wedge6", "wedge15")
    ), f"expected wedge cells, got {cell_types}"

    for name in ("L1", "L2", "L3", "P1", "P2", "P3"):
        assert name in m.field_data, f"missing physical {name}"

    # Same-z ring↔plane interfaces must appear as interface physical groups.
    interface_names = set(m.field_data.keys())
    for ring, plane in [("L1", "P1"), ("L2", "P2"), ("L3", "P3")]:
        a = f"{ring}___{plane}"
        b = f"{plane}___{ring}"
        assert a in interface_names or b in interface_names, (
            f"missing same-z interface physical group {a!r} or {b!r}; "
            f"available groups: {sorted(interface_names)}"
        )

    # Zero duplicate mesh nodes (snap tolerance covers shared boundaries).
    rounded = np.round(m.points, 9)
    n_unique = len(np.unique(rounded, axis=0))
    n_dup = len(m.points) - n_unique
    assert n_dup == 0, f"{n_dup} duplicate mesh nodes"

    # Manifold check: every triangle face is referenced by 1 or 2 cells
    # (no 3+ sharings). Wedge bot/top triangles + tet faces all contribute.
    wedges = (
        np.concatenate([cb.data for cb in m.cells if cb.type == "wedge"], axis=0)
        if any(cb.type == "wedge" for cb in m.cells)
        else np.empty((0, 6), dtype=int)
    )
    tets = (
        np.concatenate([cb.data for cb in m.cells if cb.type == "tetra"], axis=0)
        if any(cb.type == "tetra" for cb in m.cells)
        else np.empty((0, 4), dtype=int)
    )
    face_count: dict = {}
    for w in wedges:
        for tri in [(w[0], w[1], w[2]), (w[3], w[4], w[5])]:
            k = frozenset(int(v) for v in tri)
            face_count[k] = face_count.get(k, 0) + 1
    for t in tets:
        for i, j, k in [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]:
            ff = frozenset(int(v) for v in (t[i], t[j], t[k]))
            face_count[ff] = face_count.get(ff, 0) + 1
    pathological = sum(1 for c in face_count.values() if c > 2)
    assert (
        pathological == 0
    ), f"{pathological} triangle faces shared by 3+ cells (non-manifold)"

    orphans = _count_orphan_triangles(m)
    assert (
        orphans == 0
    ), f"{orphans} non-conformal boundary triangles in ring+plane stack"
