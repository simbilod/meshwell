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
    z-interface. The mesh must remain conformal across those interfaces.

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


@pytest.mark.xfail(
    reason=(
        "Misaligned per-layer seams break the structured planner: OCC "
        "fragmentation imprints both above- and below-neighbour seams onto "
        "interface faces, producing 5-corner faces that the structured "
        "pipeline still tries to mesh as transfinite (gmsh: 'Surface N is "
        "transfinite but has 5 corners'). The current face_partition logic "
        "considers each interface face independently; a clean fix requires "
        "computing the UNION of cut lines across ALL stacked layers and "
        "splitting each slab into pieces accordingly."
    ),
    strict=True,
    raises=Exception,
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
            resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
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
