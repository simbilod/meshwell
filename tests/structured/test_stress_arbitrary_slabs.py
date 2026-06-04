"""Stress tests exposing the multi-slab decomposition bug in decompose_cohorts.

The bug: when multiple structured slabs overlap in XY at the same z-interval,
zinterval_footprint unions them, and cut_sources only receives the union
boundary (never per-slab outlines).  polygonize then emits one sub-piece per
union region, and build.py picks source_slab_indices[0] (lowest mesh_order) for
all of them.  Higher-mesh_order slabs silently disappear from the output.

These tests are EXPECTED TO FAIL on the current code.  Do not fix the bug here.
"""
from __future__ import annotations

from shapely.geometry import Polygon

import meshio
from meshwell.orchestrator import generate_mesh
from meshwell.polyprism import PolyPrism
from meshwell.resolution import StructuredExtrusionResolutionSpec

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def square(x: float, y: float, w: float, h: float) -> Polygon:
    """Axis-aligned rectangle with lower-left corner (x, y)."""
    return Polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])


def hexagon(cx: float, cy: float, r: float) -> Polygon:
    """Regular hexagon centred at (cx, cy) with circumradius r."""
    import math

    pts = [
        (cx + r * math.cos(math.pi / 3 * i), cy + r * math.sin(math.pi / 3 * i))
        for i in range(6)
    ]
    return Polygon(pts)


# ---------------------------------------------------------------------------
# Mesh inspection helpers
# ---------------------------------------------------------------------------


def _wedge_count_for(m: meshio.Mesh, group_name: str) -> int:
    """Sum wedge cells assigned to the given physical group."""
    sets = m.cell_sets.get(group_name)
    if sets is None:
        return 0
    total = 0
    for block, indices in zip(m.cells, sets):
        if block.type == "wedge" and indices is not None:
            total += len(indices)
    return total


def _resolution_specs_for(*names: str, n_layers: int = 2) -> dict:
    return {
        name: [StructuredExtrusionResolutionSpec(n_layers=n_layers)] for name in names
    }


# ---------------------------------------------------------------------------
# Test 1 -- two overlapping slabs, both meshed (the user's original scene)
# ---------------------------------------------------------------------------


def test_two_overlapping_slabs_both_meshed(tmp_path):
    """a1 (1x1) inside a2 (2x2) at z=[0,1]; b (2x2) at z=[1,2].

    Bug: a2 disappears from cell_sets, only a1 wedges appear (all 4 units^2 worth).
    Expected: a1 area=1, a2 area=3, b area=4, all with non-zero wedge counts.
    """
    a1 = PolyPrism(
        square(0, 0, 1, 1),
        {0.0: 0.0, 1.0: 0.0},
        physical_name="a1",
        structured=True,
        mesh_order=1.0,
    )
    a2 = PolyPrism(
        square(0, 0, 2, 2),
        {0.0: 0.0, 1.0: 0.0},
        physical_name="a2",
        structured=True,
        mesh_order=2.0,
    )
    b = PolyPrism(
        square(0, 0, 2, 2),
        {1.0: 0.0, 2.0: 0.0},
        physical_name="b",
        structured=True,
        mesh_order=1.0,
    )
    base = PolyPrism(
        square(0, 0, 2, 2),
        {-1.0: 0.0, 0.0: 0.0},
        physical_name="base",
    )
    top = PolyPrism(
        square(0, 0, 2, 2),
        {2.0: 0.0, 3.0: 0.0},
        physical_name="top",
    )

    generate_mesh(
        [a1, a2, b, base, top],
        dim=3,
        output_mesh=tmp_path / "out.msh",
        default_characteristic_length=0.5,
        resolution_specs=_resolution_specs_for("a1", "a2", "b", n_layers=2),
    )
    m = meshio.read(tmp_path / "out.msh")

    # --- presence checks ---
    assert "a1" in m.cell_sets, "a1 missing from physical groups"
    assert (
        "a2" in m.cell_sets
    ), "a2 missing from physical groups (BUG: lost by decompose)"
    assert "b" in m.cell_sets, "b missing from physical groups"

    # --- wedge-count checks ---
    wedge_a1 = _wedge_count_for(m, "a1")
    wedge_a2 = _wedge_count_for(m, "a2")
    wedge_b = _wedge_count_for(m, "b")

    assert wedge_a1 > 0, f"a1 has 0 wedges (got {wedge_a1})"
    assert wedge_a2 > 0, f"a2 has 0 wedges (got {wedge_a2})"
    assert wedge_b > 0, f"b has 0 wedges (got {wedge_b})"

    # --- area proportionality check (+-10%) ---
    # a1 survives as 1x1=1.0; a2 survives as 2x2-1x1=3.0; b=2x2=4.0
    # With n_layers=2 and triangular wedge count proportional to area, ratios should hold.
    total_wedges = wedge_a1 + wedge_a2 + wedge_b
    frac_a1 = wedge_a1 / total_wedges
    frac_a2 = wedge_a2 / total_wedges
    frac_b = wedge_b / total_wedges

    expected_total_area = 1.0 + 3.0 + 4.0  # 8.0
    assert (
        abs(frac_a1 - 1.0 / expected_total_area) < 0.10
    ), f"a1 wedge fraction {frac_a1:.3f} far from expected {1.0/expected_total_area:.3f}"
    assert (
        abs(frac_a2 - 3.0 / expected_total_area) < 0.10
    ), f"a2 wedge fraction {frac_a2:.3f} far from expected {3.0/expected_total_area:.3f}"
    assert (
        abs(frac_b - 4.0 / expected_total_area) < 0.10
    ), f"b wedge fraction {frac_b:.3f} far from expected {4.0/expected_total_area:.3f}"


# ---------------------------------------------------------------------------
# Test 2 -- three nested slabs, mesh_order decides carving
# ---------------------------------------------------------------------------


def test_three_nested_slabs_mesh_order_decides(tmp_path):
    """Three concentric squares at z=[0,1] + a cap at z=[1,2].

    outer: 4x4, mesh_order=3  -> gets 4x4 - 2x2 = 12
    mid:   2x2, mesh_order=2  -> gets 2x2 - 1x1 = 3
    inner: 1x1, mesh_order=1  -> gets 1x1 = 1
    cap:   4x4 at z=[1,2]     -> gets 4x4 = 16

    Bug: only the lowest-mesh_order slab (inner) shows up; outer+mid disappear.
    """
    outer = PolyPrism(
        square(0, 0, 4, 4),
        {0.0: 0.0, 1.0: 0.0},
        physical_name="outer",
        structured=True,
        mesh_order=3.0,
    )
    mid = PolyPrism(
        square(1, 1, 2, 2),
        {0.0: 0.0, 1.0: 0.0},
        physical_name="mid",
        structured=True,
        mesh_order=2.0,
    )
    inner = PolyPrism(
        square(1.5, 1.5, 1, 1),
        {0.0: 0.0, 1.0: 0.0},
        physical_name="inner",
        structured=True,
        mesh_order=1.0,
    )
    cap = PolyPrism(
        square(0, 0, 4, 4),
        {1.0: 0.0, 2.0: 0.0},
        physical_name="cap",
        structured=True,
        mesh_order=1.0,
    )
    base = PolyPrism(
        square(0, 0, 4, 4),
        {-1.0: 0.0, 0.0: 0.0},
        physical_name="base",
    )
    top = PolyPrism(
        square(0, 0, 4, 4),
        {2.0: 0.0, 3.0: 0.0},
        physical_name="top",
    )

    generate_mesh(
        [outer, mid, inner, cap, base, top],
        dim=3,
        output_mesh=tmp_path / "out.msh",
        default_characteristic_length=0.5,
        resolution_specs=_resolution_specs_for(
            "outer", "mid", "inner", "cap", n_layers=2
        ),
    )
    m = meshio.read(tmp_path / "out.msh")

    # --- presence checks ---
    for name in ("outer", "mid", "inner", "cap"):
        assert name in m.cell_sets, f"{name} missing from physical groups"

    # --- wedge counts ---
    w = {name: _wedge_count_for(m, name) for name in ("outer", "mid", "inner", "cap")}
    for name, wc in w.items():
        assert wc > 0, f"{name} has 0 wedges"

    # --- area proportionality (+-10%) ---
    area_outer = 4 * 4 - 2 * 2  # 12
    area_mid = 2 * 2 - 1 * 1  # 3
    area_inner = 1 * 1  # 1
    area_cap = 4 * 4  # 16

    total_area = area_outer + area_mid + area_inner + area_cap  # 32
    total_wedges = sum(w.values())

    for name, area in zip(
        ("outer", "mid", "inner", "cap"),
        (area_outer, area_mid, area_inner, area_cap),
    ):
        frac = w[name] / total_wedges
        expected_frac = area / total_area
        assert (
            abs(frac - expected_frac) < 0.10
        ), f"{name}: wedge fraction {frac:.3f}, expected {expected_frac:.3f}"


# ---------------------------------------------------------------------------
# Test 3 -- void carves a solid
# ---------------------------------------------------------------------------


def test_void_carves_solid(tmp_path):
    """A solid 4x4 with a void 2x2 punching through it + a cap above.

    solid: 4x4, mesh_order=2, mesh_bool=True  -> surviving area = 12
    void:  2x2, mesh_order=1, mesh_bool=False  -> no 3D group
    cap:   4x4 at z=[1,2], mesh_bool=True      -> area = 16

    Bug (secondary): even if decompose correctly handles the void,
    the union footprint in cut_sources loses the void boundary, so
    the solid sub-piece may still be wrong (area=16 instead of 12).
    """
    solid = PolyPrism(
        square(0, 0, 4, 4),
        {0.0: 0.0, 1.0: 0.0},
        physical_name="solid",
        structured=True,
        mesh_order=2.0,
        mesh_bool=True,
    )
    void = PolyPrism(
        square(1, 1, 2, 2),
        {0.0: 0.0, 1.0: 0.0},
        physical_name="void",
        structured=True,
        mesh_order=1.0,
        mesh_bool=False,
    )
    cap = PolyPrism(
        square(0, 0, 4, 4),
        {1.0: 0.0, 2.0: 0.0},
        physical_name="cap",
        structured=True,
        mesh_order=1.0,
    )
    base = PolyPrism(
        square(0, 0, 4, 4),
        {-1.0: 0.0, 0.0: 0.0},
        physical_name="base",
    )
    top = PolyPrism(
        square(0, 0, 4, 4),
        {2.0: 0.0, 3.0: 0.0},
        physical_name="top",
    )

    generate_mesh(
        [solid, void, cap, base, top],
        dim=3,
        output_mesh=tmp_path / "out.msh",
        default_characteristic_length=0.5,
        resolution_specs=_resolution_specs_for("solid", "cap", n_layers=2),
    )
    m = meshio.read(tmp_path / "out.msh")

    # void must NOT be a 3D physical group
    assert "void" not in m.cell_sets, "void should NOT appear as a physical group"

    # solid and cap must be present with non-zero wedges
    assert "solid" in m.cell_sets, "solid missing from physical groups"
    assert "cap" in m.cell_sets, "cap missing from physical groups"

    w_solid = _wedge_count_for(m, "solid")
    w_cap = _wedge_count_for(m, "cap")
    assert w_solid > 0, "solid has 0 wedges"
    assert w_cap > 0, "cap has 0 wedges"

    # Area check: solid surviving area = 4x4 - 2x2 = 12; cap = 16.
    # With n_layers=2, wedge count is proportional to base area.
    # The void should have carved a 2x2 hole in solid, so solid area = 12.
    # If the bug is present, solid gets the full 16 (void hole not carved).
    # Use a +-5% tolerance so that the bug (solid area=16) actually fails.
    total_area = 12 + 16  # 28 (if void works correctly)
    total_wedges = w_solid + w_cap
    frac_solid = w_solid / total_wedges
    frac_cap = w_cap / total_wedges
    assert abs(frac_solid - 12 / total_area) < 0.05, (
        f"solid wedge fraction {frac_solid:.3f}, expected {12/total_area:.3f} "
        f"(BUG: void not carving -- solid may have full 4x4 area instead of 4x4-2x2)"
    )
    assert (
        abs(frac_cap - 16 / total_area) < 0.05
    ), f"cap wedge fraction {frac_cap:.3f}, expected {16/total_area:.3f}"


# ---------------------------------------------------------------------------
# Test 4 -- multi-z-interval mixed overlaps
# ---------------------------------------------------------------------------


def test_multi_z_interval_mixed_overlaps(tmp_path):
    """Two-level cohort: nested slabs at z=[0,1]; side-by-side at z=[1,2]; cap at z=[2,3].

    Level z=[0,1] -- three nested squares (same as test 2):
        outer3: 4x4, mesh_order=3
        mid2:   2x2, mesh_order=2
        inn1:   1x1, mesh_order=1

    Level z=[1,2] -- two side-by-side non-overlapping rectangles:
        s_left:  x=[0,2], y=[0,4], mesh_order=1  -> area = 8
        s_right: x=[2,4], y=[0,4], mesh_order=1  -> area = 8

    Level z=[2,3] -- single cap covering 4x4:
        cap: 4x4, mesh_order=1  -> area = 16

    Expected: all 6 non-void slabs present with correct wedge counts.
    No leftover empty sub-pieces (every physical group has >0 wedges).
    """
    outer3 = PolyPrism(
        square(0, 0, 4, 4),
        {0.0: 0.0, 1.0: 0.0},
        physical_name="outer3",
        structured=True,
        mesh_order=3.0,
    )
    mid2 = PolyPrism(
        square(1, 1, 2, 2),
        {0.0: 0.0, 1.0: 0.0},
        physical_name="mid2",
        structured=True,
        mesh_order=2.0,
    )
    inn1 = PolyPrism(
        square(1.5, 1.5, 1, 1),
        {0.0: 0.0, 1.0: 0.0},
        physical_name="inn1",
        structured=True,
        mesh_order=1.0,
    )
    s_left = PolyPrism(
        square(0, 0, 2, 4),
        {1.0: 0.0, 2.0: 0.0},
        physical_name="s_left",
        structured=True,
        mesh_order=1.0,
    )
    s_right = PolyPrism(
        square(2, 0, 2, 4),
        {1.0: 0.0, 2.0: 0.0},
        physical_name="s_right",
        structured=True,
        mesh_order=1.0,
    )
    cap = PolyPrism(
        square(0, 0, 4, 4),
        {2.0: 0.0, 3.0: 0.0},
        physical_name="cap4",
        structured=True,
        mesh_order=1.0,
    )
    base = PolyPrism(
        square(0, 0, 4, 4),
        {-1.0: 0.0, 0.0: 0.0},
        physical_name="base",
    )
    top = PolyPrism(
        square(0, 0, 4, 4),
        {3.0: 0.0, 4.0: 0.0},
        physical_name="top",
    )

    generate_mesh(
        [outer3, mid2, inn1, s_left, s_right, cap, base, top],
        dim=3,
        output_mesh=tmp_path / "out.msh",
        default_characteristic_length=0.5,
        resolution_specs=_resolution_specs_for(
            "outer3", "mid2", "inn1", "s_left", "s_right", "cap4", n_layers=2
        ),
    )
    m = meshio.read(tmp_path / "out.msh")

    # --- presence checks ---
    for name in ("outer3", "mid2", "inn1", "s_left", "s_right", "cap4"):
        assert name in m.cell_sets, f"{name} missing from physical groups"

    # --- wedge counts ---
    w = {
        name: _wedge_count_for(m, name)
        for name in ("outer3", "mid2", "inn1", "s_left", "s_right", "cap4")
    }
    for name, wc in w.items():
        assert wc > 0, f"{name} has 0 wedges"

    # --- loose area-proportionality sanity check ---
    # gmsh's 2D mesher gives small features more triangles per area, so strict
    # ±10% proportionality does not hold in general.  We use a factor-of-3
    # window: each slab's actual wedge fraction must be within [0.33x, 3.0x]
    # of the area-proportional expectation.  This catches "totally missing"
    # slabs (fraction ≈ 0) while accepting normal triangulation variance.
    area_map = {
        "outer3": 4 * 4 - 2 * 2,  # 12
        "mid2": 2 * 2 - 1 * 1,  # 3
        "inn1": 1 * 1,  # 1
        "s_left": 2 * 4,  # 8
        "s_right": 2 * 4,  # 8
        "cap4": 4 * 4,  # 16
    }
    total_area = sum(area_map.values())  # 48
    total_wedges = sum(w.values())

    for name, area in area_map.items():
        frac = w[name] / total_wedges
        expected_frac = area / total_area
        assert 0.33 * expected_frac <= frac <= 3.0 * expected_frac, (
            f"{name}: wedge fraction {frac:.3f} outside [0.33x, 3.0x] of "
            f"expected {expected_frac:.3f}"
        )

    # Sanity: no empty sub-pieces -> total wedge count is non-zero.
    # All survivors fill a total base area of 48 sq units over one z-unit each.
    # With n_layers=2, wedge count is proportional to area.
    assert total_wedges > 0, "zero total wedges -- mesh is empty"
