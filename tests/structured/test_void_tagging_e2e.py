"""End-to-end void boundary tagging tests."""
from __future__ import annotations

import math
from pathlib import Path

import gmsh
from shapely.geometry import Polygon

import meshio
from meshwell.orchestrator import generate_mesh
from meshwell.polyprism import PolyPrism
from meshwell.resolution import StructuredExtrusionResolutionSpec


def _disc(cx: float, cy: float, r: float, n: int = 48) -> Polygon:
    return Polygon(
        [
            (
                cx + r * math.cos(2 * math.pi * i / n),
                cy + r * math.sin(2 * math.pi * i / n),
            )
            for i in range(n)
        ]
    )


def _square(x: float, y: float, w: float, h: float) -> Polygon:
    return Polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])


def _physical_names(path: Path) -> set[str]:
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(str(path))
        return {
            gmsh.model.getPhysicalName(dim, tag)
            for dim, tag in gmsh.model.getPhysicalGroups()
        }
    finally:
        gmsh.finalize()


def _has_interface(names: set[str], a: str, b: str) -> bool:
    return f"{a}___{b}" in names or f"{b}___{a}" in names


def _structured_spec(*names: str, n_layers: int = 2) -> dict:
    return {n: [StructuredExtrusionResolutionSpec(n_layers=n_layers)] for n in names}


def test_void_inside_single_structured_slab(tmp_path: Path):
    """Bg square at z=[0,1] with a hole disc void. No neighbours.

    Expected: bg___hole lateral. No hole 3D group. No hole___None.
    """
    bg = PolyPrism(
        _square(-3, -3, 6, 6),
        {0.0: 0.0, 1.0: 0.0},
        physical_name="bg",
        structured=True,
        mesh_order=2.0,
    )
    hole = PolyPrism(
        _disc(0, 0, 1.0),
        {0.0: 0.0, 1.0: 0.0},
        physical_name="hole",
        structured=True,
        mesh_order=1.0,
        mesh_bool=False,
        identify_arcs=True,
    )
    msh = tmp_path / "out.msh"
    generate_mesh(
        [bg, hole],
        dim=3,
        output_mesh=msh,
        default_characteristic_length=0.4,
        resolution_specs=_structured_spec("bg"),
    )
    m = meshio.read(msh)
    names = _physical_names(msh)
    assert "bg" in m.cell_sets
    assert "hole" not in m.cell_sets, "void should not have 3D group"
    assert "hole___None" not in names, "void should not have boundary group"
    assert _has_interface(
        names, "bg", "hole"
    ), f"expected bg___hole lateral; got groups: {sorted(names)}"


def test_void_below_unstructured_cap(tmp_path: Path):
    """Bg + void + cap above.

    Expected: bg___hole + cap___hole.
    """
    bg = PolyPrism(
        _square(-3, -3, 6, 6),
        {0.0: 0.0, 1.0: 0.0},
        physical_name="bg",
        structured=True,
        mesh_order=2.0,
    )
    hole = PolyPrism(
        _disc(0, 0, 1.0),
        {0.0: 0.0, 1.0: 0.0},
        physical_name="hole",
        structured=True,
        mesh_order=1.0,
        mesh_bool=False,
        identify_arcs=True,
    )
    cap = PolyPrism(
        _square(-3, -3, 6, 6),
        {1.0: 0.0, 2.0: 0.0},
        physical_name="cap",
        mesh_order=3.0,
    )
    msh = tmp_path / "out.msh"
    generate_mesh(
        [bg, hole, cap],
        dim=3,
        output_mesh=msh,
        default_characteristic_length=0.4,
        resolution_specs=_structured_spec("bg"),
    )
    m = meshio.read(msh)
    names = _physical_names(msh)
    assert "bg" in m.cell_sets
    assert "cap" in m.cell_sets
    assert "hole" not in m.cell_sets
    assert _has_interface(
        names, "bg", "hole"
    ), f"missing bg___hole; groups: {sorted(names)}"
    assert _has_interface(
        names, "cap", "hole"
    ), f"missing cap___hole; groups: {sorted(names)}"


def test_void_above_unstructured_base(tmp_path: Path):
    """Base + bg + void.

    Expected: bg___hole + base___hole.
    """
    base = PolyPrism(
        _square(-3, -3, 6, 6),
        {-1.0: 0.0, 0.0: 0.0},
        physical_name="base",
        mesh_order=3.0,
    )
    bg = PolyPrism(
        _square(-3, -3, 6, 6),
        {0.0: 0.0, 1.0: 0.0},
        physical_name="bg",
        structured=True,
        mesh_order=2.0,
    )
    hole = PolyPrism(
        _disc(0, 0, 1.0),
        {0.0: 0.0, 1.0: 0.0},
        physical_name="hole",
        structured=True,
        mesh_order=1.0,
        mesh_bool=False,
        identify_arcs=True,
    )
    msh = tmp_path / "out.msh"
    generate_mesh(
        [base, bg, hole],
        dim=3,
        output_mesh=msh,
        default_characteristic_length=0.4,
        resolution_specs=_structured_spec("bg"),
    )
    names = _physical_names(msh)
    assert _has_interface(names, "bg", "hole")
    assert _has_interface(
        names, "base", "hole"
    ), f"missing base___hole; groups: {sorted(names)}"


def test_void_sandwiched_between_unstructured(tmp_path: Path):
    """Base + bg + void + cap.

    All three void interfaces expected.
    """
    base = PolyPrism(
        _square(-3, -3, 6, 6),
        {-1.0: 0.0, 0.0: 0.0},
        physical_name="base",
        mesh_order=3.0,
    )
    bg = PolyPrism(
        _square(-3, -3, 6, 6),
        {0.0: 0.0, 1.0: 0.0},
        physical_name="bg",
        structured=True,
        mesh_order=2.0,
    )
    hole = PolyPrism(
        _disc(0, 0, 1.0),
        {0.0: 0.0, 1.0: 0.0},
        physical_name="hole",
        structured=True,
        mesh_order=1.0,
        mesh_bool=False,
        identify_arcs=True,
    )
    cap = PolyPrism(
        _square(-3, -3, 6, 6),
        {1.0: 0.0, 2.0: 0.0},
        physical_name="cap",
        mesh_order=3.0,
    )
    msh = tmp_path / "out.msh"
    generate_mesh(
        [base, bg, hole, cap],
        dim=3,
        output_mesh=msh,
        default_characteristic_length=0.4,
        resolution_specs=_structured_spec("bg"),
    )
    names = _physical_names(msh)
    assert _has_interface(names, "bg", "hole"), "lateral"
    assert _has_interface(names, "base", "hole"), "void bot"
    assert _has_interface(names, "cap", "hole"), "void top"


def test_void_through_stacked_cohort(tmp_path: Path):
    """A void appearing in two stacked slabs (one void per slab).

    Each slab has its own co-planar void. Lateral interface expected on BOTH.
    The void must be split to match each slab's z-interval — a single void
    spanning both slabs would violate the z-stack constraint because the
    surrounding stack introduces z=1 as an interior plane of the hole cohort.
    """
    lower = PolyPrism(
        _square(-3, -3, 6, 6),
        {0.0: 0.0, 1.0: 0.0},
        physical_name="lower",
        structured=True,
        mesh_order=2.0,
    )
    upper = PolyPrism(
        _square(-3, -3, 6, 6),
        {1.0: 0.0, 2.0: 0.0},
        physical_name="upper",
        structured=True,
        mesh_order=2.0,
    )
    # Two voids, one per slab, with the same disc footprint.
    hole_lower = PolyPrism(
        _disc(0, 0, 1.0),
        {0.0: 0.0, 1.0: 0.0},
        physical_name="hole",
        structured=True,
        mesh_order=1.0,
        mesh_bool=False,
        identify_arcs=True,
    )
    hole_upper = PolyPrism(
        _disc(0, 0, 1.0),
        {1.0: 0.0, 2.0: 0.0},
        physical_name="hole",
        structured=True,
        mesh_order=1.0,
        mesh_bool=False,
        identify_arcs=True,
    )
    msh = tmp_path / "out.msh"
    generate_mesh(
        [lower, upper, hole_lower, hole_upper],
        dim=3,
        output_mesh=msh,
        default_characteristic_length=0.4,
        resolution_specs=_structured_spec("lower", "upper"),
    )
    names = _physical_names(msh)
    assert _has_interface(
        names, "lower", "hole"
    ), f"missing lower___hole; groups: {sorted(names)}"
    assert _has_interface(
        names, "upper", "hole"
    ), f"missing upper___hole; groups: {sorted(names)}"


def test_void_below_structured_cohort_slab(tmp_path: Path):
    """Lower has a void; upper is solid above.

    Expected: lower___hole (lateral) + upper___hole (void top at z=1 touching
    upper's bot).
    """
    lower = PolyPrism(
        _square(-3, -3, 6, 6),
        {0.0: 0.0, 1.0: 0.0},
        physical_name="lower",
        structured=True,
        mesh_order=2.0,
    )
    upper = PolyPrism(
        _square(-3, -3, 6, 6),
        {1.0: 0.0, 2.0: 0.0},
        physical_name="upper",
        structured=True,
        mesh_order=2.0,
    )
    hole = PolyPrism(
        _disc(0, 0, 1.0),
        {0.0: 0.0, 1.0: 0.0},
        physical_name="hole",
        structured=True,
        mesh_order=1.0,
        mesh_bool=False,
        identify_arcs=True,
    )
    msh = tmp_path / "out.msh"
    generate_mesh(
        [lower, upper, hole],
        dim=3,
        output_mesh=msh,
        default_characteristic_length=0.4,
        resolution_specs=_structured_spec("lower", "upper"),
    )
    names = _physical_names(msh)
    assert _has_interface(names, "lower", "hole"), "lateral"
    assert _has_interface(
        names, "upper", "hole"
    ), f"void top at z=1 should touch upper's bot; got: {sorted(names)}"


def test_void_square_no_arcs(tmp_path: Path):
    """Square void (polyline only).

    Expected: lateral walls bg___hole.
    """
    bg = PolyPrism(
        _square(-3, -3, 6, 6),
        {0.0: 0.0, 1.0: 0.0},
        physical_name="bg",
        structured=True,
        mesh_order=2.0,
    )
    hole = PolyPrism(
        _square(-0.5, -0.5, 1, 1),
        {0.0: 0.0, 1.0: 0.0},
        physical_name="hole",
        structured=True,
        mesh_order=1.0,
        mesh_bool=False,
    )
    msh = tmp_path / "out.msh"
    generate_mesh(
        [bg, hole],
        dim=3,
        output_mesh=msh,
        default_characteristic_length=0.4,
        resolution_specs=_structured_spec("bg"),
    )
    names = _physical_names(msh)
    assert _has_interface(
        names, "bg", "hole"
    ), f"missing bg___hole (square void); groups: {sorted(names)}"


def test_two_separate_voids_policy_b(tmp_path: Path):
    """Two separate voids side by side in the same solid: each gets its own interface.

    void_a is at x<0 and void_b is at x>0; they do not overlap.
    Both should produce bg___void_a and bg___void_b interfaces.
    Neither void should appear as a 3D group.
    """
    bg = PolyPrism(
        _square(-3, -3, 6, 6),
        {0.0: 0.0, 1.0: 0.0},
        physical_name="bg",
        structured=True,
        mesh_order=3.0,
    )
    # Void A: disc centred at (-1.5, 0).
    void_a = PolyPrism(
        _disc(-1.5, 0, 0.8),
        {0.0: 0.0, 1.0: 0.0},
        physical_name="void_a",
        structured=True,
        mesh_order=2.0,
        mesh_bool=False,
        identify_arcs=True,
    )
    # Void B: disc centred at (+1.5, 0).
    void_b = PolyPrism(
        _disc(1.5, 0, 0.8),
        {0.0: 0.0, 1.0: 0.0},
        physical_name="void_b",
        structured=True,
        mesh_order=1.0,
        mesh_bool=False,
        identify_arcs=True,
    )
    msh = tmp_path / "out.msh"
    generate_mesh(
        [bg, void_a, void_b],
        dim=3,
        output_mesh=msh,
        default_characteristic_length=0.4,
        resolution_specs=_structured_spec("bg"),
    )
    m = meshio.read(msh)
    names = _physical_names(msh)
    # Neither void appears as a 3D group.
    assert "void_a" not in m.cell_sets
    assert "void_b" not in m.cell_sets
    # Both voids interface with bg.
    assert _has_interface(
        names, "bg", "void_a"
    ), f"void_a boundary bg___void_a missing; groups: {sorted(names)}"
    assert _has_interface(
        names, "bg", "void_b"
    ), f"void_b boundary bg___void_b missing; groups: {sorted(names)}"


def test_void_with_arc_neighbour_pre_cut(tmp_path: Path):
    """A void with an arc-bearing disc cap above.

    The bidirectional pre-cut + unified arc detection should make both sides
    have matching OCC arc edges so BOP merges the shared disc face at z=1.
    """
    bg = PolyPrism(
        _square(-3, -3, 6, 6),
        {0.0: 0.0, 1.0: 0.0},
        physical_name="bg",
        structured=True,
        mesh_order=2.0,
    )
    hole = PolyPrism(
        _disc(0, 0, 1.0),
        {0.0: 0.0, 1.0: 0.0},
        physical_name="hole",
        structured=True,
        mesh_order=1.0,
        mesh_bool=False,
        identify_arcs=True,
    )
    # Arc-bearing cap (a larger disc) above. Pre-cut splits it into
    # disc-over-hole + annular-ring-over-bg.
    cap = PolyPrism(
        _disc(0, 0, 2.5),
        {1.0: 0.0, 2.0: 0.0},
        physical_name="cap",
        mesh_order=3.0,
        identify_arcs=True,
    )
    msh = tmp_path / "out.msh"
    generate_mesh(
        [bg, hole, cap],
        dim=3,
        output_mesh=msh,
        default_characteristic_length=0.4,
        resolution_specs=_structured_spec("bg"),
    )
    names = _physical_names(msh)
    assert _has_interface(
        names, "bg", "hole"
    ), f"missing bg___hole lateral; groups: {sorted(names)}"
    assert _has_interface(
        names, "cap", "hole"
    ), f"missing cap___hole (arc-vs-arc); groups: {sorted(names)}"
