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
