"""Regression tests: synthetic 2D annotators don't disrupt natural interfaces.

Background
----------
The structured post-pass emits one synthetic 2D ``OCCLabeledEntity`` per
tracked cohort face (named ``__cohort_<ci>__slab_<si>__{bot,top,lat_N}``).
These provide a stable lookup from pre-BOP ShapeKey to post-load gmsh tag
via the physical-group name.

An earlier filter in ``occ_xao_writer._compute_physical_groups`` skipped
any interface pair where EITHER side carried a ``__cohort_*`` name. That
filter inadvertently caught real cohort sub-solids too — their tuple is
``("lower", "__cohort_0__slab_0")`` — so the natural ``lower___upper``
interface for the shared z=1 face was never emitted, and BOTH ``lower``
and ``upper`` claimed it as a free-boundary face in their ``___None`` group.

This test fixes the contract: real cohort sub-solids must produce
``A___B`` interface groups for shared faces, even though their
physical_name tuple has a synthetic name appended.
"""
from __future__ import annotations

import math

import gmsh
from shapely.geometry import Polygon

from meshwell.orchestrator import generate_mesh
from meshwell.polyprism import PolyPrism
from meshwell.resolution import StructuredExtrusionResolutionSpec


def _disc(r: float, n: int = 48) -> Polygon:
    return Polygon(
        [
            (r * math.cos(2 * math.pi * i / n), r * math.sin(2 * math.pi * i / n))
            for i in range(n)
        ]
    )


def _annulus(r_out: float, r_in: float, n: int = 48) -> Polygon:
    outer = _disc(r_out, n)
    inner = _disc(r_in, n)
    return Polygon(outer.exterior.coords, holes=[inner.exterior.coords])


def _physical_names(path) -> set[str]:
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(str(path))
        out: set[str] = set()
        for dim, tag in gmsh.model.getPhysicalGroups():
            out.add(gmsh.model.getPhysicalName(dim, tag))
        return out
    finally:
        gmsh.finalize()


def test_stacked_discs_share_interface_group(tmp_path):
    """Two stacked discs sharing z=1 must produce a ``lower___upper`` group."""
    lower = PolyPrism(
        _disc(1.0),
        {0.0: 0.0, 1.0: 0.0},
        physical_name="lower",
        structured=True,
        identify_arcs=True,
    )
    upper = PolyPrism(
        _disc(1.0),
        {1.0: 0.0, 2.0: 0.0},
        physical_name="upper",
        structured=True,
        identify_arcs=True,
    )
    base = PolyPrism(
        _disc(1.0),
        {-1.0: 0.0, 0.0: 0.0},
        physical_name="base",
        identify_arcs=True,
    )
    cap = PolyPrism(
        _disc(1.0),
        {2.0: 0.0, 3.0: 0.0},
        physical_name="cap",
        identify_arcs=True,
    )
    msh = tmp_path / "x.msh"
    generate_mesh(
        [lower, upper, base, cap],
        dim=3,
        output_mesh=msh,
        default_characteristic_length=0.3,
        resolution_specs={
            "lower": [StructuredExtrusionResolutionSpec(n_layers=2)],
            "upper": [StructuredExtrusionResolutionSpec(n_layers=2)],
        },
    )
    names = _physical_names(msh)
    assert (
        "lower___upper" in names or "upper___lower" in names
    ), f"shared interface face missing; physical groups: {sorted(names)}"
    # No synthetic name should leak.
    assert not any(
        n.startswith("__cohort_") for n in names
    ), f"synthetic groups leaked into .msh: {sorted(n for n in names if n.startswith('__cohort_'))}"


def test_annulus_on_disc_partial_interface(tmp_path):
    """Annulus on disc: only the ring overlap is a shared interface.

    The lower disc's top face at z=1 covers radius [0, 2]. The upper annulus's
    bot face at z=1 covers radius [0.8, 2]. The shared annular ring should be
    in ``lower_disc___upper_annulus``; the inner exposed disc patch at z=1
    must remain in ``lower_disc___None`` only.
    """
    disc = PolyPrism(
        _disc(2.0),
        {0.0: 0.0, 1.0: 0.0},
        physical_name="lower_disc",
        structured=True,
        identify_arcs=True,
    )
    annulus = PolyPrism(
        _annulus(r_out=2.0, r_in=0.8),
        {1.0: 0.0, 2.0: 0.0},
        physical_name="upper_annulus",
        structured=True,
        identify_arcs=True,
    )
    base = PolyPrism(
        _disc(2.0),
        {-1.0: 0.0, 0.0: 0.0},
        physical_name="base",
        identify_arcs=True,
    )
    cap = PolyPrism(
        _annulus(r_out=2.0, r_in=0.8),
        {2.0: 0.0, 3.0: 0.0},
        physical_name="cap",
        identify_arcs=True,
    )
    msh = tmp_path / "x.msh"
    generate_mesh(
        [disc, annulus, base, cap],
        dim=3,
        output_mesh=msh,
        default_characteristic_length=0.3,
        resolution_specs={
            "lower_disc": [StructuredExtrusionResolutionSpec(n_layers=2)],
            "upper_annulus": [StructuredExtrusionResolutionSpec(n_layers=2)],
        },
    )
    names = _physical_names(msh)
    interface_name = (
        "lower_disc___upper_annulus"
        if "lower_disc___upper_annulus" in names
        else "upper_annulus___lower_disc"
    )
    assert (
        interface_name in names
    ), f"partial-overlap interface missing; physical groups: {sorted(names)}"
    assert not any(n.startswith("__cohort_") for n in names)


def test_single_disc_only_boundary(tmp_path):
    """One disc: no interfaces, all faces in ``disc___None``."""
    disc = PolyPrism(
        _disc(1.0),
        {0.0: 0.0, 1.0: 0.0},
        physical_name="disc",
        structured=True,
        identify_arcs=True,
    )
    base = PolyPrism(
        _disc(1.0),
        {-1.0: 0.0, 0.0: 0.0},
        physical_name="base",
        identify_arcs=True,
    )
    cap = PolyPrism(
        _disc(1.0),
        {1.0: 0.0, 2.0: 0.0},
        physical_name="cap",
        identify_arcs=True,
    )
    msh = tmp_path / "x.msh"
    generate_mesh(
        [disc, base, cap],
        dim=3,
        output_mesh=msh,
        default_characteristic_length=0.3,
        resolution_specs={"disc": [StructuredExtrusionResolutionSpec(n_layers=3)]},
    )
    names = _physical_names(msh)
    assert "disc___None" in names
    # Spurious ``disc___X`` interface checks limited to disc-vs-disc; base+cap
    # are expected to share interfaces with disc.
    spurious = [
        n
        for n in names
        if "___" in n
        and not n.endswith("___None")
        and n
        not in (
            "disc",
            "base",
            "cap",
            "disc___base",
            "base___disc",
            "disc___cap",
            "cap___disc",
        )
    ]
    assert not spurious, f"unexpected interface groups: {spurious}"
    assert not any(n.startswith("__cohort_") for n in names)


def test_overlapping_squares_share_interface(tmp_path):
    """``a1`` and ``a2`` on z=[0,1], ``b`` on z=[1,2]: both must interface ``b``."""
    SQa1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    SQa2 = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
    SQb = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])

    a1 = PolyPrism(
        SQa1, {0.0: 0.0, 1.0: 0.0}, physical_name="a1", structured=True, mesh_order=1
    )
    a2 = PolyPrism(
        SQa2, {0.0: 0.0, 1.0: 0.0}, physical_name="a2", structured=True, mesh_order=2
    )
    b = PolyPrism(SQb, {1.0: 0.0, 2.0: 0.0}, physical_name="b", structured=True)
    base = PolyPrism(SQa2, {-1.0: 0.0, 0.0: 0.0}, physical_name="base")
    cap = PolyPrism(SQb, {2.0: 0.0, 3.0: 0.0}, physical_name="cap")

    msh = tmp_path / "x.msh"
    generate_mesh(
        entities=[a1, a2, b, base, cap],
        dim=3,
        output_mesh=msh,
        default_characteristic_length=0.4,
        resolution_specs={
            "a1": [StructuredExtrusionResolutionSpec(n_layers=2)],
            "a2": [StructuredExtrusionResolutionSpec(n_layers=2)],
            "b": [StructuredExtrusionResolutionSpec(n_layers=2)],
        },
    )
    names = _physical_names(msh)
    assert "a1___b" in names or "b___a1" in names, sorted(names)
    assert "a2___b" in names or "b___a2" in names, sorted(names)
    # ``b`` is split into two slabs by the cohort below, but the internal
    # slab boundary is NOT a physical group.
    assert "b___b" not in names
    assert not any(n.startswith("__cohort_") for n in names)
