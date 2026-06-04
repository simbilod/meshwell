"""Tests for shared EdgeRegistry across structured + unstructured paths.

Goal: cohort↔neighbour boundary arc edges share TShapes by construction
when both sides go through the same EdgeRegistry. Today's BOP fuzzy
detection + AABB fallback is the workaround; this feature replaces it
for the edge case (face case still uses AABB).

These tests run BEFORE the refactor lands (Tasks 1-5) to lock in the
baseline mesh + group set the refactor must preserve. After the
refactor, the same tests pass AND the AABB rescue counter on the
complex scene drops.
"""
import sys
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon

import meshio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from meshwell.orchestrator import generate_mesh
from meshwell.polyprism import PolyPrism
from meshwell.resolution import StructuredExtrusionResolutionSpec


def _rect(x1, y1, x2, y2):
    return Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])


def _circle(cx, cy, r, n=48):
    a = np.linspace(0, 2 * np.pi, n + 1)[:-1]
    return Polygon([(cx + r * np.cos(t), cy + r * np.sin(t)) for t in a])


def test_arc_cohort_meets_unstructured_base_produces_interface(tmp_path):
    """An arc-bearing structured cohort meets unstructured base at z=0.

    Verifies that the bg___base interface group exists with at least
    one face, and base has both the expected interface and an
    exterior boundary. After the refactor, BOP should unify the arc
    edges by construction (shared TShapes) rather than via fuzzy +
    AABB fallback.
    """
    bg = PolyPrism(
        _circle(0, 0, 2),
        {0.0: 0.0, 1.0: 0.0},
        physical_name="bg",
        structured=True,
        mesh_order=2.0,
        identify_arcs=True,
    )
    base = PolyPrism(
        _rect(-5, -5, 5, 5),
        {-2.0: 0.0, 0.0: 0.0},
        physical_name="base",
        mesh_order=3.0,
        identify_arcs=True,
    )
    cap = PolyPrism(
        _rect(-5, -5, 5, 5),
        {1.0: 0.0, 2.0: 0.0},
        physical_name="cap",
        mesh_order=3.0,
        identify_arcs=True,
    )
    generate_mesh(
        [bg, base, cap],
        dim=3,
        output_mesh=tmp_path / "out.msh",
        default_characteristic_length=0.5,
        resolution_specs={
            "bg": [StructuredExtrusionResolutionSpec(n_layers=2)],
        },
    )
    m = meshio.read(tmp_path / "out.msh")

    assert "bg" in m.cell_sets
    assert "base" in m.cell_sets
    iface = m.cell_sets.get("bg___base") or m.cell_sets.get("base___bg")
    assert (
        iface is not None
    ), f"expected bg___base interface; groups: {sorted(m.cell_sets)}"
    iface_faces = sum(
        len(s)
        for s, c in zip(iface, m.cells)
        if c.type in ("triangle", "quad") and s is not None
    )
    assert iface_faces >= 1


def test_polyline_cohort_meets_unstructured_neighbour(tmp_path):
    """A polyline structured cohort meets an unstructured neighbour at z=0.

    Same as arc case but with rectangular boundaries. Both should
    produce an interface group regardless of whether edges are shared
    via registry or unified by BOP.
    """
    bg = PolyPrism(
        _rect(-2, -2, 2, 2),
        {0.0: 0.0, 1.0: 0.0},
        physical_name="bg",
        structured=True,
        mesh_order=2.0,
    )
    base = PolyPrism(
        _rect(-5, -5, 5, 5),
        {-2.0: 0.0, 0.0: 0.0},
        physical_name="base",
        mesh_order=3.0,
    )
    cap = PolyPrism(
        _rect(-5, -5, 5, 5),
        {1.0: 0.0, 2.0: 0.0},
        physical_name="cap",
        mesh_order=3.0,
    )
    generate_mesh(
        [bg, base, cap],
        dim=3,
        output_mesh=tmp_path / "out.msh",
        default_characteristic_length=0.5,
        resolution_specs={
            "bg": [StructuredExtrusionResolutionSpec(n_layers=2)],
        },
    )
    m = meshio.read(tmp_path / "out.msh")
    assert "bg" in m.cell_sets
    assert "base" in m.cell_sets
    iface = m.cell_sets.get("bg___base") or m.cell_sets.get("base___bg")
    assert (
        iface is not None
    ), f"expected bg___base interface; groups: {sorted(m.cell_sets)}"


def test_aabb_rescue_count_reduced_under_sharing(tmp_path):
    """Assert zero AABB rescues under shared EdgeRegistry.

    Under the refactor, a minimal arc cohort + unstructured base scene
    should produce ZERO AABB rescues — the cohort↔neighbour shared edges
    have identical TShapes by construction. Before the refactor this case
    needed at least one rescue.
    """
    from itertools import combinations

    import meshwell.occ_xao_writer as xao_mod

    rescues: list[tuple[str, str]] = []
    original = xao_mod._compute_physical_groups

    def instrumented(
        entities,
        interface_delimiter,
        boundary_delimiter,
        interface_aabb_tolerance=xao_mod._DEFAULT_AABB_INTERFACE_TOL,
    ):
        max_dim = max((e.dim for e in entities if e.shapes), default=0)
        ebs = []
        for ent in entities:
            b = {}
            if ent.dim == max_dim and ent.dim > 0:
                for s in ent.shapes:
                    for sub, sid in xao_mod._leaf_subshapes(s, ent.dim - 1):
                        b.setdefault(sid, sub)
            elif ent.dim == max_dim - 1 and ent.dim > 0:
                for s in ent.shapes:
                    for sub, sid in xao_mod._leaf_subshapes(s, ent.dim):
                        b.setdefault(sid, sub)
            ebs.append(b)
        eas = []
        for i in range(len(entities)):
            d = {}
            for sid, face in ebs[i].items():
                box = xao_mod._shape_aabb(face)
                if box is not None:
                    d[sid] = box
            eas.append(d)
        for (i1, e1), (i2, e2) in combinations(enumerate(entities), 2):
            if e1.dim <= 0 or e2.dim <= 0:
                continue
            if set(ebs[i1].keys()) & set(ebs[i2].keys()):
                continue
            if not eas[i1] or not eas[i2]:
                continue
            arr2 = np.array(list(eas[i2].values()), dtype=float)
            for b1 in eas[i1].values():
                b1_arr = np.asarray(b1, dtype=float)
                if np.any(
                    np.abs(arr2 - b1_arr).max(axis=1) < interface_aabb_tolerance
                ) and not (
                    xao_mod._is_purely_synthetic(e1) or xao_mod._is_purely_synthetic(e2)
                ):
                    n1 = (
                        xao_mod._filter_real_names(e1.physical_name) or e1.physical_name
                    )
                    n2 = (
                        xao_mod._filter_real_names(e2.physical_name) or e2.physical_name
                    )
                    rescues.append((n1[0], n2[0]))
        return original(
            entities,
            interface_delimiter,
            boundary_delimiter,
            interface_aabb_tolerance=interface_aabb_tolerance,
        )

    xao_mod._compute_physical_groups = instrumented
    try:
        bg = PolyPrism(
            _circle(0, 0, 2),
            {0.0: 0.0, 1.0: 0.0},
            physical_name="bg",
            structured=True,
            mesh_order=2.0,
            identify_arcs=True,
        )
        base = PolyPrism(
            _rect(-5, -5, 5, 5),
            {-2.0: 0.0, 0.0: 0.0},
            physical_name="base",
            mesh_order=3.0,
            identify_arcs=True,
        )
        cap = PolyPrism(
            _rect(-5, -5, 5, 5),
            {1.0: 0.0, 2.0: 0.0},
            physical_name="cap",
            mesh_order=3.0,
            identify_arcs=True,
        )
        generate_mesh(
            [bg, base, cap],
            dim=3,
            output_mesh=tmp_path / "out.msh",
            default_characteristic_length=0.5,
            resolution_specs={
                "bg": [StructuredExtrusionResolutionSpec(n_layers=2)],
            },
        )
    finally:
        xao_mod._compute_physical_groups = original

    assert (
        len(rescues) == 0
    ), f"expected 0 AABB rescues with registry sharing; got {rescues}"
