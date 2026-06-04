"""Interlocking-meander stress test.

A meander A (3 strips, 2 U-turns) and an outer-U B wrap so A's outer
U-turn arc IS B's inner U-turn arc, plus shared straight edges at
y=0, y=3, y=4. Voids carve into A, B, embed. Embed (lower-priority
structured) surrounds both. Cladding above and below as unstructured
PolyPrisms.

Status:
- All scenarios pass under the cohort-global arrangement decompose
  (2026-06-04). Prior to that refactor, the full scene reliably
  triggered CohortShellModifiedError when both voids + embed + both
  claddings were present, because cohort sub-pieces and unstructured
  pre-cuts ran independent polygonize+filter passes that could disagree
  on polygon partition.

Exercises:
- intra-cohort arc TShape sharing across distinct user entities (A and B
  share the U-turn arc at (L, 1.5), r=1.5 - the EdgeRegistry must produce
  a single TShape for both)
- many sub-pieces from a single cohort z-interval with arcs + voids
- cohort sub-piece <-> unstructured cladding face matching at z=0 and z=1
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import Polygon

import meshio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from meshwell.orchestrator import generate_mesh
from meshwell.polyprism import PolyPrism
from meshwell.resolution import StructuredExtrusionResolutionSpec

# --- Geometry parameters (fixed for the test) --------------------------------

L = 10.0
N_ARC = 24  # sample points per half-arc


def _arc_pts(cx, cy, r, t_start, t_end, n):
    """Inclusive endpoints, n+1 points evenly spaced in [t_start, t_end]."""
    ts = np.linspace(t_start, t_end, n + 1)
    return [(cx + r * np.cos(t), cy + r * np.sin(t)) for t in ts]


def meander_A_polygon():
    """Three strips at y=[0,1],[2,3],[4,5]; U1 +x at (L,1.5); U2 -x at (0,3.5).

    Boundary CCW (interior on left).
    """
    pts: list[tuple[float, float]] = [(0.0, 0.0)]
    # Outer arc U1: (L, 0) -> (L, 3) CCW around (L, 1.5), r=1.5
    pts.extend(_arc_pts(L, 1.5, 1.5, -np.pi / 2, np.pi / 2, N_ARC))
    # Strip 2 top: -> (0, 3)
    pts.append((0.0, 3.0))
    # Inner arc U2: (0, 3) -> (0, 4) via (-0.5, 3.5), CW around (0, 3.5), r=0.5
    pts.extend(
        (-0.5 * np.sin(s), 3.5 - 0.5 * np.cos(s))
        for s in np.linspace(0, np.pi, N_ARC + 1)[1:]
    )
    # Strip 3 bottom: -> (L, 4)
    pts.append((L, 4.0))
    # Strip 3 right: -> (L, 5)
    pts.append((L, 5.0))
    # Strip 3 top: -> (0, 5)
    pts.append((0.0, 5.0))
    # Outer arc U2: (0, 5) -> (0, 2) via (-1.5, 3.5), CCW around (0, 3.5), r=1.5
    pts.extend(
        (-1.5 * np.sin(s), 3.5 + 1.5 * np.cos(s))
        for s in np.linspace(0, np.pi, N_ARC + 1)[1:]
    )
    # Strip 2 bottom: -> (L, 2)
    pts.append((L, 2.0))
    # Inner arc U1: (L, 2) -> (L, 1) via (L+0.5, 1.5), CW around (L, 1.5), r=0.5
    pts.extend(
        (L + 0.5 * np.sin(s), 1.5 + 0.5 * np.cos(s))
        for s in np.linspace(0, np.pi, N_ARC + 1)[1:]
    )
    # Strip 1 top: -> (0, 1)
    pts.append((0.0, 1.0))
    return Polygon(pts)


def outer_U_polygon():
    """B wraps A from outside. Strip B1 [y=-1..0], Strip B2 [y=3..4], U-turn around (L, 1.5) inner=1.5 outer=2.5 (+x side)."""
    pts: list[tuple[float, float]] = [(0.0, -1.0)]
    # Outer arc: (L, -1) -> (L, 4) CCW around (L, 1.5), r=2.5
    pts.extend(_arc_pts(L, 1.5, 2.5, -np.pi / 2, np.pi / 2, N_ARC))
    # Strip B2 top: -> (0, 4)
    pts.append((0.0, 4.0))
    # Strip B2 left: -> (0, 3)
    pts.append((0.0, 3.0))
    # Strip B2 bottom: -> (L, 3)
    pts.append((L, 3.0))
    # Inner arc: (L, 3) -> (L, 0) via (L+1.5, 1.5), CW around (L, 1.5), r=1.5
    pts.extend(
        (L + 1.5 * np.sin(s), 1.5 + 1.5 * np.cos(s))
        for s in np.linspace(0, np.pi, N_ARC + 1)[1:]
    )
    # Strip B1 top: -> (0, 0)
    pts.append((0.0, 0.0))
    return Polygon(pts)


def _rect(x1, y1, x2, y2):
    return Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])


def _disc(cx, cy, r, n=24):
    return Polygon(_arc_pts(cx, cy, r, 0.0, 2 * np.pi, n)[:-1])


# --- Scene fixture -----------------------------------------------------------


@pytest.fixture
def meander_entities():
    A = PolyPrism(
        meander_A_polygon(),
        {0.0: 0.0, 1.0: 0.0},
        physical_name="meander_A",
        structured=True,
        mesh_order=3.0,
        identify_arcs=True,
    )
    B = PolyPrism(
        outer_U_polygon(),
        {0.0: 0.0, 1.0: 0.0},
        physical_name="meander_B",
        structured=True,
        mesh_order=3.0,
        identify_arcs=True,
    )
    void_A = PolyPrism(
        _disc(L + 0.8, 1.5, 0.3),  # inside A's U-turn 1 half-annulus
        {0.0: 0.0, 1.0: 0.0},
        physical_name="void_A",
        structured=True,
        mesh_order=1.0,
        mesh_bool=False,
        identify_arcs=True,
    )
    void_B = PolyPrism(
        _rect(L / 2 - 0.25, 3.25, L / 2 + 0.25, 3.75),  # inside strip B2
        {0.0: 0.0, 1.0: 0.0},
        physical_name="void_B",
        structured=True,
        mesh_order=1.0,
        mesh_bool=False,
        identify_arcs=True,
    )
    embed = PolyPrism(
        _rect(-3.0, -2.0, L + 4.0, 6.0),
        {0.0: 0.0, 1.0: 0.0},
        physical_name="embed",
        structured=True,
        mesh_order=10.0,
        identify_arcs=True,
    )
    cladding_below = PolyPrism(
        _rect(-3.0, -2.0, L + 4.0, 6.0),
        {-1.0: 0.0, 0.0: 0.0},
        physical_name="cladding_below",
        mesh_order=20.0,
        identify_arcs=True,
    )
    cladding_above = PolyPrism(
        _rect(-3.0, -2.0, L + 4.0, 6.0),
        {1.0: 0.0, 2.0: 0.0},
        physical_name="cladding_above",
        mesh_order=20.0,
        identify_arcs=True,
    )
    return [A, B, void_A, void_B, embed, cladding_below, cladding_above]


def _resolution_specs():
    return {
        name: [StructuredExtrusionResolutionSpec(n_layers=2)]
        for name in ("meander_A", "meander_B", "embed")
    }


# --- Tests -------------------------------------------------------------------


def test_meander_minimal(tmp_path):
    """A + B + cladding_below meshes cleanly; intra-cohort arc sharing works."""
    A = PolyPrism(
        meander_A_polygon(),
        {0.0: 0.0, 1.0: 0.0},
        physical_name="meander_A",
        structured=True,
        mesh_order=3.0,
        identify_arcs=True,
    )
    B = PolyPrism(
        outer_U_polygon(),
        {0.0: 0.0, 1.0: 0.0},
        physical_name="meander_B",
        structured=True,
        mesh_order=3.0,
        identify_arcs=True,
    )
    clad = PolyPrism(
        _rect(-3.0, -2.0, L + 4.0, 6.0),
        {-1.0: 0.0, 0.0: 0.0},
        physical_name="cladding_below",
        mesh_order=20.0,
        identify_arcs=True,
    )
    clad_above = PolyPrism(
        _rect(-3.0, -2.0, L + 4.0, 6.0),
        {1.0: 0.0, 2.0: 0.0},
        physical_name="cladding_above",
        mesh_order=20.0,
        identify_arcs=True,
    )
    specs = {
        n: [StructuredExtrusionResolutionSpec(n_layers=2)]
        for n in ("meander_A", "meander_B")
    }
    generate_mesh(
        [A, B, clad, clad_above],
        dim=3,
        output_mesh=tmp_path / "out.msh",
        default_characteristic_length=0.5,
        resolution_specs=specs,
    )
    m = meshio.read(tmp_path / "out.msh")
    field = set(m.cell_sets.keys())
    assert {"meander_A", "meander_B", "cladding_below"}.issubset(field), field


def test_meander_meshes(meander_entities, tmp_path):
    generate_mesh(
        meander_entities,
        dim=3,
        output_mesh=tmp_path / "out.msh",
        default_characteristic_length=0.5,
        resolution_specs=_resolution_specs(),
    )
    m = meshio.read(tmp_path / "out.msh")
    wedges = sum(cb.data.shape[0] for cb in m.cells if cb.type == "wedge")
    tets = sum(cb.data.shape[0] for cb in m.cells if cb.type == "tetra")
    assert wedges > 0, "expected wedge elements from structured cohort"
    assert tets > 0, "expected tet elements from unstructured cladding"


def test_meander_physical_groups_present(meander_entities, tmp_path):
    generate_mesh(
        meander_entities,
        dim=3,
        output_mesh=tmp_path / "out.msh",
        default_characteristic_length=0.5,
        resolution_specs=_resolution_specs(),
    )
    m = meshio.read(tmp_path / "out.msh")
    field = set(m.cell_sets.keys())
    expected = {"meander_A", "meander_B", "embed", "cladding_below", "cladding_above"}
    missing = expected - field
    assert not missing, f"missing physical groups: {missing}"
    assert "void_A" not in field, "void_A should be carved away"
    assert "void_B" not in field, "void_B should be carved away"


def test_meander_face_registry_pre_bop_sharing(meander_entities, tmp_path):
    """Cladding pre-cut faces share TShape with cohort sub-piece faces pre-BOP.

    The FaceRegistry establishes ``TopoDS_Face`` TShape sharing between
    cohort sub-piece bot/top faces and adjacent unstructured PolyPrism
    touched-plane faces by construction. We verify the routing is correct
    by counting cohort cache hits during
    ``CohortNeighbourUnstructured.instanciate_occ``.

    NOTE: ``CohortNeighbourUnstructured.instanciate_occ`` then runs
    ``BRepAlgoAPI_Fuse`` over the per-tile prisms, which REBUILDS
    geometry and discards the shared TShapes downstream. As a result,
    the cohort↔unstructured AABB rescue count is unchanged from the
    pre-refactor baseline (6 on this scene). Eliminating that residual
    is a future refactor — would require replacing ``BRepAlgoAPI_Fuse``
    in ``instanciate_occ`` with a TShape-preserving combiner when the
    prisms are non-overlapping (which they are when the polygons came
    from the cohort arrangement pre-cut).
    """
    from meshwell.orchestrator import generate_mesh
    from meshwell.structured.build import FaceRegistry

    cohort_callsite_hits = 0
    neighbour_callsite_calls = 0
    neighbour_callsite_hits = 0
    original = FaceRegistry.face_xy

    def instrumented(
        self,
        polygon,
        z,
        identify_arcs,
        min_arc_points,
        arc_tolerance,
    ):
        import traceback

        nonlocal cohort_callsite_hits, neighbour_callsite_calls, neighbour_callsite_hits
        key = self.key_for_polygon(polygon, z)
        is_hit = key in self._store
        stack = traceback.extract_stack(limit=3)
        caller = stack[-2].filename
        result = original(
            self, polygon, z, identify_arcs, min_arc_points, arc_tolerance
        )
        if "cohort_neighbour" in caller:
            neighbour_callsite_calls += 1
            if is_hit:
                neighbour_callsite_hits += 1
        elif "build.py" in caller and is_hit:
            cohort_callsite_hits += 1
        return result

    FaceRegistry.face_xy = instrumented
    try:
        generate_mesh(
            meander_entities,
            dim=3,
            output_mesh=tmp_path / "out.msh",
            default_characteristic_length=0.5,
            resolution_specs=_resolution_specs(),
        )
    finally:
        FaceRegistry.face_xy = original

    # Every CohortNeighbourUnstructured face_xy call for a cohort-adjacent
    # cladding tile must hit a cohort-cached face (FaceRegistry's contract).
    assert neighbour_callsite_calls > 0, (
        "expected CohortNeighbourUnstructured.instanciate_occ to consult "
        "FaceRegistry for cohort-adjacent claddings; got 0 calls"
    )
    assert neighbour_callsite_hits == neighbour_callsite_calls, (
        f"every CohortNeighbourUnstructured face_xy call should hit the "
        f"cohort cache; got {neighbour_callsite_hits}/{neighbour_callsite_calls}"
    )


def test_meander_aabb_rescue_count_bounded(meander_entities, tmp_path):
    """Assert the AABB rescue count on the meander stress scene is bounded.

    After the CohortNeighbourUnstructured refactor + FaceRegistry routing,
    every PolyPrism.instanciate_occ call goes through face_xy and hits the
    cohort cache (verified by test_meander_face_registry_pre_bop_sharing).

    For SINGLE-TILE cladding tiles, MakePrism preserves the cached face's
    TShape and BOP merges by identity — no AABB rescue.

    For MULTI-TILE cladding (the meander scene falls here because the
    cladding gets split into 6 tiles by the arrangement), BRepAlgoAPI_Fuse
    inside build_neighbour_shell rebuilds inter-tile interfaces and
    discards the shared TShapes. The AABB fallback fires for each
    cohort↔cladding face pair that didn't survive Fuse.

    This test asserts the count is BOUNDED to the historical baseline.
    A future shell-assembly refactor that avoids Fuse (resolving the
    arc-PLC and multi-solid BOP issues) would drop the count to 0.
    """
    from collections import Counter
    from itertools import combinations

    import numpy as np

    import meshwell.occ_xao_writer as xao_mod
    from meshwell.orchestrator import generate_mesh

    rescues: Counter[str] = Counter()
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
            if xao_mod._is_purely_synthetic(e1) or xao_mod._is_purely_synthetic(e2):
                continue
            arr2 = np.array(list(eas[i2].values()), dtype=float)
            for b1 in eas[i1].values():
                b1_arr = np.asarray(b1, dtype=float)
                if np.any(np.abs(arr2 - b1_arr).max(axis=1) < interface_aabb_tolerance):
                    n1 = (
                        xao_mod._filter_real_names(e1.physical_name) or e1.physical_name
                    )
                    n2 = (
                        xao_mod._filter_real_names(e2.physical_name) or e2.physical_name
                    )
                    rescues["___".join(sorted((n1[0], n2[0])))] += 1
        return original(
            entities,
            interface_delimiter,
            boundary_delimiter,
            interface_aabb_tolerance=interface_aabb_tolerance,
        )

    xao_mod._compute_physical_groups = instrumented
    try:
        generate_mesh(
            meander_entities,
            dim=3,
            output_mesh=tmp_path / "out.msh",
            default_characteristic_length=0.5,
            resolution_specs=_resolution_specs(),
        )
    finally:
        xao_mod._compute_physical_groups = original

    # Historical baseline: 6 rescues at this scene under the
    # FaceRegistry refactor without custom shell assembly. If the
    # count INCREASES, something regressed. If it DECREASES, the
    # comment block above is out of date (good problem).
    total = sum(rescues.values())
    assert total <= 6, (
        f"meander AABB rescue count regressed: expected ≤6 (baseline), "
        f"got {total}: {dict(rescues)}"
    )
