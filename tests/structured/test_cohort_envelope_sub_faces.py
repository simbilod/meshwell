"""Per-piece top/bot OCC sub-face registry tests."""

from __future__ import annotations

import shapely

from meshwell.polyprism import PolyPrism
from meshwell.structured import StructuredExtrusionResolutionSpec
from meshwell.structured.cohort_envelope import build_cohort_envelope
from meshwell.structured.plan import build_plan
from meshwell.structured.spec import FaceKey, PieceLineEdge, PieceProvenance


def _square_slab(zlo, zhi, name, side=1.0):
    return PolyPrism(
        polygons=shapely.box(0, 0, side, side),
        buffers={zlo: 0.0, zhi: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        physical_name=name,
    )


def test_sub_faces_indexed_by_face_key():
    """Two stacked single-piece slabs: 2 top + 2 bot sub-faces."""
    plan = build_plan([_square_slab(0.0, 1.0, "L1"), _square_slab(1.0, 2.0, "L2")])

    # Manually set face_partition_provenance for each slab
    # (in normal usage, build_plan would populate this from arc detection)
    for slab in plan.slabs:
        # Create simple line-edge provenance for the square boundary
        coords = list(slab.face_partition[0].exterior.coords)[
            :-1
        ]  # strip closing duplicate
        edges = []
        for i in range(len(coords)):
            p1 = (coords[i][0], coords[i][1], 0)
            p2 = (coords[(i + 1) % len(coords)][0], coords[(i + 1) % len(coords)][1], 0)
            edges.append(PieceLineEdge(points=(p1, p2)))
        slab.face_partition_provenance = [
            PieceProvenance(exterior_edges=edges, interior_edges=[])
        ]

    env = build_cohort_envelope(plan, component_index=0)
    assert FaceKey(0, "top", 0) in env.top_sub_faces
    assert FaceKey(1, "top", 0) in env.top_sub_faces
    assert FaceKey(0, "bot", 0) in env.bottom_sub_faces
    assert FaceKey(1, "bot", 0) in env.bottom_sub_faces
    assert len(env.top_sub_faces) == 2
    assert len(env.bottom_sub_faces) == 2


def test_sub_face_is_planar_at_correct_z():
    """Top sub-face is planar at z=zhi; bottom at z=zlo."""
    from OCP.BRep import BRep_Tool
    from OCP.TopAbs import TopAbs_VERTEX
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    plan = build_plan([_square_slab(0.0, 1.0, "L1")])

    # Manually set face_partition_provenance
    slab = plan.slabs[0]
    coords = list(slab.face_partition[0].exterior.coords)[
        :-1
    ]  # strip closing duplicate
    edges = []
    for i in range(len(coords)):
        p1 = (coords[i][0], coords[i][1], 0)
        p2 = (coords[(i + 1) % len(coords)][0], coords[(i + 1) % len(coords)][1], 0)
        edges.append(PieceLineEdge(points=(p1, p2)))
    slab.face_partition_provenance = [
        PieceProvenance(exterior_edges=edges, interior_edges=[])
    ]

    env = build_cohort_envelope(plan, component_index=0)
    top = env.top_sub_faces[FaceKey(0, "top", 0)]
    exp = TopExp_Explorer(top, TopAbs_VERTEX)
    while exp.More():
        v = TopoDS.Vertex_s(exp.Current())
        z = BRep_Tool.Pnt_s(v).Z()
        assert abs(z - 1.0) < 1e-9
        exp.Next()
