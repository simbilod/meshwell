"""assemble_cohort_sub_prism produces a valid solid + populated PhantomShape."""

from __future__ import annotations

import shapely
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.TopAbs import TopAbs_FACE
from OCP.TopExp import TopExp_Explorer

from meshwell.polyprism import PolyPrism
from meshwell.structured import StructuredExtrusionResolutionSpec
from meshwell.structured.cohort_topology import (
    assemble_cohort_sub_prism,
    build_cohort_topology,
)
from meshwell.structured.plan import build_plan


def _square(x0, y0, x1, y1):
    return shapely.Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


def _polyprism(name, z0, z1, mesh_order, x0=0, y0=0, x1=1, y1=1):
    return PolyPrism(
        polygons=_square(x0, y0, x1, y1),
        buffers={float(z0): 0.0, float(z1): 0.0},
        physical_name=name,
        mesh_order=mesh_order,
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
    )


def _faces(shape):
    out = []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        out.append(exp.Current())
        exp.Next()
    return out


def test_assemble_produces_valid_solid():
    """Two vertically-stacked PolyPrisms produce a proper 4-edge arrangement.

    A single isolated polygon produces a one-edge arrangement (degenerate for
    assembly), so we use two stacked same-footprint entities to get proper
    per-side arrangement edges.
    """
    plan = build_plan([_polyprism("A", 0, 1, 1), _polyprism("B", 1, 2, 2)])
    topology = build_cohort_topology(plan, component_index=0)
    slab = next(s for s in plan.slabs if s.physical_name == ("A",))
    ps = assemble_cohort_sub_prism(topology, slab, piece_index=0)
    analyzer = BRepCheck_Analyzer(ps.solid)
    assert analyzer.IsValid(), "Assembled solid failed BRepCheck"


def test_assembled_solid_has_six_faces_for_square_cohort():
    """Assembled square sub-prism has 6 faces: 1 bot + 1 top + 4 laterals.

    Uses two vertically-stacked same-footprint PolyPrisms so the arrangement
    produces 4 separate line edges (one per side), giving 4 lateral faces.
    """
    plan = build_plan([_polyprism("A", 0, 1, 1), _polyprism("B", 1, 2, 2)])
    topology = build_cohort_topology(plan, component_index=0)
    slab = next(s for s in plan.slabs if s.physical_name == ("A",))
    ps = assemble_cohort_sub_prism(topology, slab, piece_index=0)
    assert len(_faces(ps.solid)) == 6  # 1 bot + 1 top + 4 laterals


def test_input_faces_by_key_uses_registry():
    """input_faces_by_key references the same TShapes as the horizontal_faces registry."""
    plan = build_plan([_polyprism("A", 0, 1, 1), _polyprism("B", 1, 2, 2)])
    topology = build_cohort_topology(plan, component_index=0)
    slab = next(s for s in plan.slabs if s.physical_name == ("A",))
    ps = assemble_cohort_sub_prism(topology, slab, piece_index=0)
    registry_face_hashes = {hash(f) for f in topology.horizontal_faces.values()}
    for face_key, face in ps.input_faces_by_key.items():
        if face_key.side in ("bot", "top"):
            assert hash(face) in registry_face_hashes


def test_lateral_neighbors_share_interface_face_tshape():
    """Two laterally-adjacent PolyPrisms in the same cohort -> assembled sub-prisms share the interface lateral face TShape.

    The seam lateral face is registered once per z-interval and reused for
    both adjacent slabs, so their assembled solids reference the same TShape.
    """
    A = _polyprism("A", 0, 1, 1, x0=0, y0=0, x1=1, y1=1)
    B = _polyprism("B", 0, 1, 2, x0=1, y0=0, x1=2, y1=1)  # adjacent at x=1
    plan = build_plan([A, B])
    topology = build_cohort_topology(plan, component_index=0)
    slab_A = next(s for s in plan.slabs if s.physical_name == ("A",))
    slab_B = next(s for s in plan.slabs if s.physical_name == ("B",))
    ps_A = assemble_cohort_sub_prism(topology, slab_A, piece_index=0)
    ps_B = assemble_cohort_sub_prism(topology, slab_B, piece_index=0)
    a_faces = {hash(f) for f in _faces(ps_A.solid)}
    b_faces = {hash(f) for f in _faces(ps_B.solid)}
    assert a_faces & b_faces, (
        "Laterally-adjacent assembled sub-prisms do not share interface " "face TShape."
    )
