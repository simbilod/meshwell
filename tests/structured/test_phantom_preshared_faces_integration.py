"""End-to-end: build_phantom_shapes produces shared TShapes between vertically stacked sub-prisms."""

from __future__ import annotations

import shapely
from OCP.TopAbs import TopAbs_FACE
from OCP.TopExp import TopExp_Explorer

from meshwell.polyprism import PolyPrism
from meshwell.structured import StructuredExtrusionResolutionSpec
from meshwell.structured.phantom import build_phantom_shapes
from meshwell.structured.plan import build_plan


def _square(x0, y0, x1, y1):
    return shapely.Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


def _faces(shape):
    out = []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        out.append(exp.Current())
        exp.Next()
    return out


def test_stacked_sub_prisms_share_interface_face_tshape():
    """Two vertically-stacked PolyPrisms (same XY, touching in z) -> shared TShape."""
    A = PolyPrism(
        polygons=_square(0, 0, 1, 1),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="A",
        mesh_order=1,
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
    )
    B = PolyPrism(
        polygons=_square(0, 0, 1, 1),
        buffers={1.0: 0.0, 2.0: 0.0},
        physical_name="B",
        mesh_order=2,
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
    )
    plan = build_plan([A, B])
    result = build_phantom_shapes(plan)
    by_slab = {ps.slab_index: ps for ps in result.shapes}
    assert len(by_slab) == 2
    a_faces = {hash(f) for f in _faces(by_slab[0].solid)}
    b_faces = {hash(f) for f in _faces(by_slab[1].solid)}
    assert a_faces & b_faces, (
        "Vertically-stacked sub-prisms do NOT share interface face TShape "
        "after build_phantom_shapes — pre-sharing not active."
    )
