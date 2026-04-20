"""Post-fragmentation topology canonicalization.

``BOPAlgo_Builder`` (run in :mod:`meshwell.cad_occ`) merges TShapes for
sub-shapes that fall within its Fuzzy tolerance. Anything outside that
tolerance — e.g. two adjacent OCC primitives whose shared vertices drift
by more than the fuzzy value — keeps distinct TShapes even though the
geometry is identical. gmsh then imports both and tetgen's 3D boundary
recovery flags the coincident faces as "overlapping facets" (we saw this
in ``gmsh/src/mesh/meshGRegionBoundaryRecovery.cpp:1343``).

Neither ``BRepBuilderAPI_Sewing`` (which only stitches *shell* edges) nor
``ShapeUpgrade_UnifySameDomain`` (which only unifies coplanar faces
*within* one solid) addresses cross-solid duplicates. This module fills
that gap with a bottom-up canonicalization:

1. Hash every sub-vertex across all entities by quantized coordinates.
   Pick one canonical TShape per coord bucket, substitute every other
   occurrence through a single :class:`BRepTools_ReShape`.
2. Re-walk sub-edges using the now-canonical vertex identities plus
   quantized length and curve type. Substitute duplicate edges.
3. Re-walk sub-faces using canonical edge boundaries plus surface type
   and area. Substitute duplicate faces.

Each pass cascades: after applying the vertex ReShape, every entity's
``shapes`` list holds new ``TopoDS_Shape`` handles whose vertices are
shared; the edge pass then operates on those. Same for faces.

The canonicalization leaves geometry bit-identical — it only rewires
TShape identity. Cost is ~O(total sub-shape count) per dim. Safe to run
unconditionally on small models; opt-in for large ones.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCP.BRepGProp import BRepGProp
from OCP.BRepTools import BRepTools_ReShape
from OCP.GProp import GProp_GProps
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_VERTEX
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS
from OCP.TopTools import TopTools_ShapeMapHasher

if TYPE_CHECKING:
    from OCP.TopoDS import TopoDS_Edge, TopoDS_Face, TopoDS_Shape, TopoDS_Vertex

    from meshwell.cad_occ import OCCLabeledEntity


_HASHER = TopTools_ShapeMapHasher()


def _shape_id(s: TopoDS_Shape) -> int:
    """TShape-pointer hash. Two handles to the same TShape compare equal."""
    return _HASHER(s)


def _ndigits_for(tol: float) -> int:
    """Coordinate-rounding digit count derived from a linear tolerance."""
    from math import floor, log10

    return max(0, int(-floor(log10(tol))))


def _vertex_key(v: TopoDS_Vertex, n: int) -> tuple[float, float, float]:
    p = BRep_Tool.Pnt_s(TopoDS.Vertex_s(v))
    return (round(p.X(), n), round(p.Y(), n), round(p.Z(), n))


def _edge_key(e: TopoDS_Edge, n: int, vert_canon_id: dict[int, int]) -> tuple:
    """Key combining canonical endpoint IDs, curve type, and quantized length.

    Endpoint canonical IDs come from the vertex pass — two edges with the
    same endpoints (regardless of traversal direction) and same curve kind
    and length should share a TShape.
    """
    ids: list[int] = []
    exp = TopExp_Explorer(e, TopAbs_VERTEX)
    while exp.More():
        v = exp.Current()
        vid = vert_canon_id.get(_shape_id(v), _shape_id(v))
        ids.append(vid)
        exp.Next()
    ids.sort()  # undirected endpoint set
    try:
        curve_type = int(BRepAdaptor_Curve(TopoDS.Edge_s(e)).GetType())
    except Exception:
        curve_type = -1
    props = GProp_GProps()
    BRepGProp.LinearProperties_s(e, props)
    length = round(props.Mass(), n)
    return (tuple(ids), curve_type, length)


def _face_key(f: TopoDS_Face, n: int, edge_canon_id: dict[int, int]) -> tuple:
    """Key combining canonical boundary edge IDs, surface type, and area."""
    ids: list[int] = []
    exp = TopExp_Explorer(f, TopAbs_EDGE)
    while exp.More():
        e = exp.Current()
        eid = edge_canon_id.get(_shape_id(e), _shape_id(e))
        ids.append(eid)
        exp.Next()
    ids.sort()
    try:
        surf_type = int(BRepAdaptor_Surface(TopoDS.Face_s(f)).GetType())
    except Exception:
        surf_type = -1
    props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(f, props)
    # ``Mass()`` is signed for faces: +1 for outward normal, -1 for inward.
    # Two volumes sharing an interface traverse the same surface with opposite
    # orientations, so take abs to bucket them together.
    area = round(abs(props.Mass()), n)
    return (tuple(ids), surf_type, area)


def _iter_subshapes(shape: TopoDS_Shape, topabs):
    exp = TopExp_Explorer(shape, topabs)
    while exp.More():
        yield exp.Current()
        exp.Next()


def _canonicalize_pass(
    entities: list[OCCLabeledEntity],
    topabs,
    key_fn,
) -> tuple[dict[int, int], int]:
    """One bottom-up pass over sub-shapes of a given TopAbs class.

    Returns (canonical_id_map, num_substituted). ``canonical_id_map`` maps
    every seen TShape hash → its chosen canonical TShape hash. The caller
    uses this map as input when hashing the next (higher) TopAbs class,
    so an edge that previously looked "different" because its endpoints
    had different TShapes now keys on the shared canonical endpoints.
    """
    canon_by_key: dict[tuple, TopoDS_Shape] = {}
    canon_id: dict[int, int] = {}
    duplicates: list[tuple[TopoDS_Shape, TopoDS_Shape]] = []

    for ent in entities:
        for s in ent.shapes:
            for sub in _iter_subshapes(s, topabs):
                k = key_fn(sub)
                existing = canon_by_key.get(k)
                if existing is None:
                    canon_by_key[k] = sub
                    canon_id[_shape_id(sub)] = _shape_id(sub)
                else:
                    if _shape_id(sub) != _shape_id(existing):
                        duplicates.append((sub, existing))
                        canon_id[_shape_id(sub)] = _shape_id(existing)
                    else:
                        canon_id[_shape_id(sub)] = _shape_id(existing)

    if not duplicates:
        return canon_id, 0

    reshape = BRepTools_ReShape()
    seen_replacements: set[int] = set()
    for old, new in duplicates:
        oid = _shape_id(old)
        if oid in seen_replacements:
            continue
        seen_replacements.add(oid)
        reshape.Replace(old, new)

    for ent in entities:
        ent.shapes = [reshape.Apply(s) for s in ent.shapes]

    return canon_id, len(seen_replacements)


def canonicalize_topology(
    entities: list[OCCLabeledEntity],
    point_tolerance: float,
) -> dict[str, int]:
    """Unify TShapes across entities for coincident sub-vertices/edges/faces.

    Runs vertex → edge → face passes, each cascading its canonical-ID
    map into the next level's hash so identity propagates bottom-up.

    Returns a small stats dict: ``{"vertices": k, "edges": k, "faces": k}``
    counting how many TShape substitutions each pass made. Useful for
    debug/logging but callers can ignore it.
    """
    if not entities:
        return {"vertices": 0, "edges": 0, "faces": 0}

    n = _ndigits_for(point_tolerance)

    v_canon, n_v = _canonicalize_pass(
        entities,
        TopAbs_VERTEX,
        key_fn=lambda s: _vertex_key(s, n),
    )
    e_canon, n_e = _canonicalize_pass(
        entities,
        TopAbs_EDGE,
        key_fn=lambda s: _edge_key(s, n, v_canon),
    )
    _, n_f = _canonicalize_pass(
        entities,
        TopAbs_FACE,
        key_fn=lambda s: _face_key(s, n, e_canon),
    )
    return {"vertices": n_v, "edges": n_e, "faces": n_f}
