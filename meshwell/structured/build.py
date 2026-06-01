"""Stage 4 — assemble cohort TopoDS_Compound of N sub-solids.

Bottom-up build: unique vertices → unique edges (with arc detection)
→ unique faces (horizontal interior/boundary, lateral) → per-subpiece
TopoDS_Solid → TopoDS_Compound per cohort.

Shared TShapes by CONSTRUCTION (not post-hoc sewing): every face and
edge is built once and referenced by every solid that needs it. This
is what makes cohort internal interfaces conformal without BOP.
"""
from __future__ import annotations

from dataclasses import dataclass

from OCP.BRep import BRep_Builder
from OCP.BRepBuilderAPI import (
    BRepBuilderAPI_MakeEdge,
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_MakeVertex,
    BRepBuilderAPI_MakeWire,
)
from OCP.GC import GC_MakeArcOfCircle
from OCP.gp import gp_Pnt
from OCP.TopoDS import (
    TopoDS_Compound,
    TopoDS_Edge,
    TopoDS_Face,
    TopoDS_Shell,
    TopoDS_Solid,
    TopoDS_Vertex,
)

from meshwell.geometry_entity import decompose_vertices_2d
from meshwell.structured.types import (
    Cohort,
    ShapeKey,
    SlabMeta,
    StructuredSlab,
    SubPiece,
)


@dataclass
class VertexRegistry:
    """Snap-and-dedup vertex store.

    Coordinates are quantized to `point_tolerance` so near-coincident
    vertices map to the same TopoDS_Vertex.
    """

    point_tolerance: float

    def __post_init__(self):
        """Initialise the internal vertex store."""
        self._store: dict[tuple[int, int, int], TopoDS_Vertex] = {}

    def _key(self, x: float, y: float, z: float) -> tuple[int, int, int]:
        s = self.point_tolerance
        return (round(x / s), round(y / s), round(z / s))

    def get_or_create(self, x: float, y: float, z: float) -> TopoDS_Vertex:
        """Return the unique vertex at (x, y, z), creating it if necessary."""
        k = self._key(x, y, z)
        if k not in self._store:
            self._store[k] = BRepBuilderAPI_MakeVertex(gp_Pnt(x, y, z)).Vertex()
        return self._store[k]

    def __len__(self):
        """Return the number of unique vertices in the registry."""
        return len(self._store)


@dataclass
class EdgeRegistry:
    """Unique edge store with arc detection.

    Two flavours:
      - polyline_xy: a 2D polyline at fixed z; runs of vertices on a
        circle (when identify_arcs) build a GC_MakeArcOfCircle edge.
      - vertical: a single edge between two z values at one (x,y).
    """

    vertices: VertexRegistry
    point_tolerance: float

    def __post_init__(self):
        """Initialise the internal edge store."""
        self._store: dict[tuple, TopoDS_Edge] = {}

    def vertical(self, x: float, y: float, z_a: float, z_b: float) -> TopoDS_Edge:
        """Return the unique vertical edge at (x, y) between z_a and z_b."""
        a = self.vertices.get_or_create(x, y, z_a)
        b = self.vertices.get_or_create(x, y, z_b)
        key = ("V", self.vertices._key(x, y, z_a), self.vertices._key(x, y, z_b))
        if key not in self._store:
            self._store[key] = BRepBuilderAPI_MakeEdge(a, b).Edge()
        return self._store[key]

    def line_xy(
        self, x1: float, y1: float, x2: float, y2: float, z: float
    ) -> TopoDS_Edge:
        """Return the unique straight edge between (x1,y1,z) and (x2,y2,z)."""
        a = self.vertices.get_or_create(x1, y1, z)
        b = self.vertices.get_or_create(x2, y2, z)
        k_a = self.vertices._key(x1, y1, z)
        k_b = self.vertices._key(x2, y2, z)
        key = ("L", tuple(sorted([k_a, k_b])))
        if key not in self._store:
            self._store[key] = BRepBuilderAPI_MakeEdge(a, b).Edge()
        return self._store[key]

    def arc_xy(
        self,
        start: tuple[float, float],
        mid: tuple[float, float],
        end: tuple[float, float],
        z: float,
    ) -> TopoDS_Edge:
        """Return a unique arc edge through start, mid, end at height z."""
        sv = self.vertices.get_or_create(*start, z)
        ev = self.vertices.get_or_create(*end, z)
        k_s = self.vertices._key(*start, z)
        k_m = self.vertices._key(*mid, z)
        k_e = self.vertices._key(*end, z)
        key = ("A", k_s, k_m, k_e)
        if key not in self._store:
            # Guard: GC_MakeArcOfCircle cannot handle a full circle (start==end).
            # Caller should split the circle into sub-arcs before calling here.
            p_start = gp_Pnt(start[0], start[1], z)
            p_mid = gp_Pnt(mid[0], mid[1], z)
            p_end = gp_Pnt(end[0], end[1], z)
            builder = GC_MakeArcOfCircle(p_start, p_mid, p_end)
            if not builder.IsDone():
                raise ValueError(
                    f"GC_MakeArcOfCircle failed for start={start} mid={mid} end={end}. "
                    "Full-circle arcs must be split into two half-arcs before calling arc_xy."
                )
            self._store[key] = BRepBuilderAPI_MakeEdge(builder.Value(), sv, ev).Edge()
        return self._store[key]

    def polyline_xy(
        self,
        coords: list[tuple[float, float]],
        z: float,
        identify_arcs: bool,
        min_arc_points: int = 5,
        arc_tolerance: float = 1e-3,
    ) -> list[TopoDS_Edge]:
        """Return the list of edges (lines and/or arcs) covering coords.

        Uses ``meshwell.geometry_entity.decompose_vertices_2d`` as the
        canonical arc detector so the structured cohort builder and the
        legacy ``PolyPrism``/``PolySurface`` paths agree on segmentation.
        Both paths feeding into the same BOP fragment pass produces the
        same OCC edges, so coincident-but-topologically-distinct curves
        no longer fragment shared faces.
        """
        segments = decompose_vertices_2d(
            coords,
            z=z,
            point_tolerance=self.point_tolerance,
            identify_arcs=identify_arcs,
            min_arc_points=min_arc_points,
            arc_tolerance=arc_tolerance,
        )
        edges: list[TopoDS_Edge] = []
        for seg in segments:
            pts = seg.points
            if seg.is_arc:
                start = pts[0]
                end = pts[-1]
                is_closed = self.vertices._key(
                    start[0], start[1], z
                ) == self.vertices._key(end[0], end[1], z)
                if is_closed:
                    # Full-circle: GC_MakeArcOfCircle can't build start==end.
                    # Split into two half-arcs using the points at indices
                    # 1/4, 1/2, 3/4 of the run as the parametric anchors.
                    mid_idx = len(pts) // 2
                    quarter_idx = len(pts) // 4
                    three_quarter_idx = (len(pts) * 3) // 4
                    edges.append(
                        self.arc_xy(
                            (pts[0][0], pts[0][1]),
                            (pts[quarter_idx][0], pts[quarter_idx][1]),
                            (pts[mid_idx][0], pts[mid_idx][1]),
                            z,
                        )
                    )
                    edges.append(
                        self.arc_xy(
                            (pts[mid_idx][0], pts[mid_idx][1]),
                            (pts[three_quarter_idx][0], pts[three_quarter_idx][1]),
                            (pts[-1][0], pts[-1][1]),
                            z,
                        )
                    )
                else:
                    mid_idx = len(pts) // 2
                    edges.append(
                        self.arc_xy(
                            (pts[0][0], pts[0][1]),
                            (pts[mid_idx][0], pts[mid_idx][1]),
                            (pts[-1][0], pts[-1][1]),
                            z,
                        )
                    )
            else:
                # ``_decompose_vertices_3d`` always emits straight runs as
                # individual two-point segments, but tolerate longer runs
                # defensively by stitching consecutive pairs into edges.
                edges.extend(
                    self.line_xy(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1], z)
                    for i in range(len(pts) - 1)
                )
        return edges


@dataclass(frozen=True)
class _PolylineSegment:
    """One segment of a 2D polyline: a straight line or a circular arc.

    ``kind`` is either ``"line"`` (mid is unused) or ``"arc"`` (mid is
    a point lying on the arc between start and end).

    This is the lateral-face-builder's local view of a polyline segment;
    ``polyline_segments`` adapts the canonical ``DecompositionSegment``
    output of ``decompose_vertices_2d`` into this 2D-flat form so that
    the lateral builder code can stay simple.
    """

    kind: str
    start: tuple[float, float]
    end: tuple[float, float]
    mid: tuple[float, float] | None = None


def polyline_segments(
    coords: list[tuple[float, float]],
    identify_arcs: bool,
    min_arc_points: int,
    arc_tolerance: float,
    point_tolerance: float,
) -> list[_PolylineSegment]:
    """Decompose a 2D polyline into line and arc segments.

    Thin adapter over ``meshwell.geometry_entity.decompose_vertices_2d``
    that flattens the canonical 3D ``DecompositionSegment`` form into the
    lateral-builder's ``_PolylineSegment`` form. Routing both the
    horizontal-face (``polyline_xy``) and lateral-face callers through
    the same canonical detector keeps the bot/top boundary of every
    lateral matching the adjacent horizontal face's edge by construction.
    """
    raw = decompose_vertices_2d(
        coords,
        z=0.0,  # z is irrelevant — we only need (x, y) downstream
        point_tolerance=point_tolerance,
        identify_arcs=identify_arcs,
        min_arc_points=min_arc_points,
        arc_tolerance=arc_tolerance,
    )
    out: list[_PolylineSegment] = []
    for seg in raw:
        pts = seg.points
        if seg.is_arc:
            start_xy = (pts[0][0], pts[0][1])
            end_xy = (pts[-1][0], pts[-1][1])
            mid_idx = len(pts) // 2
            # Full-circle: split into two half-arcs so each maps to a
            # well-defined ``GC_MakeArcOfCircle`` build downstream.
            if (
                abs(start_xy[0] - end_xy[0]) < point_tolerance
                and abs(start_xy[1] - end_xy[1]) < point_tolerance
            ):
                q1_idx = mid_idx // 2
                q3_idx = (mid_idx + len(pts) - 1) // 2
                mid_xy = (pts[mid_idx][0], pts[mid_idx][1])
                out.append(
                    _PolylineSegment(
                        kind="arc",
                        start=start_xy,
                        end=mid_xy,
                        mid=(pts[q1_idx][0], pts[q1_idx][1]),
                    )
                )
                out.append(
                    _PolylineSegment(
                        kind="arc",
                        start=mid_xy,
                        end=end_xy,
                        mid=(pts[q3_idx][0], pts[q3_idx][1]),
                    )
                )
            else:
                out.append(
                    _PolylineSegment(
                        kind="arc",
                        start=start_xy,
                        end=end_xy,
                        mid=(pts[mid_idx][0], pts[mid_idx][1]),
                    )
                )
        else:
            out.extend(
                _PolylineSegment(
                    kind="line",
                    start=(pts[i][0], pts[i][1]),
                    end=(pts[i + 1][0], pts[i + 1][1]),
                )
                for i in range(len(pts) - 1)
            )
    return out


# ---------------------------------------------------------------------------
# Stage 4 parts 3-6: face registry + sub-solid + cohort compound assembly
# ---------------------------------------------------------------------------


def _shape_key(shape) -> ShapeKey:
    from OCP.TopTools import TopTools_ShapeMapHasher

    hasher = TopTools_ShapeMapHasher()
    return ShapeKey(tshape_id=hasher(shape), orientation=int(shape.Orientation()))


def _ring_coords(ring) -> list[tuple[float, float]]:
    return list(ring.coords)


def _ring_coords_ccw(ring) -> list[tuple[float, float]]:
    """Return ring coords ensuring CCW orientation (positive signed area).

    OCC ``BRepBuilderAPI_MakeFace`` expects the outer wire to be CCW when
    viewed from the face normal (+Z for a horizontal face).
    """
    coords = list(ring.coords)
    # Signed area using the shoelace formula.  Positive => CCW.
    n = len(coords)
    area2 = sum(
        coords[i][0] * coords[(i + 1) % n][1] - coords[(i + 1) % n][0] * coords[i][1]
        for i in range(n)
    )
    if area2 < 0:
        coords = list(reversed(coords))
    return coords


def _ring_coords_cw(ring) -> list[tuple[float, float]]:
    """Return ring coords ensuring CW orientation (negative signed area).

    OCC ``BRepBuilderAPI_MakeFace.Add(hole_wire)`` expects the hole wire
    to be CW when viewed from the face normal (+Z for a horizontal face).
    """
    coords = list(ring.coords)
    n = len(coords)
    area2 = sum(
        coords[i][0] * coords[(i + 1) % n][1] - coords[(i + 1) % n][0] * coords[i][1]
        for i in range(n)
    )
    if area2 > 0:
        coords = list(reversed(coords))
    return coords


def _build_horizontal_face(
    polygon,
    z: float,
    ereg: EdgeRegistry,
    identify_arcs: bool,
    min_arc_points: int,
    arc_tolerance: float,
) -> TopoDS_Face:
    """Build a horizontal TopoDS_Face for a polygon at fixed z."""
    outer_coords = _ring_coords(polygon.exterior)
    outer_edges = ereg.polyline_xy(
        outer_coords,
        z,
        identify_arcs,
        min_arc_points,
        arc_tolerance,
    )
    mw = BRepBuilderAPI_MakeWire()
    for e in outer_edges:
        mw.Add(e)
    outer_wire = mw.Wire()
    mf = BRepBuilderAPI_MakeFace(outer_wire)
    for interior in polygon.interiors:
        hole_coords = _ring_coords(interior)
        hole_edges = ereg.polyline_xy(
            hole_coords,
            z,
            identify_arcs,
            min_arc_points,
            arc_tolerance,
        )
        mw_h = BRepBuilderAPI_MakeWire()
        for e in hole_edges:
            mw_h.Add(e)
        mf.Add(mw_h.Wire())
    return mf.Face()


def _is_arc_edge(edge: "TopoDS_Edge") -> bool:
    """Return True iff the edge's underlying curve is a circular arc."""
    from OCP.BRep import BRep_Tool
    from OCP.GeomAbs import GeomAbs_Circle
    from OCP.GeomAdaptor import GeomAdaptor_Curve

    fp, lp = 0.0, 1.0
    curv = BRep_Tool.Curve_s(edge, fp, lp)
    if curv is None:
        return False
    try:
        adaptor = GeomAdaptor_Curve(curv)
        return adaptor.GetType() == GeomAbs_Circle
    except Exception:
        return False


def _build_lateral_face(
    edge_xy_low: "TopoDS_Edge",
    edge_xy_high: "TopoDS_Edge",
    v_left: "TopoDS_Edge",
    v_right: "TopoDS_Edge",
) -> "TopoDS_Face":
    """Stitch four edges into a lateral face.

    For straight edges the face is planar.  For arc edges (circular
    curves in the XY-plane) the face is built on the matching
    ``Geom_CylindricalSurface`` so that the OCC BRep is topologically
    closed and passes BRepCheck validation.
    """
    if _is_arc_edge(edge_xy_low):
        return _build_cylindrical_lateral_face(
            edge_xy_low, edge_xy_high, v_left, v_right
        )
    mw = BRepBuilderAPI_MakeWire()
    mw.Add(edge_xy_low)
    mw.Add(v_right)
    mw.Add(edge_xy_high)
    mw.Add(v_left)
    wire = mw.Wire()
    return BRepBuilderAPI_MakeFace(wire).Face()


def _build_cylindrical_lateral_face(
    edge_low: "TopoDS_Edge",
    edge_high: "TopoDS_Edge",
    v_left: "TopoDS_Edge",
    v_right: "TopoDS_Edge",
) -> "TopoDS_Face":
    """Build a cylindrical lateral face on an explicit ``Geom_CylindricalSurface``.

    The four input edges (two arc XY edges at z=z_low/z_high plus two
    vertical edges at the arc endpoints) are reused as-is — their TShapes
    remain shared with neighbouring sub-pieces and with the horizontal
    faces above/below. We construct a cylinder whose axis is vertical
    through the arc's circle center and whose radius matches the arc's,
    build a wire from the four edges, then build the face on that surface
    with parametric (PCurve) representations added for every edge.

    Building the face on an explicit canonical surface (rather than letting
    ``BRepOffsetAPI_ThruSections`` infer a ruled surface from the wire)
    prevents ``BOPAlgo_Builder`` from later splitting the face into multiple
    fragments at the seam line on the cylinder, which is what caused the
    "5 boundary edges" failure for the structured transfinite hint.
    """
    from OCP.BRep import BRep_Tool
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeWire
    from OCP.BRepOffsetAPI import BRepOffsetAPI_ThruSections
    from OCP.Geom import Geom_CylindricalSurface
    from OCP.GeomAbs import GeomAbs_Circle
    from OCP.GeomAdaptor import GeomAdaptor_Curve
    from OCP.gp import gp_Ax3, gp_Dir, gp_Pnt
    from OCP.ShapeFix import ShapeFix_Edge
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE
    from OCP.TopExp import TopExp, TopExp_Explorer
    from OCP.TopoDS import TopoDS

    def _ruled_fallback() -> "TopoDS_Face":
        mw_lo = BRepBuilderAPI_MakeWire()
        mw_lo.Add(edge_low)
        mw_hi = BRepBuilderAPI_MakeWire()
        mw_hi.Add(edge_high)
        tss = BRepOffsetAPI_ThruSections(False, True)
        tss.AddWire(mw_lo.Wire())
        tss.AddWire(mw_hi.Wire())
        tss.Build()
        if tss.IsDone():
            exp = TopExp_Explorer(tss.Shape(), TopAbs_FACE)
            if exp.More():
                return TopoDS.Face_s(exp.Current())
        mw = BRepBuilderAPI_MakeWire()
        mw.Add(edge_low)
        mw.Add(v_right)
        mw.Add(edge_high)
        mw.Add(v_left)
        return BRepBuilderAPI_MakeFace(mw.Wire()).Face()

    # Extract the underlying gp_Circ from the low arc edge.
    fp, lp = 0.0, 1.0
    curv_handle = BRep_Tool.Curve_s(edge_low, fp, lp)
    if curv_handle is None:
        return _ruled_fallback()
    try:
        adaptor = GeomAdaptor_Curve(curv_handle)
        if adaptor.GetType() != GeomAbs_Circle:
            return _ruled_fallback()
        gp_circ = adaptor.Circle()
    except Exception:
        return _ruled_fallback()

    # z of the bot edge (vertical cylinder origin).
    v_bot_first = TopExp.FirstVertex_s(edge_low)
    bot_z = BRep_Tool.Pnt_s(v_bot_first).Z()

    # Construct a vertical cylindrical surface through the arc's axis
    # location with the arc's radius. We orient the cylinder's X-axis (its
    # u=0 reference direction) to point at the arc's start vertex.  This
    # guarantees the PCurve of the arc on this surface has u starting at 0
    # and lying in [0, 2π) rather than crossing the u=0/u=2π seam of the
    # cylinder's intrinsic parameterization — which is what triggered
    # "BRepCheck_UnorientableShape" for the second half-arc of a closed
    # circle (its arc spans π → 2π on a cylinder whose +X reference points
    # at u=0).
    center = gp_circ.Location()
    start_pt = BRep_Tool.Pnt_s(v_bot_first)
    dx = start_pt.X() - center.X()
    dy = start_pt.Y() - center.Y()
    norm = (dx * dx + dy * dy) ** 0.5
    # Degenerate (start coincident with center) shouldn't happen; fall
    # back to global +X.
    x_dir = gp_Dir(1, 0, 0) if norm < 1e-12 else gp_Dir(dx / norm, dy / norm, 0.0)
    cyl_axis = gp_Ax3(
        gp_Pnt(center.X(), center.Y(), bot_z),
        gp_Dir(0, 0, 1),
        x_dir,
    )
    radius = gp_circ.Radius()
    surf = Geom_CylindricalSurface(cyl_axis, radius)

    # Build the closing wire from the 4 existing edges, traversing CCW in the
    # face's UV space. For both arc orientations (CCW outer, CW hole) we try
    # one orientation first and fall back to the reverse if BRepBuilderAPI
    # complains.
    def _try_wire(wire_edges) -> "TopoDS_Face | None":
        from OCP.BRepCheck import BRepCheck_Analyzer

        mw = BRepBuilderAPI_MakeWire()
        for e in wire_edges:
            mw.Add(e)
        if not mw.IsDone():
            return None
        wire = mw.Wire()
        mf = BRepBuilderAPI_MakeFace(surf, wire, True)
        if not mf.IsDone():
            return None
        face = mf.Face()
        # Build PCurves for every edge on the cylindrical surface.
        sfe = ShapeFix_Edge()
        exp = TopExp_Explorer(face, TopAbs_EDGE)
        while exp.More():
            e = TopoDS.Edge_s(exp.Current())
            # FixAddPCurve returning False is non-fatal: a PCurve may
            # already exist on the edge for this face.
            sfe.FixAddPCurve(e, face, False, 1e-7)
            exp.Next()
        # Reject "unorientable" faces — these arise when the wire's PCurves
        # straddle the cylinder's u=0/u=2π seam.  The caller will try the
        # reversed wire orientation next, then fall back to a ruled surface.
        if not BRepCheck_Analyzer(face).IsValid():
            return None
        return face

    # Order 1: bot forward, right up, top reversed, left down.
    face = _try_wire(
        [
            edge_low,
            v_right,
            TopoDS.Edge_s(edge_high.Reversed()),
            TopoDS.Edge_s(v_left.Reversed()),
        ]
    )
    if face is None:
        # Order 2: bot reversed, left up, top forward, right down.
        face = _try_wire(
            [
                TopoDS.Edge_s(edge_low.Reversed()),
                v_left,
                edge_high,
                TopoDS.Edge_s(v_right.Reversed()),
            ]
        )
    if face is None:
        return _ruled_fallback()
    return face


def build_cohort_compound(
    cohort: Cohort,
    subpieces: list[SubPiece],
    point_tolerance: float,
) -> tuple[TopoDS_Compound, dict[ShapeKey, SlabMeta]]:
    """Stage 4 driver — assemble compound + slab_meta.

    The compound is one TopoDS_Compound containing N sub-solids, where
    N == len(subpieces). Faces and edges shared between sub-solids are
    constructed once and referenced from both — guaranteeing shared
    TShapes without BOP.

    Arc-detection conformality: when two subpieces from adjacent z-levels
    share a z-plane, one source slab may have ``identify_arcs=True`` while
    the other does not.  We propagate ``identify_arcs=True`` across shared
    z-planes so that BOTH faces at the interface use the same arc
    representation.
    """
    vreg = VertexRegistry(point_tolerance=point_tolerance)
    ereg = EdgeRegistry(vertices=vreg, point_tolerance=point_tolerance)

    slab_by_source: dict[int, StructuredSlab] = {
        s.source_index: s for s in cohort.slabs
    }

    # Pre-compute, for each z-plane in the cohort, whether ANY source slab
    # active AT that plane (either touching from below or from above) has
    # identify_arcs=True.  A subpiece's bottom face uses the z=zlo plane
    # setting and its top face uses the z=zhi plane setting.
    z_plane_id_arcs: dict[float, bool] = {}
    z_plane_min_arc_pts: dict[float, int] = {}
    z_plane_arc_tol: dict[float, float] = {}
    for slab in cohort.slabs:
        for z in (slab.zlo, slab.zhi):
            if slab.identify_arcs:
                z_plane_id_arcs[z] = True
                # Use the most sensitive (smallest) arc_tolerance and
                # largest min_arc_points when multiple slabs contribute.
                if z not in z_plane_arc_tol or slab.arc_tolerance < z_plane_arc_tol[z]:
                    z_plane_arc_tol[z] = slab.arc_tolerance
                if (
                    z not in z_plane_min_arc_pts
                    or slab.min_arc_points > z_plane_min_arc_pts[z]
                ):
                    z_plane_min_arc_pts[z] = slab.min_arc_points
            else:
                z_plane_id_arcs.setdefault(z, False)
                z_plane_arc_tol.setdefault(z, slab.arc_tolerance)
                z_plane_min_arc_pts.setdefault(z, slab.min_arc_points)

    # First pass: identify shared horizontal interior faces.
    # Two subpieces share an interior face when their z_intervals are
    # adjacent (one's zhi == other's zlo) and their sub_polygons
    # intersect with non-zero area.
    sub_idx_by_z: dict[float, list[int]] = {}
    for i, sp in enumerate(subpieces):
        sub_idx_by_z.setdefault(sp.z_interval[0], []).append(i)
        sub_idx_by_z.setdefault(sp.z_interval[1], []).append(i)

    shared_horizontal: dict[tuple[int, int], object] = {}
    for z in sorted(sub_idx_by_z.keys()):
        below = [i for i in sub_idx_by_z[z] if subpieces[i].z_interval[1] == z]
        above = [i for i in sub_idx_by_z[z] if subpieces[i].z_interval[0] == z]
        for b in below:
            for a in above:
                inter = subpieces[b].sub_polygon.intersection(subpieces[a].sub_polygon)
                if inter.area > 0:
                    shared_horizontal[(b, a)] = inter

    # Cache built horizontal faces by (subpiece_idx, side) for quick lookup.
    # side: "bot" or "top".
    horiz_faces: dict[tuple[int, str], TopoDS_Face] = {}

    def arc_params_for_z(z: float, sp_idx: int):
        s = slab_by_source[subpieces[sp_idx].source_slab_indices[0]]
        id_arcs = z_plane_id_arcs.get(z, s.identify_arcs)
        min_p = z_plane_min_arc_pts.get(z, s.min_arc_points)
        arc_tol = z_plane_arc_tol.get(z, s.arc_tolerance)
        return id_arcs, min_p, arc_tol

    # Build shared interior faces first.
    # A face is shared (same TShape) only when the intersection covers the
    # FULL polygon of the sub-piece being assigned.  If inter_poly is only a
    # fragment of the sub-piece's polygon (e.g., an above-sub-piece is larger
    # than the below-sub-piece), assigning the tiny intersection as the face
    # would cause stamp_wedges to read only that fragment's triangulation.
    # In that situation we skip the assignment here and let the "remaining
    # bot/top faces" loop below build the correct full-polygon face.
    for (b, a), inter_poly in shared_horizontal.items():
        z = subpieces[b].z_interval[1]
        id_arcs, min_p, arc_tol = arc_params_for_z(z, b)
        area_tol = 1e-8
        below_full = abs(inter_poly.area - subpieces[b].sub_polygon.area) < area_tol
        above_full = abs(inter_poly.area - subpieces[a].sub_polygon.area) < area_tol
        face = _build_horizontal_face(inter_poly, z, ereg, id_arcs, min_p, arc_tol)
        if below_full:
            horiz_faces[(b, "top")] = face
        if above_full:
            horiz_faces[(a, "bot")] = face

    # Build remaining bot/top faces (those that weren't shared).
    for i, sp in enumerate(subpieces):
        if (i, "bot") not in horiz_faces:
            id_arcs, min_p, arc_tol = arc_params_for_z(sp.z_interval[0], i)
            horiz_faces[(i, "bot")] = _build_horizontal_face(
                sp.sub_polygon,
                sp.z_interval[0],
                ereg,
                id_arcs,
                min_p,
                arc_tol,
            )
        if (i, "top") not in horiz_faces:
            id_arcs, min_p, arc_tol = arc_params_for_z(sp.z_interval[1], i)
            horiz_faces[(i, "top")] = _build_horizontal_face(
                sp.sub_polygon,
                sp.z_interval[1],
                ereg,
                id_arcs,
                min_p,
                arc_tol,
            )

    # Build lateral faces per subpiece. Each polygon-edge of the
    # subpiece's sub_polygon becomes one lateral face. The edge
    # registry shares vertical edges between laterally-adjacent
    # subpieces in the same z-interval automatically.
    #
    # A lateral face cache keyed by the same vertex identity used by the
    # edge registry shares the FACE TShape between two subpieces that
    # have a coincident xy-edge at the same z-interval. The cohort
    # compound is passed to BOPAlgo_Builder as one argument
    # (keep_compound_for_bop=True) so it does NOT fragment internal
    # coincident-but-distinct-TShape faces. Without this cache the
    # n_layers-mismatch lateral-touch detection in
    # ``apply_lateral_transfinite_hints`` cannot see that two adjacent
    # slabs share a lateral face.
    lateral_face_cache: dict[tuple, TopoDS_Face] = {}

    def _lateral_key(
        seg: _PolylineSegment,
        zlo: float,
        zhi: float,
    ) -> tuple:
        """Build a lateral-face cache key for one polyline segment.

        The key distinguishes lateral faces that genuinely differ even
        when their endpoints coincide. Crucially, for a closed circle
        ``polyline_segments`` returns two half-arcs whose endpoint pair
        ``(start, end)`` are identical when sorted (start = coords[0],
        end = coords[mid]; second arc swaps them). Without the segment
        ``mid`` in the key, the two half-cylinder faces would collapse
        to one TShape and the lateral wall would only be half-covered
        in the final mesh.

        For straight-line segments shared by two side-by-side subpieces
        (e.g., two squares touching at x=10) the key is endpoint-order-
        invariant so the same face is shared between subpieces. That
        sharing is what lets ``apply_lateral_transfinite_hints`` detect
        n_layers mismatches and lets two adjacent solids hold the same
        TShape.
        """
        a, b = seg.start, seg.end
        k_a = vreg._key(a[0], a[1], zlo)
        k_b = vreg._key(b[0], b[1], zlo)
        # Endpoint-order-invariant; same lateral whether traversed
        # left->right or right->left.
        endpoints = tuple(sorted([k_a, k_b]))
        if seg.kind == "arc":
            # Arcs with the same endpoints but different midpoints
            # (the two halves of a full circle) MUST get distinct keys.
            mid = seg.mid
            k_mid = vreg._key(mid[0], mid[1], zlo)
            disambig: tuple = ("arc", k_mid)
        else:
            disambig = ("line",)
        return (
            "F",
            endpoints,
            disambig,
            vreg._key(0, 0, zlo)[2],
            vreg._key(0, 0, zhi)[2],
        )

    lateral_faces: dict[int, list[TopoDS_Face]] = {i: [] for i in range(len(subpieces))}
    for i, sp in enumerate(subpieces):
        zlo, zhi = sp.z_interval
        coords = _ring_coords(sp.sub_polygon.exterior)
        # When arc-detection is active for either z-plane of this sub-piece,
        # decompose the polygon boundary into the same arc/line segments
        # used by the horizontal faces above and below.  This makes each
        # lateral face's bot/top boundary edge be the SAME shared TShape
        # already used by the horizontal face, so the shell is closed and
        # BOP doesn't need to re-discover the arc topology.
        id_arcs_lo, min_p_lo, arc_tol_lo = arc_params_for_z(zlo, i)
        id_arcs_hi, min_p_hi, arc_tol_hi = arc_params_for_z(zhi, i)
        # Cohort-wide arc propagation guarantees both planes agree on
        # whether identify_arcs is in effect; we just need to detect the
        # segmentation once.
        use_arcs = id_arcs_lo or id_arcs_hi
        segments = polyline_segments(
            coords,
            identify_arcs=use_arcs,
            min_arc_points=max(min_p_lo, min_p_hi),
            arc_tolerance=min(arc_tol_lo, arc_tol_hi),
            point_tolerance=point_tolerance,
        )

        for seg in segments:
            a, b = seg.start, seg.end
            if seg.kind == "arc":
                e_lo = ereg.arc_xy(a, seg.mid, b, zlo)
                e_hi = ereg.arc_xy(a, seg.mid, b, zhi)
            else:
                e_lo = ereg.line_xy(a[0], a[1], b[0], b[1], zlo)
                e_hi = ereg.line_xy(a[0], a[1], b[0], b[1], zhi)
            v_left = ereg.vertical(a[0], a[1], zlo, zhi)
            v_right = ereg.vertical(b[0], b[1], zlo, zhi)
            cache_key = _lateral_key(seg, zlo, zhi)
            face = lateral_face_cache.get(cache_key)
            if face is None:
                face = _build_lateral_face(e_lo, e_hi, v_left, v_right)
                lateral_face_cache[cache_key] = face
            lateral_faces[i].append(face)

    # Build each sub-solid.
    builder = BRep_Builder()
    compound = TopoDS_Compound()
    builder.MakeCompound(compound)
    slab_meta: dict[ShapeKey, SlabMeta] = {}
    for i, sp in enumerate(subpieces):
        shell = TopoDS_Shell()
        builder.MakeShell(shell)
        bot = horiz_faces[(i, "bot")]
        top = horiz_faces[(i, "top")]
        laterals = lateral_faces[i]
        for f in [bot, top, *laterals]:
            builder.Add(shell, f)
        solid = TopoDS_Solid()
        builder.MakeSolid(solid)
        builder.Add(solid, shell)
        builder.Add(compound, solid)
        source_slab = slab_by_source[sp.source_slab_indices[0]]
        slab_meta[_shape_key(solid)] = SlabMeta(
            slab_index=source_slab.source_index,
            physical_name=source_slab.physical_name,
            bot_face_key=_shape_key(bot),
            top_face_key=_shape_key(top),
            lateral_face_keys=tuple(_shape_key(f) for f in laterals),
        )
    return compound, slab_meta
