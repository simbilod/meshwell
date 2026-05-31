"""Stage 4 — assemble cohort TopoDS_Compound of N sub-solids.

Bottom-up build: unique vertices → unique edges (with arc detection)
→ unique faces (horizontal interior/boundary, lateral) → per-subpiece
TopoDS_Solid → TopoDS_Compound per cohort.

Shared TShapes by CONSTRUCTION (not post-hoc sewing): every face and
edge is built once and referenced by every solid that needs it. This
is what makes cohort internal interfaces conformal without BOP.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
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
        min_arc_points: int = 4,
        arc_tolerance: float = 1e-3,
    ) -> list[TopoDS_Edge]:
        """Return the list of edges (lines and/or arcs) covering coords.

        Uses the same arc-detection as GeometryEntity.decompose_vertices
        but inlined to avoid pulling that dep just for the 2D case.
        """
        if not identify_arcs or len(coords) < min_arc_points:
            return [
                self.line_xy(
                    coords[i][0], coords[i][1], coords[i + 1][0], coords[i + 1][1], z
                )
                for i in range(len(coords) - 1)
            ]
        edges: list[TopoDS_Edge] = []
        i, n = 0, len(coords)
        while i < n - 1:
            best = None
            best_j = i + 1
            for j in range(i + min_arc_points, n + 1):
                pts = np.array(coords[i:j])
                cx, cy, r, residual = _fit_circle_2d(pts)
                if residual <= arc_tolerance and r < 1e6:
                    best = (cx, cy, r)
                    best_j = j
                else:
                    break
            if best is not None:
                seg_start = coords[i]
                seg_end = coords[best_j - 1]
                mid_idx = (i + best_j - 1) // 2
                # Full-circle: start and end coincide → split into two half-arcs.
                tol = self.point_tolerance
                if (
                    abs(seg_start[0] - seg_end[0]) < tol
                    and abs(seg_start[1] - seg_end[1]) < tol
                ):
                    q1_idx = (i + mid_idx) // 2
                    q3_idx = (mid_idx + best_j - 1) // 2
                    edges.append(
                        self.arc_xy(seg_start, coords[q1_idx], coords[mid_idx], z)
                    )
                    edges.append(
                        self.arc_xy(coords[mid_idx], coords[q3_idx], seg_end, z)
                    )
                else:
                    edges.append(self.arc_xy(seg_start, coords[mid_idx], seg_end, z))
                i = best_j - 1
            else:
                edges.append(
                    self.line_xy(
                        coords[i][0],
                        coords[i][1],
                        coords[i + 1][0],
                        coords[i + 1][1],
                        z,
                    )
                )
                i += 1
        return edges


def _fit_circle_2d(pts: np.ndarray) -> tuple[float, float, float, float]:
    """Algebraic circle fit. Returns (cx, cy, r, residual)."""
    x, y = pts[:, 0], pts[:, 1]
    A = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
    b = x * x + y * y
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, c = sol
    r = float(np.sqrt(c + cx * cx + cy * cy))
    residual = float(np.std(np.hypot(x - cx, y - cy) - r))
    return float(cx), float(cy), r, residual


# ---------------------------------------------------------------------------
# Stage 4 parts 3-6: face registry + sub-solid + cohort compound assembly
# ---------------------------------------------------------------------------


def _shape_key(shape) -> ShapeKey:
    from OCP.TopTools import TopTools_ShapeMapHasher

    hasher = TopTools_ShapeMapHasher()
    return ShapeKey(tshape_id=hasher(shape), orientation=int(shape.Orientation()))


def _ring_coords(ring) -> list[tuple[float, float]]:
    return list(ring.coords)


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


def _build_lateral_face(
    edge_xy_low: TopoDS_Edge,
    edge_xy_high: TopoDS_Edge,
    v_left: TopoDS_Edge,
    v_right: TopoDS_Edge,
) -> TopoDS_Face:
    """Stitch four edges into a lateral quad face."""
    mw = BRepBuilderAPI_MakeWire()
    mw.Add(edge_xy_low)
    mw.Add(v_right)
    mw.Add(edge_xy_high)
    mw.Add(v_left)
    wire = mw.Wire()
    return BRepBuilderAPI_MakeFace(wire).Face()


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
    """
    vreg = VertexRegistry(point_tolerance=point_tolerance)
    ereg = EdgeRegistry(vertices=vreg, point_tolerance=point_tolerance)

    slab_by_source: dict[int, StructuredSlab] = {
        s.source_index: s for s in cohort.slabs
    }

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

    def arc_params_for(sp_idx: int):
        s = slab_by_source[subpieces[sp_idx].source_slab_indices[0]]
        return s.identify_arcs, s.min_arc_points, s.arc_tolerance

    # Build shared interior faces first.
    for (b, a), inter_poly in shared_horizontal.items():
        z = subpieces[b].z_interval[1]
        id_arcs, min_p, arc_tol = arc_params_for(b)
        face = _build_horizontal_face(inter_poly, z, ereg, id_arcs, min_p, arc_tol)
        horiz_faces[(b, "top")] = face
        horiz_faces[(a, "bot")] = face

    # Build remaining bot/top faces (those that weren't shared).
    for i, sp in enumerate(subpieces):
        id_arcs, min_p, arc_tol = arc_params_for(i)
        if (i, "bot") not in horiz_faces:
            horiz_faces[(i, "bot")] = _build_horizontal_face(
                sp.sub_polygon,
                sp.z_interval[0],
                ereg,
                id_arcs,
                min_p,
                arc_tol,
            )
        if (i, "top") not in horiz_faces:
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
    lateral_faces: dict[int, list[TopoDS_Face]] = {i: [] for i in range(len(subpieces))}
    for i, sp in enumerate(subpieces):
        zlo, zhi = sp.z_interval
        coords = _ring_coords(sp.sub_polygon.exterior)
        for a, b in itertools.pairwise(coords):
            e_lo = ereg.line_xy(a[0], a[1], b[0], b[1], zlo)
            e_hi = ereg.line_xy(a[0], a[1], b[0], b[1], zhi)
            v_left = ereg.vertical(a[0], a[1], zlo, zhi)
            v_right = ereg.vertical(b[0], b[1], zlo, zhi)
            lateral_faces[i].append(_build_lateral_face(e_lo, e_hi, v_left, v_right))

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
