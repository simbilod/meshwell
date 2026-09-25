"""Re-bind unstructured fragment pieces onto the (unmodified) cohort topology.

``BOPAlgo_Builder`` receives each structured cohort as a single compound
argument (``keep_compound_for_bop``). ``Modified(compound)`` is empty, so
:meth:`meshwell.cad_occ.CAD_OCC._fragment_all` keeps the *pre-BOP* cohort
compound, while unstructured neighbours receive their *post-BOP* images.

Wherever BOP merged a cohort sub-shape with a coincident neighbour sub-shape
it may have produced a brand-new shape (typical for edge-on-edge common
blocks). The neighbour then references the new shape and the
cohort the old one, so the interface face is duplicated instead of shared
and gmsh meshes it twice.

:func:`rebind_to_cohort_topology` walks BOP's own history for every cohort
vertex / edge / face. Each 1:1 replacement ``S -> S'`` is reversed in the
non-cohort output shapes (``S' -> S``) with ``BRepTools_ReShape``. Only
exact BOP provenance is used — no geometric matching or fuzzy tolerance.
1:n splits of cohort sub-shapes are left alone (a genuine cohort shell
modification, reported by ``validate_cohort_shells``).
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Any

from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCP.BRepTools import BRepTools_ReShape
from OCP.GeomLProp import GeomLProp_SLProps
from OCP.gp import gp_Pnt, gp_Vec
from OCP.ShapeAnalysis import ShapeAnalysis_Surface
from OCP.TopAbs import (
    TopAbs_EDGE,
    TopAbs_FACE,
    TopAbs_FORWARD,
    TopAbs_REVERSED,
    TopAbs_VERTEX,
)
from OCP.TopExp import TopExp
from OCP.TopoDS import TopoDS, TopoDS_Shape
from OCP.TopTools import TopTools_IndexedMapOfShape, TopTools_ShapeMapHasher

if TYPE_CHECKING:
    from OCP.BOPAlgo import BOPAlgo_Builder

logger = logging.getLogger(__name__)

_HASHER = TopTools_ShapeMapHasher()


def _edge_endpoints(edge: TopoDS_Shape) -> tuple[gp_Pnt, gp_Pnt]:
    e = TopoDS.Edge_s(edge)
    return (
        BRep_Tool.Pnt_s(TopExp.FirstVertex_s(e, True)),
        BRep_Tool.Pnt_s(TopExp.LastVertex_s(e, True)),
    )


def _edge_start_tangent(edge: TopoDS_Shape) -> gp_Vec:
    e = TopoDS.Edge_s(edge)
    curve = BRepAdaptor_Curve(e)
    reversed_ = e.Orientation() == TopAbs_REVERSED
    u = curve.LastParameter() if reversed_ else curve.FirstParameter()
    p, v = gp_Pnt(), gp_Vec()
    curve.D1(u, p, v)
    return v.Reversed() if reversed_ else v


def _same_sense_edge(a: TopoDS_Shape, b: TopoDS_Shape) -> bool:
    """True if FORWARD-oriented edges ``a`` and ``b`` traverse the same geometry in the same direction."""
    a0, a1 = _edge_endpoints(a)
    b0, b1 = _edge_endpoints(b)
    if a0.Distance(a1) > 1e-9 and b0.Distance(b1) > 1e-9:
        return a0.Distance(b0) <= a0.Distance(b1)
    # Closed edge: compare start tangents.
    return _edge_start_tangent(a).Dot(_edge_start_tangent(b)) > 0.0


def _face_normal_at(face: TopoDS_Shape, pnt: gp_Pnt) -> gp_Vec | None:
    f = TopoDS.Face_s(face)
    surf = BRep_Tool.Surface_s(f)
    uv = ShapeAnalysis_Surface(surf).ValueOfUV(pnt, 1e-7)
    props = GeomLProp_SLProps(surf, uv.X(), uv.Y(), 1, 1e-9)
    if not props.IsNormalDefined():
        return None
    n = gp_Vec(props.Normal())
    return n.Reversed() if f.Orientation() == TopAbs_REVERSED else n


def _same_sense_face(a: TopoDS_Shape, b: TopoDS_Shape) -> bool:
    """True if FORWARD-oriented faces ``a`` and ``b`` have the same outward normal."""
    ad = BRepAdaptor_Surface(TopoDS.Face_s(a), True)
    u = 0.5 * (ad.FirstUParameter() + ad.LastUParameter())
    v = 0.5 * (ad.FirstVParameter() + ad.LastVParameter())
    p = ad.Value(u, v)
    na, nb = _face_normal_at(a, p), _face_normal_at(b, p)
    if na is None or nb is None:
        return True
    return na.Dot(nb) > 0.0


def _aligned_original(
    original: TopoDS_Shape, image: TopoDS_Shape, kind
) -> TopoDS_Shape:
    """Return ``original`` oriented to match ``image.Oriented(FORWARD)``."""
    orig_f = original.Oriented(TopAbs_FORWARD)
    img_f = image.Oriented(TopAbs_FORWARD)
    if kind == TopAbs_VERTEX:
        return orig_f
    same = (
        _same_sense_edge(orig_f, img_f)
        if kind == TopAbs_EDGE
        else _same_sense_face(orig_f, img_f)
    )
    return orig_f if same else original.Oriented(TopAbs_REVERSED)


def _collect_replacements(
    cohort_shapes: list[TopoDS_Shape],
    builder: BOPAlgo_Builder,
) -> tuple[list[tuple[TopoDS_Shape, TopoDS_Shape, Any]], dict[Any, int]]:
    """``(image, original, kind)`` for every 1:1 BOP replacement of a cohort sub-shape."""
    candidates: dict[int, list[tuple[TopoDS_Shape, TopoDS_Shape, Any]]] = defaultdict(
        list
    )
    for kind in (TopAbs_VERTEX, TopAbs_EDGE, TopAbs_FACE):
        for shape in cohort_shapes:
            sub_map = TopTools_IndexedMapOfShape()
            TopExp.MapShapes_s(shape, kind, sub_map)
            for i in range(1, sub_map.Extent() + 1):
                original = sub_map.FindKey(i)
                if builder.IsDeleted(original):
                    continue
                images = list(builder.Modified(original))
                if len(images) != 1:
                    continue  # unchanged (0) or split (n) — nothing to re-bind
                image = images[0]
                if image.IsSame(original):
                    continue
                candidates[_HASHER(image)].append((image, original, kind))

    replacements: list[tuple[TopoDS_Shape, TopoDS_Shape, Any]] = []
    skipped: dict[Any, int] = defaultdict(int)
    for entries in candidates.values():
        # Several distinct cohort sub-shapes collapsing onto one image would
        # make the mapping ambiguous; leave those alone.
        distinct = {_HASHER(o) for _, o, _ in entries}
        if len(distinct) == 1:
            replacements.append(entries[0])
        else:
            skipped[entries[0][2]] += 1
    return replacements, skipped


def rebind_to_cohort_topology(entities: list[Any], builder: BOPAlgo_Builder) -> int:
    """Rewrite non-cohort entity shapes to reuse cohort vertices / edges / faces BOP re-created.

    Args:
        entities: ``OCCLabeledEntity`` list after fragment ownership resolution.
        builder: The ``BOPAlgo_Builder`` that produced the fragments.

    Returns:
        Number of sub-shape replacements registered.
    """
    cohort_shapes = [s for ent in entities if ent._is_cohort for s in ent.shapes]
    if not cohort_shapes:
        return 0
    replacements, skipped = _collect_replacements(cohort_shapes, builder)
    if skipped:
        logger.warning(
            "Cohort re-bind: skipped ambiguous replacements (several cohort "
            "sub-shapes mapped to one BOP image): %s",
            dict(skipped),
        )
    if not replacements:
        return 0

    reshaper = BRepTools_ReShape()
    counts: dict[str, int] = defaultdict(int)
    for image, original, kind in replacements:
        reshaper.Replace(
            image.Oriented(TopAbs_FORWARD), _aligned_original(original, image, kind)
        )
        counts[
            {TopAbs_VERTEX: "vertex", TopAbs_EDGE: "edge", TopAbs_FACE: "face"}[kind]
        ] += 1

    for ent in entities:
        if ent._is_cohort:
            continue
        ent.shapes = [reshaper.Apply(s) for s in ent.shapes]
    logger.info("Cohort re-bind: reused cohort sub-shapes %s", dict(counts))
    return len(replacements)
