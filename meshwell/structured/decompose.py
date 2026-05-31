"""Stage 3 — cohort decomposition in shapely.

Stage 3b: collect z-planes (just the cohort's own slab boundaries
after the 3a validator ran).
Stage 3c: per-z-interval footprint with Policy B carving (this task).
Stage 3d: bidirectional pre-cut at shared z-planes (Task 7).
"""

from __future__ import annotations

from shapely.geometry import Polygon
from shapely.ops import unary_union

from meshwell.structured.types import StructuredSlab


def zinterval_footprint(slabs_here: list[StructuredSlab]) -> Polygon:
    """Resolve Policy B carving for one z-interval.

    Sort slabs by (mesh_order, source_index) ascending; lower wins.
    For mesh_bool=True: union (footprint - accumulated).
    For mesh_bool=False (void): subtract footprint from accumulated.
    """
    ordered = sorted(
        slabs_here,
        key=lambda s: (
            s.mesh_order if s.mesh_order is not None else float("inf"),
            s.source_index,
        ),
    )
    acc = Polygon()  # empty
    for s in ordered:
        if s.mesh_bool:
            new = s.footprint.difference(acc)
            acc = unary_union([acc, new])
        else:
            acc = acc.difference(s.footprint)
    return acc
