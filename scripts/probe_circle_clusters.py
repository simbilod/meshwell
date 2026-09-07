"""Report canonical-circle clusters + chord-run exposure for a scene.

Usage:
    uv run python scripts/probe_circle_clusters.py <entities.json>

Run on production scenes: cluster fit disagreements are the sliver sites
the canonical pipeline removes; ambiguous single-chord candidates are
NOT auto-promoted and deserve manual review.
"""
import json
import sys
from pathlib import Path

from meshwell.cad_common import apply_arc_params
from meshwell.circle_registry import build_circle_registry
from meshwell.geometry_entity import (
    # Private, but the probe must run the pipeline's EXACT promotion scan
    # (greedy sub-window growth inside each line run) to stay predictive --
    # re-implementing it as a whole-run match would miss embedded remnants
    # (arc in the middle of a run, corners at the ends) that the real
    # pipeline still promotes.
    _promote_chord_runs,
    decompose_vertices_2d,
)
from meshwell.orchestrator import deserialize


def main() -> None:
    """Load entities, build circle registry, report clusters and chord exposure."""
    with Path(sys.argv[1]).open() as f:
        entities = deserialize(json.load(f))
    apply_arc_params(entities, identify_arcs=True)
    reg = build_circle_registry(entities)
    print(f"{len(reg.clusters)} canonical circle cluster(s)")
    for k, cl in enumerate(reg.clusters):
        print(
            f"  cluster {k}: center=({cl.center[0]:.9g}, {cl.center[1]:.9g}) "
            f"radius={cl.radius:.9g}"
        )

    promotable = ambiguous = 0
    for e in entities:
        polygons = getattr(e, "polygons", None)
        if polygons is None:
            continue
        tol = getattr(e, "arc_tolerance", 1e-3) + getattr(e, "point_tolerance", 1e-3)
        polys = (
            polygons.geoms
            if hasattr(polygons, "geoms")
            else (polygons if isinstance(polygons, list) else [polygons])
        )
        for poly in polys:
            for ring in [poly.exterior, *poly.interiors]:
                segments = decompose_vertices_2d(
                    list(ring.coords),
                    z=0.0,
                    point_tolerance=getattr(e, "point_tolerance", 1e-3),
                    identify_arcs=getattr(e, "identify_arcs", False),
                    min_arc_points=getattr(e, "min_arc_points", 5),
                    arc_tolerance=getattr(e, "arc_tolerance", 1e-3),
                )
                # Run the REAL pipeline promotion scan (greedy sub-window
                # growth inside each maximal line run), not a whole-run
                # match: only this reproduces embedded-remnant promotions.
                arcs_before = sum(1 for s in segments if s.is_arc)
                promoted = _promote_chord_runs(segments, reg, tolerance=tol)
                arcs_after = sum(1 for s in promoted if s.is_arc)
                promotable += arcs_after - arcs_before
                # Ambiguous: single-chord (2-vertex) line segments that
                # survive promotion untouched but whose own two endpoints
                # alone already match a registered circle -- the pipeline
                # deliberately never auto-promotes these (indistinguishable
                # from a straight edge grazing the circle).
                for seg in promoted:
                    if seg.is_arc:
                        continue
                    xy = [(p[0], p[1]) for p in seg.points]
                    if (
                        len(xy) == 2
                        and reg.match_chord_run(xy, tolerance=tol) is not None
                    ):
                        ambiguous += 1
    print(
        f"chord-run exposure: {promotable} promotable run(s) (auto-fixed), "
        f"{ambiguous} ambiguous single-chord candidate(s) (manual review)"
    )


if __name__ == "__main__":
    main()
