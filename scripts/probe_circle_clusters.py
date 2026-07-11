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
from meshwell.geometry_entity import decompose_vertices_2d
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
                i, n = 0, len(segments)
                while i < n:
                    if segments[i].is_arc:
                        i += 1
                        continue
                    j = i
                    while j < n and not segments[j].is_arc:
                        j += 1
                    pts = [s.points[0] for s in segments[i:j]]
                    pts.append(segments[j - 1].points[-1])
                    xy = [(p[0], p[1]) for p in pts]
                    if reg.match_chord_run(xy, tolerance=tol) is not None:
                        if len(xy) >= 3:
                            promotable += 1
                        else:
                            ambiguous += 1
                    i = j
    print(
        f"chord-run exposure: {promotable} promotable run(s) (auto-fixed), "
        f"{ambiguous} ambiguous single-chord candidate(s) (manual review)"
    )


if __name__ == "__main__":
    main()
