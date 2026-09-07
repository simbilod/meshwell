"""Cross-entity canonical circle registry (fit on NOMINAL geometry).

Entities discretize shared circular boundaries independently; fitted
circles disagree by up to arc_tolerance between entities, which dwarfs
any boolean clearance and produces graze/gap slivers. This module fits
on the nominal (unbuffered, grid-snapped) polygons, clusters the fits
across entities, and gives each cluster ONE canonical circle. Analytic
offsets (R +/- perturbation) are applied later at emission -- never by
buffering polylines and refitting.

Geometry only: no OCP imports.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from meshwell.geometry_entity import decompose_vertices_2d, fit_circle_2d


@dataclass(frozen=True)
class ArcFit:
    """Arc fit result: center, radius, and point count."""

    center: tuple[float, float]
    radius: float
    npoints: int


@dataclass
class CircleCluster:
    """Canonical circle cluster: center and radius."""

    center: tuple[float, float]
    radius: float


@dataclass
class CircleRegistry:
    """Registry of canonical circles clustered by fit proximity."""

    clusters: list[CircleCluster] = field(default_factory=list)
    cluster_tolerance: float = 1e-3

    @classmethod
    def from_fits(
        cls, fits: list[ArcFit], *, cluster_tolerance: float
    ) -> "CircleRegistry":
        """Build registry by clustering arc fits within tolerance."""
        reg = cls(cluster_tolerance=cluster_tolerance)
        if not fits:
            return reg
        parent = list(range(len(fits)))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i in range(len(fits)):
            for j in range(i + 1, len(fits)):
                a, b = fits[i], fits[j]
                d = np.hypot(a.center[0] - b.center[0], a.center[1] - b.center[1])
                if (
                    d <= cluster_tolerance
                    and abs(a.radius - b.radius) <= cluster_tolerance
                ):
                    ri, rj = find(i), find(j)
                    if ri != rj:
                        parent[rj] = ri

        groups: dict[int, list[ArcFit]] = {}
        for i, f in enumerate(fits):
            groups.setdefault(find(i), []).append(f)
        for members in groups.values():
            w = float(sum(m.npoints for m in members))
            reg.clusters.append(
                CircleCluster(
                    center=(
                        sum(m.center[0] * m.npoints for m in members) / w,
                        sum(m.center[1] * m.npoints for m in members) / w,
                    ),
                    radius=sum(m.radius * m.npoints for m in members) / w,
                )
            )
        return reg

    def lookup(
        self,
        center: tuple[float, float],
        radius: float,
        *,
        slack: float = 0.0,
    ) -> tuple[tuple[float, float], float] | None:
        """Nearest cluster within tolerance (+slack for the point-grid rounding).

        Accounts for the point-grid rounding of DecompositionSegment fields,
        or None if no match.
        """
        tol = self.cluster_tolerance + slack
        best: CircleCluster | None = None
        best_d = tol
        for cl in self.clusters:
            d = float(np.hypot(center[0] - cl.center[0], center[1] - cl.center[1]))
            if d <= best_d and abs(radius - cl.radius) <= tol:
                best, best_d = cl, d
        return (best.center, best.radius) if best else None

    def match_chord_run(
        self,
        points: list[tuple[float, float]],
        *,
        tolerance: float,
    ) -> tuple[tuple[float, float], float] | None:
        """Best cluster such that every point lies within tolerance of circle.

        Vertices of a chorded arc lie ON the circle (only chord midpoints
        sag); interior vertices of a genuine straight edge lie on the
        chord -- the all-vertices test separates the two given >= 1 interior
        vertex. Never re-fits.
        """
        pts = np.asarray(points, dtype=float)
        best: tuple[tuple[float, float], float] | None = None
        best_dev = tolerance
        for cl in self.clusters:
            d = np.hypot(pts[:, 0] - cl.center[0], pts[:, 1] - cl.center[1])
            dev = float(np.abs(d - cl.radius).max())
            if dev <= best_dev:
                best, best_dev = (cl.center, cl.radius), dev
        return best


def build_circle_registry(
    entities: list, *, cluster_tolerance: float | None = None
) -> CircleRegistry:
    """Fit arcs on every arc-identified entity's NOMINAL rings and cluster."""
    fits: list[ArcFit] = []
    max_arc_tol = 0.0
    for e in entities:
        if not getattr(e, "identify_arcs", False):
            continue
        polygons = getattr(e, "polygons", None)
        if polygons is None:
            continue
        max_arc_tol = max(max_arc_tol, getattr(e, "arc_tolerance", 1e-3))
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
                    point_tolerance=e.point_tolerance,
                    identify_arcs=True,
                    min_arc_points=getattr(e, "min_arc_points", 5),
                    arc_tolerance=getattr(e, "arc_tolerance", 1e-3),
                )
                for seg in segments:
                    if not seg.is_arc:
                        continue
                    pts = np.array(seg.points)
                    center, radius, residual = fit_circle_2d(pts[:, :2])
                    if np.isfinite(residual):
                        fits.append(
                            ArcFit(
                                center=(float(center[0]), float(center[1])),
                                radius=float(radius),
                                npoints=len(seg.points),
                            )
                        )
    tol = cluster_tolerance if cluster_tolerance is not None else max(max_arc_tol, 1e-3)
    return CircleRegistry.from_fits(fits, cluster_tolerance=tol)
