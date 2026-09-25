"""Stage 2: Union-Find over StructuredSlabs.

Two slabs merge if they share a z-plane with XY-overlap (face-touch)
or share a z-interval with XY-overlap (lateral-touch). Output cohorts
are disjoint by construction.
"""
from __future__ import annotations

from dataclasses import replace

from meshwell.structured._zmath import approx_in
from meshwell.structured.types import Cohort, StructuredPlane, StructuredSlab


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def _xy_overlaps(a: StructuredSlab, b: StructuredSlab) -> bool:
    # Use intersects (touching boundaries count) — face-touch with shared
    # edge but no interior overlap should still couple cohorts because
    # shared edges become shared OCC edges in the cohort solid.
    inter = a.footprint.intersection(b.footprint)
    return not inter.is_empty


def build_cohorts(
    slabs: list[StructuredSlab],
    planes: list[StructuredPlane] | None = None,
) -> list[Cohort]:
    """Group slabs into cohorts and attach matching StructuredPlanes."""
    n = len(slabs)
    if n == 0:
        return []
    uf = _UnionFind(n)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = slabs[i], slabs[j]
            # Lateral-touch: same z-interval, XY overlap.
            same_interval = (a.zlo == b.zlo) and (a.zhi == b.zhi)
            # Face-touch: share a z-plane (top-of-a == bot-of-b or vice
            # versa), XY overlap.
            face_touch = (a.zhi == b.zlo) or (b.zhi == a.zlo)
            if (same_interval or face_touch) and _xy_overlaps(a, b):
                uf.union(i, j)

    groups: dict[int, list[StructuredSlab]] = {}
    for i, s in enumerate(slabs):
        groups.setdefault(uf.find(i), []).append(s)

    cohorts: list[Cohort] = []
    for members in groups.values():
        z_planes = tuple(sorted({m.zlo for m in members} | {m.zhi for m in members}))
        cohorts.append(Cohort(slabs=tuple(members), z_planes=z_planes))

    if planes:
        cohorts = attach_planes_to_cohorts(cohorts, planes)

    return cohorts


def _snap_z_to_cohort(
    z: float, z_planes: tuple[float, ...], tol: float = 1e-9
) -> float | None:
    for zp in z_planes:
        if abs(z - zp) <= tol:
            return zp
    return None


def attach_planes_to_cohorts(
    cohorts: list[Cohort],
    planes: list[StructuredPlane],
) -> list[Cohort]:
    """Attach each StructuredPlane to the cohort(s) it intersects and snap its z to cohort.z_planes."""
    if not planes:
        return cohorts

    planes_by_cohort: list[list[StructuredPlane]] = [[] for _ in cohorts]
    for plane in planes:
        for ci, cohort in enumerate(cohorts):
            z_set = set(cohort.z_planes)
            if plane.orientation == "horizontal":
                if not approx_in(plane.zmin, z_set):
                    continue
                z_snap = _snap_z_to_cohort(plane.zmin, cohort.z_planes)
                if z_snap is None:
                    continue
                active_slabs = [s for s in cohort.slabs if s.zlo <= z_snap <= s.zhi]
                if any(s.footprint.intersects(plane.footprint) for s in active_slabs):
                    planes_by_cohort[ci].append(
                        replace(plane, zmin=z_snap, zmax=z_snap)
                    )
            else:
                if not (approx_in(plane.zmin, z_set) and approx_in(plane.zmax, z_set)):
                    continue
                zmin_snap = _snap_z_to_cohort(plane.zmin, cohort.z_planes)
                zmax_snap = _snap_z_to_cohort(plane.zmax, cohort.z_planes)
                if zmin_snap is None or zmax_snap is None or zmin_snap >= zmax_snap:
                    continue
                active_slabs = [
                    s for s in cohort.slabs if s.zlo < zmax_snap and s.zhi > zmin_snap
                ]
                if any(s.footprint.intersects(plane.footprint) for s in active_slabs):
                    planes_by_cohort[ci].append(
                        replace(plane, zmin=zmin_snap, zmax=zmax_snap)
                    )

    return [
        replace(cohort, planes=tuple(planes_by_cohort[ci]))
        for ci, cohort in enumerate(cohorts)
    ]
