"""Stage 2: Union-Find over StructuredSlabs.

Two slabs merge if they share a z-plane with XY-overlap (face-touch)
or share a z-interval with XY-overlap (lateral-touch). Output cohorts
are disjoint by construction.
"""
from __future__ import annotations

from meshwell.structured.types import Cohort, StructuredSlab


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


def build_cohorts(slabs: list[StructuredSlab]) -> list[Cohort]:
    """Group slabs into cohorts."""
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
    return cohorts
