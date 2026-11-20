"""Pin the vectorized arc-fit pair scan against a plain pairwise reference."""
from __future__ import annotations

import numpy as np
import pytest

import meshwell.circle_registry as cr
from meshwell.circle_registry import ArcFit, CircleRegistry

TOL = 1e-3


def _reference_clusters(
    fits: list[ArcFit], cluster_tolerance: float
) -> list[tuple[tuple[float, float], float]]:
    """A plain pairwise reference, returned as an ordered cluster list."""
    if not fits:
        return []
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
            if d <= cluster_tolerance and abs(a.radius - b.radius) <= cluster_tolerance:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri

    groups: dict[int, list[ArcFit]] = {}
    for i, f in enumerate(fits):
        groups.setdefault(find(i), []).append(f)
    out = []
    for members in groups.values():
        w = float(sum(m.npoints for m in members))
        out.append(
            (
                (
                    sum(m.center[0] * m.npoints for m in members) / w,
                    sum(m.center[1] * m.npoints for m in members) / w,
                ),
                sum(m.radius * m.npoints for m in members) / w,
            )
        )
    return out


def _actual_clusters(fits: list[ArcFit]) -> list[tuple[tuple[float, float], float]]:
    reg = CircleRegistry.from_fits(fits, cluster_tolerance=TOL)
    return [(c.center, c.radius) for c in reg.clusters]


def _random_fits(rng, count: int, n_groups: int) -> list[ArcFit]:
    """Fits clumped around a few centres so merges actually happen."""
    base = rng.uniform(0, 10, size=(max(1, n_groups), 2))
    fits = []
    for _ in range(count):
        group = int(rng.integers(0, max(1, n_groups)))
        if rng.random() < 0.8:
            center = base[group] + rng.uniform(-0.5e-3, 0.5e-3, size=2)
        else:
            center = rng.uniform(0, 10, size=2)
        radius = float(rng.choice([5.0, 5.0, 5.0, 5.0005, 5.002, 7.0]))
        fits.append(
            ArcFit(
                center=(float(center[0]), float(center[1])),
                radius=radius,
                npoints=int(rng.integers(3, 12)),
            )
        )
    return fits


def test_clustering_matches_the_pairwise_formulation():
    """Centres, radii, membership and cluster order all match the pairwise scan."""
    rng = np.random.default_rng(0)
    for _ in range(150):
        count = int(rng.integers(0, 40))
        fits = _random_fits(rng, count, count // 3)
        assert _actual_clusters(fits) == _reference_clusters(fits, TOL)


def test_clustering_matches_with_exact_duplicates():
    """Exactly coincident fits (the common case for shared boundaries) merge."""
    rng = np.random.default_rng(1)
    fits = []
    for _ in range(40):
        x, y = rng.uniform(0, 5, size=2)
        fits.append(ArcFit(center=(float(x), float(y)), radius=5.0, npoints=6))
        fits.append(ArcFit(center=(float(x), float(y)), radius=5.0, npoints=6))
    reg = CircleRegistry.from_fits(fits, cluster_tolerance=TOL)
    assert len(reg.clusters) == 40
    assert _actual_clusters(fits) == _reference_clusters(fits, TOL)


N_CHUNK_FIXTURE = 61


@pytest.mark.parametrize(
    "chunk_elements", [1, 2 * N_CHUNK_FIXTURE, 3 * N_CHUNK_FIXTURE]
)
def test_chunking_does_not_change_the_result(monkeypatch, chunk_elements):
    """Chunk sizes that split the rows unevenly must not change the clustering."""
    rng = np.random.default_rng(2)
    fits = _random_fits(rng, N_CHUNK_FIXTURE, 8)
    expected = _actual_clusters(fits)
    monkeypatch.setattr(cr, "_PAIR_CHUNK_ELEMENTS", chunk_elements)

    chunk = max(1, chunk_elements // N_CHUNK_FIXTURE)
    assert chunk in {1, 2, 3}
    if chunk > 1:
        # The final chunk is partial, which is where a dropped row offset shows.
        assert N_CHUNK_FIXTURE % chunk != 0
    assert _actual_clusters(fits) == expected


def _reference_pairs(fits, cluster_tolerance):
    """Close pairs (i < j) via ``cdist``, an independent distance implementation."""
    from scipy.spatial.distance import cdist

    centers = np.asarray([f.center for f in fits], dtype=float)
    radii = np.asarray([f.radius for f in fits], dtype=float)
    close = (cdist(centers, centers) <= cluster_tolerance) & (
        np.abs(radii[:, None] - radii[None, :]) <= cluster_tolerance
    )
    rows, cols = np.nonzero(np.triu(close, k=1))
    return list(zip(rows.tolist(), cols.tolist(), strict=True))


def _reference_clusters_from_pairs(fits, pairs):
    """Weighted-mean clusters built from an independently derived pair set."""
    parent = list(range(len(fits)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in pairs:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    groups: dict[int, list[ArcFit]] = {}
    for i, f in enumerate(fits):
        groups.setdefault(find(i), []).append(f)
    out = []
    for members in groups.values():
        w = float(sum(m.npoints for m in members))
        out.append(
            (
                (
                    sum(m.center[0] * m.npoints for m in members) / w,
                    sum(m.center[1] * m.npoints for m in members) / w,
                ),
                sum(m.radius * m.npoints for m in members) / w,
            )
        )
    return out


def test_clustering_at_the_shipped_chunk_cap_spans_multiple_chunks():
    """Production settings really do chunk, and the clustering is unchanged.

    The chunk size is ``_PAIR_CHUNK_ELEMENTS // n``, so exercising more than one
    chunk needs n above the square root of the cap. Comparing partitions (not
    order) keeps the independent oracle affordable at that size.
    """
    count = 2100
    rng = np.random.default_rng(3)
    fits = _random_fits(rng, count, count // 3)

    chunk = max(1, cr._PAIR_CHUNK_ELEMENTS // count)
    assert 1 < chunk < count, f"fixture does not chunk: chunk={chunk}"

    actual = _actual_clusters(fits)
    expected = _reference_clusters_from_pairs(fits, _reference_pairs(fits, TOL))
    assert sorted(actual) == sorted(expected)
    assert len(actual) < count, "the fixture must produce merges"


@pytest.mark.parametrize("count", [0, 1])
def test_too_few_fits_yields_no_merges(count):
    """Empty and single-fit inputs are handled without touching pair machinery."""
    fits = [
        ArcFit(center=(float(i), float(i)), radius=1.0, npoints=4) for i in range(count)
    ]
    assert len(_actual_clusters(fits)) == count


def test_distant_fits_stay_separate():
    """Fits beyond tolerance on either axis do not merge."""
    fits = [
        ArcFit(center=(0.0, 0.0), radius=5.0, npoints=4),
        # Within centre tolerance but radius differs by more than tolerance.
        ArcFit(center=(0.0005, 0.0), radius=5.5, npoints=4),
        # Within radius tolerance but centre differs by more than tolerance.
        ArcFit(center=(1.0, 0.0), radius=5.0, npoints=4),
    ]
    assert len(_actual_clusters(fits)) == 3
    assert _actual_clusters(fits) == _reference_clusters(fits, TOL)


def test_lookup_finds_the_canonical_cluster():
    """Registered clusters stay queryable after the vectorized scan."""
    fits = [
        ArcFit(center=(1.0, 2.0), radius=3.0, npoints=4),
        ArcFit(center=(1.0 + 4e-4, 2.0), radius=3.0 + 2e-4, npoints=6),
    ]
    reg = CircleRegistry.from_fits(fits, cluster_tolerance=TOL)
    assert len(reg.clusters) == 1
    hit = reg.lookup((1.0, 2.0), 3.0)
    assert hit is not None
    assert reg.lookup((50.0, 50.0), 3.0) is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
