"""WP4 Task 7 — wedge error ordering + placeholder hardening.

Unit-level tests. The structured integration suite (tests/structured/)
is the behavioral guard for full wedge assembly; here we pin the pure
helper logic and the fail-before-mutation ordering in ``_stamp_one``
using a minimal fake gmsh so no OCC kernel is required.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from meshwell.structured import wedge
from meshwell.structured.exceptions import WedgeBotNodeMismatchError

# ---------------------------------------------------------------------------
# (e) Non-collinear placeholder-triangle selection
# ---------------------------------------------------------------------------


def test_pick_noncollinear_triangle_rejects_collinear():
    tags = [1, 2, 3, 4]
    xy = {1: (0.0, 0.0), 2: (1.0, 0.0), 3: (2.0, 0.0), 4: (3.0, 0.0)}
    assert wedge._pick_noncollinear_triangle(tags, xy) is None


def test_pick_noncollinear_triangle_finds_corner():
    # Three of these lie on y=0; the fourth (0,1) is off the line.
    tags = [1, 2, 3, 4]
    xy = {1: (0.0, 0.0), 2: (1.0, 0.0), 3: (2.0, 0.0), 4: (0.0, 1.0)}
    tri = wedge._pick_noncollinear_triangle(tags, xy)
    assert tri is not None
    p0, p1, p2 = (np.array(xy[t]) for t in tri)
    area = abs(np.cross(p1 - p0, p2 - p0))
    assert area > 1e-9
    # The off-line node must participate; the first-three collinear nodes
    # alone would give a degenerate triangle.
    assert 4 in tri


def test_pick_noncollinear_triangle_too_few():
    assert wedge._pick_noncollinear_triangle([1, 2], {1: (0, 0), 2: (1, 0)}) is None


# ---------------------------------------------------------------------------
# (d) Left/right vertical-edge assignment — no silent collapse into one slot
# ---------------------------------------------------------------------------


def test_choose_left_right_verticals_splits():
    reps = [(11, 0.0, 0.0), (22, 1.0, 0.0)]
    assert wedge._choose_left_right_verticals((0.0, 0.0), (1.0, 0.0), reps) == (11, 22)


def test_choose_left_right_verticals_orientation_independent():
    # Same edges presented in swapped order still resolve identically.
    reps = [(22, 1.0, 0.0), (11, 0.0, 0.0)]
    assert wedge._choose_left_right_verticals((0.0, 0.0), (1.0, 0.0), reps) == (11, 22)


def test_choose_left_right_verticals_degenerate_returns_none():
    # Both verticals sit at the same corner: one edge is nearest to BOTH
    # endpoints. The old code silently dropped one and returned; now we
    # signal the ambiguity so the caller can warn + fall back.
    reps = [(11, 0.0, 0.0), (22, 0.0, 0.001)]
    assert wedge._choose_left_right_verticals((0.0, 0.0), (1.0, 0.0), reps) is None


# ---------------------------------------------------------------------------
# (b)/(c) Fail before mutation in _stamp_one
# ---------------------------------------------------------------------------


class _FakeMesh:
    def __init__(self, parent: "_FakeGmshModel"):
        self._p = parent

    def getElements(self, dim, tag):
        return self._p.elements.get((dim, tag), ([], [], []))

    def getNodes(self, dim, tag, **_kwargs):
        tags, coord = self._p.nodes.get((dim, tag), ([], []))
        return list(tags), list(coord), []

    def getMaxNodeTag(self):
        return self._p.max_node_tag

    def addNodes(self, dim, tag, node_tags, _coords):
        self._p.log.append(("addNodes", dim, tag, list(node_tags)))
        if node_tags:
            self._p.max_node_tag = max(
                self._p.max_node_tag, max(int(t) for t in node_tags)
            )

    def removeElements(self, dim, tag):
        self._p.log.append(("removeElements", dim, tag))

    def addElementsByType(self, tag, etype, _elem_tags, _node_tags):
        self._p.log.append(("addElementsByType", tag, etype))


class _FakeGmshModel:
    def __init__(self):
        self.elements: dict = {}
        self.nodes: dict = {}
        self.boundary: dict = {}
        self.bbox: dict = {}
        self.max_node_tag = 0
        self.log: list = []
        self.mesh = _FakeMesh(self)

    # gmsh.model.getBoundary
    def getBoundary(self, dimtags, **_kwargs):
        (dim, tag) = dimtags[0]
        return self.boundary.get((dim, tag), [])

    def getBoundingBox(self, _dim, tag):
        return self.bbox[tag]


class _FakeGmsh:
    def __init__(self):
        self.model = _FakeGmshModel()


def _meta(lateral_face_keys=()):
    return SimpleNamespace(
        slab_index=0,
        lateral_face_keys=tuple(lateral_face_keys),
    )


def test_stamp_one_top_mismatch_fails_before_mutation(monkeypatch):
    """(b) Top-face bot-node mismatch must raise BEFORE any element mutation.

    The gmsh model must be left untouched (no removeElements, no
    addElementsByType) so it is never half-stamped.
    """
    g = _FakeGmsh()
    bot_tag, top_tag, vol_tag = 10, 20, 30
    # bot: one triangle, all three nodes on the boundary.
    g.model.elements[(2, bot_tag)] = ([2], [[1]], [[1, 2, 3]])
    g.model.nodes[(2, bot_tag)] = ([1, 2, 3], [0, 0, 0, 1, 0, 0, 0, 1, 0])
    g.model.boundary[(2, bot_tag)] = [(1, 101), (1, 102), (1, 103)]
    g.model.nodes[(1, 101)] = ([1, 2], [0, 0, 0, 1, 0, 0])
    g.model.nodes[(1, 102)] = ([2, 3], [1, 0, 0, 0, 1, 0])
    g.model.nodes[(1, 103)] = ([3, 1], [0, 1, 0, 0, 0, 0])
    # existing top nodes exist but sit FAR from the bot boundary XY, so
    # every boundary bot node is unmatched -> mismatched > 0.
    g.model.bbox[top_tag] = (0, 0, 1, 1, 1, 1)
    g.model.nodes[(2, top_tag)] = (
        [4, 5, 6],
        [100, 100, 1, 101, 100, 1, 100, 101, 1],
    )
    g.model.max_node_tag = 6

    monkeypatch.setattr(wedge, "gmsh", g)

    with pytest.raises(WedgeBotNodeMismatchError):
        wedge._stamp_one(bot_tag, top_tag, vol_tag, _meta(), 1, 1e-3, {})

    kinds = {c[0] for c in g.model.log}
    assert "removeElements" not in kinds, g.model.log
    assert "addElementsByType" not in kinds, g.model.log


def test_stamp_one_intermediate_mismatch_fails_before_wedge_emit(monkeypatch):
    """(c) Intermediate-layer mismatch must raise before any wedge is emitted."""
    g = _FakeGmsh()
    bot_tag, top_tag, vol_tag, lf_tag = 10, 20, 30, 40
    lfk = ("__lat",)
    g.model.elements[(2, bot_tag)] = ([2], [[1]], [[1, 2, 3]])
    g.model.nodes[(2, bot_tag)] = ([1, 2, 3], [0, 0, 0, 1, 0, 0, 0, 1, 0])
    g.model.boundary[(2, bot_tag)] = [(1, 101), (1, 102), (1, 103)]
    g.model.nodes[(1, 101)] = ([1, 2], [0, 0, 0, 1, 0, 0])
    g.model.nodes[(1, 102)] = ([2, 3], [1, 0, 0, 0, 1, 0])
    g.model.nodes[(1, 103)] = ([3, 1], [0, 1, 0, 0, 0, 0])
    # Top nodes coincide with bot boundary XY at top_z -> top match succeeds.
    g.model.bbox[top_tag] = (0, 0, 1, 1, 1, 1)
    g.model.nodes[(2, top_tag)] = ([4, 5, 6], [0, 0, 1, 1, 0, 1, 0, 1, 1])
    # Lateral nodes exist at z_layer=0.5 but FAR in XY -> intermediate
    # boundary nodes cannot snap -> intermediate mismatch > 0.
    g.model.nodes[(2, lf_tag)] = ([7, 8, 9], [50, 50, 0.5, 51, 50, 0.5, 50, 51, 0.5])
    g.model.max_node_tag = 9

    monkeypatch.setattr(wedge, "gmsh", g)

    with pytest.raises(WedgeBotNodeMismatchError):
        wedge._stamp_one(
            bot_tag, top_tag, vol_tag, _meta([lfk]), 2, 1e-3, {lfk: lf_tag}
        )

    # No 6-node prism (wedge) element must have been emitted into the volume.
    wedge_emits = [
        c
        for c in g.model.log
        if c[0] == "addElementsByType" and c[1] == vol_tag and c[2] == 6
    ]
    assert wedge_emits == [], g.model.log


def test_stamp_one_clean_emits_wedges(monkeypatch):
    """Happy path: matching nodes -> wedges emitted, no error raised."""
    g = _FakeGmsh()
    bot_tag, top_tag, vol_tag = 10, 20, 30
    g.model.elements[(2, bot_tag)] = ([2], [[1]], [[1, 2, 3]])
    g.model.nodes[(2, bot_tag)] = ([1, 2, 3], [0, 0, 0, 1, 0, 0, 0, 1, 0])
    g.model.boundary[(2, bot_tag)] = [(1, 101), (1, 102), (1, 103)]
    g.model.nodes[(1, 101)] = ([1, 2], [0, 0, 0, 1, 0, 0])
    g.model.nodes[(1, 102)] = ([2, 3], [1, 0, 0, 0, 1, 0])
    g.model.nodes[(1, 103)] = ([3, 1], [0, 1, 0, 0, 0, 0])
    g.model.bbox[top_tag] = (0, 0, 1, 1, 1, 1)
    # Top nodes coincide with bot boundary XY -> all match, mismatched == 0.
    g.model.nodes[(2, top_tag)] = ([4, 5, 6], [0, 0, 1, 1, 0, 1, 0, 1, 1])
    g.model.max_node_tag = 6

    monkeypatch.setattr(wedge, "gmsh", g)

    wedge._stamp_one(bot_tag, top_tag, vol_tag, _meta(), 1, 1e-3, {})

    wedge_emits = [
        c
        for c in g.model.log
        if c[0] == "addElementsByType" and c[1] == vol_tag and c[2] == 6
    ]
    assert len(wedge_emits) == 1, g.model.log
