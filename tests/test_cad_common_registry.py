"""Tests for :mod:`meshwell.cad_common`'s shared pre-pass.

Covers two bugs in ``prepare_entities``:

1. The ``polygon_ents`` registry built for ``InterfaceTag.resolve()`` was
   keyed only by an entity's FIRST physical name. A two-name entity (a
   common pattern for shared/aliased regions) silently vanished from
   ``targets=`` lookups on its second name.
2. The global bbox used to clip each buffered polygon was inflated by
   only ``1 * perturbation``, but shapely's mitre join (the default
   ``join_style=2`` used here) can extend a sharp convex corner's offset
   up to ``mitre_limit * perturbation`` (shapely default ``mitre_limit=5``).
   The under-sized inflation shaved sharp corners sitting at the scene's
   bounding-box edge.
"""
from __future__ import annotations

import math

import shapely
from shapely.geometry import LineString

from meshwell.cad_common import prepare_entities
from meshwell.interface_tag import InterfaceTag
from meshwell.polysurface import PolySurface


def test_prepare_entities_registers_entity_under_all_physical_names():
    """An InterfaceTag targeting the SECOND name of a two-name entity must resolve.

    Entity carries physical_name=("A", "B") -- e.g. a shared region known
    under two aliases. Before the fix, ``polygon_ents`` only had an "A"
    key, so ``targets=["B"]`` silently matched nothing and the tag
    resolved to zero segments (with a warning). After the fix, the
    entity is registered under both "A" and "B", so it resolves.
    """
    square = shapely.Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
    entity = PolySurface(polygons=square, physical_name=("A", "B"), mesh_order=1)

    tag = InterfaceTag(
        linestrings=LineString([(5, 0), (5, 5)]),
        zmin=0.0,
        zmax=1.0,
        physical_name="iface",
        targets=["B"],
    )

    prepare_entities([entity, tag], perturbation=1e-3)

    assert tag.resolved_linestrings, (
        "InterfaceTag targeting the second physical name of a two-name "
        "entity failed to resolve"
    )
    # Sanity: the resolved trace sits on the buffered right edge of the
    # square (x = 5 + perturbation), not somewhere spurious.
    xs = {round(x, 6) for ls in tag.resolved_linestrings for x, _ in ls.coords}
    assert xs == {round(5 + 1e-3, 6)}, xs


def test_prepare_entities_registers_entity_under_both_names_no_duplicate_cut():
    """A two-name entity must not be double-counted when targets=None.

    Registering an entity under every physical name means it can appear
    under multiple keys in ``polygon_ents``. The "any target" branch
    (targets=None) sums entities across all keys, so without
    deduplication a two-name entity would be processed twice by the cut
    cascade. Compare against an equivalent single-name scene: the
    resolved trace must be identical, proving the second registration
    didn't perturb the result.
    """
    square = shapely.Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])

    entity_two_names = PolySurface(
        polygons=square, physical_name=("A", "A_alias"), mesh_order=1
    )
    tag_two_names = InterfaceTag(
        linestrings=LineString([(5, 0), (5, 5)]),
        zmin=0.0,
        zmax=1.0,
        physical_name="iface",
        targets=None,
    )
    prepare_entities([entity_two_names, tag_two_names], perturbation=1e-3)

    entity_one_name = PolySurface(polygons=square, physical_name="A", mesh_order=1)
    tag_one_name = InterfaceTag(
        linestrings=LineString([(5, 0), (5, 5)]),
        zmin=0.0,
        zmax=1.0,
        physical_name="iface",
        targets=None,
    )
    prepare_entities([entity_one_name, tag_one_name], perturbation=1e-3)

    assert len(tag_two_names.resolved_linestrings) == len(
        tag_one_name.resolved_linestrings
    )
    len_two = sum(ls.length for ls in tag_two_names.resolved_linestrings)
    len_one = sum(ls.length for ls in tag_one_name.resolved_linestrings)
    assert math.isclose(len_two, len_one, rel_tol=1e-9)


def test_prepare_entities_mitre_bbox_inflation_preserves_sharp_corner_tip():
    """A sharp convex spike's mitre-buffer tip must survive the bbox clip.

    Constructed so the interior angle at the spike's apex gives an
    (uncapped, since below shapely's default ``mitre_limit=5``) mitre
    extension of ``perturbation / sin(half_angle) ~= 3.33 * perturbation``
    past the vertex -- comfortably more than the OLD single-perturbation
    bbox inflation (which would clip it) and comfortably less than the
    NEW ``5 * perturbation`` inflation (which must preserve it fully).
    """
    sin_half_angle = 0.3
    half_height = 1.0
    depth = half_height * math.sqrt(1.0 / sin_half_angle**2 - 1.0)
    pert = 0.1

    apex_x = 4.0 + depth
    spike = shapely.Polygon(
        [
            (0, 0),
            (4, 0),
            (apex_x, half_height),
            (4, 2 * half_height),
            (0, 2 * half_height),
        ]
    )
    entity = PolySurface(polygons=spike, physical_name="spike", mesh_order=1)

    prepare_entities([entity], perturbation=pert)

    buffered = (
        entity.polygons[0] if isinstance(entity.polygons, list) else entity.polygons
    )
    tip_x = buffered.bounds[2]  # xmax

    # ``prepare_entities`` relaxes shapely's precision grid to
    # ``perturbation / 100`` before buffering (so sub-tolerance buffers
    # take effect); GEOS snaps buffer output to that same grid, so the
    # analytic mitre-tip prediction is only exact up to ~that grid size.
    # abs_tol is generous relative to that noise floor while still far
    # tighter than the ~0.23 gap between the OLD (shaved) and NEW
    # (preserved) bbox inflation asserted below.
    expected_tip_x = apex_x + pert / sin_half_angle
    assert math.isclose(tip_x, expected_tip_x, abs_tol=2e-3), (tip_x, expected_tip_x)

    # Regression guard: the OLD inflation (1 * perturbation) would have
    # clipped the tip well short of the true mitre extension.
    old_inflated_xmax = apex_x + pert
    assert (
        old_inflated_xmax < expected_tip_x - 1e-6
    ), "test scenario does not actually exercise a shave case"
