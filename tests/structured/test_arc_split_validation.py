"""Tests for the arc-partition alignment validator (StructuredArcSplitError)."""
from __future__ import annotations

import math

import pytest
from shapely.geometry import Polygon


def _disc(cx=0.0, cy=0.0, r=1.5, n=32):
    return Polygon(
        [
            (
                cx + r * math.cos(2 * math.pi * i / n),
                cy + r * math.sin(2 * math.pi * i / n),
            )
            for i in range(n)
        ]
    )


def _square(x, y, w, h):
    return Polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])


@pytest.mark.skip(
    reason=(
        "The arc-neighbour-alignment validator was removed by the planar-"
        "arrangement refactor (spec line 192: 'There is nothing to validate'). "
        "Canonical-edge sharing makes chord-vs-arc divergence impossible. "
        "Test will be deleted in Task 15 along with the validator."
    )
)
def test_arc_split_at_non_polygon_vertex_raises():
    """Two overlapping non-aligned contacts that would split an arc disc raise."""
    from meshwell.polyprism import PolyPrism
    from meshwell.structured import (
        StructuredArcSplitError,
        StructuredExtrusionResolutionSpec,
        build_plan,
    )

    disc = PolyPrism(
        polygons=_disc(0, 0, 1.5, n=32),
        buffers={0.5: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
        identify_arcs=True,
        physical_name="disc",
        mesh_order=1.0,
    )
    contact_a = PolyPrism(
        polygons=_square(-1.5, -1.5, 1.5, 3),  # right edge at x=0 - aligned
        buffers={1.0: 0.0, 1.3: 0.0},
        physical_name="contact_a",
    )
    contact_b = PolyPrism(
        polygons=_square(
            -0.5, -1.5, 1.5, 3
        ),  # left edge at x=-0.5 NOT a polygon vertex
        buffers={1.0: 0.0, 1.3: 0.0},
        physical_name="contact_b",
    )

    with pytest.raises(
        StructuredArcSplitError, match="not an original arc polygon vertex"
    ):
        build_plan([disc, contact_a, contact_b])


def test_arc_split_aligned_at_polygon_vertex_succeeds():
    """A cut aligned with a polygon vertex (x=0) does NOT raise."""
    from meshwell.polyprism import PolyPrism
    from meshwell.structured import StructuredExtrusionResolutionSpec, build_plan

    disc = PolyPrism(
        polygons=_disc(0, 0, 1.5, n=32),
        buffers={0.5: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
        identify_arcs=True,
        physical_name="disc",
        mesh_order=1.0,
    )
    contact = PolyPrism(
        polygons=_square(-1.5, -1.5, 1.5, 3),  # cut at x=0 = polygon vertex
        buffers={1.0: 0.0, 1.3: 0.0},
        physical_name="contact",
    )
    plan = build_plan([disc, contact])
    assert len(plan.slabs) == 1
    assert plan.slabs[0].face_partition is not None


def test_arc_disc_no_neighbours_does_not_raise():
    """Single-piece arc disc never triggers the split validation."""
    from meshwell.polyprism import PolyPrism
    from meshwell.structured import StructuredExtrusionResolutionSpec, build_plan

    disc = PolyPrism(
        polygons=_disc(),
        buffers={0.5: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
        identify_arcs=True,
        physical_name="disc",
    )
    plan = build_plan([disc])
    assert len(plan.slabs) == 1
    assert len(plan.slabs[0].face_partition) == 1


def test_non_arc_slab_split_does_not_raise():
    """A non-arc structured slab can be split arbitrarily — validation is skipped."""
    from meshwell.polyprism import PolyPrism
    from meshwell.structured import StructuredExtrusionResolutionSpec, build_plan

    rect = PolyPrism(
        polygons=_square(0, 0, 4, 4),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
        physical_name="rect",
        mesh_order=1.0,
    )
    cap = PolyPrism(
        polygons=_square(0, 0, 1.7, 4),  # arbitrary x=1.7 cut, but rect isn't arc
        buffers={1.0: 0.0, 2.0: 0.0},
        physical_name="cap",
    )
    plan = build_plan([rect, cap])
    assert len(plan.slabs) == 1
    assert len(plan.slabs[0].face_partition) == 2
