"""Tests for Phase 5(c) mid-height-cut detection at plan stage."""
from __future__ import annotations

import pytest
from shapely.geometry import Polygon


def _square(x=0, y=0, w=1, h=1) -> Polygon:
    return Polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])


def test_midheight_cut_by_non_structured_neighbour_raises():
    """Non-structured neighbour with z-endpoint inside slab z-extent is rejected.

    Tests the case where neighbour's zmin/zmax falls strictly inside the
    structured slab's z-extent and the xy footprints overlap.
    """
    from meshwell.polyprism import PolyPrism
    from meshwell.structured import (
        StructuredExtrusionResolutionSpec,
        StructuredMidHeightCutError,
        build_plan,
    )

    slab = PolyPrism(
        polygons=_square(0, 0, 4, 4),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
        physical_name="slab",
    )
    intruder = PolyPrism(
        polygons=_square(3, 1, 2, 2),
        buffers={0.3: 0.0, 1.5: 0.0},  # zmin=0.3 inside slab's (0, 1)
        physical_name="intruder",
    )

    with pytest.raises(StructuredMidHeightCutError, match="mid-height-cut"):
        build_plan([slab, intruder])


def test_midheight_cut_no_xy_overlap_does_not_raise():
    """If neighbour z-endpoint is mid-height but xy doesn't overlap, no cut."""
    from meshwell.polyprism import PolyPrism
    from meshwell.structured import StructuredExtrusionResolutionSpec, build_plan

    slab = PolyPrism(
        polygons=_square(0, 0, 4, 4),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
        physical_name="slab",
    )
    # Disjoint xy footprint.
    disjoint = PolyPrism(
        polygons=_square(10, 10, 2, 2),
        buffers={0.3: 0.0, 1.5: 0.0},
        physical_name="elsewhere",
    )
    plan = build_plan([slab, disjoint])
    assert len(plan.slabs) == 1


def test_neighbour_at_slab_zlo_zhi_planes_does_not_raise():
    """Stacked / capping neighbours sharing z-planes are fine — not mid-height."""
    from meshwell.polyprism import PolyPrism
    from meshwell.structured import StructuredExtrusionResolutionSpec, build_plan

    slab = PolyPrism(
        polygons=_square(0, 0, 4, 4),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
        physical_name="slab",
    )
    cap = PolyPrism(
        polygons=_square(0, 0, 4, 4),
        buffers={1.0: 0.0, 2.0: 0.0},  # zmin=1.0 == slab.zhi (NOT strictly inside)
        physical_name="cap",
    )
    build_plan([slab, cap])  # no raise


def test_neighbour_z_envelopes_slab_does_not_raise():
    """Enveloping z-range (extends beyond slab on both sides) is fine.

    Neither zmin nor zmax falls inside the slab z-extent, so no
    mid-height cut from this neighbour.

    Note: xy footprint is disjoint from the slab. An unstructured
    neighbour that overlapped the slab in xy would trip the lateral-
    neighbour conformality check (see test_lateral_neighbour.py),
    which is a different concern from mid-height cuts.
    """
    from meshwell.polyprism import PolyPrism
    from meshwell.structured import StructuredExtrusionResolutionSpec, build_plan

    slab = PolyPrism(
        polygons=_square(0, 0, 4, 4),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
        physical_name="slab",
    )
    cladding = PolyPrism(
        polygons=_square(10, 10, 4, 4),  # xy disjoint from slab
        buffers={-1.0: 0.0, 2.0: 0.0},  # zmin=-1 and zmax=2 both outside (0,1)
        physical_name="cladding",
    )
    build_plan([slab, cladding])  # no raise


def test_midheight_cut_zmax_inside_slab_raises():
    """Zmax inside the slab also triggers the check."""
    from meshwell.polyprism import PolyPrism
    from meshwell.structured import (
        StructuredExtrusionResolutionSpec,
        StructuredMidHeightCutError,
        build_plan,
    )

    slab = PolyPrism(
        polygons=_square(0, 0, 4, 4),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
        physical_name="slab",
    )
    intruder = PolyPrism(
        polygons=_square(1, 1, 2, 2),
        buffers={-0.5: 0.0, 0.7: 0.0},  # zmax=0.7 inside (0, 1)
        physical_name="below_partial",
    )
    with pytest.raises(StructuredMidHeightCutError, match="mid-height-cut"):
        build_plan([slab, intruder])
