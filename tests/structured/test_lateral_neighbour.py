"""Tests for unstructured-neighbour lateral-face conformality check."""
from __future__ import annotations

import pytest
from shapely.geometry import Polygon


def _square(x=0, y=0, w=1, h=1) -> Polygon:
    return Polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])


def test_unstructured_touching_lateral_raises():
    """Unstructured neighbour abutting slab side: quad/tri lateral mismatch."""
    from meshwell.polyprism import PolyPrism
    from meshwell.structured import (
        StructuredExtrusionResolutionSpec,
        StructuredLateralUnstructuredNeighbourError,
        build_plan,
    )

    slab = PolyPrism(
        polygons=_square(0, 0, 4, 4),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
        physical_name="slab",
    )
    neighbour = PolyPrism(
        polygons=_square(4, 0, 4, 4),  # shares the slab's x=4 edge
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="neighbour",
    )
    with pytest.raises(
        StructuredLateralUnstructuredNeighbourError, match="lateral face"
    ):
        build_plan([slab, neighbour])


def test_unstructured_enveloping_slab_raises():
    """Unstructured cladding surrounding slab shares all four laterals."""
    from meshwell.polyprism import PolyPrism
    from meshwell.structured import (
        StructuredExtrusionResolutionSpec,
        StructuredLateralUnstructuredNeighbourError,
        build_plan,
    )

    slab = PolyPrism(
        polygons=_square(0, 0, 4, 4),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
        physical_name="slab",
    )
    cladding = PolyPrism(
        polygons=_square(-2, -2, 8, 8),
        buffers={-1.0: 0.0, 2.0: 0.0},
        physical_name="cladding",
    )
    with pytest.raises(StructuredLateralUnstructuredNeighbourError):
        build_plan([slab, cladding])


def test_unstructured_disjoint_xy_does_not_raise():
    """Unstructured neighbour with disjoint xy footprint is fine."""
    from meshwell.polyprism import PolyPrism
    from meshwell.structured import StructuredExtrusionResolutionSpec, build_plan

    slab = PolyPrism(
        polygons=_square(0, 0, 4, 4),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
        physical_name="slab",
    )
    elsewhere = PolyPrism(
        polygons=_square(10, 10, 2, 2),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="elsewhere",
    )
    build_plan([slab, elsewhere])  # no raise


def test_unstructured_z_separated_does_not_raise():
    """Unstructured neighbour stacked above slab (no z-overlap) is fine."""
    from meshwell.polyprism import PolyPrism
    from meshwell.structured import StructuredExtrusionResolutionSpec, build_plan

    slab = PolyPrism(
        polygons=_square(0, 0, 4, 4),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
        physical_name="slab",
    )
    above = PolyPrism(
        polygons=_square(4, 0, 4, 4),  # would share lateral if z-overlapped
        buffers={1.0: 0.0, 2.0: 0.0},  # z=[1,2], slab z=[0,1] — just touching
        physical_name="above",
    )
    build_plan([slab, above])  # no raise: z_overlap == 0


def test_structured_neighbour_sharing_lateral_does_not_raise():
    """Two structured slabs sharing a lateral face are fine (both quad)."""
    from meshwell.polyprism import PolyPrism
    from meshwell.structured import StructuredExtrusionResolutionSpec, build_plan

    slab_a = PolyPrism(
        polygons=_square(0, 0, 4, 4),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
        physical_name="slab_a",
    )
    slab_b = PolyPrism(
        polygons=_square(4, 0, 4, 4),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
        physical_name="slab_b",
    )
    build_plan([slab_a, slab_b])  # no raise


def test_unstructured_keep_false_helper_does_not_raise():
    """keep=False entities are carving helpers, not in final mesh — allowed."""
    from meshwell.polyprism import PolyPrism
    from meshwell.structured import StructuredExtrusionResolutionSpec, build_plan

    slab = PolyPrism(
        polygons=_square(0, 0, 4, 4),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
        physical_name="slab",
    )
    helper = PolyPrism(
        polygons=_square(4, 0, 4, 4),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="helper",
        mesh_bool=False,  # keep=False in the meshed model
    )
    build_plan([slab, helper])  # no raise


def test_unstructured_point_contact_does_not_raise():
    """0D corner-only contact (no shared lateral curve) is fine."""
    from meshwell.polyprism import PolyPrism
    from meshwell.structured import StructuredExtrusionResolutionSpec, build_plan

    slab = PolyPrism(
        polygons=_square(0, 0, 4, 4),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
        physical_name="slab",
    )
    corner = PolyPrism(
        polygons=_square(4, 4, 4, 4),  # touches slab only at (4, 4)
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="corner",
    )
    build_plan([slab, corner])  # no raise: shared.length == 0
