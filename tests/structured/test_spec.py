"""Tests for meshwell.structured.spec dataclasses + validators."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from meshwell.structured.spec import StructuredExtrusionResolutionSpec


def test_spec_minimal_valid():
    spec = StructuredExtrusionResolutionSpec(n_layers=[3, 5])
    assert spec.n_layers == [3, 5]
    assert spec.recombine is False


def test_spec_recombine_true():
    spec = StructuredExtrusionResolutionSpec(n_layers=[2], recombine=True)
    assert spec.recombine is True


def test_spec_rejects_empty_n_layers():
    with pytest.raises(ValidationError, match="n_layers"):
        StructuredExtrusionResolutionSpec(n_layers=[])


def test_spec_rejects_non_positive_layer_count():
    with pytest.raises(ValidationError, match="positive"):
        StructuredExtrusionResolutionSpec(n_layers=[3, 0, 4])

    with pytest.raises(ValidationError, match="positive"):
        StructuredExtrusionResolutionSpec(n_layers=[-1])


def test_spec_equality_on_identical_fields():
    """Two specs with identical fields compare equal (pydantic value semantics)."""
    a = StructuredExtrusionResolutionSpec(n_layers=[2])
    b = StructuredExtrusionResolutionSpec(n_layers=[2])
    assert a == b


def test_structured_partition_convergence_error_is_runtime_error():
    """The new convergence error must be a RuntimeError subclass and exportable."""
    from meshwell.structured import StructuredPartitionConvergenceError

    assert issubclass(StructuredPartitionConvergenceError, RuntimeError)
    err = StructuredPartitionConvergenceError("did not converge")
    assert "did not converge" in str(err)


def test_canonical_circle_is_hashable():
    """CanonicalCircle is a frozen dataclass usable as a dict/set key."""
    from meshwell.structured import CanonicalCircle

    a = CanonicalCircle(center=(1.0, 2.0), radius=3.0)
    b = CanonicalCircle(center=(1.0, 2.0), radius=3.0)
    assert a == b
    assert hash(a) == hash(b)
    assert a in {b}


def test_arrangement_edge_carries_circle_or_none():
    """ArrangementEdge is a line when circle is None, an arc when not."""
    from meshwell.structured import ArrangementEdge, CanonicalCircle

    line = ArrangementEdge(edge_id=0, vertices=((0.0, 0.0), (1.0, 0.0)), circle=None)
    arc = ArrangementEdge(
        edge_id=1,
        vertices=((1.0, 0.0), (0.707, 0.707), (0.0, 1.0)),
        circle=CanonicalCircle(center=(0.0, 0.0), radius=1.0),
    )
    assert line.circle is None
    assert arc.circle is not None
    assert arc.circle.radius == 1.0


def test_arrangement_face_holds_polygon_and_boundary():
    """ArrangementFace carries a Polygon and an ordered edge-id list."""
    from shapely.geometry import Polygon

    from meshwell.structured import ArrangementFace

    f = ArrangementFace(
        face_id=0,
        polygon=Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        boundary=[(0, False), (1, False), (2, False), (3, False)],
    )
    assert f.face_id == 0
    assert f.polygon.area == 1.0
    assert len(f.boundary) == 4


def test_stack_arrangement_holds_edges_and_faces():
    """StackArrangement is the per-component output type."""
    from meshwell.structured import StackArrangement

    s = StackArrangement(edges=[], faces=[])
    assert s.edges == []
    assert s.faces == []


def test_slab_has_resolved_footprint_and_face_partition_edges():
    """Slab gains two new optional fields for the arrangement pipeline."""
    from shapely.geometry import Polygon

    from meshwell.structured.spec import Slab

    s = Slab(
        footprint=Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        zlo=0.0,
        zhi=1.0,
        physical_name=("X",),
        source_index=0,
        z_interval_index=0,
        mesh_order=1.0,
    )
    # New fields default to safe values.
    assert s.resolved_footprint is None
    assert s.face_partition_edges is None
