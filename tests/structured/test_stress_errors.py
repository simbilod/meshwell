"""Negative stress tests: every planner failure mode raises with the correct exception class.

Each test targets one documented error class; assertions are on the exception
type (not message) so they remain valid across message changes.
"""
import pytest
from shapely.geometry import Polygon

from meshwell.orchestrator import generate_mesh
from meshwell.polyprism import PolyPrism
from meshwell.resolution import StructuredExtrusionResolutionSpec
from meshwell.structured.exceptions import (
    MixedIdentifyArcsError,
    StructuredEntityTypeError,
    StructuredExtrudeRequiredError,
    StructuredLateralNLayersMismatchError,
    StructuredZStackError,
)

SQ = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
SQ2 = Polygon([(10, 0), (20, 0), (20, 10), (10, 10)])  # shares edge x=10
FAR = Polygon([(100, 100), (110, 100), (110, 110), (100, 110)])


def test_structured_on_buffered_raises():
    with pytest.raises(StructuredExtrudeRequiredError):
        PolyPrism(SQ, {0.0: 0.0, 1.0: 0.5}, physical_name="x", structured=True)


def test_non_polyprism_structured_raises(tmp_path):
    class Fake:
        structured = True
        physical_name = "fake"

        def instanciate_occ(self):
            return None

    with pytest.raises(StructuredEntityTypeError):
        generate_mesh(
            [Fake()],
            dim=3,
            output_mesh=tmp_path / "x.msh",
            default_characteristic_length=1.0,
        )


def test_zstack_violation_raises(tmp_path):
    s = PolyPrism(SQ, {0.0: 0.0, 2.0: 0.0}, physical_name="s", structured=True)
    bad = PolyPrism(SQ, {1.0: 0.0, 3.0: 0.0}, physical_name="bad")
    with pytest.raises(StructuredZStackError):
        generate_mesh(
            [s, bad],
            dim=3,
            output_mesh=tmp_path / "x.msh",
            default_characteristic_length=1.0,
        )


def test_n_layers_mismatch_lateral_touch_raises(tmp_path):
    a = PolyPrism(SQ, {0.0: 0.0, 1.0: 0.0}, physical_name="a", structured=True)
    b = PolyPrism(SQ2, {0.0: 0.0, 1.0: 0.0}, physical_name="b", structured=True)
    # Cladding to satisfy wrapping; spans the union of a + b footprints.
    union = Polygon([(0, 0), (20, 0), (20, 10), (0, 10)])
    below = PolyPrism(union, {-1.0: 0.0, 0.0: 0.0}, physical_name="below")
    above = PolyPrism(union, {1.0: 0.0, 2.0: 0.0}, physical_name="above")
    with pytest.raises(StructuredLateralNLayersMismatchError):
        generate_mesh(
            [a, b, below, above],
            dim=3,
            output_mesh=tmp_path / "x.msh",
            default_characteristic_length=1.0,
            resolution_specs={
                "a": [StructuredExtrusionResolutionSpec(n_layers=2)],
                "b": [StructuredExtrusionResolutionSpec(n_layers=5)],
            },
        )


def test_mixed_identify_arcs_raises(tmp_path):
    """When structured entities exist and any PolyPrism opts into arcs, all must."""
    a = PolyPrism(
        SQ, {0.0: 0.0, 1.0: 0.0}, physical_name="a", structured=True, identify_arcs=True
    )
    b = PolyPrism(
        SQ2,
        {0.0: 0.0, 1.0: 0.0},
        physical_name="b",
        structured=True,
        identify_arcs=False,
    )
    with pytest.raises(MixedIdentifyArcsError):
        generate_mesh(
            [a, b],
            dim=3,
            output_mesh=tmp_path / "x.msh",
            default_characteristic_length=1.0,
        )
