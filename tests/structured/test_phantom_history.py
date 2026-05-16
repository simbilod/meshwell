"""Tests for meshwell.structured.phantom.extract_phantom_map.

Tests construct a small BOP scene with OCP directly:
  - One phantom sub-prism (a 1x1x1 box).
  - One neighbour (a 0.3x0.3 stick passing through, or a half-cover top box).
We run BOPAlgo_Builder, then call extract_phantom_map, then assert the
output map's structure.
"""
from __future__ import annotations

from shapely.geometry import Polygon


def _square(x=0, y=0, w=1, h=1) -> Polygon:
    return Polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])


def _run_bop(*shapes):
    """Run BOPAlgo_Builder on the given input shapes; return the builder."""
    from OCP.BOPAlgo import BOPAlgo_Builder

    builder = BOPAlgo_Builder()
    for s in shapes:
        builder.AddArgument(s)
    builder.Perform()
    return builder


def test_extract_phantom_map_no_neighbours_identity():
    """With no neighbour shape, every input maps to itself (one-element list)."""
    from meshwell.structured.phantom import _build_sub_prism, extract_phantom_map
    from meshwell.structured.spec import (
        FaceKey,
        PhantomBuildResult,
    )

    shape = _build_sub_prism(_square(), zlo=0.0, zhi=1.0, slab_index=0, piece_index=0)
    build_result = PhantomBuildResult(shapes=(shape,))
    builder = _run_bop(shape.solid)

    pmap = extract_phantom_map(build_result, builder)
    # Every input face should map to exactly one output (itself).
    assert all(len(v) == 1 for v in pmap.output_faces.values())
    assert FaceKey(0, "bot", 0) in pmap.output_faces
    assert FaceKey(0, "top", 0) in pmap.output_faces


def test_extract_phantom_map_neighbour_cuts_top_face():
    """A neighbour box overlapping the phantom's top half-plane splits the top face."""
    from meshwell.structured.phantom import _build_sub_prism, extract_phantom_map
    from meshwell.structured.spec import FaceKey, PhantomBuildResult
    from tests.structured._occ_helpers import make_box

    # Phantom box from z=0 to z=1 with footprint [0,4]x[0,4].
    shape = _build_sub_prism(
        _square(0, 0, 4, 4), zlo=0.0, zhi=1.0, slab_index=0, piece_index=0
    )
    # Neighbour box sits z=[1,2] over half the phantom: covers x in [0,2], y in [0,4].
    # Its bottom face at z=1 partially-covers the phantom's top face at z=1.
    neighbour = make_box(0, 0, 1, 2, 4, 1)
    build_result = PhantomBuildResult(shapes=(shape,))
    builder = _run_bop(shape.solid, neighbour)

    pmap = extract_phantom_map(build_result, builder)
    top_key = FaceKey(0, "top", 0)
    # The top face was cut into 2 pieces by the neighbour bottom-face boundary.
    assert (
        len(pmap.output_faces[top_key]) >= 2
    ), f"Expected the top face to split, got {len(pmap.output_faces[top_key])} pieces"
    # The bottom face was untouched.
    bot_key = FaceKey(0, "bot", 0)
    assert len(pmap.output_faces[bot_key]) == 1


def test_extract_phantom_map_lateral_face_through_cut():
    """A stick overlapping a phantom lateral face cuts it."""
    from meshwell.structured.phantom import _build_sub_prism, extract_phantom_map
    from meshwell.structured.spec import PhantomBuildResult
    from tests.structured._occ_helpers import make_stick

    shape = _build_sub_prism(
        _square(0, 0, 4, 4), zlo=0.0, zhi=1.0, slab_index=0, piece_index=0
    )
    # Stick spans x in [1,3], y in [3.5,4.5], z in [-0.5, 1.5] — overlaps the
    # y=4 lateral face of the phantom.
    stick = make_stick(1.0, 3.5, -0.5, 1.5, 2.0, 1.0)
    build_result = PhantomBuildResult(shapes=(shape,))
    builder = _run_bop(shape.solid, stick)

    pmap = extract_phantom_map(build_result, builder)
    cut_counts = [len(v) for v in pmap.output_laterals.values()]
    assert any(
        c > 1 for c in cut_counts
    ), f"Expected at least one lateral face to be cut, got counts {cut_counts}"


def test_extract_phantom_map_records_all_input_keys():
    """Every key in the PhantomShape's input_*_by_key should appear in the map."""
    from meshwell.structured.phantom import _build_sub_prism, extract_phantom_map
    from meshwell.structured.spec import LateralKey, PhantomBuildResult

    shape = _build_sub_prism(_square(), zlo=0.0, zhi=1.0, slab_index=0, piece_index=0)
    builder = _run_bop(shape.solid)
    pmap = extract_phantom_map(PhantomBuildResult(shapes=(shape,)), builder)

    assert set(pmap.output_faces.keys()) == set(shape.input_faces_by_key.keys())
    assert set(pmap.output_edges.keys()) == set(shape.input_edges_by_key.keys())
    assert set(pmap.output_vertices.keys()) == set(shape.input_vertices_by_key.keys())
    expected_lateral_keys = {
        LateralKey(slab_index=0, outer_edge_index=i)
        for i in shape.input_laterals_by_outer_edge
    }
    assert set(pmap.output_laterals.keys()) == expected_lateral_keys
    assert set(pmap.lateral_has_midheight_cut.keys()) == expected_lateral_keys


def test_lateral_no_midheight_cut_default():
    """A phantom with no neighbours should have no mid-height cuts."""
    from meshwell.structured.phantom import _build_sub_prism, extract_phantom_map
    from meshwell.structured.spec import PhantomBuildResult

    shape = _build_sub_prism(_square(), zlo=0.0, zhi=1.0, slab_index=0, piece_index=0)
    builder = _run_bop(shape.solid)
    pmap = extract_phantom_map(PhantomBuildResult(shapes=(shape,)), builder)
    # All four laterals should have no mid-height cut.
    assert all(not v for v in pmap.lateral_has_midheight_cut.values())


def test_lateral_midheight_cut_detected_from_partial_neighbour():
    """A neighbour that touches only part of a lateral face is a mid-height cut."""
    from meshwell.structured.phantom import _build_sub_prism, extract_phantom_map
    from meshwell.structured.spec import PhantomBuildResult
    from tests.structured._occ_helpers import make_box

    # Phantom z=[0,2], 4x4 footprint.
    shape = _build_sub_prism(
        _square(0, 0, 4, 4), zlo=0.0, zhi=2.0, slab_index=0, piece_index=0
    )
    # Neighbour box: protrudes into the y=4 lateral face but only at z in [0.5, 1.5].
    # This puts new vertices at z=0.5 and z=1.5 on the lateral face -> mid-height cut.
    neighbour = make_box(1.0, 3.5, 0.5, 2.0, 1.0, 1.0)
    builder = _run_bop(shape.solid, neighbour)
    pmap = extract_phantom_map(PhantomBuildResult(shapes=(shape,)), builder)

    # At least one lateral should have a mid-height cut.
    assert any(pmap.lateral_has_midheight_cut.values()), (
        f"Expected mid-height cut on a lateral, got "
        f"{dict(pmap.lateral_has_midheight_cut)}"
    )
