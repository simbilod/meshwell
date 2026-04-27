"""Test multi-dimensional entity processing capabilities.

This test module validates the ability to process entities of different
dimensions (3D volumes, 2D surfaces, 1D curves, 0D points) in a single model
using fragment-based integration.
"""

from __future__ import annotations

import pytest

from meshwell.cad_occ import cad_occ
from meshwell.mesh import mesh
from meshwell.occ_entity import OCC_entity
from meshwell.occ_xao_writer import write_xao
from meshwell.resolution import ConstantInField
from tests.test_occ_helpers import (
    _occ_box,
    _occ_circle,
    _occ_point,
    _occ_rectangle,
    _occ_sphere,
)


def test_volume_with_internal_surface():
    """Test 3D volume with 2D surface fragment integration."""
    box_obj = OCC_entity(
        occ_function=_occ_box(x=0, y=0, z=0, dx=2, dy=2, dz=1),
        physical_name="box",
        mesh_order=1,
        dimension=3,
    )

    surface_obj = OCC_entity(
        occ_function=_occ_rectangle(x=0.5, y=0.5, z=0.5, dx=1, dy=1),
        physical_name="internal_surface",
        mesh_order=2,
        dimension=2,
    )

    entities = [box_obj, surface_obj]
    write_xao(cad_occ(entities, progress_bars=True), "test_volume_with_surface.xao")
    mesh(
        dim=3,
        input_file="test_volume_with_surface.xao",
        output_file="test_volume_with_surface.msh",
        default_characteristic_length=1,
    )


def test_lower_dim_with_multiple_higher_entities():
    """Test lower-dimension object interacting with multiple higher-level entities."""
    box1_obj = OCC_entity(
        occ_function=_occ_box(x=0, y=0, z=0, dx=1, dy=1, dz=1),
        physical_name="box1",
        mesh_order=1,
        dimension=3,
    )

    box2_obj = OCC_entity(
        occ_function=_occ_box(x=1.5, y=0, z=0, dx=1, dy=1, dz=1),
        physical_name="box2",
        mesh_order=2,
        dimension=3,
    )

    spanning_surface = OCC_entity(
        occ_function=_occ_rectangle(x=0.5, y=0.5, z=0.5, dx=1.5, dy=0.5),
        physical_name="spanning_surface",
        mesh_order=3,
        dimension=2,
    )

    entities = [box1_obj, box2_obj, spanning_surface]
    write_xao(
        cad_occ(entities, progress_bars=True),
        "test_lower_dim_multiple_higher.xao",
    )
    mesh(
        dim=3,
        input_file="test_lower_dim_multiple_higher.xao",
        output_file="test_lower_dim_multiple_higher.msh",
        default_characteristic_length=1,
        resolution_specs={
            "spanning_surface": [ConstantInField(apply_to="surfaces", resolution=0.1)],
        },
    )


def test_surface_boundary_overlap():
    """Test surface that lies exactly on boundary of volume."""
    box_obj = OCC_entity(
        occ_function=_occ_box(x=0, y=0, z=0, dx=2, dy=2, dz=1),
        physical_name="box",
        mesh_order=1,
        dimension=3,
    )

    boundary_surface = OCC_entity(
        occ_function=_occ_rectangle(x=0.5, y=0.5, z=1.0, dx=1, dy=1),
        physical_name="boundary_surface",
        mesh_order=2,
        dimension=2,
    )

    entities = [box_obj, boundary_surface]
    write_xao(cad_occ(entities, progress_bars=True), "test_boundary_overlap.xao")
    mesh(
        dim=3,
        input_file="test_boundary_overlap.xao",
        output_file="test_boundary_overlap.msh",
        default_characteristic_length=1,
    )


@pytest.mark.skip
def test_point_in_multiple_entities():
    """Test 0D point inside multiple overlapping entities."""
    box_obj = OCC_entity(
        occ_function=_occ_box(x=0, y=0, z=0, dx=2, dy=2, dz=2),
        physical_name="box",
        mesh_order=1,
        dimension=3,
    )

    sphere_obj = OCC_entity(
        occ_function=_occ_sphere(xc=1, yc=1, zc=1, radius=1.2),
        physical_name="sphere",
        mesh_order=2,
        dimension=3,
    )

    surface_obj = OCC_entity(
        occ_function=_occ_rectangle(x=0.5, y=0.5, z=1, dx=1, dy=1),
        physical_name="surface",
        mesh_order=3,
        dimension=2,
    )

    point_obj = OCC_entity(
        occ_function=_occ_point(x=1, y=1, z=1),
        physical_name="central_point",
        mesh_order=4,
        dimension=0,
    )

    entities = [box_obj, sphere_obj, surface_obj, point_obj]
    write_xao(
        cad_occ(entities, progress_bars=True),
        "test_point_multiple_entities.xao",
    )
    mesh(
        dim=3,
        input_file="test_point_multiple_entities.xao",
        output_file="test_point_multiple_entities.msh",
        default_characteristic_length=1,
    )


def test_sequential_fragmentation_complex():
    """Test sequential fragmentation with multiple lower-dim entities."""
    base_volume = OCC_entity(
        occ_function=_occ_box(x=0, y=0, z=0, dx=3, dy=3, dz=2),
        physical_name="base_volume",
        mesh_order=1,
        dimension=3,
    )

    surface1 = OCC_entity(
        occ_function=_occ_rectangle(x=0.5, y=0.5, z=1, dx=2, dy=2),
        physical_name="surface1",
        mesh_order=2,
        dimension=2,
    )

    surface2 = OCC_entity(
        occ_function=_occ_rectangle(x=1, y=1, z=0.5, dx=1.5, dy=1.5),
        physical_name="surface2",
        mesh_order=3,
        dimension=2,
    )

    curve = OCC_entity(
        occ_function=_occ_circle(x=1.5, y=1.5, z=1.2, r=0.8),
        physical_name="curve",
        mesh_order=4,
        dimension=1,
    )

    point1 = OCC_entity(
        occ_function=_occ_point(x=1.2, y=1.2, z=1.2),
        physical_name="point1",
        mesh_order=5,
        dimension=0,
    )

    point2 = OCC_entity(
        occ_function=_occ_point(x=2, y=2, z=0.5),
        physical_name="point2",
        mesh_order=6,
        dimension=0,
    )

    entities = [base_volume, surface1, surface2, curve, point1, point2]
    write_xao(cad_occ(entities, progress_bars=True), "test_sequential_complex.xao")
    mesh(
        dim=3,
        input_file="test_sequential_complex.xao",
        output_file="test_sequential_complex.msh",
        default_characteristic_length=1,
    )


if __name__ == "__main__":
    test_volume_with_internal_surface()
    test_lower_dim_with_multiple_higher_entities()
    test_surface_boundary_overlap()
    test_sequential_fragmentation_complex()
