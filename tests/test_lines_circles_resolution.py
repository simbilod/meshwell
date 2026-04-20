from __future__ import annotations

import shapely

from meshwell.cad_occ import cad_occ
from meshwell.mesh import mesh
from meshwell.occ_xao_writer import write_xao
from meshwell.polysurface import PolySurface
from meshwell.resolution import ConstantInField


def test_lines_circles_disambiguation():
    square = shapely.box(0, 0, 10, 10)
    circle = shapely.Point(5, 5).buffer(2)
    geometry = square.difference(circle)

    entity = PolySurface(
        polygons=geometry,
        physical_name="entity",
        mesh_order=1,
    )

    write_xao(cad_occ([entity]), "test_lc.xao")

    # Case 1: straight lines coarse (res=5), circles fine (res=0.1)
    # → dominated by dense circle discretization
    mesh_1 = mesh(
        dim=2,
        input_file="test_lc.xao",
        resolution_specs={
            "entity": [
                ConstantInField(apply_to="lines", resolution=5.0),
                ConstantInField(apply_to="circles", resolution=0.1),
            ]
        },
        default_characteristic_length=10,
    )

    # Case 2: straight lines fine (res=0.1), circles coarse (res=5)
    # → dominated by dense line discretization
    mesh_2 = mesh(
        dim=2,
        input_file="test_lc.xao",
        resolution_specs={
            "entity": [
                ConstantInField(apply_to="lines", resolution=0.1),
                ConstantInField(apply_to="circles", resolution=5.0),
            ]
        },
        default_characteristic_length=10,
    )

    # Case 3: all curves fine (res=0.1) — most points overall
    mesh_3 = mesh(
        dim=2,
        input_file="test_lc.xao",
        resolution_specs={
            "entity": [
                ConstantInField(apply_to="curves", resolution=0.1),
            ]
        },
        default_characteristic_length=10,
    )

    n1, n2, n3 = len(mesh_1.points), len(mesh_2.points), len(mesh_3.points)
    print(f"Points (lines=5, circles=0.1): {n1}")
    print(f"Points (lines=0.1, circles=5): {n2}")
    print(f"Points (curves=0.1):           {n3}")

    # Swapping which subtype is fine/coarse must produce different meshes
    assert n1 != n2, "Line vs circle resolution swap should produce different meshes"

    # Fine on everything should dominate whichever single subtype is fine
    assert n3 > n1, "All curves fine should have more points than only circles fine"
    assert n3 > n2, "All curves fine should have more points than only lines fine"


if __name__ == "__main__":
    test_lines_circles_disambiguation()
