"""Regression: a named 0D OCC point is carried through to the mesh as a named node."""

from __future__ import annotations

import numpy as np
import shapely
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
from OCP.gp import gp_Pnt

from meshwell.occ_entity import OCC_entity
from meshwell.orchestrator import generate_mesh
from meshwell.polysurface import PolySurface


def _named_point(x, y, name):
    """A named 0D OCC point at (x, y): the documented recipe."""
    return OCC_entity(
        occ_function=lambda: BRepBuilderAPI_MakeVertex(gp_Pnt(x, y, 0.0)).Vertex(),
        physical_name=name,
        dimension=0,
    )


def _stack():
    return [
        PolySurface(
            polygons=shapely.box(0, 0, 4, 1), physical_name="lower", mesh_order=2
        ),
        PolySurface(
            polygons=shapely.box(0, 1, 4, 2), physical_name="upper", mesh_order=1
        ),
    ]


def _has_node_near(mesh, x, y, tol=1e-3):
    pts = mesh.points[:, :2]
    return bool((np.hypot(pts[:, 0] - x, pts[:, 1] - y) < tol).any())


def _named_vertex_present(mesh, name):
    """True if `name` is a physical group backed by a vertex cell block."""
    if name not in mesh.cell_sets:
        return False
    for block, idx in zip(mesh.cells, mesh.cell_sets[name]):
        if block.type == "vertex" and idx is not None and len(idx):
            return True
    return False


def test_named_point_inside_region(tmp_path):
    pt = _named_point(2.0, 0.5, "pin")
    mesh = generate_mesh(
        entities=[*_stack(), pt],
        dim=2,
        output_mesh=str(tmp_path / "pin.msh"),
        default_characteristic_length=0.5,
    )
    assert _named_vertex_present(mesh, "pin")
    assert _has_node_near(mesh, 2.0, 0.5)


def test_named_point_on_interface(tmp_path):
    pt = _named_point(2.0, 1.0, "oniface")
    mesh = generate_mesh(
        entities=[*_stack(), pt],
        dim=2,
        output_mesh=str(tmp_path / "oniface.msh"),
        default_characteristic_length=0.5,
    )
    assert _named_vertex_present(mesh, "oniface")
    assert _has_node_near(mesh, 2.0, 1.0)


def test_named_point_on_corner(tmp_path):
    pt = _named_point(0.0, 1.0, "oncorner")
    mesh = generate_mesh(
        entities=[*_stack(), pt],
        dim=2,
        output_mesh=str(tmp_path / "oncorner.msh"),
        default_characteristic_length=0.5,
    )
    assert _named_vertex_present(mesh, "oncorner")
    assert _has_node_near(mesh, 0.0, 1.0)


def test_named_point_group_in_xao(tmp_path):
    """The dim-0 physical group is written into the XAO checkpoint too."""
    import gmsh

    xao = tmp_path / "pin.xao"
    generate_mesh(
        entities=[*_stack(), _named_point(2.0, 0.5, "pin")],
        dim=2,
        checkpoint_cad=xao,
        output_mesh=str(tmp_path / "pin2.msh"),
        default_characteristic_length=0.5,
    )
    gmsh.initialize()
    try:
        gmsh.merge(str(xao))
        names = {
            gmsh.model.getPhysicalName(d, t) for d, t in gmsh.model.getPhysicalGroups()
        }
    finally:
        gmsh.finalize()
    assert "pin" in names
