# %% [markdown]
# # Multi-entity models
# Multiple OCC entities (and polysurfaces or prisms) can be combined in a single model.

# %%
import shapely
from OCP.BRepBuilderAPI import (
    BRepBuilderAPI_MakeEdge,
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_MakePolygon,
    BRepBuilderAPI_MakeWire,
)
from OCP.GC import GC_MakeCircle
from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt

from meshwell.cad_occ import cad_occ
from meshwell.mesh import mesh
from meshwell.occ_entity import OCC_entity
from meshwell.occ_xao_writer import occ_to_xao
from meshwell.polysurface import PolySurface
from meshwell.visualization import plot2D


def _rectangle(x, y, z, dx, dy):
    def build():
        poly = BRepBuilderAPI_MakePolygon()
        poly.Add(gp_Pnt(x, y, z))
        poly.Add(gp_Pnt(x + dx, y, z))
        poly.Add(gp_Pnt(x + dx, y + dy, z))
        poly.Add(gp_Pnt(x, y + dy, z))
        poly.Close()
        return BRepBuilderAPI_MakeFace(poly.Wire()).Shape()

    return build


def _disk(xc, yc, zc, r):
    """Planar disk (circular face) centered at (xc, yc, zc), axis +Z."""

    def build():
        ax = gp_Ax2(gp_Pnt(xc, yc, zc), gp_Dir(0, 0, 1))
        circle = GC_MakeCircle(ax, r).Value()
        edge = BRepBuilderAPI_MakeEdge(circle).Edge()
        wire = BRepBuilderAPI_MakeWire(edge).Wire()
        return BRepBuilderAPI_MakeFace(wire).Shape()

    return build


# %%
polygon_hull = shapely.Polygon(
    [[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]],
)
line1 = shapely.LineString([[0.5, 0.5], [1.5, 1.5]])
polygon_hole1 = shapely.buffer(line1, 0.2)
line2 = shapely.LineString([[1.5, 0.5], [0.5, 1.5]])
polygon_hole2 = shapely.buffer(line2, 0.2)
polygon = polygon_hull - polygon_hole1 - polygon_hole2


s_curve = shapely.LineString([[-1, 0], [0, 0.5], [-1, 1], [0, 1.5], [-1, 2]])
s_shape = shapely.buffer(s_curve, 0.2)

poly2D = PolySurface(
    polygons=polygon,
    physical_name="meshwell_polygon",
    mesh_order=1,
)

s = PolySurface(
    polygons=s_shape,
    physical_name="meshwell_s",
    mesh_order=2,
)

disk_entity = OCC_entity(
    occ_function=_disk(xc=2, yc=2, zc=0, r=1),
    physical_name="occ_disk",
    mesh_order=3,
    dimension=2,
)

rectangle = OCC_entity(
    occ_function=_rectangle(x=1.5, y=0, z=0, dx=1, dy=1),
    physical_name="occ_rectangle",
    mesh_order=4,
    dimension=2,
)

entities_list = [poly2D, s, disk_entity, rectangle]

occ_to_xao(cad_occ(entities_list), "complicated.xao")

output_mesh = mesh(
    dim=2,
    input_file="complicated.xao",
    output_file="complicated.msh",
    default_characteristic_length=0.5,
    mesh_element_order=1,  # set to 2 to generate a curved mesh with the disk
)

# %%
plot2D(output_mesh, wireframe=False)
# %%
plot2D(output_mesh, wireframe=False, physicals=["meshwell_polygon___occ_disk"])
# %% [markdown]
# mesh_order specifies which entity takes precedence if there is a conflict; lower numbers override higher numbers.

# %%
poly2D = PolySurface(
    polygons=polygon,
    physical_name="meshwell_polygon",
    mesh_order=4,
)

s = PolySurface(
    polygons=s_shape,
    physical_name="meshwell_s",
    mesh_order=3,
)

disk_entity = OCC_entity(
    occ_function=_disk(xc=2, yc=2, zc=0, r=1),
    physical_name="occ_disk",
    mesh_order=2,
    dimension=2,
)

rectangle = OCC_entity(
    occ_function=_rectangle(x=1.5, y=0, z=0, dx=1, dy=1),
    physical_name="occ_rectangle",
    mesh_order=1,
    dimension=2,
)

entities_list = [poly2D, s, disk_entity, rectangle]

occ_to_xao(cad_occ(entities_list), "model.xao")

output_mesh = mesh(
    dim=2,
    input_file="model.xao",
    output_file="model.msh",
    default_characteristic_length=0.5,
    mesh_element_order=1,
)

# %%
plot2D(output_mesh, wireframe=True)


# %% [markdown]
# By default, all CAD entities get meshed. By setting mesh_bool to False, a CAD entity can be inserted for the purposes of cutting out regions / tagging interfaces, without adding a mesh within a region.


# %%
poly2D = PolySurface(
    polygons=polygon,
    physical_name="meshwell_polygon",
    mesh_order=4,
)

s = PolySurface(
    polygons=s_shape,
    physical_name="meshwell_s",
    mesh_order=3,
    mesh_bool=False,
)

disk_entity = OCC_entity(
    occ_function=_disk(xc=2, yc=2, zc=0, r=1),
    physical_name="occ_disk",
    mesh_order=2,
    mesh_bool=False,
    dimension=2,
)

rectangle = OCC_entity(
    occ_function=_rectangle(x=1.5, y=0, z=0, dx=1, dy=1),
    physical_name="occ_rectangle",
    mesh_order=1,
    dimension=2,
)

entities_list = [poly2D, s, disk_entity, rectangle]

occ_to_xao(cad_occ(entities_list), "model.xao")

output_mesh = mesh(
    dim=2,
    input_file="model.xao",
    output_file="model.msh",
    default_characteristic_length=0.5,
    mesh_element_order=1,
)

# %%
plot2D(output_mesh, wireframe=True)
