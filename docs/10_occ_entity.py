# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # OCC entities for arbitrary 3D geometry
#
# While meshwell provides convenient polygon-based entities (``PolySurface``,
# ``PolyPrism``), you can also wrap arbitrary OpenCASCADE shapes via
# ``OCC_entity``. This gives you full access to the OCP geometric modeler
# while still benefiting from meshwell's fragment/tagging/meshing workflow.

# %%
from functools import partial

from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCP.gp import gp_Pnt

from meshwell.cad_occ import cad_occ
from meshwell.occ_entity import OCC_entity
from meshwell.occ_xao_writer import write_xao


# %%
def _make_box():
    return BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 1, 1, 1).Shape()


box_entity = OCC_entity(
    occ_function=_make_box,
    physical_name="box",
    mesh_order=1,
    dimension=3,
)

write_xao(cad_occ([box_entity]), "box.xao")

# %% [markdown]
# You can wrap *any* zero-argument callable that returns a ``TopoDS_Shape``:
# primitives, boolean results, imported STEP bodies, etc. For parameterized
# shapes, use ``functools.partial`` or a closure.


# %%
def cylinder(radius, height):
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder

    return BRepPrimAPI_MakeCylinder(radius, height).Shape()


custom_entity = OCC_entity(
    occ_function=partial(cylinder, radius=0.5, height=2.0),
    physical_name="cyl",
    mesh_order=1,
    dimension=3,
)

write_xao(cad_occ([custom_entity]), "cyl.xao")

# %% [markdown]
# Use ``OCC_entity`` when you need:
# - Primitive shapes beyond polygon extrusion (spheres, cones, tori)
# - Shapes from external STEP / BREP files
# - Any OCP boolean composition you want processed as a single labeled entity

# %% [markdown]
# ## Naming individual points (0D entities)
#
# meshwell meshes are keyed by physical name. Regions (``PolySurface``), curves
# (``PolyLine``), and auto-generated interfaces/boundaries all get names -- and
# so can an individual **point**, by wrapping a vertex in an ``OCC_entity`` with
# ``dimension=0`` and a ``physical_name``. Build the vertex with
# ``BRepBuilderAPI_MakeVertex(gp_Pnt(x, y, 0.0)).Vertex()``. The named vertex
# survives CAD fragmentation and becomes a named mesh node (a ``vertex`` cell
# block) within meshwell's point-tolerance (~1e-5) of the requested coordinate.
# Works inside a region, on an interface, or on a corner.
#
# This is the building block for features that reference a specific vertex by
# name (e.g. boundary-layer fan points at a convex corner -- see notebook 25).

# %%
import shapely  # noqa: E402

from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeVertex  # noqa: E402

from meshwell.orchestrator import generate_mesh  # noqa: E402
from meshwell.polysurface import PolySurface  # noqa: E402


def named_point(x, y, name):
    return OCC_entity(
        occ_function=lambda: BRepBuilderAPI_MakeVertex(gp_Pnt(x, y, 0.0)).Vertex(),
        physical_name=name,
        dimension=0,
    )


sheet = PolySurface(polygons=shapely.box(0, 0, 4, 2), physical_name="sheet", mesh_order=1)
point_mesh = generate_mesh(
    entities=[sheet, named_point(2.0, 1.0, "probe")],
    dim=2,
    output_mesh="named_point.msh",
    default_characteristic_length=0.5,
)
print("named groups:", sorted(k for k in point_mesh.cell_sets if not k.startswith("gmsh:")))
