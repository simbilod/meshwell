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
from meshwell.occ_xao_writer import occ_to_xao


# %%
def _make_box():
    return BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 1, 1, 1).Shape()


box_entity = OCC_entity(
    occ_function=_make_box,
    physical_name="box",
    mesh_order=1,
    dimension=3,
)

occ_to_xao(cad_occ([box_entity]), "box.xao")

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

occ_to_xao(cad_occ([custom_entity]), "cyl.xao")

# %% [markdown]
# Use ``OCC_entity`` when you need:
# - Primitive shapes beyond polygon extrusion (spheres, cones, tori)
# - Shapes from external STEP / BREP files
# - Any OCP boolean composition you want processed as a single labeled entity
