# %% [markdown]
# # GDSFactory integration
# GDSFactory has an interface to meshwell (v1.0.7)

# %% [markdown]
# > **Warning**
# > The GDSFactory plugin for meshwell/gmsh has not been updated since meshwell version 1.0.7! Some features may be missing, and some more bugs might be present.
#
# %%[markdown]
# GDSFactory has the concept of a "LayerStack", which contains the information about the final fabricated layers:

# %% [markdown]
# ```python
# import gmsh
# import gdsfactory as gf
# from gplugins.gmsh.get_mesh import get_mesh
# from gdsfactory.generic_tech.layer_stack import get_layer_stack
#
# PDK = gf.generic_tech.get_generic_pdk()
# PDK.bend_points_distance = 0.5
#
# LAYER_STACK = get_layer_stack()
# LAYER_STACK.pprint()
# ```


# %% [markdown]
# ```python
# c = gf.components.spiral_heater.spiral_racetrack_heater_metal(num=3)
# c.plot()

# %% [markdown]
# In meshwell 1.0.7, only a ThresholdField as a distance from the surfaces/volumes was supported. From the GDSFactory plugin, this is entered as a dict for the LayerLevels:
# %% [markdown]
# ```python
# resolutions = {}
# resolutions["core"] = {"resolution": 0.1, "distance": 5}
# resolutions["heater"] = {"resolution": 0.2, "distance": 10}
# ```
# %% [markdown]
# the "type" argument in get_mesh can be "xy", "uz", or "3D", depending on if a 2D in-plane, 2D out-of-plane cross-section, or 3D mesh si desired:

# %% [markdown]
# ```python
# xbound = (c.dxmin + c.dxmax) / 2
# ymax = c.dymin
# ymin = (c.dymin + c.dymax) / 2
#
# mesh = get_mesh(
#     c,
#     type="uz",
#     xsection_bounds=[[xbound, ymin], [xbound, ymax]],
#     layer_stack=LAYER_STACK,
#     filename="heater_uz.msh",
#     resolutions=resolutions,
#     default_characteristic_length=10,
#     wafer_padding=50,
#     interface_delimiter="___",
# )
# ```

# %% [markdown]
# ```python
# try:
#     gmsh.initialize()
#     gmsh.open("heater_uz.msh")
#     gmsh.fltk.run()
# except:
#     print("Skipping CAD GUI visualization - only available when running locally")
# ```

# %% [markdown]
# ```python
# resolutions = {}
# resolutions["core"] = {"resolution": 0.5, "distance": 10}
# resolutions["heater"] = {"resolution": 5, "distance": 10}

# mesh = get_mesh(
#     c,
#     type="3D",
#     layer_stack=LAYER_STACK,
#     filename="heater_3D.msh",
#     resolutions=resolutions,
#     default_characteristic_length=20,
#     wafer_padding=50,
#     interface_delimiter="___",
# )
# ```

# %% [markdown]
# ```python
# try:
#     gmsh.initialize()
#     gmsh.open("heater_3D.msh")
#     gmsh.fltk.run()
# except:
#     print("Skipping CAD GUI visualization - only available when running locally")
# ```
# %%
