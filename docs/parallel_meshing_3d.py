import gmsh
import meshio
from shapely.geometry import Polygon, box
from meshwell.model import ModelManager
from meshwell.polyprism import PolyPrism
from meshwell.resolution import ThresholdField

# 1. Define global 3D geometry using PolyPrism
# We define a 10x10x2 background volume
device_poly = box(0, 0, 10, 10)
device = PolyPrism(
    polygons=device_poly, 
    buffers={0.0: 0, 2.0: 0}, 
    physical_name="background", 
    mesh_order=3
)

# A triangular prism feature spanning the full height
triangle_poly = Polygon([(2, 4), (8, 4), (8, 6)])
triangle = PolyPrism(
    polygons=triangle_poly, 
    buffers={0.0: 0, 2.0: 0}, 
    physical_name="triangle", 
    mesh_order=1
)

# A rectangular waveguide feature suspended in the middle (z from 0.5 to 1.5)
waveguide_poly = box(2, 4, 8, 6)
waveguide = PolyPrism(
    polygons=waveguide_poly, 
    buffers={0.5: 0, 1.5: 0}, 
    physical_name="waveguide", 
    mesh_order=2
)

entities = [device, triangle, waveguide]

# 2. Decompose into 2D subdomains (parallel meshing currently decomposes in XY plane)
subdomains = [
    box(0, 0, 5, 5),    # Bottom-Left
    box(5, 0, 10, 5),   # Bottom-Right
    box(0, 5, 5, 10),   # Top-Left
    box(5, 5, 10, 10),  # Top-Right
]

# 3. Set a 3D resolution field to refine the mesh inside the features
# Note: ThresholdField currently requires applying to surfaces/curves in 3D 
# as GMSH Distance fields work on boundaries.
res = ThresholdField(
    sizemin=0.2, 
    sizemax=1.0, 
    distmin=0.0, 
    distmax=1.0, 
    apply_to="surfaces"
)
resolution_specs = {"triangle": [res], "waveguide": [res]}

# 4. Run parallel meshing in 3D (dim=3)
model = ModelManager()

print("Starting 3D Parallel Meshing...")
final_mesh = model.mesh_parallel(
    entities_list=entities,
    subdomains=subdomains,
    halo_buffer=1.0,
    n_jobs=4,
    default_characteristic_length=1.0,
    resolution_specs=resolution_specs,
    dim=3,
)

# 5. Save and Visualize using Gmsh GUI
msh_filename = "parallel_mesh_3d.msh"
final_mesh.write(msh_filename, file_format="gmsh", binary=False)

print(f"Mesh saved to {msh_filename}. Opening Gmsh GUI...")
gmsh.initialize()
gmsh.open(msh_filename)
gmsh.fltk.run()
gmsh.finalize()
