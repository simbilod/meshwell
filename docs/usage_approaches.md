# Usage Approaches

Meshwell provides two main approaches for CAD geometry generation and meshing:

## 1. Functional

Hermetic functions with clear inputs and outputs that handle everything automatically, including file I/O:

```python
from meshwell.cad import cad
from meshwell.mesh import mesh
from meshwell.polysurface import PolySurface
import shapely

# Define geometry
box1 = PolySurface(
    polygons=[shapely.box(0, 0, 5, 3)],
    physical_name="region_1",
    mesh_order=1,
)

box2 = PolySurface(
    polygons=[shapely.box(-1, -1, 6, 4)],
    physical_name="region_2",
    mesh_order=2,
)

# Generate CAD → automatically saves to file
cad(entities_list=[box1, box2], output_file="geometry.xao")

# Generate mesh → automatically loads and saves files
mesh_obj = mesh(
    input_file="geometry.xao",
    output_file="geometry.msh",
    default_characteristic_length=0.2,
    dim=2
)
```

**Characteristics:**
- Each function creates its own GMSH model
- Automatic file handling
- No shared state between operations
- Useful to reuse a single expensive CAD model across multiple meshing schemes

## 2. Object-oriented

Single model instance with flexible processing and optional file I/O. Has convenience .cad and .mesh attributes to call the respective generators:

```python
from meshwell.model import Model
from meshwell.polysurface import PolySurface
from meshwell.resolution import ThresholdField
import shapely

# Create shared model
model = Model(filename="my_project", n_threads=4)

# Define geometry
inner = PolySurface(
    polygons=[shapely.box(2, 2, 6, 6)],
    physical_name="inner_region",
    mesh_order=1,
)

outer = PolySurface(
    polygons=[shapely.box(0, 0, 8, 8)],
    physical_name="outer_region",
    mesh_order=2,
)

# Option A: In-memory processing (no files)
entities = model.cad.process_entities([inner, outer])
resolutions = {
    "inner_region": [ThresholdField(sizemin=0.05, distmax=1, sizemax=0.2)]
}
mesh_obj = model.mesh.process_geometry(
    dim=2,
    default_characteristic_length=0.1,
    resolution_specs=resolutions
)

# Option B: Save/load files when needed
model.cad.save_to_file("geometry.xao")  # Save geometry
model.mesh.save_to_file("geometry.msh") # Save mesh
```

**Characteristics:**
- Single GMSH model shared between CAD and Mesh
- Work entirely in memory or save/load selectively
- Fine-grained control over each step
- Useful when both CAD and Mesh will live within the same session
