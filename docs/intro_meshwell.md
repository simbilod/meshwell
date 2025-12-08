# Meshwell Pipeline

Meshwell provides a streamlined workflow for generating high-quality meshes from geometric definitions. The typical pipeline consists of three stages:

```mermaid
flowchart LR
    Self-contained inputs --> CAD["CAD\n(Geometry)"] --> MESH["Mesh\n(Initial)"] --> REMESH["Remesh\n(Adaptive)"]
```

---

## 1. Polygons (and more) --> CAD

The first step is to define your geometry using built-in GMSH or meshwell entity classes. Meshwell entities are built from polygons (using [Shapely](https://shapely.readthedocs.io/)) and can be:

- **PolyLine**: 1D lines defined by LineStrings
- **PolySurface**: 2D surfaces defined by polygons
- **PolyPrism**: 3D volumes created by extruding polygons along the z-axis

Key concepts:
- **`physical_name`**: A label for the entity, used for the GMSH physical group
- **`mesh_order`**: Controls how overlapping entities of the same dimension interact (lower order takes precedence)

The geometry is exported using the `cad()` function, which writes a `.xao` file containing the complete geometric definition.

```python
from meshwell.cad import cad
from meshwell.polysurface import PolySurface
import shapely

polygon = shapely.Polygon([[-5, -5], [5, -5], [5, 5], [-5, 5]])
entity = PolySurface(polygons=polygon, physical_name="my_surface", mesh_order=1)

cad(entities_list=[entity], output_file="geometry.xao")
```

For more details on the CAD options, see:
- [GMSH entities](cad), which can be arbitrarily complex
- [Polysurfaces](polysurfaces)
- [Prisms](prisms)
- [Models](models)

---

## 2. CAD --> mesh

The `mesh()` function takes the geometry from the CAD stage and generates an initial mesh. You can control:

- **`default_characteristic_length`**: The base mesh size
- **`resolution_specs`**: Fine-grained control over mesh size per entity or globally
- **`dim`**: Dimension of the mesh -- 2D (planes, or surfaces of 3D objects) or full 3D

```python
from meshwell.mesh import mesh

initial_mesh = mesh(
    dim=2,
    input_file="geometry.xao",
    output_file="mesh.msh",
    default_characteristic_length=1.0,
)
```

For more advanced meshing options, see:
- [Resolution Basics](resolution_basic)
- [Resolution Advanced](resolution_advanced)
- [Direct Size Specification](direct_size_specification)

---

## 3. mesh --> (re)mesh

DirectSizeSpecification allows fine grained control of mesh sizing over space. Often, however, we want  this process to be guided by some data field we have computed over an existing mesh. The remeshing utilities are here for that.

Remeshing is controlled by `RemeshingStrategy` objects that define how mesh sizes should change. Generic strategies have the following attributes:

| Attribute | Description |
|-----------|-------------|
| `refinement_data` | (N, 4) array of points `(x, y, z, data)` where data is a solution field, error estimator, etc. |
| `func` | Optional callable to transform the data (e.g., compute gradients, differences) |
| `threshold_func` | Maps transformed data to new mesh sizes |
| `min_size` | Minimum allowed mesh size |
| `max_size` | Maximum allowed mesh size |

Note that while `refinement_data` is often specified at the existing mesh nodes, it can actually be any set of points in space.

For example, here is a simple pre-implemented strategy that refines the mesh by some factor where data exceeds a threshold:

```python
from meshwell.remesh import BinaryScalingStrategy

strategy = BinaryScalingStrategy(
    refinement_data=data,  # (N, 4) array
    threshold=0.5,         # Consider nodes where data > 0.5
    factor=0.2,            # Make new mesh size 20% of original where data > threshold
    min_size=0.05,
    max_size=2.0,
)
```

Meshwell supports two remeshing backends:

- **GMSH** (`remesh_gmsh`): Uses GMSH for remeshing (by transforming the strategy into the appropriate DirectSizeSpecification), can reuse the original CAD file
- **MMG** (`remesh_mmg`): Uses MMG library, often produces higher quality meshes

See the detailed examples:
- [Adaptive Remeshing (GMSH)](adaptive_remeshing_gmsh)
- [Adaptive Remeshing (MMG)](adaptive_remeshing_mmg)
