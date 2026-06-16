# Meshwell Pipeline

Meshwell provides a streamlined workflow for generating high-quality meshes from geometric definitions. The typical pipeline consists of three stages:

```mermaid
flowchart LR
    Self-contained inputs --> CAD["CAD\n(Geometry)"] --> MESH["Mesh\n(Initial)"] --> REMESH["Remesh\n(Adaptive)"]
```

## Feature highlights

A quick map of everything meshwell can do; each item links to a dedicated page below.

**Geometry (CAD)**
- [1D/2D/3D entities](11_polysurfaces) — `PolyLine`, `PolySurface`, and `PolyPrism` built straight from Shapely.
- [Arbitrary OCC shapes](10_occ_entity) — drop in any OpenCASCADE construction when polygons aren't enough.
- [Arc identification](13_arc_identification) — `identify_arcs=True` recovers true circular boundaries from point-sampled polygons.
- [Multi-entity models](14_models) — combine OCC entities, polysurfaces, and prisms in one model.
- **`mesh_order` ownership** — painter's-algorithm resolution of overlapping entities (lower order wins); see [Usage Approaches](03_usage_approaches).

**Meshing**
- **2D & 3D meshing** — planes, cross-sections, surfaces of 3D objects, or full volumes.
- [Resolution control](20_resolution_basic) — global sizing plus per-entity [`ThresholdField`](21_resolution_advanced) distance-based refinement.
- [Direct size specification](22_direct_size_specification) — prescribe the mesh-size field explicitly over space.
- [Structured meshing](23_structured) — wedge (prism) elements through a layer's thickness, conformal with the surrounding unstructured tets.
- [Mesh quality analysis](40_mesh_quality) — `MeshQualityAnalyzer` reports element quality before you simulate.

**Remeshing**
- [Adaptive remeshing](30_adaptive_remeshing_gmsh) — drive mesh size from a computed field via `RemeshingStrategy`, with [GMSH](30_adaptive_remeshing_gmsh) and [MMG](31_adaptive_remeshing_mmg) backends.

**Integrations**
- [GDS import](41_importing_gds) — load `.gds` layers as Shapely polygons ready to mesh.
- [gdswell](42_gdswell_interface) — turn a gdswell layout `Stackup` into a 3D mesh and 2D cross-sections.

---

## 1. Polygons (and more) --> CAD

The first step is to define your geometry using built-in meshwell entity classes or arbitrary OCC shapes. Meshwell entities are built from polygons (using [Shapely](https://shapely.readthedocs.io/)) and can be:

- **PolyLine**: 1D lines defined by LineStrings
- **PolySurface**: 2D surfaces defined by polygons
- **PolyPrism**: 3D volumes created by extruding polygons along the z-axis — with z-dependent `buffers`, the footprint can grow or shrink with height to make tapered or slanted sidewalls

Beyond these, geometry can also come from [arbitrary OCC entities](10_occ_entity). Assemblies of entities can form [multi-entity models](14_models).

Key concepts:
- **`physical_name`**: A label for the entity, used for the GMSH physical group
- **`mesh_order`**: Controls how overlapping entities of the same dimension interact (lower order takes precedence)

The geometry is processed by OCC via `cad_occ()` and written to a `.xao` file using `write_xao()`.

```python
import shapely

from meshwell.cad_occ import cad_occ
from meshwell.occ_xao_writer import write_xao
from meshwell.polysurface import PolySurface

polygon = shapely.Polygon([[-5, -5], [5, -5], [5, 5], [-5, 5]])
entity = PolySurface(polygons=polygon, physical_name="my_surface", mesh_order=1)

write_xao(cad_occ([entity]), "geometry.xao")
```

For more details on the CAD options, see:
- [OCC entities](10_occ_entity), which can be arbitrarily complex
- [Polysurfaces](11_polysurfaces)
- [Prisms](12_prisms)
- [Arc identification](13_arc_identification)
- [Models](14_models)

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

Meshwell offers [three ways to drive this](03_usage_approaches): the **functional** helpers shown above, an **object-oriented** `Model` that shares a single GMSH session between CAD and meshing, and the **`generate_mesh`** orchestrator that runs the whole pipeline in one call.

A few meshing features worth calling out:
- **Resolution fields** — combine a global size with per-entity refinement such as `ThresholdField` (mesh size grows with distance from an entity). See [Resolution Basics](20_resolution_basic) and [Resolution Advanced](21_resolution_advanced).
- **Direct size specification** — prescribe the size field explicitly over space when you already know where you want resolution: [Direct Size Specification](22_direct_size_specification).
- **Structured meshing** — fill a layer with wedge (triangular-prism) elements through its thickness while staying conformal with the surrounding unstructured tetrahedra: [Structured meshing](23_structured).
- **Quality analysis** — inspect element quality with the `MeshQualityAnalyzer` before using the mesh downstream: [Mesh Quality](40_mesh_quality).

---

## 3. mesh --> (re)mesh

DirectSizeSpecification allows fine grained control of mesh sizing over space. Often, however, we want  this process to be guided by some data field we have computed over an existing mesh. The remeshing utilities are here for that (in case your solver does not have a native way to do this, which is preferred!).

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
- [Adaptive Remeshing (GMSH)](30_adaptive_remeshing_gmsh)
- [Adaptive Remeshing (MMG)](31_adaptive_remeshing_mmg)

---

## Integrations

Meshwell is the meshing backend for higher-level layout tools:

- **[gdswell](42_gdswell_interface)** — convert a gdswell layout `Stackup` (the 3D extension of a 2D layout) into a watertight 3D mesh, and slice 2D cross-sections at a cell's ports. meshwell's `mesh_order` reproduces gdswell's painter's-order cutting for free.
