# Meshing and Resolution Optimization Plan

## Overview
This document details specific algorithmic and architectural optimizations for the meshing and resolution pipeline (`meshwell/mesh.py`, `meshwell/labeledentity.py`, `meshwell/resolution.py`). The primary goals are to eliminate $O(N^2)$ entity resolution loops, reduce GMSH kernel calls, and minimize the computational overhead of the GMSH `Min` field during mesh generation.

## Optimization Strategies

### 1. The $O(N^2)$ Resolution Handoff (`_apply_entity_refinement`)
**Problem:** In `LabeledEntities.add_refinement_fields_to_model`, resolution specs that use `sharing` or `not_sharing` trigger an iteration over all other `LabeledEntities` in the model. Inside this loop, tag intersections and boundary logic (`filter_mesh_boundary_tags_by_target_dimension`) are computed repeatedly. For models with many physical names, this becomes an $O(N^2)$ bottleneck that constantly queries the GMSH kernel.
**Solution:** 
- In `Mesh._apply_entity_refinement`, pre-compute a global reverse index before iterating through entities.
- Map `Tag -> List[PhysicalNames]` and pre-cache `Tag -> BoundaryTags`.
- Perform set intersections purely in-memory using standard Python dictionaries, changing the complexity from a kernel-bound $O(N^2)$ operation to a memory-bound $O(N)$ operation.

### 2. GMSH Field Creation Overhead (The "Min" Field Bottleneck)
**Problem:** `meshwell` creates a separate `MathEval` + `Restrict` field pair for *every* resolution spec applied to *every* physical entity. These indices are all appended to a global `Min` field. If a `Min` field contains $M$ sub-fields, GMSH evaluates $M$ equations for *every single point* in the generated mesh. This drastically slows down `gmsh.model.mesh.generate(dim)`.
**Solution:** 
- Consolidate identical `ConstantInField` specifications.
- Group entity tags by their target resolution size.
- Create a single `MathEval` field for a given constant size (e.g., `0.5`), and apply a single `Restrict` field targeting the combined list of all entity tags sharing that resolution. This minimizes the length of the `FieldsList` provided to the `Min` field.

### 3. Redundant Mass Filtering Calls
**Problem:** Inside `LabeledEntities.filter_by_mass`, `self.model.occ.getMass(dim, tag)` is called for every entity. If an entity has multiple resolution specs (e.g., a constant field and an exponential distance field), the OpenCASCADE mass property algorithm is invoked repeatedly for the same shape.
**Solution:** 
- Implement a mass property cache at the `LabeledEntities` or `ModelManager` level.
- When `getMass` is called for a specific `(dim, tag)` tuple, cache the result so subsequent calls for the same entity return instantly.

### 4. `DirectSizeSpecification` Inefficiency
**Problem:** `DirectSizeSpecification.apply` writes a GMSH `.pos` file to a temporary location on disk by iterating through a NumPy array and performing string formatting (`SP({x}, {y}, {z}){{{s}}};`). It then uses `gmsh.merge()` to read the file back into memory. This is highly inefficient for large point clouds.
**Solution:** 
- Utilize the direct, in-memory GMSH Python API for views.
- Use `view_tag = gmsh.view.add("size_field")` followed by `gmsh.view.addListData(view_tag, "SP", num_elements, data.flatten().tolist())`.
- This bypasses disk I/O and string serialization entirely, passing the numerical data directly to the GMSH C++ core.