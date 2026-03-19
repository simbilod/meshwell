# Design Spec: Unified CAD Boolean Logic (Approach 3)

*   **Date**: 2026-03-19
*   **Status**: Draft
*   **Topic**: Refactoring `CAD_GMSH` to use unified boolean fragmentation instead of sequential cuts and global caching.

## 1. Context & Problem Statement

The current `CAD_GMSH` implementation relies on two brittle mechanisms:
1.  **Global Point/Line Caching**: Coordinate snapping (rounding) is used to deduplicate GMSH entities across different objects. This often conflicts with the OpenCASCADE (OCC) kernel's own "fuzzy" logic, leading to misconnected vertices.
2.  **Sequential Cuts**: Entities are processed in `mesh_order` groups and cut against accumulated tools. This sequential approach is sensitive to ID instability and often leaves behind slivers or topological artifacts.

These issues recently caused a failure in `tests/test_polysurface.py` where a vertex was incorrectly connected to a center point instead of a box corner.

## 2. Objective

Replace manual cross-entity caching and sequential cuts with a **Unified Fragmentation** strategy. This will align `CAD_GMSH` with the robustness of `CAD_OCC` while preserving GMSH-specific features like `interface` and `boundary` tagging.

## 3. Proposed Approach: Pure Boolean Integration

The core principle is to let the OCC kernel handle all geometric intersections and point merging in one "giant" operation per dimension, using its native tolerance settings.

### 3.1. Entity-Local Connectivity
*   **Mechanism**: Caches (`_points`, `_lines`) in `GeometryEntity` will be strictly instance-local.
*   **Goal**: Ensure closed loops (Concern C) within a single `PolySurface` or `PolyPrism` by using consistent tags for shared vertices of that entity.
*   **Benefit**: Eliminates the "stale ID" problem and coordinate-snapping errors across different entities.

### 3.2. Unified Fragmentation Per Dimension
For each dimension group (e.g., all 2D Surfaces):
1.  **Instantiation**: Create every entity independently (no shared global tags).
2.  **Fuzzy Tolerance**: Set `Geometry.OCCTolerance` to the user-defined `point_tolerance`.
3.  **Global Fragment**: Call `gmsh.model.occ.fragment(all_dimtags, [])`. This resolves all overlaps and coincident boundaries in one step.
4.  **Synchronization**: Immediately call `gmsh.model.occ.synchronize()`.

### 3.3. Priority Mapping (Lowest Order Wins)
We will use the GMSH `outDimTagsMap` to resolve ownership of the fragments:
*   Each fragment tag `T` from the result is mapped to the list of original entities that contained it.
*   The fragment `T` is assigned **only** to the owner with the **lowest `mesh_order`**.
*   All other owners discard that fragment tag. This correctly implements the "override" logic.
*   **No Self-Fusion**: If an entity is represented by multiple fragments, they will be kept as a list of tags. `LabeledEntities` and GMSH `Physical Groups` handle multiple tags natively. This avoids "fake" interfaces because internal fragment boundaries are ignored by the `combined=True` flag in `getBoundary`.
*   **Revisiting Self-Fuse**: If these internal fragment boundaries introduce undesired mesh constraints that significantly alter node positions (causing regressions against references), we will revisit the idea of performing a `fuse` operation on all fragments belonging to the same logical entity.

### 3.4. Hierarchical Integration
After resolving a dimension, the resulting shapes are fragmented against all previously resolved **higher-dimensional** entities. This ensures that a 1D Polyline correctly "cuts" or "embeds" into a 2D Surface, maintaining full topological connectivity for meshing.

## 4. Implementation Plan

1.  **Refactor `GeometryEntity`**: Change caching from shared dicts to instance-local dicts.
2.  **Refactor `CAD_GMSH._process_dimension_group_cuts`**:
    *   Remove the `mesh_order` loop.
    *   Implement the `fragment` + `outDimTagsMap` priority logic.
3.  **Refactor `CAD_GMSH._process_dimension_group_fragments`**: Ensure it handles the re-mapping correctly for cross-dimensional integration.
4.  **Update `ModelManager`**: Ensure `Geometry.OCCTolerance` is set during initialization.

## 5. Verification Plan

### 5.1. Automated Regression
*   Run `tests/test_polysurface.py` and `tests/test_polyline.py`.
*   Verify that node counts and cell counts match references exactly.

### 5.2. Targeted Failure Case
*   Reproduce the "wrong node connection" scenario reported in the order 2 polysurface.
*   Verify that the new logic connects vertices correctly without coordinate snapping artifacts.

## 6. Trade-offs

*   **Pros**: High robustness; simplified code; utilizes OCC's native geometric matching.
*   **Cons**: Higher peak memory during the `fragment` operation for very large models compared to sequential cuts (though sequential cuts often create more temporary garbage).
