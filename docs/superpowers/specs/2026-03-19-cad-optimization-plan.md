# CAD Engine Optimization Plan

## Overview
This document details specific algorithmic and architectural optimizations for the CAD Boolean processing pipelines in `cad_gmsh.py` and `cad_occ.py`. The primary goal is to reduce computational complexity from $O(N^2)$ to $O(N)$ where possible, and properly leverage OpenCASCADE/GMSH batch processing capabilities.

## Optimization Strategies

### 1. Batching Boolean Tool Operands (The $O(N^2)$ Cut Problem)
**Problem:** Both backends currently iterate through new entities one-by-one and cut them against previously accumulated shapes one-by-one. If there are $M$ new entities and $N$ accumulated shapes, this requires $M 	imes N$ boolean operations.
**Solution (OCC):** Create a single `TopoDS_Compound` out of the accumulated tool shapes. This allows cutting $M$ entities against 1 tool compound, reducing operations to $M$.
**Solution (GMSH):** Group entities by their `mesh_order` and perform a single batch `gmsh.model.occ.cut(objectDimTags, toolDimTags)` passing all entities as objects and all accumulated entities as tools. Use the returned `outDimTagsMap` to re-associate the resulting shapes back to their source `LabeledEntity`.

### 2. Resolving Same-Mesh-Order Overlaps
**Problem:** `cad_gmsh.py` processes entities with the same `mesh_order` sequentially, making the outcome dependent on the arbitrary list index (Entity A cuts Entity B). `cad_occ.py` handles them in parallel via `BOPAlgo_Builder`, resulting in different behavior between the two backends.
**Solution:** Align `cad_gmsh.py` with `cad_occ.py`. For a given `mesh_order` group, use `gmsh.model.occ.fragment` to resolve overlaps between peers *before* cutting the entire group against higher-priority (lower `mesh_order`) entities.

### 3. Parallelizing Instantiation
**Problem:** Entity instantiation (`instanciate_occ()`) runs sequentially.
**Solution:** Because OCP shape generation (`instanciate_occ`) is a pure in-memory operation that does not mutate global state, dispatch the instantiation loop to a thread pool (using the existing `n_threads` parameter) to build the base un-cut shapes concurrently.

### 4. Maximizing Geometry Caches
**Problem:** In `cad_gmsh.py`, the `_shared_point_cache` is cleared after *every* boolean cut because boolean operations can re-tag or swallow points, invalidating the cache.
**Solution:** By shifting to the batch-processing strategy outlined in point 1, we can instantiate *all* entities of a specific dimension and mesh order before calling `sync()` and executing boolean cuts. This allows the point and line deduplication caches to work across a much wider swath of entities, reducing the total point count handed to the OCC kernel.
