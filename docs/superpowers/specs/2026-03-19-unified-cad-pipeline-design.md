# Unified CAD Backend Pipeline Design

## Overview
The goal is to consolidate the execution of CAD operations (OpenCASCADE via OCP vs GMSH OCC) behind a unified backend protocol. This allows users to interchange CAD engines seamlessly, eliminates redundant disk I/O when keeping the pipeline in-memory, and provides a clear architectural boundary between geometry definitions and CAD boolean processing.

## Architecture

1. **Common CAD Backend Protocol**
   A standard interface that any CAD engine must implement:
   - `process_entities(entities: list[GeometryEntity], **kwargs) -> None`: Evaluates geometries.
   - `save_checkpoint(filepath: Path) -> None`: Exports the CAD model (e.g., .xao, .brep).
   - `to_gmsh_model() -> ModelManager`: Returns a hydrated `ModelManager` ready for the meshing stage.

2. **Engine Implementations**
   - `GmshBackend`: Wraps `meshwell.cad_gmsh.CAD`. Mutates a `ModelManager` directly and returns it.
   - `OccBackend`: Wraps `meshwell.cad_occ.CAD_OCC`. Performs OCP booleans, then bridges the final `TopoDS_Shape`s into a fresh `ModelManager`.

3. **In-Memory Handoff to Mesher**
   `meshwell.mesh.mesh(...)` will be updated to accept a pre-populated `ModelManager` instead of strictly requiring a disk `.xao` file.

4. **Unified Orchestrator**
   A top-level API (e.g., `generate_mesh()`) that accepts an engine parameter (`backend="occ" | "gmsh"`), processes entities, optionally checkpoints to disk, and delegates to the mesher seamlessly.

## Trade-offs & Benefits
- **Pros:** Completely eliminates unnecessary disk I/O when meshing immediately. Cleans up user API. Makes falling back to a different CAD engine trivial.
- **Cons:** Requires slight adjustments to how `mesh.py` is called internally by users who might be relying on the strict file I/O interface, though the API can remain backward compatible.