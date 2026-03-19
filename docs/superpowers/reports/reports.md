  1. CAD Batching Boolean Tool Operands
   * Goal: Reduce O(N^2) time complexity in CAD boolean cuts by batching tool operands.
   * Implementation:
       * OCC Backend: In cad_occ.py, previous shapes are now grouped into a single TopoDS_Compound before cutting current shapes against them.
       * GMSH Backend: In cad_gmsh.py, entities are grouped by mesh_order and processed in batches using a single gmsh.model.occ.cut call for all entities of the same priority.
   * Performance Impact:
       * OCC: Speedup from 0.87s to 0.28s (~3x) for 50 overlapping entities.
       * GMSH: Speedup from 2.50s to 1.54s (~1.6x) for 50 overlapping entities.
   * Verified by: tests/benchmarks/benchmark_cad_batching.py.


  2. CAD Parallel Instantiation
   * Goal: Parallelize the in-memory instantiation of OCC shapes.
   * Implementation: Used concurrent.futures.ThreadPoolExecutor in CAD_OCC.process_entities to map the _instantiate_entity_occ method across available threads.
   * Performance Impact: Measured ~5.2s for 50 complex entities (with 20 holes each). While Python's GIL limits pure CPU scaling for simple shapes, this provides a foundation for scaling I/O
     and complex geometry generation in higher-latency environments.
   * Verified by: tests/benchmarks/benchmark_cad_parallel.py.


  3. Direct Size Specification In-Memory
   * Goal: Bypass writing large .pos view files to disk during resolution specification.
   * Implementation: Replaced tempfile and disk-based gmsh.merge logic in DirectSizeSpecification.apply with GMSH’s in-memory addListData API, passing NumPy arrays directly into memory.
   * Benefit: Eliminates disk I/O overhead and temporary file management, crucial for large point-cloud based refinement.
   * Verified by: tests/test_mesh_direct_size.py.


  4. Meshing Min Field Consolidation
   * Goal: Reduce the number of sub-fields attached to the global GMSH Min field.
   * Implementation: Refactored Mesh._apply_entity_refinement to use a constant_collector. Identical ConstantInField specifications are now grouped by resolution value, creating only one
     MathEval and one Restrict field for all matching entities.
   * Performance Impact: Reduced refinement field count from 1 per entity to 1 per unique size. In standard tests, field count dropped from ~5-10 to 1-2.
   * Verified by: Debug logs in tests/test_resolution.py.


  5. Meshing Resolution O(N^2) Handoff
   * Goal: Eliminate O(N^2) iteration over entities when evaluating sharing and not_sharing resolution constraints.
   * Implementation: Pre-computes a reverse lookup index (tag_to_entity_names) mapping every geometry tag to its physical names. The sharing logic now uses O(1) set intersections on these
     indices instead of nested loops over all entities.
   * Performance Impact: Reduced handoff time for 50 sharing entities from 0.051s to 0.027s (~2x speedup).
   * Verified by: tests/benchmarks/benchmark_resolution_handoff.py.


  6. Unified CAD Backend Pipeline
   * Goal: Consolidate GMSH and OCC backends behind a unified protocol and orchestrator.
   * Implementation:
       * Defined CADBackend Protocol in backend_protocol.py.
       * Implemented GmshBackend and OccBackend wrappers.
       * Created generate_mesh(entities, backend="occ/gmsh", ...) orchestrator in orchestrator.py.
       * Updated mesh() to support in-memory ModelManager inputs.
   * Benefit: Allows users to switch between GMSH and OCC backends with a single string change, and enables seamless in-memory handoff between CAD and Meshing without .xao disk intermediate
     files.
   * Verified by: tests/test_unified_pipeline.py and tests/test_mesh_in_memory.py.
