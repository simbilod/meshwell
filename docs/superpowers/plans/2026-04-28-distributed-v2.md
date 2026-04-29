# Distributed Domain Decomposition v2 Plan

**Goal:** Replace the v1 seam-mesh pipeline with a single-phase pipeline based on independent per-tile meshing + hashed-tag merge. Unlocks arbitrary NxM grids; removes the v1 corner-tile limitation.

**Architecture:** Master clips entities to subdomains. Workers mesh each subdomain independently in any order, assigning physical-group tags by hashing `(dim, name)` so all tiles agree on which integer represents "silicon". Master `gmsh.merge`s all per-tile `.msh` files (auto-consolidates by shared tag), runs `removeDuplicateNodes` for seam conformity, writes the final mesh.

**Validated by:** `tests/test_merge_spike.py` — 12 passing tests covering 2x1 / 2x2 / 3x3 strip + grid layouts, single + multi-material per tile, internal-feature tiles, both meshio_concat and hashed-tag gmsh.merge strategies.

**Tech stack:** Python 3.10+, gmsh, shapely, hashlib, pytest. Drops the meshio-based stitch dependency (we keep meshio as a transitive dep for other meshwell features).

---

## File Structure

| File | Change |
|---|---|
| `meshwell/distributed.py` | REWRITE the body. Keep public API (`generate_mesh_distributed`, `subdomains_from_grid`, `Executor`, `SubprocessExecutor`, `InProcessExecutor`, `run_job`, `cli_main`). Remove: `Slab`, junction detection, `_write_seam_bundle`, `_resolution_only_proxy`, `run_plan` two-phase scheduling, meshio-based stitch. Add: `_name_to_tag` helper, `_assign_hashed_tags`, simplified single-phase scheduler, gmsh-merge-based stitch. |
| `meshwell/mesh.py` | Add `_hashed_physical_tags: bool = False` kwarg through the existing path; when set, replace auto-generated tags with `_name_to_tag(name, dim)` before writing the .msh. Drop the dead `_emit_only_seam_surfaces` + `_interface_constraints` plumbing entirely (no longer needed). Keep `_pre_buffered`, `_global_physical_names`. |
| `meshwell/orchestrator.py` | Drop `_emit_only_seam_surfaces` + `_interface_constraints` forwarding. Add `_hashed_physical_tags`. |
| `tests/test_distributed.py` | Remove tests for removed features. Add NxM grid tests (replacing the v1 corner-tile xfail comment block). |
| `docs/distributed.md` | Rewrite v1-limitations section: NxM grids supported; mismatched lc across tiles produces non-conformal seams (a real, much narrower limitation). |
| `docs/distributed_example.py` | Promote from 2x1 to 3x3 grid with multiple materials per tile. |
| `meshwell/distributed_spike` (test_distributed_spike.py) | Keep as historical evidence; the `_seed_occ_face_from_seam` recipe is now unused but the spike file documents WHY we tried it. Mark with a header note. |

---

## Tasks

### Task 1: `_name_to_tag` helper in `meshwell/distributed.py`

Add at module scope:

```python
import hashlib

_TAG_SPACE = 1_000_000  # safe gap from gmsh's auto-tag range

def _name_to_tag(name: str, dim: int) -> int:
    """Deterministic positive int tag from (dim, name).

    Same name + dim across processes / runs / files produces the same
    integer, so independently-written .msh files all agree on which
    tag represents 'silicon' at dim=3. After ``gmsh.merge``, entities
    from different files contributing to the same (dim, tag) group get
    auto-unioned by gmsh under the shared name — no post-merge
    consolidation pass needed.
    """
    h = hashlib.sha1(
        f"{dim}:{name}".encode(), usedforsecurity=False
    ).digest()
    n = int.from_bytes(h[:4], "big") % _TAG_SPACE
    return max(1, n)
```

Test:

```python
def test_name_to_tag_is_deterministic():
    from meshwell.distributed import _name_to_tag
    assert _name_to_tag("silicon", 3) == _name_to_tag("silicon", 3)
    assert _name_to_tag("silicon", 3) != _name_to_tag("silicon", 2)
    assert _name_to_tag("silicon", 3) != _name_to_tag("oxide", 3)
    # Tag is positive int32-safe.
    assert 1 <= _name_to_tag("anything", 3) < 2**31
```

Commit: `feat(distributed): _name_to_tag for stable cross-file physical-group tags`

### Task 2: `_hashed_physical_tags` flag in `mesh.py`

When True, after CAD load + before mesh.write, walk all physical groups and re-tag each via `_name_to_tag(name, dim)`. Concretely: for each existing `(dim, tag, name)`, call `removePhysicalGroups([(dim, tag)])`, capture the entities, then `addPhysicalGroup(dim, entities, tag=hashed, name=name)`.

Helper in `mesh.py`:

```python
def _retag_physical_groups_with_hash(name_to_tag) -> None:
    """Replace all auto-generated physical-group tags with hash-derived ones."""
    snapshots = []
    for dim in (3, 2, 1, 0):
        for d, tag in list(gmsh.model.getPhysicalGroups(dim)):
            name = gmsh.model.getPhysicalName(d, tag)
            ents = list(gmsh.model.getEntitiesForPhysicalGroup(d, tag))
            snapshots.append((d, tag, name, ents))
    for d, tag, name, ents in snapshots:
        gmsh.model.removePhysicalGroups([(d, tag)])
    for d, _old, name, ents in snapshots:
        if not name:
            continue
        gmsh.model.addPhysicalGroup(d, ents, tag=name_to_tag(name, d), name=name)
```

Plumb `_hashed_physical_tags: bool = False` through `mesh()` and `orchestrator.generate_mesh()`. Call retag right before `gmsh.write(output_file)` when the flag is set. Forward `name_to_tag` callable from `meshwell.distributed._name_to_tag` (avoid circular import — pass via kwarg).

Test (in `tests/test_distributed.py`):

```python
def test_generate_mesh_hashed_physical_tags(tmp_path):
    """Two independent meshwell runs with the same physical_name produce
    .msh files where the physical-group tag is identical."""
    import shapely
    from meshwell.distributed import _name_to_tag
    from meshwell.orchestrator import generate_mesh
    from meshwell.polyprism import PolyPrism
    import meshio

    def _emit(out_path, x_offset):
        prism = PolyPrism(
            polygons=shapely.box(x_offset, 0, x_offset + 1, 1),
            buffers={0.0: 0.0, 1.0: 0.0},
            physical_name="silicon", mesh_order=1,
        )
        generate_mesh(
            entities=[prism], dim=3, output_mesh=out_path,
            default_characteristic_length=0.5,
            _hashed_physical_tags=True,
        )

    out_a = tmp_path / "a.msh"
    out_b = tmp_path / "b.msh"
    _emit(out_a, 0.0)
    _emit(out_b, 5.0)

    ma = meshio.read(out_a)
    mb = meshio.read(out_b)
    silicon_a = int((ma.field_data or {})["silicon"][0])
    silicon_b = int((mb.field_data or {})["silicon"][0])
    assert silicon_a == silicon_b == _name_to_tag("silicon", 3)
```

Commit: `feat(mesh): _hashed_physical_tags emits stable cross-file tags`

### Task 3: Drop dead seam machinery from mesh.py + orchestrator.py

Remove:
- `_emit_only_seam_surfaces` parameter and the `_filter_msh_to_seam_groups` call in mesh.py
- `_interface_constraints` parameter and the `_seed_occ_face_from_seam` calls
- The `_apply_entity_refinement` cross-reference comment about the seeding override (no longer applies)
- `_seed_occ_face_from_seam` and `_seed_one_seam_into_active_model` helpers in mesh.py
- `_filter_msh_to_seam_groups` helper in mesh.py
- The `Mesh.MeshOnlyEmpty` toggle in process_mesh

Same removals in orchestrator.py's kwarg forwarding.

Re-enable any gmsh options that were toggled for seeding (`MeshSizeFromPoints`, `MeshSizeExtendFromBoundary`).

Update tests/test_distributed.py: remove `test_emit_only_seam_surfaces_filters_output` and `test_interface_constraints_seed_volume_seam`. Verify the rest still pass.

Commit: `refactor(mesh): remove dead seam-mesh + OCC-seeding machinery (v1 leftovers)`

### Task 4: Simplify `meshwell/distributed.py` plan + bundle layout

Drop `Slab` dataclass and `interfaces`/`junctions` fields on `SubdomainPlan`:

```python
@dataclass
class SubdomainPlan:
    volumes: list[VolumeRegion]
    physical_names_seen: list[str]
    perturbation: float
    point_tolerance: float
```

Drop the `interface_width` parameter from `build_subdomain_plan` and `generate_mesh_distributed`. Document the change.

Drop `_write_seam_bundle`, `_resolution_only_proxy` (nothing seeds anymore so resolution-only proxies are dead). Keep `_clip_entity_to_polygon`, `_write_volume_bundle`, `subdomains_from_grid`, `_entity_within`.

Update `write_bundles` to only write per-volume bundles (no interface or junction bundles). Update `manifest.json` schema accordingly: drop `subdomains.interface_*` / `subdomains.junction_*`, drop `phase_order`.

Update `run_plan` to a single-phase scheduler:

```python
def run_plan(work_dir, plan, executor):
    futures = {v.id: executor.submit(work_dir / "jobs" / v.id) for v in plan.volumes}
    failures = []
    for vid, f in futures.items():
        try:
            res = f.result()
            if isinstance(res, dict) and res.get("returncode", 0) != 0:
                failures.append((vid, res.get("stderr", "")))
        except Exception as e:
            failures.append((vid, repr(e)))
    if failures:
        raise RuntimeError(f"Job failures: {failures}")
```

Update `run_job` to set `_hashed_physical_tags=True` on the worker's `generate_mesh` call (and drop `_emit_only_seam_surfaces` / `_interface_constraints`).

Tests to update / drop:
- `test_build_subdomain_plan_creates_interface_slabs` → drop
- `test_build_subdomain_plan_no_interface_for_disjoint_subdomains` → drop
- `test_build_subdomain_plan_creates_junction_for_2x2_grid` → drop
- `test_build_subdomain_plan_interface_width_dict` → drop
- `test_clip_with_perturbation_drops_neighbor_buffer_halo` → keep, ensure clip_polygon still works
- `test_clip_drops_slivers_below_area_threshold` → keep
- `test_resolution_only_proxy_contributes_no_geometry` → drop
- `test_write_bundles_creates_expected_layout` → update assertion (no interface_0000 bundle)
- `test_run_plan_two_phases_completes` → rewrite as `test_run_plan_completes_single_phase`

Commit: `refactor(distributed): single-phase plan + bundle layout (no seam slabs)`

### Task 5: New `stitch_meshes` via gmsh.merge

Replace the meshio-based stitch:

```python
def stitch_meshes(work_dir, plan, output_mesh, point_tolerance) -> None:
    """Merge all per-volume .msh files via gmsh.merge.

    Workers wrote each .msh with hashed physical-group tags, so
    silicon's tag is identical across files. After gmsh.merge, gmsh
    auto-consolidates entities under each shared (dim, tag), and we
    just removeDuplicateNodes to glue coincident nodes at the seams.
    """
    import gmsh

    work_dir = Path(work_dir)
    output_mesh = Path(output_mesh)

    owns_gmsh = not gmsh.is_initialized()
    if owns_gmsh:
        gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("stitched")
        for v in plan.volumes:
            path = work_dir / "jobs" / v.id / "result.msh"
            if path.exists():
                gmsh.merge(str(path))
        gmsh.option.setNumber("Geometry.Tolerance", point_tolerance / 2)
        gmsh.model.mesh.removeDuplicateNodes()
        gmsh.write(str(output_mesh))
    finally:
        if owns_gmsh:
            gmsh.finalize()
```

Tests: extend `test_stitch_meshes_produces_one_unified_mesh` to also check
that NxM and multi-material scenarios consolidate by name correctly.

Commit: `refactor(distributed): stitch via gmsh.merge with hashed tags`

### Task 6: NxM integration tests

Add new tests in `tests/test_distributed.py` (these were the v1 corner-tile failures, now expected to pass):

```python
def test_distributed_2x2_grid_with_junction(tmp_path): ...
def test_distributed_consolidates_same_material_across_4_tiles(tmp_path): ...
def test_distributed_3x3_mixed_materials(tmp_path): ...
def test_distributed_4_tile_strip(tmp_path): ...
```

Each follows the spec-test pattern: build entities, call `generate_mesh_distributed` with the appropriate subdomains, read the merged .msh with meshio, assert that all expected physical-group names are present and that material counts match the serial baseline (where applicable).

Commit: `test(distributed): NxM grid integration tests (formerly v1-blocked)`

### Task 7: Update docs + example

- `docs/distributed.md`: rewrite limitations section. NxM grids work; the only real limit is mismatched-lc across tiles producing non-conformal seams. Drop the corner-tile language.
- `docs/distributed_example.py`: change to a 3x2 or 4x2 grid with multiple materials per tile. Re-run, verify, update example output in docstring.
- `README.md`: drop the v1 strip-only caveat.
- `docs/superpowers/specs/2026-04-28-distributed-domain-decomposition-design.md`: append a "v2 amendment" section pointing at the new pipeline; keep the v1 design as historical record.

Commit: `docs(distributed): v2 supports NxM grids; drop corner-tile language`

### Task 8: Final validation

Run full suite. Run example. Verify everything green. Final commit if any tweaks.

---

## Execution

I'll dispatch the agent for each task in order, reviewing between. Tasks 1-5 are mechanical (hashed-tag plumbing + dead-code removal); Tasks 6-7 are integration verification.
