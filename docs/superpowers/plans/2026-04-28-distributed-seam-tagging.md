# Distributed: seam-id tagging for cross-tile interfaces

**Goal:** Restore meshwell's v1 interface-naming convention (`silicon___oxide` for material-material interfaces; `material___None` only on true outer-domain edges) on top of the v2 single-phase pipeline.

**Approach:** Workers tag subdomain-internal boundary faces as `<material>___seam_<i>_<j>` (where `(i, j)` is the sorted pair of subdomain ids); true outer-domain faces stay `<material>___None`. Master post-merge consolidates by seam id: same material both sides → drop (invisible interior face); different materials → retag as `A___B`.

**Validated by:** the architecture follows directly from the merge spike at `tests/test_merge_spike.py` + the empirical naming-by-hashed-tag finding (commit `18c1628`).

---

## Files

| File | Change |
|---|---|
| `meshwell/distributed.py` | Add neighbour-map computation in `build_subdomain_plan`. Ship per-tile neighbour list in `job.json`. Add `_retag_subdomain_seams` worker helper. Add `_consolidate_seam_groups` master post-merge pass invoked from `stitch_meshes`. |
| `meshwell/mesh.py` | (No change.) |
| `tests/test_distributed.py` | Update `test_distributed_two_materials_shared_interface` to assert the v1 `silicon___oxide` convention (drop the v2-shift docstring). Add `test_distributed_seam_drops_invisible_interior_faces` (single material across 2 tiles → no `mat___seam` group remains). |

---

## Tasks

### Task 1: Compute neighbour map in `build_subdomain_plan`

In `meshwell/distributed.py`, extend `VolumeRegion`:

```python
@dataclass
class VolumeRegion:
    id: str
    polygon: Polygon
    neighbors: list[tuple[str, str]] = field(default_factory=list)
    # neighbors: list of (neighbour_volume_id, shared_boundary_wkt) pairs.
    # Empty boundary means no contact; populated only for actual seams.
```

In `build_subdomain_plan`, after constructing `volumes`, compute pairwise shared boundaries:

```python
for i in range(len(subdomains)):
    for j in range(i + 1, len(subdomains)):
        shared = subdomains[i].boundary.intersection(subdomains[j].boundary)
        if shared.is_empty or shared.length < point_tolerance:
            continue
        volumes[i].neighbors.append((volumes[j].id, shared.wkt))
        volumes[j].neighbors.append((volumes[i].id, shared.wkt))
```

Test: `test_build_subdomain_plan_populates_neighbors` — 2x1 grid, expect each volume's `.neighbors` to contain a single entry pointing at the other.

Commit: `feat(distributed): populate VolumeRegion.neighbors with shared-boundary geometry`

### Task 2: Ship neighbour map in `job.json`

In `_write_volume_bundle`, add a `neighbors` field (list of `{id, shared_boundary_wkt}` dicts) to `job.json`.

In `run_job`, deserialize the neighbour list and pass it as `_subdomain_neighbours` to the worker call. (This is one more private kwarg, similar to existing `_pre_buffered`, `_global_physical_names`, `_hashed_physical_tags`.)

Test: `test_write_bundles_ships_neighbors_in_job_json`.

Commit: `feat(distributed): ship subdomain neighbour list in job.json`

### Task 3: Worker post-CAD-tag retag pass

Add `_retag_subdomain_seams` to `meshwell/distributed.py`:

```python
def _retag_subdomain_seams(
    subdomain_polygon: Polygon,
    neighbours: list[tuple[str, str]],   # [(neighbour_id, shared_boundary_wkt), ...]
    own_id: str,
    point_tolerance: float,
) -> None:
    """Rename <material>___None faces that lie on subdomain seams.

    For each existing physical group ending in '___None', partition its
    face entities by which subdomain seam (if any) they lie on:
      - face on the seam shared with neighbour N:
          regroup as <material>___seam_<min(own,N)>_<max(own,N)>
      - face not on any seam (true outer boundary):
          stays in <material>___None group

    Multiple physical groups may share a material name across different
    seams; gmsh allows multiple physical groups per face entity.
    """
    import shapely.wkt
    import gmsh

    nb_lines = [
        (nb_id, shapely.wkt.loads(wkt))
        for nb_id, wkt in neighbours
    ]
    own_index = int(own_id.split("_")[-1])  # "volume_0001" -> 1

    for d, tag in list(gmsh.model.getPhysicalGroups(2)):
        name = gmsh.model.getPhysicalName(2, tag)
        if not name.endswith("___None"):
            continue
        material = name[: -len("___None")]
        ents = list(gmsh.model.getEntitiesForPhysicalGroup(2, tag))

        outer = []
        seam_buckets: dict[str, list[int]] = {}  # neighbour_id -> face tags

        for ent in ents:
            bb = gmsh.model.getBoundingBox(2, ent)
            assigned = False
            for nb_id, nb_geom in nb_lines:
                if _bbox_on_geometry(bb, nb_geom, point_tolerance):
                    seam_buckets.setdefault(nb_id, []).append(int(ent))
                    assigned = True
                    break
            if not assigned:
                outer.append(int(ent))

        gmsh.model.removePhysicalGroups([(2, tag)])
        if outer:
            gmsh.model.addPhysicalGroup(
                2, outer,
                tag=_name_to_tag(name, 2),
                name=name,
            )
        for nb_id, seam_ents in seam_buckets.items():
            nb_index = int(nb_id.split("_")[-1])
            lo, hi = sorted([own_index, nb_index])
            seam_name = f"{material}___seam_{lo:04d}_{hi:04d}"
            gmsh.model.addPhysicalGroup(
                2, seam_ents,
                tag=_name_to_tag(seam_name, 2),
                name=seam_name,
            )


def _bbox_on_geometry(bb, geom: BaseGeometry, tol: float) -> bool:
    """Return True iff the OCC face's xy bbox sits on the subdomain seam line.

    bb is gmsh's 6-tuple (xmin, ymin, zmin, xmax, ymax, zmax). The seam
    geometry is a 2D LineString or MultiLineString in xy. A face is "on"
    the seam if its xy projection (a degenerate segment when the face is
    vertical) lies within tol of the seam line.
    """
    xmin, ymin, _zmin, xmax, ymax, _zmax = bb
    from shapely.geometry import box
    face_xy = box(xmin - tol, ymin - tol, xmax + tol, ymax + tol)
    return geom.intersects(face_xy) and geom.distance(face_xy) <= tol
```

In `run_job`, call `_retag_subdomain_seams` after `generate_mesh` finishes but before the result.msh is written. **Important:** this must happen on the gmsh model that `generate_mesh` produced — we may need to factor out the .msh write step so the retag fits in between, OR run `_retag_subdomain_seams` as a separate post-processing pass that opens result.msh, retags, and rewrites.

Easier path: post-process the .msh after generate_mesh writes it. Open the file in a fresh gmsh session, retag, rewrite. Implementation:

```python
def _retag_subdomain_seams_in_msh(
    msh_path: Path,
    subdomain_polygon: Polygon,
    neighbours: list[tuple[str, str]],
    own_id: str,
    point_tolerance: float,
) -> None:
    import gmsh
    owns = not gmsh.is_initialized()
    if owns:
        gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("retag")
        gmsh.merge(str(msh_path))
        _retag_subdomain_seams(subdomain_polygon, neighbours, own_id, point_tolerance)
        gmsh.write(str(msh_path))
        gmsh.model.remove()
    finally:
        if owns:
            gmsh.finalize()
```

In `run_job`, call this right after the `generate_mesh` call writes `result.msh`.

Test: `test_run_job_tags_subdomain_seams` — small 2x1 case, run the worker on volume_0000, verify the result.msh has `mat___seam_0000_0001` (no `mat___None` for the seam face) and `mat___None` for the true outer faces.

Commit: `feat(distributed): worker retags subdomain-seam faces with seam IDs`

### Task 4: Master post-merge consolidation in `stitch_meshes`

Extend `stitch_meshes` to add a consolidation pass after `removeDuplicateNodes`:

```python
import re

def _consolidate_seam_groups() -> None:
    """Walk merged-model physical groups; rename '___seam_i_j'-tagged groups
    based on the materials that contributed to each seam.

    For each seam id (i, j) appearing in any physical-group name suffix:
      collect all groups matching '*___seam_i_j'
      collect the materials (the part before '___seam_')
      if all materials are the same: drop the groups (invisible interior face)
      if multiple materials: union all their entities into a single new group
        named '<A>___<B>' (alphabetical). Drop the per-material seam groups.
    """
    import gmsh

    pat = re.compile(r"^(.+)___seam_(\d+)_(\d+)$")
    by_seam: dict[tuple[int, int], list[tuple[int, str]]] = {}
    for d, tag in gmsh.model.getPhysicalGroups(2):
        name = gmsh.model.getPhysicalName(2, tag)
        m = pat.match(name)
        if not m:
            continue
        material, i, j = m.group(1), int(m.group(2)), int(m.group(3))
        by_seam.setdefault((i, j), []).append((tag, material))

    for (i, j), group_list in by_seam.items():
        materials = sorted({mat for _, mat in group_list})
        all_ents: list[int] = []
        for tag, _ in group_list:
            all_ents.extend(int(e) for e in gmsh.model.getEntitiesForPhysicalGroup(2, tag))
        for tag, _ in group_list:
            gmsh.model.removePhysicalGroups([(2, tag)])
        if len(materials) == 1:
            # interior face — drop entirely
            continue
        new_name = "___".join(materials)
        gmsh.model.addPhysicalGroup(
            2, list(set(all_ents)),
            tag=_name_to_tag(new_name, 2),
            name=new_name,
        )
```

Call `_consolidate_seam_groups()` in `stitch_meshes` after `removeDuplicateNodes` and before `gmsh.write`.

Test: extend `test_distributed_two_materials_shared_interface` to assert `silicon___oxide` is present and `silicon___None` / `oxide___None` are not present on the seam (only on true outer faces).

Add `test_distributed_same_material_drops_invisible_seam` — single material spans 2 tiles, after stitch there's no `mat___seam_*` and no `mat___None` on the interior cut.

Commit: `feat(distributed): post-merge consolidate ___seam_i_j into A___B or drop`

### Task 5: Final validation + run example

```bash
pytest tests/test_distributed.py -v --no-cov
pytest tests/ -q --ignore=tests/test_distributed_spike.py --ignore=tests/test_merge_spike.py
rm -rf docs/distributed_example_work docs/distributed_example*.msh
python docs/distributed_example.py
```

Inventory comparison should now show identical material AND interface inventories between serial and distributed.

Commit: `docs(distributed): seam-id tagging completes the v1 convention parity`
