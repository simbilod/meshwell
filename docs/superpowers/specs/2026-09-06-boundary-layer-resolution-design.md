# BoundaryLayerResolutionSpec Design

**Goal:** Add a `ResolutionSpec` that, when attached to a 1D physical name,
configures a gmsh **BoundaryLayer** mesh field on those curves — giving an
anisotropic, graded (optionally quad) layer grown off the curve, meshed by
gmsh's frontal-Delaunay mesher.

This complements `StructuredSweep`, not replaces it: StructuredSweep
hand-stamps an *exact* tensor grid at straight material interfaces;
`BoundaryLayerResolutionSpec` delegates to gmsh's *heuristic* boundary layer,
which handles curved walls, corner fans, and 3D — at the cost of exactness.

## Tech Stack

Python, pydantic (v1-style `class Config` where needed), gmsh Python API,
pytest via `uv run pytest`. Branch: `feat/boundary-layer-resolution`, stacked
on `feat/structured-sweeps`.

## Scope decisions (from brainstorming)

- **Target:** any 1D physical name — a `PolyLine`'s own name, an
  auto-generated interface `a___b`, or a boundary `a___None`. Keyed in
  `resolution_specs` exactly like every other spec.
- **Parameter surface:** full gmsh BoundaryLayer passthrough.
- **Element type:** raw `quads: bool` (gmsh `Quads` 0/1). No built-in
  quad→triangle split; callers can run `gmsh.model.mesh.splitQuadrangles()`
  downstream if they want it.
- **Fan points:** DEFERRED to a follow-up PR. Fans are identified by 0D
  physical name, but meshwell cannot currently name a single 0D point in a 2D
  mesh (the XAO boundary pass emits only `dim-1` groups; dim-0 groups appear
  only in polyline-only models, as endpoint sets). The follow-up PR first adds
  a **named 0D OCC point entity** (via the existing generic `OCC_entity` with
  `dimension=0`), ensuring the named vertex survives BOP fragmentation and is
  emitted as a dim-0 physical group / embedded mesh node, then wires
  `fan_points` into this spec. See "Follow-up: fan points" below. **This
  design (PR 1) ships the core BL spec without fan points.**
- **Multiplicity:** each spec becomes its own gmsh BoundaryLayer field.
  Verified empirically that the installed gmsh accepts multiple boundary-layer
  fields (different curves/params) in one model and meshes them together.

## The class (`meshwell/resolution.py`)

```python
class BoundaryLayerResolutionSpec(ResolutionSpec):
    """Configure a gmsh BoundaryLayer field on a 1D physical name.

    Attached (via resolution_specs) to any dim-1 physical group — a PolyLine
    name, an interface `a___b`, or a boundary `a___None`. Produces an
    anisotropic graded layer off those curves. Registered with gmsh via
    setAsBoundaryLayer, NOT combined into the Min background size field.
    """

    apply_to: Literal["curves"] = "curves"        # dim-1 target
    size: float = Field(gt=0)                       # gmsh Size: first-layer normal size
    thickness: float = Field(gt=0)                  # gmsh Thickness: total layer thickness
    ratio: float = Field(default=1.0, ge=1.0)       # gmsh Ratio: geometric growth
    quads: bool = False                             # gmsh Quads (0/1)
    size_far: float | None = None                   # gmsh SizeFar
    nb_layers: int | None = None                    # gmsh NbLayers
    intersect_metrics: bool = False                 # gmsh IntersectMetrics
    aniso_max: float | None = None                  # gmsh AnisoMax
    beta: float | None = None                       # gmsh Beta
    # fan_points / fan_points_sizes -> follow-up PR (needs named 0D points)
```

Validation:
- `size > 0`, `thickness > 0`, `ratio >= 1` (pydantic `Field`).
- `name` / physical-name rules unchanged (this reuses the standard key).

Optional params (`size_far`, `nb_layers`, `aniso_max`, `beta`) are pushed to
gmsh **only when set**, so gmsh defaults hold when they are `None`.

Verified gmsh option names (all accepted on the installed gmsh): `Size`,
`Ratio`, `Thickness`, `Quads`, `SizeFar`, `NbLayers`, `IntersectMetrics`,
`AnisoMax`, `Beta`, `CurvesList`, `PointsList`, `FanPointsList`,
`FanPointsSizesList`.

## `apply()` — self-registering, returns `None`

The per-name dispatch in `_mesh_entity.add_refinement_fields_to_model` already:
(1) resolves this physical name to its dim-1 tags (`entities_mass_dict`), and
(2) filters out `None` returns (`_mesh_entity.py:481`, `if field_idx is not
None`). So `apply()` does its own registration and returns `None`:

```python
def apply(self, model, entities_mass_dict, **kwargs) -> None:
    f = model.mesh.field.add("BoundaryLayer")
    model.mesh.field.setNumbers(f, "CurvesList", list(entities_mass_dict.keys()))
    model.mesh.field.setNumber(f, "Size", self.size)
    model.mesh.field.setNumber(f, "Thickness", self.thickness)
    model.mesh.field.setNumber(f, "Ratio", self.ratio)
    model.mesh.field.setNumber(f, "Quads", 1 if self.quads else 0)
    if self.size_far is not None:
        model.mesh.field.setNumber(f, "SizeFar", self.size_far)
    if self.nb_layers is not None:
        model.mesh.field.setNumber(f, "NbLayers", self.nb_layers)
    model.mesh.field.setNumber(f, "IntersectMetrics", 1 if self.intersect_metrics else 0)
    if self.aniso_max is not None:
        model.mesh.field.setNumber(f, "AnisoMax", self.aniso_max)
    if self.beta is not None:
        model.mesh.field.setNumber(f, "Beta", self.beta)
    model.mesh.field.setAsBoundaryLayer(f)
    return None
```

Because `apply()` returns `None`, the BL field never enters the `Min`
background field — correct, since a BL field is registered via
`setAsBoundaryLayer` separately from the size-field pipeline.

## Integration approach (chosen)

**A — self-register in `apply()`, return `None` (recommended, chosen).**
Minimal; fits the existing `ResolutionSpec` dispatch; the `None`-filter
already exists.

Rejected:
- **B — dedicated discovery pass** (like `sweeps=`): unnecessary
  infrastructure; the per-name path already yields the tags.
- **C — merge into the `Min` size field:** not viable — a BL field is not a
  Min-combinable size field.

## Edge cases / limitations

- **Global (`None`-keyed) BL specs:** the global-spec path
  (`mesh.py:336`) appends `apply()` returns **without** the `None`-filter,
  so a `None` there would corrupt the `Min` field. Fix: add the same
  `if field_index is not None` guard to that path (one line). BL specs are
  expected to be name-keyed regardless.
- **BL + StructuredSweep on the same model:** the sweep stamping kernel sets
  `Mesh.MeshOnlyEmpty=1`, which would suppress gmsh's boundary-layer
  generation. v1 raises a clear error if both are requested for the same
  model rather than silently producing a bad mesh. Combining them is out of
  scope for v1.
- **Fan-point name not found, or `fan_points_sizes` length mismatch:** clear
  `ValueError` at apply / construction time.

## Testing (`tests/test_boundary_layer.py`)

Real behavior, not mocks:
- BL on a `PolyLine` name and on an `a___b` interface → mesh has anisotropic
  cells hugging the curve; first-layer offset ≈ `size`; spacing grows by
  ≈ `ratio`.
- `quads=True` → quad cells present in the layer; `quads=False` → triangles.
- Two BL specs (different curves/params) → both layers present in one mesh.
- Regression: a mesh with no BL spec is unchanged; full suite stays green.

## Files

- `meshwell/resolution.py` — `BoundaryLayerResolutionSpec` (+ any package
  export it needs).
- `meshwell/mesh.py` — one-line `None` guard on the global-spec apply path.
- `tests/test_boundary_layer.py` — new.
- `docs/25_boundary_layer.py` — short worked-example notebook (in scope;
  mirrors `docs/23`/`24`).

## Follow-up: fan points (separate, stacked PR)

Fan points wrap a boundary layer around a sharp convex vertex. They are
identified by a single 0D vertex, but meshwell has no way today to name a
single point in a 2D mesh. The follow-up PR (stacked on this one) will:

1. **Named 0D OCC point entity.** Use the existing generic `OCC_entity`
   (`meshwell/occ_entity.py`) with `dimension=0` and an `occ_function` that
   builds a `TopoDS_Vertex` at a coordinate. Ensure the CAD pipeline
   (`cad_occ.py`) carries a dim-0 entity through BOP fragmentation without
   dropping the vertex, and that `occ_xao_writer.py` emits its `(0, name)`
   physical group (the `tag_entities` pass already keys groups by `ent.dim`,
   so a surviving dim-0 leaf should group naturally — the risk is the
   standalone vertex being dropped/uncaptured by the fragment, which the PR
   must verify and, if needed, embed into the containing surface).
2. **Wire `fan_points` into `BoundaryLayerResolutionSpec`.** Add
   `fan_points: list[str]` (0D physical names) + `fan_points_sizes: list[int]`,
   a `_resolve_point_names(model, names)` helper (query gmsh 0D physical
   groups; require each name to resolve to exactly one point tag; clear error
   otherwise; `len(fan_points_sizes) == len(fan_points)` or empty), and set
   `FanPointsList`/`FanPointsSizesList` in `apply()` before
   `setAsBoundaryLayer`.

That work gets its own spec + plan when this PR lands.

## Non-goals (this PR)

- Fan points (deferred to the follow-up above).
- Curved-source *exactness* (that's StructuredSweep's domain; BL is
  heuristic by nature).
- quad→triangle split option (do it downstream via `splitQuadrangles`).
- Combining BL and StructuredSweep on one model.
