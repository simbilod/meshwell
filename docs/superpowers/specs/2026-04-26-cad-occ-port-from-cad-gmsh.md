# Port cad_gmsh logic into cad_occ

**Status:** approved spec, ready for implementation plan
**Date:** 2026-04-26
**Scope:** Phase A + B (pipeline + InterfaceTag). Phase C (test parametrization) is a separate session.

## Problem

`cad_gmsh` and `cad_occ` are supposed to be drop-in replacements but currently have very different pipelines. `cad_gmsh` evolved more aggressively to handle:

1. Almost-coincident faces (via outward shapely buffer + sequential `occ.cut` cascade)
2. InterfaceTag entities (resolve nominal trace against post-buffer polygon boundaries, instantiate as vertical 2D faces, fold into the fragment pass)

`cad_occ` does only a flat `BOPAlgo_Builder.Perform()` over instantiated `TopoDS_Shape`s, with no buffering, no cuts, and no InterfaceTag support. Users picking between backends therefore get materially different behaviour on complex scenes.

`occ_xao_writer` already does the tagging (physical groups, A___B interfaces, A___None exteriors) at serialization time, so we do **not** need to port the tagging step.

## Solution

1. Extract the shapely-only pre-pass shared by both backends into a new `meshwell/cad_common.py`.
2. Refactor `cad_gmsh.process_entities` to call it (no behaviour change).
3. Refactor `cad_occ.process_entities` to call it, then run a sequential `BRepAlgoAPI_Cut` cascade before the existing `BOPAlgo_Builder` fragment.
4. Add `InterfaceTag.instanciate_occ()` so InterfaceTag works in the OCC backend.

`occ_xao_writer` is unchanged. `_remove_keep_false_top_dim` (cad_gmsh-only) is not ported because the XAO writer already excludes keep=False bodies at serialization time.

## Architecture

### New module: `meshwell/cad_common.py`

```python
def prepare_entities(
    entities_list: list[Any],
    perturbation: float,
) -> None:
    """Run the shapely pre-pass shared by cad_gmsh and cad_occ:
    1. Compute a global bbox slightly larger than the union of inputs.
    2. For each polygon-bearing entity: relax precision via
       set_precision(perturbation/100), buffer outward by perturbation,
       clip to bbox.
    3. For each InterfaceTag: call resolve(polygon_ents, default_snap).

    Mutates entities in place; returns None. Idempotent only in the
    sense that calling twice will compound the buffer -- callers
    must not call this twice.
    """
```

Both `cad_gmsh.process_entities` and `cad_occ.process_entities` call this at the start.

### `cad_occ` pipeline (new)

```
1. cad_common.prepare_entities(entities, perturbation)
2. Sort by mesh_order (lowest first)
3. For each entity:
   a. Instantiate via instanciate_occ() -> TopoDS_Shape
   b. Sequentially cut against all previously-instantiated same-dim shapes
      using BRepAlgoAPI_Cut (mirror cad_gmsh's cut step)
4. Final BOPAlgo_Builder fragment over all cut shapes (existing logic)
5. Resolve ownership by mesh_order (existing logic)
```

### `CAD_OCC` parameter additions

- `perturbation: float | None = None` (default `1e-5`, matching cad_gmsh)
- `fuzzy_value` keeps its existing role as the BOPAlgo fuzzy value
- A new `tolerance_boolean` knob is **not** added on this backend — the OCC fragment fuzziness is already controllable via `fuzzy_value`

### `InterfaceTag.instanciate_occ()`

Builds vertical 2D `TopoDS_Face` objects from `self.resolved_linestrings`. One face per linestring segment (mirroring the per-segment `addPlaneSurface` approach in `instanciate()`):

```python
def instanciate_occ(self) -> TopoDS_Shape:
    if not self.resolved_linestrings:
        return _empty_compound()  # see Compound below

    shapes = []
    for ls in self.resolved_linestrings:
        coords = list(ls.coords)
        if len(coords) < 2:
            continue
        for (x1, y1), (x2, y2) in itertools.pairwise(coords):
            v1 = BRepBuilderAPI_MakeVertex(gp_Pnt(x1, y1, self.zmin)).Vertex()
            v2 = BRepBuilderAPI_MakeVertex(gp_Pnt(x2, y2, self.zmin)).Vertex()
            edge = BRepBuilderAPI_MakeEdge(v1, v2).Edge()
            wire = BRepBuilderAPI_MakeWire(edge).Wire()
            face = BRepPrimAPI_MakePrism(
                wire, gp_Vec(0, 0, self.zmax - self.zmin),
            ).Shape()  # extruding a wire produces a face
            shapes.append(face)

    if not shapes:
        return _empty_compound()
    if len(shapes) == 1:
        return shapes[0]

    # Combine into a single compound so cad_occ sees one shape per entity.
    builder = TopoDS_Builder()
    compound = TopoDS_Compound()
    builder.MakeCompound(compound)
    for s in shapes:
        builder.Add(compound, s)
    return compound
```

`cad_occ`'s `_get_shape_dimension` helper already handles compounds (via `TopExp_Explorer`).

## Tests

### Existing tests must pass unchanged

- All `tests/test_interface_tag.py` (15 tests) — InterfaceTag tests use cad_gmsh; behaviour unchanged.
- All `tests/test_cad_occ.py` and other cad_occ-using tests — output mesh should be identical-or-better with the new pipeline (sequential cuts make BOP more reliable on complex scenes).

### New regression tests

In `tests/test_cad_occ.py`, add:

1. **`test_cad_occ_two_abutting_prisms_share_interface`** — mirror of cad_gmsh's `test_cad_gmsh_3d_adjacent_prisms_share_interface`. Two PolyPrisms abutting at x=5; assert physical groups `{A, B, A___B, A___None, B___None}` after `cad_occ` + `write_xao` + `gmsh.open`.

2. **`test_cad_occ_interface_tag_resolves_to_winning_boundary`** — mirror of the InterfaceTag e2e test. Two PolyPrisms + one InterfaceTag at the shared face; assert `iface` physical group exists with at least one face, A is not internally split.

3. **`test_cad_occ_perturbation_below_point_tolerance`** — mirror of the cad_gmsh version. Single PolySurface; assert resulting bbox is within `point_tolerance` of input.

If any new test reveals OCC-specific quirks (e.g., `BRepAlgoAPI_Cut` produces empty result on coincident shapes), fix in the implementation, not by loosening the test.

### Tests NOT changed in this scope

- No backend-parametrization (Phase C deferred). Tests stay backend-specific for now.
- `tests/test_cad_gmsh.py` — unchanged; refactor of cad_gmsh is internal-only.

## Out of scope

- **Test parametrization across both backends** (Phase C) — separate spec.
- **Tagging in cad_occ** — stays in `occ_xao_writer` per existing separation of concerns.
- **`keep=False` body removal** — XAO writer already excludes them at serialization.
- **2D-only InterfaceTag** — same limitation as in cad_gmsh; `zmin`/`zmax` required.
