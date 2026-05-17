# Structured-mesh conformality validator design

**Date:** 2026-05-17
**Branch target:** `feat/structured-clean` (follow-on work on the same branch).
**Companion of:** `2026-05-15-structured-polyprism-clean-design.md` — the structured-polyprism rewrite that this validator polices.

---

## Motivation

The structured-polyprism builder produces a mesh that mixes two element families in one file:

- **Structured slabs** — wedge/hex elements emitted by our own Python builder into a `addDiscreteEntity(3)` per slab.
- **Tet regions** — meshed by gmsh's 3D mesher around the slabs.

Visually the resulting meshes look correct, but the seam between the two families is precisely where non-conformality would hide:

- A wedge quad face shared with the tet side via 2 nodes instead of 4 (T-junction).
- A near-duplicate node pair the global `removeDuplicateNodes` failed to merge.
- A tet mesher inserting a Steiner point on a slab's lateral face mid-edge.
- A "no auto-mesh" internal seam face that gmsh meshed anyway, producing a spurious 2D layer inside a slab.
- A planned `(slab, piece)` whose elements never reached the mesh (silent drop).
- Top/bottom face triangulations whose nodes don't correspond piece-for-piece via z-translation (the spike-P2 failure mode that motivated Layer C).

None of these are caught by the existing `meshwell/quality.py` analyzer — it inspects element shape, not cross-family conformality, and it operates on a `.msh` re-parse rather than the live gmsh session that has the builder's `StructuredMeshPlan` in scope.

This spec defines a validator that runs in the live gmsh session immediately after `gmsh.model.mesh.generate(3)`, consumes the in-memory `StructuredMeshPlan`, and reports conformality failures with enough localization to be useful in CI.

---

## Decisions taken during brainstorming

| Area | Decision |
| --- | --- |
| What to catch | Both topological and geometric conformality, layered. Topology checks run first; geometric localization runs only when a topology check fails (to distinguish "builder produced two distinct face sets at the same location" from "builder produced a genuinely misaligned face"). |
| Call site | Live gmsh session only. No CLI in v1, no `.msh`-and-pickled-plan path. The validator is invoked from Python with `(plan, gmsh model active)` in scope. Re-opening a `.msh` later is deferred — most uses are CI assertions immediately after mesh generation. |
| Source of truth | Mesh + `StructuredMeshPlan`. Plan-aware checks require the plan; refuses to run without it. |
| Element quality | Delegated to `gmsh.model.mesh.getElementQualities`. The validator does not reimplement what `quality.py` does. Quality check is opt-in via `include_quality=True` and is **not** the focus of this validator. `quality.py` is left in place — deprecating it in favour of `getElementQualities` is a separate follow-up. |
| Library use | gmsh's Python API for all primitives (`createFaces`, `getFaces`, `getElementFaceNodes`, `getElementEdgeNodes`, `getDuplicateNodes`, `getElementQualities`). meshio is not used — it offers no analysis primitives, only I/O. |
| Tolerance | Mesh-derived default: `min_edge_length × 1e-6`, where `min_edge_length` comes from `getElementQualities(..., "minEdge")`. Override via `tol=` argument. The default is a footgun-free choice for the common "did the builder produce a gross mismatch" case; the override is for diagnostic deep-dives. |
| Failure model | Severity-tagged `ValidationResult`. Collect all failures rather than first-fail-and-raise. `errors` are definite breakage (hole, T-junction, missing piece, meshed seam, plan-mesh count mismatch); `warnings` are suspicious (near-duplicate nodes inside tolerance but non-zero, top/bottom z-translation residual non-zero but inside tolerance). CI assertion: `assert result, result.format_report()`. |

---

## API surface

One module, one public function, two public dataclasses, no class hierarchies.

```python
# meshwell/structured/validator.py

from dataclasses import dataclass
from typing import Literal

from meshwell.structured.spec import StructuredMeshPlan


@dataclass(frozen=True)
class Issue:
    severity: Literal["error", "warning"]
    check: str                       # e.g. "prism_tet_interface"
    message: str                     # human-readable, one-line preferred
    entities: tuple = ()             # implicated tags / keys for localization
                                     # (e.g. face tags, node tags, (slab, piece) keys)


@dataclass(frozen=True)
class ValidationResult:
    errors: tuple[Issue, ...]
    warnings: tuple[Issue, ...]

    def __bool__(self) -> bool:
        return not self.errors

    def format_report(self) -> str:
        ...                          # grouped by check, errors first, then warnings


def validate_structured_mesh(
    plan: StructuredMeshPlan,
    *,
    tol: float | None = None,
    include_quality: bool = False,
) -> ValidationResult:
    """Validate the live-gmsh-session mesh against the builder's plan.

    Must be called while a gmsh model is initialized and meshed
    (i.e. after ``gmsh.model.mesh.generate(3)``). Reads from gmsh's
    in-memory model; writes nothing.

    Args:
        plan: the StructuredMeshPlan returned by the builder's mesh-stage
            resolution. Required — plan-aware checks refuse to run without it.
        tol: absolute coordinate tolerance for geometric checks. If None,
            derived from the minimum edge length reported by gmsh.
        include_quality: if True, additionally run
            ``gmsh.model.mesh.getElementQualities`` over all elements and
            report degenerate / negative-Jacobian elements. Off by default
            to keep the validator focused on conformality.

    Returns:
        ValidationResult with collected errors/warnings. ``bool(result)``
        is True iff ``errors`` is empty.
    """
    ...
```

Typical CI usage:

```python
result = validate_structured_mesh(plan)
assert result, result.format_report()
```

The function does not raise on validation failure — only on programming errors (no active gmsh model, `plan is None`, malformed plan).

---

## The eight checks

Each check is a private function in `validator.py` returning `list[Issue]`. `validate_structured_mesh` calls them in order, accumulates results into a single `ValidationResult`. Checks 1–4 are pure topology. Check 5 is mesh-level geometry. Check 6 only runs if 2 reports failures. Check 8 mixes topology + geometry but is structured-specific. Check 7 (quality) is opt-in.

### 1. Watertight volume boundary

For every physical volume (structured slab or tet region), every face that bounds it must appear in exactly one element of that volume (boundary faces), or in exactly two elements of the *same* volume (internal faces of that volume).

**Implementation:** `gmsh.model.mesh.getElements(3, volume_tag)` → per-element node lists → enumerate face nodes via `gmsh.model.mesh.getElementFaceNodes(elementType, faceType)` → tally face occurrences with `frozenset` keys.

**Failure → ERROR.** A face appearing 0 times where the plan says a piece exists, or 3+ times (geometric overlap), is a definite bug.

### 2. Prism↔tet face matching at the structured/tet interface

For each `(slab, piece, "lateral"|"top"|"bottom")` face that the plan identifies as an interface with the tet region, the following must hold:

- A prism **quad** face → exactly 2 tet triangles on the other side, sharing all 4 prism nodes and no others. (Standard quad-split-into-two-triangles, no Steiner node.)
- A prism **triangle** face → exactly 1 tet triangle sharing all 3 nodes.
- A hex **quad** face on the slab/tet interface → exactly 2 tet triangles, same condition as the prism quad.

**Implementation:** From the plan, enumerate interface OCC face tags. For each: collect the 2D mesh elements on the OCC face (`gmsh.model.mesh.getElements(2, face_tag)`); from each side's volume mesh elements that touch this face, build the face-node sets via `getElementFaceNodes`; cross-reference.

**Failure → ERROR**, with the offending face tag and the two element IDs in `Issue.entities`.

### 3. Internal-seam faces carry no 2D elements

Layer A of the polyprism rewrite explicitly marks the OCC interior seam faces (between adjacent pieces of the same slab's `face_partition`) as "no auto-mesh" before `generate(2)`. The validator must confirm this stuck: each such face should have zero 2D elements.

**Implementation:** From the plan's recorded internal seam face tags, call `gmsh.model.mesh.getElements(2, seam_face_tag)` for each; assert empty.

**Failure → ERROR.** A spurious 2D mesh layer inside a slab indicates the no-auto-mesh hint failed to apply.

### 4. Plan↔mesh consistency

For each `(slab, piece)` in the plan, the wedge/hex element count in the corresponding discrete 3D entity must equal `n_layers × number_of_bottom_triangles` (for prisms) or `n_layers × number_of_bottom_quads` (for hexes). No extra slabs, no missing slabs.

**Implementation:** Iterate `plan.slabs`; for each, `gmsh.model.mesh.getElements(3, slab.discrete_entity_tag)`; count by element type; compare against expected count from the plan's piece-level metadata.

**Failure → ERROR.** A silent piece-drop or layer-count mismatch was one of the original failure modes the rewrite is supposed to prevent — catching it here closes the loop.

### 5. Near-duplicate nodes

Wraps `gmsh.model.mesh.getDuplicateNodes()` — gmsh's built-in detector for exact duplicates — and pairs it with a tolerance-aware sweep for near-duplicates. The near-duplicate pass uses a spatial hash over node coordinates with bin size = `tol`; any pair inside `tol` but with distinct node IDs is reported.

**Exact duplicates → ERROR** (indicates `removeDuplicateNodes` was skipped or the merge tolerance was set too tight).
**Near-duplicates (within `tol` but non-zero) → WARNING** (geometrically suspicious; may indicate a merge that should have happened but didn't).

### 6. Geometric coordinate match — runs only on topology check 2 failures

When check 2 reports that a prism interface face has the wrong number of tet triangles on the other side, run a coordinate-level localization on the failed face:

- Look up the prism face's node coordinates.
- Scan all 2D elements on neighbouring tet volume boundaries with all three node coordinates inside the bounding box (inflated by `tol`).
- Report: "no candidate at all" (definite hole) vs. "candidate exists at matching coords but with different node IDs" (definite duplicate-node bug) vs. "candidate exists with a slight coordinate offset" (definite misalignment).

This refines check 2's `ERROR` into a more specific `ERROR` message; severity is unchanged.

### 7. Element quality — opt-in (off by default)

When `include_quality=True`, call `gmsh.model.mesh.getElementQualities` with `qualityName="minSICN"` over all elements; flag negative or near-zero values as `ERROR`, very small positive values (below a heuristic threshold) as `WARNING`.

Default off. The validator's identity is conformality, not quality. This check is provided so that callers don't need to also separately invoke `quality.py` or roll their own quality loop — one validator call covers both if they want it.

### 8. Top↔bottom face symmetry (structured-specific)

For each piece in each slab, the top-face triangulation must be a z-translation of the bottom-face triangulation:

- Same node count.
- A bijection between top and bottom nodes (via the builder's stamped correspondence stored in the plan).
- Coords differ only in z; the residual `max(|x_top - x_bot|, |y_top - y_bot|)` is reported as `ERROR` if exceeding `tol`.

This is the explicit guard against the P2 spike's failure mode — the property Layer C of the polyprism rewrite was built to maintain.

---

## Module layout

```
meshwell/structured/
    validator.py           # new — public API + check implementations
    spec.py                # existing — StructuredMeshPlan lives here (no changes
                           #   needed; the plan already records every key the
                           #   validator uses: discrete_entity_tag,
                           #   interface_face_tags, internal_seam_face_tags,
                           #   piece_count_by_slab, top_to_bottom_node_map)
```

Tests:

```
tests/structured/
    test_validator_watertight.py
    test_validator_interface.py
    test_validator_seams.py
    test_validator_plan_consistency.py
    test_validator_duplicates.py
    test_validator_top_bottom_symmetry.py
    test_validator_quality.py
    test_validator_integration.py   # end-to-end: build a known-good scene,
                                    # assert validator passes; build a scene
                                    # with a deliberately injected fault,
                                    # assert validator fails with the expected
                                    # check name in the issues list.
```

Existing structured end-to-end tests (`test_end_to_end_minimal.py`, `test_end_to_end_multipiece.py`) gain a single line: `assert validate_structured_mesh(plan), result.format_report()` after mesh generation. This is the assertion that makes the validator earn its keep — every existing structured test starts policing its own conformality for free.

`quality.py` is **not** modified. Removing or rewriting it is a follow-up, out of scope here.

---

## Plan attributes the validator reads

The validator depends on these existing or planned attributes of `StructuredMeshPlan` / `Slab`:

- `plan.slabs[i].discrete_entity_tag: int` — the gmsh 3D discrete entity tag for slab `i`.
- `plan.slabs[i].pieces[k].interface_face_tags: dict[str, list[int]]` — keyed by side `"lateral_m"` / `"top"` / `"bottom"`, gives the post-BOP face tags that are shared with tet regions.
- `plan.slabs[i].internal_seam_face_tags: list[int]` — the post-BOP face tags for inter-piece seams marked no-auto-mesh.
- `plan.slabs[i].pieces[k].expected_wedge_count: int` and `expected_hex_count: int` — derived from `n_layers × #bot_tris` / `n_layers × #bot_quads` at mesh-plan resolution time.
- `plan.slabs[i].pieces[k].top_bottom_node_correspondence: dict[int, int]` — populated by Layer C's stamped top-from-bottom step.

Any of these that do not yet exist on `StructuredMeshPlan` will be added as part of the validator's implementation. The plan rewrite design already records the keys in Layer B; the validator surfaces them by name. The implementation plan will identify the precise diff, but in the worst case this is a few new fields with no behavioral changes to existing builder code.

---

## What the validator deliberately does **not** do

- It does not run mesh quality by default (delegated to `gmsh.model.mesh.getElementQualities`; opt-in only).
- It does not re-parse `.msh` files. No standalone-file mode in v1.
- It does not write or modify the mesh. Read-only against the live gmsh model.
- It does not duplicate `quality.py`. `quality.py` is left untouched.
- It does not check arc-resolution agreement between neighbours (an arc face's discretization matching its neighbour's discretization is a property of the CAD stage's edge sharing, not of conformality; if Layer B's edge sharing works, conformality follows. If it doesn't, check 2 fails and check 6 will localize it).

---

## Open questions deferred to implementation

- Whether `format_report()` should group by check or by entity. Default: group by check, errors first; revisit if reports get noisy.
- Whether the spatial-hash bin size in check 5 needs tuning for very anisotropic meshes (extremely fine in z, coarse in xy). Default: use `tol` uniformly; revisit if false-negatives appear in tests.
- Whether to expose a `strict=True` option that elevates all warnings to errors. Probably yes, but trivial; add when first user asks.
