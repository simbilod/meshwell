# Structured Slab Pipeline — Deep-Dive Notes (2026-05-13)

Companion to [2026-05-13-structured-slab-arc-periodic-correctness.md](2026-05-13-structured-slab-arc-periodic-correctness.md).

This document captures the reasoning behind three load-bearing
design choices in the structured-slab pipeline that came up during
the R2 fix session:

1. **Why `setPeriodic` can't pair bottom ↔ top corners on arc-bearing
   slabs in dense scenes** (R1 mechanism).
2. **What role the internal `BOPAlgo_Builder` plays inside
   `_StructuredPhantom.instanciate_occ`** (it is not a boolean fuse).
3. **Why `identify_arcs` is intentionally NOT propagated through the
   cascade to resolved sub-slabs.**

These notes are intended to read standalone so future debugging
sessions can pick up the context quickly.

---

## 1. Why `setPeriodic` can't pair bottom ↔ top corners

### The mechanism

1. The slab phantom builds a 3D solid via
   `BRepPrimAPI_MakePrism(footprint, dz)`. **Right after
   construction**, the OCC vertices of the top face are *exact*
   translations of the bottom face's vertices — perfect symmetry.

2. The **global cad_occ `BOPAlgo_Builder`** then fragments this solid
   against every neighbour:
   * The slab's **bottom** (at `z=zlo`) gets cut against the
     **cladding's top face** plus any 2D/3D entity touching `z=zlo`.
   * The slab's **top** (at `z=zhi`) gets cut against the
     **encapsulant's bottom face** plus any 2D/3D entity touching
     `z=zhi`.

3. Cladding and encapsulant are *different entities* with
   *independent* polygon vertex sets, snapped through the shapely
   buffer (`perturbation ≈ 1e-5`) and BOP fuzzy
   (`fragment_fuzzy_value ≈ 1e-3`). When BOP cuts the slab bottom
   against the cladding, it snaps cut points to the cladding's vertex
   grid. When it cuts the top against the encapsulant, it snaps to
   the encapsulant's vertex grid.

4. Result: the slab's bottom and top **start as exact translations**
   but **end up with drift** of
   `O(perturbation + fragment_fuzzy_value) ≈ 1e-5 to 1e-3` at the
   cut points. This drift is *geometric truth*: each cut is a
   faithful intersection with its specific neighbour, and the two
   neighbours don't agree.

### Why gmsh's `setPeriodic` can't recover

`gmsh.model.mesh.setPeriodic(2, [top], [bottom], translation)` is
**all-or-nothing on corners**:

1. It walks the slave face's bounding 1D curves.
2. For each curve, it walks the bounding 0D vertices.
3. For each slave vertex, it computes `slave_xyz - translation` and
   looks for a master vertex within `Geometry.Tolerance` (or a
   hard-coded internal tolerance — empirically a `1e-3` bump did
   *not* loosen it during the session).
4. **A single unmatched vertex aborts the entire periodic
   constraint.**

In the bench scene, struct_ring_0 had 118 bottom and 118 top curves,
with 117 xy-matched and **1 pair drifted past tolerance** at
`(2.80969, 11.6567)` vs `(2.80968, 11.6567)` — a 1e-5 drift caused
by BOP snapping to different neighbour vertices on bottom vs top.

### Why the mesh is still correct without setPeriodic

`_build_one_slab_conformal` does not actually *need* setPeriodic to
succeed. It reads the bottom 2D mesh and **stamps a translated copy
of those nodes and elements directly onto each top sub-face**,
overwriting whatever gmsh produced for the top face. The setPeriodic
call was originally added to keep the *lateral OCC faces' bottom and
top boundary curves* node-matched (otherwise `setTransfiniteSurface`
fails). For arc-bearing slabs in dense scenes, that fails — but the
conformal builder downstream rescues correctness by stamping
translations.

Hence the mesh is *correct* even with the warnings; the warnings are
noise about a constraint that was redundant for arc-bearing geometry.

### What a true fix would look like

**Force OCC-level vertex symmetry after BOP**, before
`apply_structured_slabs`. Add a *slab canonicalization* pass to
cad_occ that, for each slab:

1. Locate the bottom and top OCC sub-face pairs by xy projection.
2. For each bottom vertex, compute `(x, y, zhi)`.
3. **Replace** the corresponding top OCC vertex's coordinates with
   that exact value — eliminating the drift.

Implementing this cleanly is non-trivial because OCC TShapes are
shared: a single vertex may bound multiple faces and edges. The
operation has to use `BRepBuilderAPI_Transform` to produce a new
isolated top-face geometry that's pinned to the bottom +
translation, then re-fragment locally. Or: build the entire slab's
top face *from scratch* as `BRepBuilderAPI_Transform(bottom,
translation)` after the global BOP — bypassing the BOP's independent
top-cut entirely.

Either path is ~half a day of OCC plumbing and probably one or two
cascade interactions to chase. The current mitigation (demote
"point correspondence" warnings to DEBUG + rely on the conformal
builder to override the top mesh) gives a correct mesh today; the
above would also give a clean `setPeriodic` AND remove the
conformal builder's dependency on overriding gmsh's top mesh.

---

## 2. The role of the internal `BOPAlgo_Builder` (it's not a fuse)

The `BOPAlgo_Builder` call inside `_StructuredPhantom.instanciate_occ`
(around line 601, despite the variable name `bop`) is **not** doing
a boolean fuse — it's doing a **non-merging TShape-sharing pass**.

### Three ways to combine N sub-prisms in OCC

| approach | merges coplanar faces? | shares TShapes at partition seams? |
|---|---|---|
| `BRepAlgoAPI_Fuse` | yes (eliminates the partition we just built) | yes |
| `BRep_Builder.MakeCompound + Add` (plain wrapper) | no | **no** |
| `BOPAlgo_Builder` (no-fuse mode) | no | **yes** |

We need the third option:

* **No merging** — otherwise the partition we just spent code
  building is wiped out, and we lose the symmetric bottom/top
  sub-face decomposition the mesh-stage builder relies on.
* **TShape sharing across the partition seam** — so when sub-prism
  A and sub-prism B meet along an internal partition face, that
  face has *one* `TopoDS_Shape` referenced by both solids, not two
  near-coincident-but-distinct ones.

### Why TShape sharing matters

1. **Downstream global BOP attribution.** When the global cad_occ
   `BOPAlgo_Builder` sees that sub-prism A's right face and
   sub-prism B's left face are *the same TShape*, it does not have
   to fuzzy-rediscover that they coincide. Without the sharing the
   global BOP would still find them via fuzzy `1e-3` snapping — but
   the resulting TShape identity might or might not survive
   depending on neighbour interactions.

2. **XAO writer interface tagging.** `_compute_physical_groups` in
   `occ_xao_writer.py` uses `bid1 & bid2` set intersection on
   TShape identifiers to discover shared boundary faces between
   entities. If A's partition face and B's partition face don't
   share a TShape, that intersection is empty and the interface
   tagging falls through to the fuzzy AABB fallback (a known
   secondary risk). With shared TShapes the intersection works
   directly.

### The dark side of this same mechanism (R2 root cause)

R2 came from exactly this TShape-sharing layer. When a partition
piece had interior holes (an annulus with interior splitter
cutouts), the internal BOP fragmented the input sub-prism into
**more sub-solids than partition pieces**, all sharing TShape state
through the BOP's history map. The *global* BOP's `Modified()` lookup
then over-attributed pieces across the entire scene back to those
sub-prism inputs (97 mod_pieces from 5 originals in the bench).

**Path A's bridge cuts** eliminated the trigger by ensuring partition
pieces are always *simply-connected* — so the internal BOP doesn't
produce extra sub-solids, and the TShape sharing stays well-scoped
to legitimate partition seams.

**Summary:** `BOPAlgo_Builder` is acting as a "topological linker",
not a boolean fuse. It glues the sub-prisms together at shared
TShapes while keeping them as N separate solids. We need it for
clean interface tagging and clean global-BOP fragmentation; we
simply had to make sure (via Path A) that the input it sees is
non-pathological.

---

## 3. Why `identify_arcs` is NOT propagated through the cascade

`expand_slabs_for_entity` reads `identify_arcs` off the original
`_StructuredPolyPrism` and stamps it onto the raw `Slab` objects.
**But the cascade in `resolve_structured_slabs` then drops the flag
when constructing resolved sub-slabs.** This is intentional.

### What changes when `slab.identify_arcs=True`

`_StructuredPhantom.instanciate_occ` calls
`_make_occ_wire_from_vertices(..., identify_arcs=True)`. This walks
the partition piece's vertices, identifies runs of near-cocircular
vertices, and replaces those runs with OCC `gp_Circ`-based edges
(`BRepBuilderAPI_MakeEdge` with a circle, not two points).

The resulting sub-prism's lateral wall is then a **cylindrical
surface** (`BRepPrimAPI_MakePrism` of a circular wire produces a
cylinder), not a polyline of planar quads.

### What goes wrong in the global BOP

The global cad_occ `BOPAlgo_Builder` then fragments this slab
against straight-line entities (wires, stacks, encapsulant,
cladding). Mixing **curve geometry against straight geometry**
through BOP is where the trouble starts:

1. **TShape sharing breaks at the slab/neighbour seam.** The
   cladding's polygon edge meeting the slab's cylindrical face
   produces a curved intersection edge whose TShape doesn't match
   the cladding's straight TShape on its own edge. The
   interface-tagging fallback then has to chase the resemblance via
   fuzzy AABB, which is brittle.

2. **Post-Path-A partition pieces are *mixed-vertex*.** A piece's
   exterior consists of: some original annulus arcs + some straight
   bridge cuts + some splitter polygon boundaries. Running
   `_make_occ_wire_from_vertices` with `identify_arcs=True` on a
   mixed-vertex exterior yields a wire that's part-arc, part-line.
   The classification is sensitive: a bridge endpoint might or
   might not "look cocircular" with adjacent original-annulus
   vertices, so adjacent sub-prisms can end up with *different*
   arc-vs-line classifications for the *same* partition seam. Now
   the TShape sharing across the seam fails — the two sides are
   geometrically coincident but topologically distinct (one's an
   arc, the other's a line).

3. **The 2026-05-13 cascade experiment exhibited this concretely.**
   When `identify_arcs=True` was propagated to resolved slabs, the
   bench produced cascading "Different number of points" failures
   *across the whole scene* (wires, stacks — not just struct_rings)
   and core-dumped. The arc-bearing struct_ring's BOP output
   rippled out and disturbed everything else's fragmentation.

### The current trade-off

By dropping `identify_arcs` at the resolved-slab stage, all
sub-prisms are built with straight-line wires only. The annulus
becomes a 48-segment polyline.

We lose: **mesh quality near circular features** — lateral surfaces
are facet-stepped, not smoothly curved.

We keep: **BOP robustness** — everything is planar, TShape sharing
works cleanly across the whole scene.

Note that `identify_arcs=True` *does* still affect the user-side
polygon: in the cad-stage shapely buffer pass, arc snapping affects
the vertex positions of the polygon footprint. So the resulting
polyline is at least *uniformly* sampled around the true circular
geometry — both the slab and its neighbours see the same vertex
set. That's why dropping arc identification at the phantom level
doesn't destroy the *correctness* of the mesh, only its smoothness.

### What a real fix would look like

To restore arc-faithful lateral surfaces without breaking BOP, we'd
need either:

1. **Pre-snap arcs across all entities** so the slab's arc and its
   neighbour's polyline intersect at well-defined points that both
   sides see identically (extends cad_occ's prepare-entities
   shapely pass with arc-aware co-tessellation).

2. **Build the slab's lateral cylinder *after* the global BOP**,
   replacing the polyline lateral facet with a true cylindrical
   patch that reuses the bottom/top arc edges (extends
   apply_structured_slabs with a post-fragment cylinder-restoration
   pass).

Either path requires plumbing arc-awareness through cad_occ AND the
xao writer. Out of scope for the current session; tracked alongside
R1 as future work.

---

## Cross-references

* Phase 1 (`crosses_z`) — commit `bdc1ce6`.
* Path A (bridge cuts) — commit `9d92083`.
* R1 mitigation (log demotion) — commit `80f42ba`.
* Diagnostic scripts — `scripts/inspect_*.py`,
  `scripts/diagnose_arc_periodic.py`.
* Full bench scene — `scripts/bench_structured.py`.
