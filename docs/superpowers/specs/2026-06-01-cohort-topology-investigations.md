# Cohort topology investigations

**Date:** 2026-06-01
**Branch:** `feat/structured_discrete`
**Status:** investigation / planning — not yet a design

Two open questions about whether we can replace post-hoc reconstruction
with construction-time knowledge. Both stem from the same observation:
the planner already builds the cohort with full topological knowledge
(shared TShapes via VertexRegistry / EdgeRegistry), but downstream
stages re-derive what we already knew.

---

## Investigation 1: AABB fallback in interface detection

### What's happening today

`occ_xao_writer._compute_physical_groups` pairs every two top-dim
entities and computes their shared boundary faces — these become the
`A___B` interface group. The pairing has two paths:

1. **TShape identity (fast):** intersect the TShape-ID sets of each
   entity's boundary faces. Common IDs = shared interface.
2. **AABB fallback (slow):** when (1) returns empty but both entities
   have boundary faces, compute the per-face axis-aligned bounding
   box on each side and pair faces whose AABBs match within a
   per-corner L_inf tolerance of 1e-2.

The fallback is in `occ_xao_writer.py:237-251`. The complexity is
roughly **O(N² × F²)** with bbox queries inside:

- N² entity pairs
- For each pair: F faces on each side, full O(F_A × F_B) scan
- Each pair compares a fresh `_shape_aabb(face)` call, which walks
  the face's BRep to compute the box

For a scene with ~10 entities and ~30 faces each, that's already
~90 000 comparisons per pair × 45 pairs = ~4 M, and each comparison
allocates a `Bnd_Box` and probes the BRep.

### Why the fallback exists

1. **Interfaces don't exist before BOP.** Each entity arrives as an
   independent OCC shape. Volume A's face and Volume B's face are
   different objects even when they coincide geometrically. The
   "interface between A and B" only becomes a single TShape after
   `BOPAlgo_Builder.Perform()` fragments and re-identifies coincident
   geometry.
2. **BOP's identity tracking is imperfect.** After Perform, we query
   `Modified/Generated/IsDeleted` to map each input face → its
   output image. Usually A's input face and B's input face both
   map to the *same* output TShape and the fast path wins. But
   precision-near-miss or certain fragmenting patterns occasionally
   produce two coincident faces with *different* TShapes — and the
   fast path misses.
3. **We can't easily fix BOP.** Only its output. The AABB fallback
   is a robustness gap-closer.

### Why "come in knowing exactly" isn't trivial

Structured cohorts *already* come in knowing — the cohort compound
is pre-assembled with shared TShapes via `EdgeRegistry`, so
cohort-internal interfaces ALWAYS match via identity. The AABB
fallback fires for the **cohort↔unstructured** boundary, where:

- Each side is a separate OCC argument to `BOPAlgo_Builder`
- BOP is supposed to merge them but sometimes produces two
  coincident faces with different TShapes
- The fast path misses; AABB picks them up

So the structurally clean answer is: don't go through BOP for those
boundaries. Pre-assemble the cohort↔unstructured interface faces
the same way we pre-assemble cohort-internal ones. But that requires
deeper rework.

### Options under consideration

**Option A — gmsh-side adjacency lookup (preferred to investigate first).**
Skip computing interfaces in Python entirely. Write the XAO with
only volume-level (entity → user name) mappings, then after
`gmsh.merge(xao)`:

1. Resolve each user entity to its gmsh volume tag(s).
2. For each pair (i, j) of volume tags, call
   `gmsh.model.getAdjacencies(3, i)` and `(3, j)` to get the bounding
   face tags. The intersection is the interface in *gmsh space* —
   no OCC TShape comparison or AABB needed.
3. Call `gmsh.model.addPhysicalGroup(2, face_tags, name=f"{a}___{b}")`.

gmsh's internal post-merge topology already has the adjacency graph
we'd otherwise rebuild in Python. No bbox arithmetic. No TShape
identity issues — gmsh has unified everything by the time it returns.

Caveats:
- We still need the volume→user-name mapping to survive `gmsh.merge`.
  Today it survives via XAO physical groups; the synthetic-name lookup
  bridge handles structured sub-solids. That part stays.
- For the bookkeeping cases (purely-synthetic annotators, keep=False
  helpers), the equivalent rules apply: exclude their volume tags
  from interface computation; use their face TShapes for
  `neighbour___helper` naming via gmsh adjacency on the helper's
  faces.

**Option B — spatial index for the current AABB scan.**
R-tree-index each entity's face AABBs. O(F log F) intersection queries
instead of O(F²). Doesn't change the architecture; 10–100× speedup
without changing the protocol. Cheap to implement.

**Option C — eliminate the fallback by tightening BOP.**
Set `BOPAlgo_Builder.SetGlue(BOPAlgo_GlueShift or GlueFull)` so that
coincident shapes are guaranteed to share TShapes after Perform.
Pair with adaptive `fragment_fuzzy_value` if needed. If we can prove
the fast path never misses, the fallback becomes dead code.

### Recommended order

1. (Prereq) Add timing instrumentation to the writer so we can quantify
   the current cost on a real scene. Without measurements we don't
   know if this is the bottleneck.
2. Prototype Option A on one moderate scene. Compare wall time and
   correctness (groups produced must match the current writer's
   output exactly).
3. If Option A wins on correctness *and* timing, plan a migration.
   If it doesn't (e.g. adjacency loses face-level granularity we
   currently need), fall back to Option B for a tactical speedup
   and Option C for the structural fix.

### Status — 2026-06-01

- ❌ **Option A** (gmsh-side adjacency lookup): spike showed it
  catches only 8/14 kept-vs-kept interfaces. Misses every interface
  where BOP produced coincident-but-separate TShapes — precisely
  what the AABB fallback exists to rescue. Not a viable replacement.
  See spike commit 04a9ceb.
- ❌ **Option C** (`SetGlue` to force TShape sharing): spike showed
  Glue mode does not reduce AABB fallback need; invocation count
  actually grows under GlueShift/GlueFull. Output is identical
  across all three Glue modes. Not a fix. See spike commit 04a9ceb.
- ✅ **Option B** (numpy-vectorized inner loop + per-entity AABB
  cache): promoted. ~2.3x speedup on the complex stress scene
  (131.8ms → 58.3ms). See implementation plan
  [docs/superpowers/plans/2026-06-01-aabb-fallback-speedup.md](../plans/2026-06-01-aabb-fallback-speedup.md).

### Update — 2026-06-02 — shared EdgeRegistry refactor

- ✅ **Cohort↔neighbour shared registry**: shipped. Each cohort's
  `EdgeRegistry` is exposed via `StructuredState.cohort_registries`.
  Pre-cut unstructured entities are tagged in `decompose_cohorts`
  with their adjacent cohorts. `PolyPrism.instanciate_occ` routes
  the shared boundary wire through the cohort's registry when the
  tag is present, building the polygon face at the z-plane shared
  with the cohort and extruding in the appropriate direction so the
  user-built face IS the shared face. Result: cohort↔neighbour
  arc/line edges share `TopoDS_Edge` TShapes **by construction**;
  BOP fuzzy detection is no longer load-bearing for edges. On the
  focused arc + base test, AABB rescue count drops from "needed at
  least one" to **zero**. See implementation plan
  [docs/superpowers/plans/2026-06-02-shared-edge-registry.md](../plans/2026-06-02-shared-edge-registry.md).
- ⏸ **Face-level sharing (Sketch B)**: deferred. Face TShape
  matching at structured↔unstructured horizontal interfaces still
  goes through BOP + AABB rescue for the residual cases. If needed,
  build the interface face once in the cohort and reuse the
  `TopoDS_Face` in the neighbour's polyprism construction.

---

## Investigation 2: Why transfinite hints are needed

### What's happening today

The structured pipeline produces wedge (prism) elements in cohort
sub-solids via this dance:

1. **Planner** builds the cohort with shared TShapes — full
   topological knowledge of every sub-piece and its lateral faces.
2. **`apply_lateral_transfinite_hints`** (pre-2D-mesh hook) marks
   each lateral face's vertical edges as transfinite with `n_layers`
   nodes, and the face itself as a transfinite surface.
3. **gmsh meshes 2D** — lateral faces come out as structured quads;
   horizontal faces (bot/top) come out as free triangulation
   matching the surrounding unstructured tet mesh.
4. **`stamp_wedges`** (pre-3D-mesh hook) reads each sub-solid's bot
   triangulation, snaps intermediate-layer nodes to existing lateral
   nodes, and emits explicit wedge elements layer by layer.
5. **gmsh meshes 3D** — unstructured regions fill with tets;
   structured regions are already wedged.

The transfinite hints are the bridge: gmsh has no concept of "this
volume is a prism extrusion." We have to coerce its 2D mesher into
producing the right lateral-face topology so that the wedge stamp can
reuse it.

### Why this feels indirect

We already know:
- Every cohort sub-solid is a vertical extrusion of a polygon.
- The bot and top polygons are identical.
- The lateral surfaces are ruled cylinders (or planes) connecting them.
- `n_layers` and the bot triangulation completely determine the wedge
  mesh.

We don't need gmsh to *figure out* the structured topology — we built
it. We're using transfinite hints because gmsh's API for "mesh this
volume as a prism extrusion" doesn't quite exist in the form we want
once the volume is part of a heterogeneous compound. So we lean on
the transfinite-quad-side trick + manual wedge stamping.

### Open questions

**Q1: Is gmsh's `Extrude` API a cleaner fit?**

gmsh has `gmsh.model.geo.extrude(...)` and `gmsh.model.occ.extrude(...)`
which take a bottom surface, a translation/rotation vector, and a
number of layers — and they produce a structured prism mesh
automatically. They're used in boundary-layer meshing.

Why don't we use them?

- They build geometry, not just mesh. We'd have to build the cohort
  via gmsh-extrude rather than OCC compound — losing our OCC-level
  control (EdgeRegistry, arc detection, etc.).
- They're not friendly to "this prism shares faces with N other prisms
  pre-assembled in a compound."

But: maybe we can extrude the bot face *after* loading the cohort and
having gmsh recognise the bot as an existing 2D entity? Worth testing.

**Q2: Can we skip gmsh's 3D mesher entirely for cohorts?**

After 2D meshing produces:
- bot triangulation
- top triangulation
- lateral quad strips

…the wedge connectivity is fully determined. We could:

1. Read the 2D mesh from gmsh
2. Build wedge elements in Python directly (we already do this!
   `stamp_wedges` is exactly this)
3. Add wedges to gmsh via `gmsh.model.mesh.addElements`
4. Tell gmsh to skip 3D meshing for these volumes (e.g. set a flag,
   or pass them through a custom dim=3 generator that's a no-op for
   structured volumes)

We already do steps 1-3. The "skip gmsh's 3D mesher for these volumes"
is implicit because by the time gmsh.generate(3) runs, the volumes are
already filled with wedges — and gmsh respects existing elements.

So this is already what we're doing in spirit. The transfinite hints
are just the price of getting 2D quads on laterals.

**Q3: Why not build the lateral face mesh ourselves and skip transfinite?**

If we control the bot face triangulation (we don't today — gmsh does
it), then we can deterministically generate every node and every
element of every lateral face. Then we tell gmsh: "use this 2D mesh
for these faces; mesh the rest freely." No transfinite hints.

This requires:
- Either pre-mesh the bot face in Python (loses gmsh's good free
  meshing) OR mesh the bot first with gmsh, then take its output and
  use it to fully construct the lateral and top meshes
- A way to inject mesh elements into gmsh as the "official" mesh for
  certain entities, bypassing gmsh's mesher

Step 2 is supported by `gmsh.model.mesh.addElements` + careful tag
management. We use it for wedges already. Extending to lateral quads
would be straightforward.

**Q4: Is the issue gmsh's tet mesher seeing structured wedges and
getting confused?**

When gmsh meshes the unstructured tet volumes adjacent to a cohort,
it must respect the wedge surface mesh as a boundary. If the lateral
faces aren't transfinite, gmsh might tet-mesh them with free
triangles that don't match the wedge's tri-strip. That's the
non-trivial conformity constraint.

Transfinite ensures the lateral face has a *regular* quad-strip mesh,
which our wedge stamper can mirror. If we instead generated the
lateral mesh ourselves directly, we'd still need to ensure conformity
with the bot/top triangulation (achievable) and with the surrounding
tet mesher's expectations (also achievable since the tet mesher only
sees the boundary mesh, not the volume mesh).

### Recommended investigation order

1. **Document the *purpose* of each transfinite hint.**
   Why do we mark vertical edges transfinite? Why mark lateral faces
   transfinite? What breaks if we remove each one separately? This
   tells us which hints are load-bearing and which are belt-and-
   suspenders.
2. **Try removing lateral-face transfinite, keeping vertical-edge
   transfinite.** Does gmsh still produce a meshable cohort? What does
   the lateral mesh look like?
3. **Prototype: bot-meshes-first, then construct lateral quads in
   Python.** Build a minimal scene (single cylinder cohort + one
   unstructured neighbour). Mesh the bot in gmsh, freeze it, generate
   lateral quads via Python, addElements, then run gmsh's 3D mesher.
   Does it produce a conformal mesh?
4. **Compare wedge-count and quality** between current transfinite
   path and the construction-driven path.

### Tangential observation

The transfinite hints are an *example* of the same general pattern as
the AABB fallback: we know the topology by construction but
re-communicate it to a downstream stage that re-derives it. Both
investigations are about closing that loop — the structured pipeline
should be able to pass concrete knowledge forward, not re-derive
through indirect mechanisms.

---

## Both investigations: shared principles

| Principle | AABB fallback | Transfinite hints |
|---|---|---|
| What we know at construction | Which faces are interfaces | The full prism topology |
| What we're forcing the downstream to do | Re-find interfaces by geometry | Re-derive structured mesh from quad hints |
| Faster / cleaner alternative | Pass interface info via gmsh adjacency | Construct lateral and wedge mesh directly |
| Risk of the direct approach | Need to track volume tags across merge | Need to manage gmsh element tags carefully |
| Estimated effort | Low for spatial index; medium for adjacency | Medium for lateral construction |
| Estimated payoff | Linear → log on F; protocol simplification | Eliminates transfinite + extends to richer topologies |

Both should be revisited together. The same instinct ("stop re-deriving
what we already know") points to a broader simplification of the
boundary between meshwell and gmsh.

---

## Next concrete actions

1. **Timing instrumentation.** Add `perf_counter` blocks around
   `_compute_physical_groups`, around the AABB fallback specifically,
   around `apply_lateral_transfinite_hints`, and around
   `stamp_wedges`. Get numbers on the largest scene we have
   (probably the complex stress test) and `demo_curves.py`.
2. **Catalog transfinite hint dependencies.** Read every consumer of
   the hints in the pipeline; document what would break if each were
   removed. This produces a dependency map we can prune.
3. **Spike Option A (gmsh adjacency lookup).** One scene, side-by-side
   with the current writer's output. Compare groups, timings,
   correctness.
4. **Spike Q3 (Python-constructed lateral mesh).** Tiny cohort, manual
   lateral-quad generation, see if it passes gmsh's conformity checks.

Each spike is bounded — half a day each. After we have data, we can
decide whether either becomes a real design and implementation plan.
