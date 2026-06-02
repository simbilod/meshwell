# Transfinite hint dependencies — catalog

**Date:** 2026-06-01
**Branch:** `feat/structured_discrete`
**Status:** PROMOTED 2026-06-01 — Alt B is now the production path
  (see `meshwell/structured/wedge.py::freeze_lateral_mesh`).
  Catalog kept as historical reference and future-replacement spec.
**Sister doc:** [2026-06-01-cohort-topology-investigations.md](2026-06-01-cohort-topology-investigations.md)
**Promotion plan:** [docs/superpowers/plans/2026-06-01-alt-b-promotion.md](../plans/2026-06-01-alt-b-promotion.md)

## Why this document exists

The Q3 spike (`scripts/spike_manual_lateral.py`,
`meshwell/structured/wedge_manual_spike.py`) showed that we *can*
construct lateral-face quads in Python and run the complex stress
scene without any transfinite hints. Same 31 physical groups, valid
mesh. So a replacement is feasible.

But "valid on one scene" is a weak signal. Before we can promote the
manual lateral path to replace `apply_lateral_transfinite_hints`,
we need to know what *every consumer* of the transfinite outputs
silently depends on. A replacement that satisfies the obvious
contracts but misses an implicit one breaks differently — often only
under specific topologies — and we discover the gaps the hard way.

This document catalogs:
1. What `apply_lateral_transfinite_hints` *does* to the gmsh model.
2. What other stages *read* from that work.
3. The implicit assumptions wrapped in each call.
4. Scenarios where the assumptions might not hold.
5. The contract any replacement must satisfy.

Once we have this list, we can design the replacement against an
explicit checklist instead of "looks right on the scenes we tried."

---

## 1. What the current path produces

### Calls made by `apply_lateral_transfinite_hints`

Per lateral face of every keep=True cohort sub-solid:

```
gmsh.model.mesh.setTransfiniteCurve(vertical_edge_tag, n_layers + 1)
   for each of the 2 vertical edges
gmsh.model.mesh.setTransfiniteSurface(face_tag)
gmsh.model.mesh.setRecombine(2, face_tag)
```

### What gmsh then produces during `generate(2)`

- **Each vertical edge:** exactly `n_layers + 1` nodes, *uniformly
  spaced* along the edge from z=z_bot to z=z_top.
- **Each lateral face:** a structured (n × m)-node quad mesh, where
  n = nodes along the horizontal edges and m = `n_layers + 1`.
  Nodes are placed in the face's parametric coordinate system; for
  cylindrical surfaces this means uniform θ-step nodes at each layer.
- **Lateral face seam:** the cylinder's intrinsic seam (theta=0/2π
  line) is on the face's boundary as one of its 4 edges. Nodes on
  that seam are *shared* with the seam vertex topology — both
  sides of the periodic surface see them.
- **Recombine:** the (n × m) triangulation that the transfinite
  generator first produces is post-processed into (n-1)(m-1) quads.

### What the current path does *not* touch

- Horizontal faces (bot, top) — gmsh free-meshes them, producing
  triangulations.
- Adjacent unstructured volumes' boundary mesh.
- Interior-face mesh between two cohort sub-solids — handled by
  shared TShape (constructed once in EdgeRegistry).

---

## 2. Downstream consumers and what they assume

### Consumer A: `stamp_wedges._stamp_one` (pre_3d hook)

**Where:** `meshwell/structured/wedge.py:178-…`

**What it reads from the transfinite mesh:**

- Bot face triangulation via `gmsh.model.mesh.getElements(2, bot_tag)`
  and `getNodes(2, bot_tag, includeBoundary=True)`.
- Top face *boundary nodes* via `getNodes(2, top_tag, includeBoundary=True)` —
  these are the corner nodes placed by the lateral transfinite mesh.
- Intermediate-layer nodes via a global `gmsh.model.mesh.getNodes()`
  scan, filtered by z ≈ z_layer. These are the n_layers-1 rows of
  nodes that transfinite placed on the lateral face at intermediate
  z values.

**What it assumes:**

1. For every boundary node of the bot face, there's a node at the
   same (x, y) at z_layer for every intermediate layer 1..n_layers-1.
2. For every boundary node of the bot face, there's a node at the
   same (x, y) at z_top (placed via the top edge of the lateral
   face, recorded as a top-face boundary node).
3. The "every boundary node has a vertical counterpart at every
   layer" mapping is what makes the wedge-element emission well-
   defined: each bot triangle becomes `n_layers` wedges by mapping
   its 3 vertices into 3 vertical columns of nodes.
4. The intermediate nodes are *unique* per (x, y, z_layer) up to
   `point_tolerance` — duplicates would cause non-deterministic
   merge later.
5. The intermediate nodes are *findable* via a global model scan +
   z-band filter. If they don't exist, `_stamp_one` creates new ones
   in the volume entity (line 280-287).

**Assumption fragility:**

- If the lateral mesh places vertical nodes at non-uniform z (e.g.
  cosine-spaced for boundary layers), assumption #1 still holds —
  but the spacing must match what the user actually wants. Currently
  uniform `dz = (z_top - z_bot) / n_layers`.
- If the lateral mesh has DIFFERENT horizontal discretization on
  different lateral faces of the same sub-solid (e.g. via per-face
  transfinite curve density on the horizontal edges), the boundary
  nodes around the bot face perimeter wouldn't be aligned. This
  doesn't happen today because we don't set transfinite hints on
  *horizontal* edges of lateral faces — those edges are the bot/top
  arcs, which are shared with horizontal faces and meshed freely.

### Consumer B: `gmsh.model.mesh.generate(3)` for adjacent unstructured volumes

**Where:** runs after the pre_3d hook (`mesh.py`); fills the
non-cohort tet volumes.

**What it reads:**

- The 2D boundary mesh of every face it touches, including the
  lateral faces of cohorts and the shared horizontal faces between
  cohorts and base/cap.

**What it assumes:**

1. The 2D boundary mesh of any face it bounds is a valid 2D mesh —
   nodes and elements consistent.
2. The 2D mesh on faces shared with cohort lateral surfaces must
   provide a meshable triangle/quad boundary for the tet algorithm.
3. The shared horizontal face between cohort bot and unstructured
   neighbour top has a single triangulation used by both.

**Assumption fragility:**

- If a cohort lateral face has the same TShape as a face used as a
  boundary for an unstructured volume (does this happen? Probably
  not — laterals are interior to the cohort), the tet algorithm
  would need to see a triangulation that matches its expectations.
  Quads from recombine *should* be fine, but mixing tri-quad
  boundary on the same face can cause issues.
- The horizontal interface face mesh is shared by construction. Any
  manual replacement must not break this.

### Consumer C: orchestrator's `removeDuplicateNodes` calls

**Where:** `meshwell/orchestrator.py:213-214`
(`gmsh.model.mesh.removeDuplicateNodes(structured_vol_dimtags)`)
and `:225` (post-3D global dedup).

**What it assumes:**

- Nodes inside structured volumes are deduplicated only against each
  other (scope-limited). This relies on the assumption that within
  a structured volume, any pair of nodes at the same (x, y, z) up to
  `gmsh's` tolerance can be merged without breaking topology.
- The intermediate-layer nodes placed by `_stamp_one` and the
  lateral-face nodes placed by transfinite are *the same nodes*
  after dedup — they have to be, because they were placed at the
  same coordinates.

**Assumption fragility:**

- If the manual replacement places nodes with float-imprecision off
  from the transfinite-equivalent positions by more than the
  internal dedup tolerance, dedup won't fold them and we get
  duplicates that persist into the .msh output. This is exactly the
  scenario that motivated the scoped + post-3D dedup split.

### Consumer D: physical group assignment in `occ_xao_writer`

**Where:** writes physical groups to the XAO file.

**What it assumes:**

- Each face's gmsh tag is bound to the correct OCC face TShape.
- Lateral faces don't appear *as* user-named physical groups (they
  appear inside `A___B` interface groups via the AABB or TShape
  matching).
- Transfinite hints don't change which TShape is bound to which
  gmsh tag.

**Assumption fragility:**

- If the manual lateral construction adds new mesh entities (nodes,
  elements) but doesn't preserve the topological binding (face tag
  → TShape), then post-merge resolution of physical groups breaks.
- Currently `addElementsByType(face_tag, 3, [], quad_nodes)` and
  `addNodes(2, face_tag, ...)` preserve the binding — we add elements
  *to* an existing face tag, we don't create a new one.

### Consumer E: gmsh's internal periodic-surface mesher

**Where:** invoked transparently during `generate(2)` for cylindrical
faces.

**What it assumes (relative to transfinite):**

- A transfinite cylindrical face has its seam edge meshed once. The
  periodic-surface handler maps nodes on the seam to themselves
  (theta=0 ↔ theta=2π). Recombine + transfinite gives the mesher a
  topology it knows how to handle.

**Assumption fragility:**

- This is what bit us in the earlier segfault investigation: when
  the seam edge has TWO distinct TShape arcs at the same physical
  location (the EdgeRegistry direction-key bug), the periodic
  mesher recursively intersects the seam with itself. That bug is
  fixed; the assumption is that the seam edge is a *single* arc
  with a single set of nodes.
- The manual replacement doesn't *invoke* the periodic mesher — it
  clears the gmsh-meshed lateral face and writes elements by hand.
  So consumer E becomes irrelevant if we replace transfinite. But
  any *partial* replacement (e.g. keep transfinite for lateral
  faces, use manual elsewhere) would still rely on it.

### Consumer F: validators

**Where:** `apply_lateral_transfinite_hints` itself raises
`StructuredLateralNLayersMismatchError` and
`StructuredTransfiniteRejectedError`.

**What they enforce:**

- All cohort sub-solids sharing a lateral face must agree on n_layers.
- Lateral faces must have exactly 4 boundary edges.

**What a replacement still owes:**

- The same validations must run somewhere. They're already
  preserved in `wedge_manual_spike.manual_pre_2d_validate` for
  n_layers; the "4 boundary edges" check is currently woven into
  the transfinite path (`if len(edges) != 4: raise`). The manual
  spike skips this validation today — a replacement would need to
  re-add it.

---

## 3. Implicit topological + numerical assumptions

Beyond what specific consumers expect, the transfinite path encodes
assumptions about the cohort that we should make explicit:

### T1: Lateral faces are 4-edge quadrilaterals

The cohort builder produces 4-edge lateral faces by construction
(two arcs / lines at z_low, two arcs / lines at z_high — wait, no:
two arcs/lines + two vertical seams = 4). This holds for vertical
prisms but would break for:
- Tilted prisms (the lateral becomes a parallelogram with non-
  vertical seams — does the planner support this? No, current
  cohort definition forbids it)
- "Skirts" where the bot polygon differs from the top polygon
  (forbidden by SlabMeta: bot_face and top_face have the same XY)
- Closed-circle laterals built as a single periodic face (the
  EdgeRegistry currently splits these into half-cylinders explicitly,
  so the cohort produces *two* 4-edge laterals per circle)

**Implication for replacement:** if the cohort builder ever
introduces non-4-edge laterals (e.g. unified periodic cylinders), the
4-edge assumption baked into both transfinite and the manual spike
breaks.

### T2: Vertical edges are oriented z_low → z_high

The transfinite curve placement creates `n_layers + 1` nodes evenly
spaced from one endpoint to the other. The orientation determines
which endpoint corresponds to z_low. Today the code identifies
vertical edges by z-difference between endpoints rather than by
orientation, so this is robust.

**Implication for replacement:** the manual spike already does this
robustly (classifies by endpoint z), so no new fragility here.

### T3: Bot and top discretization match

The lateral face's `n` horizontal nodes (one row at z_bot, one at
z_top) must match the horizontal-face mesh at the same locations.
Because horizontal edges of the lateral face are *shared TShapes*
with the bot/top face boundary, gmsh meshes them once and both faces
see the same nodes. This holds by construction in the cohort.

**Implication for replacement:** any replacement must continue to
respect shared horizontal edges. Today the manual spike does this
(it walks `bot_edge` which is a TShape shared with the bot face's
boundary, and uses those nodes directly).

### T4: Uniform z spacing

`dz = (z_top - z_bot) / n_layers` is implicit. Non-uniform spacing
(boundary-layer-style) would require:
- Different vertical edge node placement
- Different intermediate-layer node lookup in stamp_wedges
- A way to specify the spacing schema per physical_name

This is a planned-future-feature concern, not a present-day
limitation, but worth noting: the replacement design should
not bake uniformity in deeper than necessary.

### T5: n_layers consistency across shared laterals

Already validated and preserved in the spike. Worth a test.

### T6: point_tolerance is large enough to merge but small enough to discriminate

The wedge-stamping path uses `point_tolerance` to match boundary
nodes between adjacent layers. If two distinct features in the
scene are closer than `point_tolerance`, false matches can fold
unrelated nodes together. Today we default to `1e-3` which is
order-of-mm for user-scale geometry.

**Implication for replacement:** the manual spike inherits this
tolerance and uses it for the same lookups; no new concern.

### T7: Lateral faces all have the same n_layers within a sub-solid

A sub-solid's n_layers is single-valued per physical_name. All its
lateral faces use this same value. This means a sub-solid can't
mix coarse and fine vertical resolution across different lateral
faces. Acceptable for the current scope; documented.

### T8: Voids have no lateral mesh

`keep=False` sub-solids are skipped in `apply_lateral_transfinite_hints`
(commit 4726ebc). Their lateral face TShapes still exist on the
neighbour kept sub-solid, which *does* set transfinite, so the
shared lateral is meshed correctly.

**Implication for replacement:** must skip voids in the manual
path too. The spike does this (`if not meta.keep: continue`).

---

## 4. Scenarios to test against any replacement

A replacement that passes the current 105 tests is necessary but not
sufficient. Beyond that, the following scenarios should be exercised
*before* declaring the transfinite hints obsolete:

### S1: Stacked cohort + arc-bearing lateral

The complex stress scene (test_stress_complex_scene) hits this. We
have evidence it works in both transfinite and manual paths.

### S2: Stacked cohort + unstructured neighbour with interior hole

The "A_square + B_circle + base-with-HOLE_BASE" scenario from the
earlier investigation. Currently in the complex stress scene.

### S3: n_layers = 1

The degenerate case: a single "layer" of wedges, top face = bot face
mapped directly. The intermediate-layer loop in
`construct_lateral_quads` doesn't execute (range(1, 1) is empty),
but the bot+top quads at faces=1 still need a single row of quads.
Need to verify the manual spike handles this.

### S4: n_layers ≥ 5

A moderately resolved structured volume. Verifies that intermediate-
node placement is correct at multiple layers.

### S5: Shared lateral between two cohort sub-solids

Two sub-solids in the same cohort sharing a lateral face (e.g.
side-by-side rectangles in the same XY z-interval). The lateral
quads must be emitted *once*, not twice. Today the manual spike
uses `lateral_info[face_tag] = (...)` which dedupes by face_tag.
The seam-edge node cache also dedupes. Worth an explicit test.

### S6: Cohort meeting another cohort at a z-plane

(stacked cohorts: e.g. A z=[0,1] meets B z=[1,2]). The cohort
builder fuses them into one cohort if XY-aligned at z=1, so they're
the same compound. If they're disjoint cohorts (no XY overlap at
z=1), they're two cohorts with no shared lateral. Both cases need
the manual replacement to work.

### S7: Cylindrical lateral with arc seam at theta=0

A full circle cohort. The cylindrical surface has a seam line at
theta=0 (where u=0=2π). The lateral face is *not* a 4-edge quad in
the topological sense — it's a periodic face with a seam. The
EdgeRegistry currently splits this into two half-cylinder faces,
each 4-edge. The replacement must handle this split correctly.

### S8: Annular lateral (interior ring)

A lateral face that's an interior ring of a polygon (a hole). Tested
in the complex stress scene via A_recth and ANNULUS_B. The
horizontal edges of this lateral are the interior ring of the bot
face. Sharing rules same as exterior.

### S9: Void carving a solid

A void's lateral face is shared with the surrounding solid's inner
ring. The void itself doesn't get transfinite (keep=False). The
solid does. The replacement must do the same: skip the void, emit
quads from the solid's side, share seam nodes.

### S10: Very fine bot face mesh (~100 nodes around perimeter)

If the lateral quad mesh inherits the bot face boundary
discretization (the manual spike does this), a very fine bot mesh
produces a very fine lateral mesh. Verifies no scaling problems
in the manual path's loop / addNodes calls.

### S11: Very coarse bot face mesh (~10 nodes around perimeter)

The other extreme: ensures the per-layer quad emission still works
when there are only 2 nodes per row.

### S12: characteristic_length variation

If `default_characteristic_length` is small enough that gmsh wants
to place INTERIOR nodes on the bot edge (it does — see the spike's
~2k wedges in complex scene), the lateral quad construction must
handle interior nodes correctly. Spike does; tested by the
complex scene.

---

## 5. Replacement contract checklist

Any replacement for `apply_lateral_transfinite_hints` must:

- [ ] Skip keep=False (void) sub-solids
- [ ] Validate n_layers consistency across owners of every shared lateral face
- [ ] Validate lateral face is a 4-edge quadrilateral (or document why not)
- [ ] Emit *exactly one* set of quad elements per face tag, even for shared laterals
- [ ] Place nodes at (x, y, z_bot + l × dz) for l in 1..n_layers-1
- [ ] Bind those intermediate nodes to the lateral face tag (so OCC↔gmsh tag binding survives)
- [ ] Bind seam-line intermediate nodes to the vertical edge tag (so adjacent laterals share them)
- [ ] Produce a 2D quad mesh on the lateral face that `stamp_wedges` can read consistently
- [ ] Produce a top-edge node set that `_stamp_one` can find via its existing-node lookup
- [ ] Preserve gmsh's adjacency mapping: face_tag boundary edges, vertex tags, etc.
- [ ] Not corrupt the 2D mesh on horizontal faces (bot, top) that gmsh just placed
- [ ] Not invalidate adjacent unstructured volumes' boundary topology
- [ ] Behave correctly under `removeDuplicateNodes` (no spurious dups within point_tolerance)
- [ ] Run before `stamp_wedges` (or be folded into it)
- [ ] Produce the same physical_group output as transfinite on every test scene

---

## 6. Open questions for the replacement design

- **Q1: Does the manual spike's node spacing matter for solver
  consumers?** Right now it's uniform dz. Likely fine for the
  general case; worth confirming with downstream solver users.

- **Q2: Should the replacement be lazy (run per face on demand) or
  eager (run all faces upfront)?** Today's transfinite is eager-set
  via gmsh API. The spike is eager-construct in pre_3d_hook. Trade-
  offs around memory and ordering.

- **Q3: How do we test "intermediate-layer node sharing across shared
  laterals" without relying on a global dedup?** The seam-edge
  cache in the spike handles two laterals sharing a vertical edge.
  But there's no test that specifically exercises it. We need one.

- **Q4: Where does the "lateral discretization should be controlled
  by user, not by bot face mesh" concern fit?** This is a real
  problem (the spike's wedge count varies with bot density). We
  probably want a `lateral_discretization` knob per physical_name,
  separate from n_layers. That's a feature, not a bug fix.

- **Q5: Does removing transfinite hints help or hurt performance
  for free-mesh-then-clear cycles?** The spike's pre_3d work
  includes `gmsh.model.mesh.clear([(2, face_tag)])` per face,
  which throws away whatever gmsh just did. Measurable cost
  on big scenes? Unknown. Worth instrumenting.

- **Q6: Could we keep transfinite for some faces and use manual for
  others?** E.g. transfinite-by-default, manual when transfinite
  rejects (5+ edges, non-4-edge topologies). Hybrid path might
  have its own bugs but covers more topologies.

- **Q7: What's the smallest scene that would let us assert "the
  manual path produces the same wedge / quad count as transfinite
  on a uniform-meshed scene"?** Would help quickly diagnose
  regressions.

---

## 6b. Alternative architectures we should weigh against

Two directions worth thinking through before committing to a
manual-quad-construction replacement.

### Alt A — Built-in gmsh extrusion APIs

gmsh exposes prism extrusion at the geometry level:

- `gmsh.model.geo.extrude(dimTags, dx, dy, dz, numElements=[...],
  heights=[...], recombine=True)` — geo kernel
- `gmsh.model.occ.extrude(dimTags, dx, dy, dz, numElements=[...],
  heights=[...], recombine=True)` — occ kernel

Both take a 2D face, extrude it by (dx, dy, dz), and produce a
structured prism mesh when `recombine=True`. This is *exactly* what
we want logically.

**Why it doesn't drop in cleanly:**

1. **Geo/OCC kernels don't mix.** Our pipeline is all OCC: the
   cohort compound is an OCC `TopoDS_Compound` imported into gmsh
   via XAO. We can't combine occ-imported geometry with geo-extruded
   geometry in the same model. So `geo.extrude` is out unless we
   abandon the OCC kernel entirely (loses BOP, EdgeRegistry, etc.).
2. **`occ.extrude` creates *new* geometry.** The cohort sub-solids
   already exist in our imported compound as TopoDS_Solid entities.
   Calling `occ.extrude` on a bot face wouldn't *mesh* the existing
   solid — it would create a *new* prism solid coincident with the
   existing one. Topology mismatch with our compound; downstream
   physical-group assignment breaks.
3. **Vertex-exact requirement.** Even if we could splice occ.extrude
   into our pipeline, the bot and top faces in our cohort are
   *already* meshed (or about to be) by gmsh's free 2D mesher.
   extrude assumes the bot face mesh is what gets swept up — so we'd
   have to ensure the bot face is meshed *first* and that the mesh
   nodes are positioned exactly where extrude expects (no
   re-meshing once extruded).

**What we'd gain if we made it work:** the extrude path is what gmsh
itself supports natively. We'd get fewer custom code paths, fewer
chances for our wedge stamping to disagree with gmsh's expectations.

**What we'd give up:** geometry construction control. OCC's
EdgeRegistry, arc unification, void-as-keep=False, multi-cohort
stacking — all of it lives upstream of gmsh, where we have full
control. Switching to gmsh-side extrude pushes that control into
gmsh's geometry kernel, where we have less.

**Tentative verdict:** not viable for the cohort body in our
architecture. The constraint of "must reuse existing OCC compound
topology" eliminates both extrude APIs. The manual-quad approach
remains the better fit because it works *with* our existing topology
rather than replacing it.

### Alt B — Freeze the cohort mesh first, conform tets around it

Both transfinite and the manual spike interleave with gmsh's normal
meshing pipeline:

```
generate(1)  ← edges
[pre_2d_hook: transfinite set OR no-op]
generate(2)  ← 2D faces (cohort laterals + horizontals + neighbours)
[pre_3d_hook: stamp_wedges, possibly construct_lateral_quads first]
generate(3)  ← tet volumes
```

The user's preferred shape:

```
[pre_2d_hook: do nothing on cohort lateral faces]
generate(1)
generate(2) for HORIZONTAL faces only (we control what gets meshed)
[pre_3d_hook BEFORE generate(2) on cohort laterals:]
  - construct lateral quads explicitly from bot face mesh
  - emit wedge elements directly
  - this "freezes" the cohort mesh
generate(2) for cohort lateral faces (skip — frozen)
generate(2) for neighbour faces (gmsh meshes them, conforming to
  the cohort's boundary)
generate(3) for unstructured volumes (gmsh tets them, conforming to
  the cohort's lateral quad boundary)
```

The key insight: **the cohort is small and fully-known; the
unstructured neighbours are large and gmsh-meshed. The natural order
is "fix the small known thing first, fit the large unknown thing
around it."** Currently we do the opposite — let gmsh free-mesh
*everything* in 2D and then stitch wedges into the cohort.

**Why this is cleaner:**

- The cohort's 2D mesh (lateral quads + horizontal tris) becomes a
  hard constraint for the tet mesher. Conformity is automatic.
- No periodic-surface concerns: by the time gmsh's 2D mesher *would*
  encounter a cohort cylinder, the lateral mesh is already there as
  fixed elements. gmsh's periodic-surface logic (and its known
  failure modes around seam topology) doesn't run on cohort faces.
- No "free-mesh then clear" cycle: today the manual spike lets gmsh
  mesh lateral faces, then clears them and rebuilds. Wasted work,
  and a race where the cleared mesh's nodes might still be
  referenced.
- Wedge stamping integrates naturally: lateral quads + wedges are
  emitted together, in one pre-2D pass, before gmsh's free mesher
  is loose.

**What we'd need to figure out:**

1. **gmsh hook ordering.** `pre_2d_hook` currently fires before
   `generate(2)`. We'd need a way to perform 2D mesh emission
   *during* the pre_2d phase such that subsequent `generate(2)`
   does not re-mesh the cohort laterals. Likely via
   `gmsh.model.mesh.setMeshSize`, mesh-already-present detection,
   or explicit per-face mesh algorithm settings.
2. **Cohort bot face mesh availability.** If we want to build
   lateral quads from the bot face mesh, the bot face must be
   meshed first. So bot face 2D meshing happens — then lateral
   construction — then everything else. This is a sub-ordering
   within pre_2d.
3. **Periodic surface avoidance.** Even if we explicitly skip cohort
   laterals in `generate(2)`, gmsh might still try to "use" them as
   periodic surfaces during its 1D periodic edge processing. Worth
   investigating whether `setTransfinite*(face, "Quadrangle")` or
   similar marker tells gmsh to leave the face alone.
4. **Tet conformity verification.** A tet boundary must precisely
   match the cohort's lateral quads. Mixed tri/quad boundary is
   supported by tet meshers in general, but worth confirming for
   gmsh's specific algorithms (`Mesh.Algorithm3D = 1` (Delaunay),
   `7` (HXT), `9` (R-tree), etc.) and that conformity holds at the
   interface.

**Tentative verdict:** this is the architecturally cleanest answer
and resolves several pain points at once (periodic surfaces, free-
mesh-then-clear waste, ordering ambiguity). The path involves
working out exactly how to disable gmsh 2D meshing of specific
faces, which is the unknown.

The manual-quad spike is a step in this direction (we already emit
the lateral quads ourselves), but it still pays the cost of letting
gmsh mesh-then-clear those faces. The truly clean version skips
gmsh's 2D mesher entirely for cohort lateral faces — and that's
worth a separate spike before promoting the manual path.

### Periodic-boundary concerns specifically

Cylindrical lateral faces in OCC are intrinsically periodic in θ.
The cohort builder splits closed circles into two half-cylinder
faces (each with 4 edges, no periodic seam at the face level),
which sidesteps gmsh's periodic-surface handling for the *face*.
But two failure modes remain:

- **Vertical seam edge duplication.** Two adjacent lateral faces
  share a vertical seam (e.g. half-circle 1 ends at theta=0, half-
  circle 2 starts at theta=0). If their seams are co-linear but
  separate edges (different TShape IDs), gmsh treats them as
  distinct and tries to mesh them twice — the segfault we hit in
  the earlier investigation. **Mitigated** by the EdgeRegistry
  direction-invariant arc cache (commit 4ae762f). Worth confirming
  the manual lateral path doesn't reintroduce it.
- **Cohort top/bot face periodic-vertex collision.** A full circle
  bot face has a single vertex per half-circle endpoint (theta=0
  vertex and theta=π vertex). Both half-cylinder lateral faces
  touch the bot face's theta=0 vertex. The bot face's *parametric*
  representation in OCC may also be periodic (if it's a face on
  Geom_PlaneSurface trimmed to a circular wire, no periodic; if
  it's a face on a disc-like surface, possibly). Worth a small
  test that a full-circle cohort's bot/top faces don't bring their
  own periodic concerns.

If we adopt Alt B (freeze cohort mesh first), the periodic surface
mesher never runs on cohort laterals at all — periodic concerns
become irrelevant for cohort topology. They'd only matter at the
*neighbour* side, where gmsh's mesher still runs.

---

## 7. Recommended next steps

1. **Write the missing tests** (S3, S5, S9 specifically) so the
   replacement has a complete safety net.
2. **Instrument timings** (Q5, also a sister-doc action) so we know
   if the manual path is faster, slower, or same as transfinite.
3. **Decide on the discretization knob** (Q4) — even a "follow the
   bot face" default is fine if explicit.
4. **Spike Alt B: freeze-cohort-first ordering.** Before promoting
   the current manual-spike module, prototype the "skip gmsh's 2D
   mesh on cohort laterals entirely" path. Confirm it works on the
   complex stress scene and avoids the free-mesh-then-clear waste.
5. **Promote whichever variant won §4** (current spike or Alt B
   spike) to a `wedge_manual.py` module, with the validators added
   and dependencies addressed per the contract in §5.
6. **Run the replacement against the test suite and demos.** If
   green, switch the orchestrator default to the new path. Keep
   the old transfinite path under a flag for one release as a
   safety hatch.
7. **Then remove the transfinite path** when no one's complaining.

---

## 8. Status after promotion (2026-06-01)

- ✅ Step 1 (missing tests): S3 added as
  `test_n_layers_1_meshes_cleanly`; S5 added as
  `test_shared_lateral_between_two_subsolids` in
  `tests/structured/test_wedge_pre2d.py`. S9 (void / keep=False) is
  covered by the existing `tests/structured/test_void_tagging_e2e.py`
  suite, which now exercises Alt B post-orchestrator switch.
- ⏸ Step 2 (timings instrumentation): deferred. Spike showed Alt B
  is ~15% slower than the old transfinite path on the complex scene
  (4.39s vs 4.15s). Acceptable for determinism trade. Re-visit if
  perf becomes a bottleneck.
- ⏸ Step 3 (discretization knob, Q4): deferred. Alt B currently
  inherits lateral discretization from bot-edge mesh density. Add a
  `lateral_discretization` parameter when a user requests it.
- ✅ Step 4 (Alt B spike): completed; see prior commits.
- ✅ Steps 5-7 (promotion, suite/demos green, transfinite path
  removed): done in
  [docs/superpowers/plans/2026-06-01-alt-b-promotion.md](../plans/2026-06-01-alt-b-promotion.md).
