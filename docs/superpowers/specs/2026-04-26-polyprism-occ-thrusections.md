# Tapered PolyPrism via `BRepOffsetAPI_ThruSections` (OCC backend)

**Status:** approved spec, ready for implementation plan
**Date:** 2026-04-26
**Scope:** OCC backend tapered-prism construction. gmsh path unchanged.

## Problem

`PolyPrism(buffers={...})` with any non-zero buffer takes the
`extrude=False` path: a tapered (z-varying-cross-section) prism. Today's
two backends construct the OCC solid very differently:

- **gmsh** uses `gmsh.model.occ.addThruSections(curve_loops,
  makeSolid=True, makeRuled=True)` — one API call lofts a continuous
  solid through a list of per-z wires. Bottom + top + lateral surface
  emerge from the algorithm. Mesh-clean output.

- **OCC** (current `_create_occ_volume` in `meshwell/polyprism.py:364-459`)
  manually builds bottom face, top face, and per-edge lateral quads,
  stitches them with `BRep_Builder.MakeShell`, wraps in
  `BRepBuilderAPI_MakeSolid`. The corner topology where adjacent quads
  meet isn't tolerance-clean enough — gmsh's PLC mesher rejects the
  resulting BREP with `Two facets intersect at point`.

Concrete failure: `tests/test_multiple_physicals.py::test_multiple_physicals`
(the only remaining failure on `feat/simple_fragments`).

## Solution

OCC has the exact equivalent of `addThruSections`:
`BRepOffsetAPI_ThruSections`. Replace the manual stitching with a loft.
Bottom + top + lateral surface come from one OCC algorithm tested for
this case, eliminating the per-edge facet brittleness.

## Code changes

### `meshwell/polyprism.py`

Replace `_create_occ_volume` (lines 364-459) and
`_create_occ_volume_with_holes` (lines 539-560) with new
`BRepOffsetAPI_ThruSections`-based implementations.

**New `_create_occ_volume`** (signature unchanged for callers):

```python
def _create_occ_volume(
    self,
    entry: list[tuple[float, Polygon]],
    exterior: bool = True,
    interior_index: int = 0,
) -> TopoDS_Shape:
    """Loft a tapered solid through the per-z wires of ``entry``.

    Mirrors gmsh's ``addThruSections(makeSolid=True, makeRuled=True)``.
    Each layer contributes one wire (built with the existing
    ``_make_occ_wire_from_vertices`` helper so arcs are preserved).
    ``BRepOffsetAPI_ThruSections`` builds bottom + top + lateral surface
    as a single closed solid -- no manual face stitching, no shell
    sealing, no MakeSolid wrapping.
    """
    from OCP.BRepOffsetAPI import BRepOffsetAPI_ThruSections

    loft = BRepOffsetAPI_ThruSections(True, True)  # isSolid, isRuled
    for z, polygon in entry:
        vertices = self.xy_surface_vertices(
            polygon=polygon,
            polygon_z=z,
            exterior=exterior,
            interior_index=interior_index,
        )
        wire = self._make_occ_wire_from_vertices(
            vertices,
            identify_arcs=self.identify_arcs,
            min_arc_points=self.min_arc_points,
            arc_tolerance=self.arc_tolerance,
        )
        loft.AddWire(wire)
    loft.Build()
    return loft.Shape()
```

**`_create_occ_volume_with_holes`** stays semantically identical (build
exterior, build each interior, cut interiors from exterior); only the
volume builder it calls changes. No code change needed since it already
calls `_create_occ_volume` and `BRepAlgoAPI_Cut`.

### Vertex-count invariant

`BRepOffsetAPI_ThruSections` requires all wires to have the same vertex
count. The existing `_validate_polygon_buffers` method at line 695
already checks this (per-z exterior coord count + per-interior coord
count). Wire validity is a precondition for the loft.

For this work: assume buffers are uniform (same vertex count per layer
— true for all current test scenes including the failing one). Do NOT
add a fallback path or call `_validate_polygon_buffers` proactively;
keep the change tight. If a future user hits a non-uniform-vertex-count
scene, OCC will raise during `Build()` with a recognizable error and we
can add the validation path then.

## Test impact

### Expected to start passing

- `tests/test_multiple_physicals.py::test_multiple_physicals` — currently
  the sole remaining failure. The tapered nested-prism scene becomes
  mesh-clean once OCC stops producing self-intersecting facets.

### Must remain unchanged

- All `tests/test_cad_gmsh.py` (gmsh path untouched).
- All `tests/test_cad_occ.py` and `tests/test_cad_occ_fragment_ownership.py`
  scenes that use `buffers={0: 0, 1: 0}` (extrude=True path,
  unchanged).
- `tests/test_backend_cross_compare.py` — its 3D scenes use extrude=True
  prisms, so no impact.
- `tests/test_buffers_prism.py` — exercises tapered prisms. Should stay
  green or improve.

### New cross-compare test (lightweight)

Add to `tests/test_backend_cross_compare.py`:

```python
def test_tapered_prism_match(tmp_path):
    """Tapered prism (sidewall buffer) must produce equivalent meshes
    on both backends. This is the case the OCC ThruSections port
    addresses; without it, OCC fails to mesh."""
    def make():
        polygon = shapely.Polygon([(-5, -5), (5, -5), (5, 5), (-5, 5)])
        return [
            PolyPrism(
                polygons=polygon,
                buffers={0.0: 0.0, 1.0: -0.1},
                physical_name="tapered",
                mesh_order=1,
            ),
        ]

    m_gmsh, m_occ = _run_both(make, tmp_path)
    s_gmsh = _mesh_summary(m_gmsh)
    s_occ = _mesh_summary(m_occ)
    _assert_summaries_equivalent(s_gmsh, s_occ)
```

## Out of scope

- Non-monotonic buffer profiles (`{0: 0, 0.5: -0.05, 1: +0.05}`).
  Solvable by per-slab lofts; deferred until needed.
- Refactoring the gmsh `_create_volume_directly` path. It works.
- Removing the now-unused helpers in `_create_occ_volume`'s old
  implementation (manual face stitching). The replacement deletes them.
