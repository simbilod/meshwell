# gdswell → meshwell integration notebook

**Date:** 2026-06-15
**Status:** Approved design, pending implementation plan
**Deliverable:** `meshwell/docs/gdswell_interface.py` (new, executed doc notebook)

## Goal

Add a docs notebook to meshwell that demonstrates **gdswell as the layout +
stackup front-end** and **meshwell as the 3D/2D meshing backend**. This is the
"downstream backend (e.g. meshwell)" that gdswell notebook `15-stackup.py`
explicitly defers painter's-algorithm cutting to. The notebook walks one
recognizable silicon-photonic example all the way from a 2D layout to a
watertight 3D mesh and a 2D cross-section mesh.

## Background / current state

- **gdswell** produces, from `stack.resolve(cell)`, a `ResolvedStackup` whose
  `prisms` are `ResolvedPrism(name, z_to_region: dict[float, kdb.Region],
  mesh_order, keep, cut_by)`. Coordinates are integer dbu; `ResolvedStackup.dbu`
  is µm-per-dbu. Painter's semantics: later entries (`j > mesh_order`) cut
  earlier ones where their bboxes overlap; `cut_by` is forward-only.
  `stack.resolve_cutline(cell, cutline)` gives the analogous 2D
  `ResolvedStackup2D` of `ResolvedPolygon2D(name, region, mesh_order, keep,
  cut_by)` in the `(s, z) → (x, y)` convention.
- **meshwell** consumes `PolyPrism(polygons=<shapely>, buffers={z: offset},
  physical_name, mesh_order, structured=…)` and `PolySurface(polygons=<shapely>,
  physical_name, mesh_order)`, then `generate_mesh(entities=[…], dim=2|3, …)`.
  Convention: **lowest `mesh_order` wins** on overlap (`cad_occ` fragment +
  ownership). Structured wedge meshing requires a pure extrusion (all-zero
  buffers) plus a `StructuredExtrusionResolutionSpec(n_layers=N)` per physical
  name in `resolution_specs`.
- The existing `gdsfactory_interface.py` page is the old (meshwell 1.0.7,
  non-executed, markdown-only) integration. The new page supersedes the
  "modern front-end" story; the gdsfactory page is left untouched.

## Key insight (the reason this integration is clean)

meshwell's native `mesh_order` ownership **reproduces gdswell's forward-only
painter `cut_by` for free** if the order is inverted. A gdswell prism at index
`i` is cut by all later prisms `j > i`; in meshwell, the prism that *keeps* an
overlap region is the one with the lowest `mesh_order`. Mapping

```
meshwell_mesh_order = N − gdswell_prism.mesh_order      # N = len(prisms)
```

makes later gdswell entries win overlaps — exactly "later cuts earlier." No
explicit boolean subtraction is performed in the adapter; meshwell's CAD stage
does it.

## Architecture

Single percent-format notebook (`# %%` cells), executed in the docs build,
matching the style of the other `docs/*.py` files. Two small **inline,
copy-pasteable adapter functions** carry the whole bridge — no library changes.

### Section 1 — Layout + stackup (gdswell)

Reuse the rib-waveguide + TiN-heater + via + metal example from gdswell
`15-stackup.py`: the `PDK(Layer, Enum)`, the bulk media via
`AllLayers().bbox()`, the 70 nm slab, the 220 nm slanted rib
(`{0.0: PDK.WG, 0.22: PDK.WG.size(-0.05)}`), heater, slanted via
(`.size(0.2)`), metal1, composed with `+`. Draw `device_cell(...)` and
**add a `gw.Port`** at the waveguide input (e.g. `position=(0, 0)`,
`angle=180`, a `gw.CrossSection` for the rib) via `cell.add_port(...)`.

### Section 2 — Resolve + the 3D adapter

```python
resolved = stack.resolve(cell)                       # ResolvedStackup
polyprisms = resolved_to_polyprisms(resolved, structured_names={...})
```

Adapter helpers (defined inline in the notebook):

- `region_to_shapely(region: kdb.Region, dbu: float) -> shapely (Multi)Polygon`
  — iterate merged polygons, scale integer dbu points by `dbu` to µm, preserve
  holes.
- `resolved_to_polyprisms(resolved, structured_names=()) -> list[PolyPrism]`:
  for each `keep=True` prism,
  - base footprint = `region_to_shapely` of the **min-z** region;
  - `buffers = {z: (bbox_width(region_z) − bbox_width(region_zmin)) / 2}` over
    the prism's z-keys. This is exact for gdswell's uniform `.size(d)` slants
    (rib top → `−0.05`, via top → `+0.2`) and `0` for uniform layers;
  - `physical_name = prism.name`;
  - `mesh_order = N − prism.mesh_order`;
  - `structured = prism.name in structured_names` (only set on uniform layers).
  `keep=False` cutters are **skipped** (documented limitation — see below).

### Section 3 — 3D structured mesh

`generate_mesh(entities=polyprisms, dim=3, default_characteristic_length=…,
resolution_specs={name: [StructuredExtrusionResolutionSpec(n_layers=N)] …})`.
Uniform-footprint layers (slab, claddings, heater, metal1) opt into
`structured=True` → wedge fill; the **slanted rib and via stay unstructured
tets** because a non-zero buffer precludes structured meshing. Teaching point:
the wedge cohort and surrounding tets share **conformal** interfaces — cross-
reference `structured.py`. Report wedge/tet counts and `plot3D(...)`.

### Section 4 — 2D cross-section mesh at a port

Derive a cutline from the waveguide `Port`: perpendicular to the port's
outward `angle`, centered at `port.position`, spanning the transverse device
extent. Then:

```python
cutline = cutline_from_port(cell.ports["input"], half_extent=W)
resolved_2d = stack.resolve_cutline(cell, cutline)   # ResolvedStackup2D
polysurfaces = resolved2d_to_polysurfaces(resolved_2d)
mesh2d = generate_mesh(entities=polysurfaces, dim=2, …)
plot2D(mesh2d)
```

`resolved2d_to_polysurfaces(resolved_2d)` mirrors the 3D adapter: for each
`keep=True` `ResolvedPolygon2D`, `region_to_shapely(region, dbu)` (note: the
`(s, z)` plane, not layout xy) → `PolySurface(polygons=…,
physical_name=name, mesh_order=N − mesh_order)`.

### Section 5 — Limitations + outlook

- `keep=False` pure-cutter prisms are dropped; meshwell expresses subtraction
  through `mesh_order`, so a faithful pure-cutter mapping is future work.
- The buffer model captures **uniform** z-offset slants only; arbitrary
  z-topology morphs in a gdswell entry can't be expressed as a single
  base-polygon + scalar buffers.
- Natural next step: graduate the inline adapters into a library helper
  (e.g. `meshwell.contrib.gdswell`) once the API settles.

## Packaging changes

- `meshwell/pyproject.toml`: add a `docs` optional-dependency extra containing
  `gdswell`, `pyvista`, and the doc-build tooling the page needs (or add
  `gdswell` + `pyvista` to the existing `dev` extra if a separate `docs` extra
  is judged redundant — decide in the plan).
- `meshwell/docs/_toc.yml`: register `gdswell_interface` under **Misc** (next to
  `gdsfactory_interface`).
- `meshwell/docs/_config.yml`: do **not** add the page to `execute.exclude_patterns`
  — it must run in the build.

## Testing / verification

- Notebook executes end-to-end under `uv run --python 3.12` (3.14 lacks a
  cadquery-ocp wheel): both adapters run, `generate_mesh` returns for `dim=3`
  and `dim=2`, wedge and tet counts are non-zero, conformal interface groups
  (`name___name`) are present, and `plot3D`/`plot2D` render.
- Confirm the inverted `mesh_order` produces the same kept-region ownership as
  gdswell's `cut_by` on the overlapping pairs (e.g. via the heater pad / metal1
  / via stack).

## Out of scope

- No changes to gdswell.
- No changes to the existing `gdsfactory_interface.py`.
- No new library module (adapters stay inline this iteration).
