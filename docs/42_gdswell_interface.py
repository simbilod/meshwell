# %% [markdown]
# # gdswell integration
#
# [gdswell](https://github.com/simbilod/gdswell) is a layout tool whose
# `Stackup` describes how 2D layout layers extend into 3D — what material lives
# at what height and how its xy footprint morphs with z. Its
# `Stackup.resolve(cell)` deliberately stops at *raw* per-entry 3D recipes and
# leaves painter's-algorithm cutting to a **downstream backend** — and meshwell
# is exactly that backend.
#
# This page is the bridge. We take a gdswell `Stackup` + `Cell`, resolve it, and
# convert the result into meshwell entities to obtain
#
# - a watertight **3D mesh** of the whole device, and
# - a **2D cross-section mesh** sliced at one of the cell's ports.
#
# The two adapter functions below (`resolved_to_polyprisms`,
# `resolved2d_to_polysurfaces`) are the whole integration — small enough to copy
# into your own project.

# %%
import math
from enum import Enum

import gdswell as gw
import klayout.db as kdb
import shapely

from meshwell.orchestrator import generate_mesh
from meshwell.polyprism import PolyPrism
from meshwell.polysurface import PolySurface
from meshwell.visualization import plot2D, plot3D

# %% [markdown]
# ## A silicon-photonic stackup (gdswell)
#
# We reuse the rib-waveguide example from gdswell's stackup notebook: a 70 nm
# silicon slab, a 220 nm rib with slanted sidewalls, a TiN heater, a via column,
# and a metal-1 pad, all embedded in substrate / buried-oxide / cladding bulk
# media. The bulk media use the `AllLayers().bbox()` smart recipe so they grow
# to the bounding box of whatever the device draws.


# %%
class PDK(gw.Layer, Enum):
    """Process layer set for the demo device."""

    WG = (1, 0)  # full-Si rib (220 nm)
    SLAB = (2, 0)  # partial-etch Si slab (70 nm)
    HEATER = (10, 0)  # TiN heater above the rib
    VIA1 = (11, 0)  # via from heater pad to METAL1
    METAL1 = (12, 0)  # routing metal


device_extent = gw.AllLayers().bbox()
substrate = gw.StackupEntry.uniform("Substrate", device_extent, -2.0, -1.0)
box = gw.StackupEntry.uniform("BOX", device_extent, -1.0, 0.0)
lower_clad = gw.StackupEntry.uniform("Lower_clad", device_extent, 0.0, 1.5)
upper_clad = gw.StackupEntry.uniform("Upper_clad", device_extent, 1.6, 2.5)

si_slab = gw.StackupEntry.uniform("Si_slab", PDK.SLAB, 0.0, 0.07)
si_rib = gw.StackupEntry("Si_rib", {0.0: PDK.WG, 0.22: PDK.WG.size(-0.05)})

heater = gw.StackupEntry.uniform("Heater", PDK.HEATER, 1.5, 1.6)
via1 = gw.StackupEntry("Via1", {1.55: PDK.VIA1, 2.5: PDK.VIA1.size(0.2)})
metal1 = gw.StackupEntry.uniform("Metal1", PDK.METAL1, 2.5, 3.5)

stack = (
    substrate
    + box
    + lower_clad
    + upper_clad
    + si_slab
    + si_rib
    + heater
    + via1
    + metal1
)
print(stack)

# %% [markdown]
# ## The device cell (gdswell)
#
# A short rib waveguide with a heater strip over it; both ends of the heater fan
# out to a metal-1 pad south of the waveguide, contacted by a via column. We add
# a `Port` at the waveguide input — its `position` and `angle` are all we need to
# derive a cross-section cutline later.

# %%
L = 10.0  # propagation length, µm
W = 8.0  # transverse half-extent, µm


@gw.cell
def device_cell(L=L) -> gw.Cell:
    """Rib waveguide with a TiN heater wired to a metal pad."""
    cell = gw.Cell()

    cell.add_polygon([(0.0, -0.25), (L, -0.25), (L, 0.25), (0.0, 0.25)], PDK.WG)
    cell.add_polygon([(0.0, -3.0), (L, -3.0), (L, 3.0), (0.0, 3.0)], PDK.SLAB)

    cell.add_polygon([(0.0, -1.0), (L, -1.0), (L, 1.0), (0.0, 1.0)], PDK.HEATER)
    cell.add_polygon(
        [(L / 2 - 3, -5.5), (L / 2 + 3, -5.5), (L / 2 + 3, -2.5), (L / 2 - 3, -2.5)],
        PDK.HEATER,
    )
    cell.add_polygon(
        [(L / 2 - 1, -5.0), (L / 2 + 1, -5.0), (L / 2 + 1, -4.5), (L / 2 - 1, -4.5)],
        PDK.VIA1,
    )
    cell.add_polygon(
        [(L / 2 - 3, -5.5), (L / 2 + 3, -5.5), (L / 2 + 3, -2.5), (L / 2 - 3, -2.5)],
        PDK.METAL1,
    )

    # A port at the waveguide input, pointing outwards (-x).
    xs = gw.CrossSection(
        layer_sections=(
            gw.LayerSection(name="slab", layer=PDK.SLAB, width=6.0),
            gw.LayerSection(name="rib", layer=PDK.WG, width=0.5),
        )
    )
    cell.add_port(
        gw.Port(name="input", position=(0.0, 0.0), angle=180, cross_section=xs)
    )

    return cell


cell = device_cell()

# %% [markdown]
# ## The adapter: gdswell → meshwell
#
# `Stackup.resolve(cell)` returns a `ResolvedStackup` whose `prisms` each carry a
# `z_to_region` (a `{z: klayout.Region}` map in integer database units), a
# `mesh_order` (its position in the painter's stack), and a `cut_by` list of
# later entries that carve it.
#
# Two facts make the conversion small:
#
# 1. **klayout regions become shapely polygons** by scaling integer coordinates
#    by the database unit `dbu` (µm per integer).
# 2. **meshwell's `mesh_order` reproduces gdswell's `cut_by` for free.** In
#    gdswell, *later* entries cut earlier ones. In meshwell, the entity with the
#    *lowest* `mesh_order` owns an overlap. So we simply invert the order —
#    `mesh_order = N - prism.mesh_order` — and meshwell's CAD stage performs the
#    painter's cuts. No explicit boolean subtraction in the adapter.


# %%
def region_to_shapely(region: kdb.Region, dbu: float):
    """Convert a klayout Region (integer dbu) to a shapely polygon in µm."""
    polys = []
    for p in region.merged().each():
        shell = [(pt.x * dbu, pt.y * dbu) for pt in p.each_point_hull()]
        holes = [
            [(pt.x * dbu, pt.y * dbu) for pt in p.each_point_hole(i)]
            for i in range(p.holes())
        ]
        polys.append(shapely.Polygon(shell, holes))
    if not polys:
        return None
    return polys[0] if len(polys) == 1 else shapely.MultiPolygon(polys)


def _buffer_at(region: kdb.Region, base: kdb.Region, dbu: float) -> float:
    """Uniform xy offset (µm) mapping ``base`` to ``region`` via bbox width.

    Exact for gdswell's uniform ``.size(d)`` slants, ``0`` for a constant
    footprint. ``.size(d)`` changes the bbox width by ``2 d``.
    """
    return (region.bbox().width() - base.bbox().width()) / 2 * dbu


def resolved_to_polyprisms(resolved: gw.ResolvedStackup) -> list[PolyPrism]:
    """Convert a resolved 3D stackup into meshwell PolyPrisms."""
    n = len(resolved.prisms)
    prisms = []
    for rp in resolved.prisms:
        if not rp.keep:
            continue  # pure cutters are not emitted as volumes (see Limitations)
        zmin = min(rp.z_to_region)
        base = rp.z_to_region[zmin]
        polygons = region_to_shapely(base, resolved.dbu)
        if polygons is None:
            continue
        buffers = {
            z: _buffer_at(rp.z_to_region[z], base, resolved.dbu) for z in rp.z_to_region
        }
        prisms.append(
            PolyPrism(
                polygons=polygons,
                buffers=buffers,
                physical_name=rp.name,
                mesh_order=n - rp.mesh_order,
            )
        )
    return prisms


# %% [markdown]
# ## 3D mesh
#
# Resolve, convert, mesh. The overlaps between the bulk media and the device
# layers are carved by the inverted `mesh_order`, and shared faces show up as
# conformal interface physical groups (`a___b`).

# %%
resolved = stack.resolve(cell)
polyprisms = resolved_to_polyprisms(resolved)

mesh3d = generate_mesh(
    entities=polyprisms,
    dim=3,
    output_mesh="gdswell_device.msh",
    default_characteristic_length=2.0,
)

tets = sum(cb.data.shape[0] for cb in mesh3d.cells if cb.type == "tetra")
print(f"tetrahedra: {tets}")
print("interface groups:", [k for k in mesh3d.cell_sets if "___" in k][:10])

# %%
plot3D(mesh3d, title="gdswell stackup meshed by meshwell (3D)")

# %% [markdown]
# ## 2D cross-section at a port
#
# A `Port` carries a `position` and an outward `angle`. The transverse
# cross-section cutline runs **perpendicular** to that direction, through a point
# nudged slightly inward (so we sample the interior rather than the exact cell
# edge). `Stackup.resolve_cutline` slices the stackup along it, returning 2D
# regions in the `(s, z)` plane — arclength `s` along the cut on x, height `z` on
# y.


# %%
def cutline_from_port(port: gw.Port, half_extent: float, inset: float = 1.0):
    """Transverse cutline (two xy points, µm) perpendicular to a port."""
    rad = math.radians(port.angle)
    dx, dy = round(math.cos(rad)), round(math.sin(rad))
    cx, cy = port.x - dx * inset, port.y - dy * inset  # nudge inward
    px, py = -dy, dx  # perpendicular
    return (
        (cx - px * half_extent, cy - py * half_extent),
        (cx + px * half_extent, cy + py * half_extent),
    )


def resolved2d_to_polysurfaces(resolved_2d: gw.ResolvedStackup2D) -> list[PolySurface]:
    """Convert a resolved 2D cross-section into meshwell PolySurfaces."""
    n = len(resolved_2d.polygons)
    surfaces = []
    for rp in resolved_2d.polygons:
        if not rp.keep:
            continue
        polygons = region_to_shapely(rp.region, resolved_2d.dbu)
        if polygons is None:
            continue
        surfaces.append(
            PolySurface(
                polygons=polygons,
                physical_name=rp.name,
                mesh_order=n - rp.mesh_order,
            )
        )
    return surfaces


# %%
cutline = cutline_from_port(cell.ports["input"], half_extent=W - 1.0)
resolved_2d = stack.resolve_cutline(cell, cutline)
polysurfaces = resolved2d_to_polysurfaces(resolved_2d)

mesh2d = generate_mesh(
    entities=polysurfaces,
    dim=2,
    output_mesh="gdswell_xsection.msh",
    default_characteristic_length=0.5,
)

# %%
plot2D(mesh2d, wireframe=True)

# %% [markdown]
# ## Limitations
#
# - **Pure cutters.** gdswell keeps `keep=False` entries only so other prisms'
#   `cut_by` can reference them; the adapter drops them. meshwell expresses
#   subtraction through `mesh_order`, so a faithful pure-cutter mapping is left
#   as future work.
# - **Footprint morphs.** The buffer model captures *uniform* z-offset slants
#   (gdswell's `.size(d)`) only. An entry whose footprint changes topology with
#   z cannot be expressed as a single base polygon plus scalar buffers.
# - **Structured meshing.** This page produces an unstructured (tetrahedral)
#   mesh. The uniform-footprint layers could opt into structured wedge meshing —
#   see [Structured meshing](23_structured).
#
# Once the adapter API settles, these helpers are natural candidates to graduate
# from notebook glue into a small library module.
