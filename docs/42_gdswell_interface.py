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
# We reuse the rib-waveguide example from gdswell's stackup notebook: a silicon
# slab, a rib with angled sidewalls, a TiN heater, a via column, and a metal-1
# pad, all embedded in substrate / buried-oxide / cladding bulk media. The bulk
# media use the `AllLayers().bbox()` smart recipe so they grow to the bounding
# box of whatever the device draws.
#
# A `build_stack` factory exposes every layer thickness and the rib/via
# **sidewall angles** (degrees from horizontal: `90` = vertical, `<90` = top
# narrower, `>90` = flared) as keyword arguments, so a process corner is a
# one-line sweep. The factory converts each angle into the per-side xy offset
# gdswell needs via `offset = -thickness / tan(angle)`.


# %%
class PDK(gw.Layer, Enum):
    """Process layer set for the demo device."""

    WG = (1, 0)  # full-Si rib (220 nm)
    SLAB = (2, 0)  # partial-etch Si slab (70 nm)
    HEATER = (10, 0)  # TiN heater above the rib
    VIA1 = (11, 0)  # via from heater pad to METAL1
    METAL1 = (12, 0)  # routing metal


def build_stack(
    *,
    slab_thickness: float = 0.07,
    rib_thickness: float = 0.22,
    rib_sidewall_angle: float = 80.0,
    heater_thickness: float = 0.10,
    via_sidewall_angle: float = 100.0,
) -> gw.Stackup:
    """Assemble the silicon-photonic stackup.

    Thicknesses are in µm. Sidewall angles are in degrees measured from the
    horizontal substrate plane: ``90`` is a vertical wall, ``<90`` makes the
    top narrower (an inward slope), ``>90`` flares the top outward. The
    per-side xy offset over a layer of height ``t`` is ``-t / tan(angle)``.
    """

    def sidewall_offset(thickness: float, angle_deg: float) -> float:
        return -thickness / math.tan(math.radians(angle_deg))

    device_extent = gw.AllLayers().bbox()
    substrate = gw.StackupEntry.uniform("Substrate", device_extent, -2.0, -1.0)
    box = gw.StackupEntry.uniform("BOX", device_extent, -1.0, 0.0)
    lower_clad = gw.StackupEntry.uniform("Lower_clad", device_extent, 0.0, 1.5)
    upper_clad = gw.StackupEntry.uniform("Upper_clad", device_extent, 1.6, 2.5)

    si_slab = gw.StackupEntry.uniform("Si_slab", PDK.SLAB, 0.0, slab_thickness)
    si_rib = gw.StackupEntry(
        "Si_rib",
        {
            0.0: PDK.WG,
            rib_thickness: PDK.WG.size(
                sidewall_offset(rib_thickness, rib_sidewall_angle)
            ),
        },
    )

    heater = gw.StackupEntry.uniform("Heater", PDK.HEATER, 1.5, 1.5 + heater_thickness)
    via_zmin, via_zmax = 1.55, 2.5
    via1 = gw.StackupEntry(
        "Via1",
        {
            via_zmin: PDK.VIA1,
            via_zmax: PDK.VIA1.size(
                sidewall_offset(via_zmax - via_zmin, via_sidewall_angle)
            ),
        },
    )
    metal1 = gw.StackupEntry.uniform("Metal1", PDK.METAL1, 2.5, 3.5)

    return (
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


# Every thickness and sidewall angle is a knob — sweep a process corner by
# passing values here (a steeper 75° rib and a more flared 110° via):
stack = build_stack(rib_sidewall_angle=75.0, via_sidewall_angle=110.0)
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
# Three facts make the conversion small:
#
# 1. **klayout regions become shapely polygons** by scaling integer coordinates
#    by the database unit `dbu` (µm per integer). This is the only helper we
#    really need — `region_to_shapely`.
# 2. **meshwell's `mesh_order` reproduces gdswell's `cut_by` for free.** In
#    gdswell, *later* entries cut earlier ones. In meshwell, the entity with the
#    *lowest* `mesh_order` owns an overlap. So we simply invert the order —
#    `mesh_order = N - prism.mesh_order` — and meshwell's CAD stage performs the
#    painter's cuts. No explicit boolean subtraction in the adapter.
# 3. **Sidewalls cross the two solid models.** A meshwell `PolyPrism` is *one*
#    base footprint plus a `{z: offset}` map of **scalar** buffers — at each z it
#    grows or shrinks the base by `polygon.buffer(offset)`. gdswell instead hands
#    us an explicit region at *every* z-key. The two agree for gdswell's uniform
#    `.size(d)` sidewalls, where the equivalent meshwell offset is just half the
#    change in bbox width. That single line (inlined below) is what carries the
#    rib's taper and the via's flare across; for a constant footprint the offset
#    is `0` and the prism is a plain extrusion. So we still only expose
#    `region_to_shapely` and `resolved_to_polyprisms` / `resolved2d_to_polysurfaces`.
#
#    NOTE: meshwell precedes gdswell; in the future we could consider moving from shapely
#    to KLayout regions directly, and use the z --> region convention instead of buffers.


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
        # meshwell offsets the single `base` footprint by a scalar buffer at each
        # z; gdswell's uniform `.size(d)` sidewall is half the bbox-width change.
        base_width = base.bbox().width()
        buffers = {
            z: (region.bbox().width() - base_width) / 2 * resolved.dbu
            for z, region in rp.z_to_region.items()
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
# ## 2D cross-section
#
# gdswell's Stackup can be resolved along a 1D cutline or given a specific cross-section. This can be used to define a 2D vertical cross-sectional mesh:


# %%
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
resolved_2d = stack.resolve_cross_section(cell.ports["input"].cross_section)
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
# ## Notes
#
# - **Structured meshing.** This page produces an unstructured (tetrahedral)
#   mesh. The uniform-footprint layers could opt into structured wedge meshing —
#   see [Structured meshing](23_structured).
