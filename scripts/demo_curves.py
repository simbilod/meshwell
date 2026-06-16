"""Demo: structured slabs with curved (arc) features.

Builds three small scenes and writes each to its own .msh under
/tmp/curves_demo. Each scene exercises a different curve case.

Run::

    python demo_curves.py
    gmsh /tmp/curves_demo/single_disc.msh
    gmsh /tmp/curves_demo/stacked_discs.msh
    gmsh /tmp/curves_demo/annulus_disc_stack.msh
"""
from __future__ import annotations

import math
from pathlib import Path

import meshio
from shapely.geometry import Polygon

from meshwell.orchestrator import generate_mesh
from meshwell.polyprism import PolyPrism
from meshwell.resolution import StructuredExtrusionResolutionSpec


def _disc(cx: float, cy: float, r: float, n: int = 48) -> Polygon:
    """Polygon sampling of a circle with n vertices (closed)."""
    return Polygon(
        [
            (
                cx + r * math.cos(2 * math.pi * i / n),
                cy + r * math.sin(2 * math.pi * i / n),
            )
            for i in range(n)
        ]
    )


def _annulus(cx: float, cy: float, r_out: float, r_in: float, n: int = 48) -> Polygon:
    outer = _disc(cx, cy, r_out, n)
    inner = _disc(cx, cy, r_in, n)
    return Polygon(outer.exterior.coords, holes=[inner.exterior.coords])


def _report(path: Path) -> None:
    m = meshio.read(path)
    counts: dict[str, int] = {}
    for cb in m.cells:
        counts[cb.type] = counts.get(cb.type, 0) + len(cb.data)
    print(f"  {path.name}: nodes={len(m.points)}  cells={counts}")
    # Per-physical-group wedge counts (skip synthetic and gmsh internals)
    for name in sorted(m.cell_sets):
        if name.startswith("__") or name.startswith("gmsh:"):
            continue
        sets = m.cell_sets[name]
        wedges = sum(
            len(s) for s, b in zip(sets, m.cells) if b.type == "wedge" and s is not None
        )
        if wedges:
            print(f"    {name:24s} wedges={wedges}")


def scene_single_disc(out_dir: Path) -> Path:
    """One structured disc, n_layers=3."""
    print("Scene 1: single_disc")
    print("  A radius-1 disc at z=[0, 1] meshed with wedge prisms.")
    print("  Lateral wall is a cylindrical face from explicit Geom_CylindricalSurface.")
    disc = PolyPrism(
        polygons=_disc(0, 0, 1.0, n=48),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="disc",
        structured=True,
        identify_arcs=True,
    )
    # Cladding above and below to satisfy the cohort wrapping invariant.
    base = PolyPrism(
        polygons=_disc(0, 0, 1.0, n=48),
        buffers={-1.0: 0.0, 0.0: 0.0},
        physical_name="base",
        identify_arcs=True,
    )
    cap = PolyPrism(
        polygons=_disc(0, 0, 1.0, n=48),
        buffers={1.0: 0.0, 2.0: 0.0},
        physical_name="cap",
        identify_arcs=True,
    )
    out = out_dir / "single_disc.msh"
    generate_mesh(
        [disc, base, cap],
        dim=3,
        output_mesh=out,
        default_characteristic_length=0.3,
        resolution_specs={"disc": [StructuredExtrusionResolutionSpec(n_layers=3)]},
    )
    _report(out)
    return out


def scene_stacked_discs(out_dir: Path) -> Path:
    """Two stacked discs sharing the z=1 plane (cohort with arc interior face)."""
    print("\nScene 2: stacked_discs")
    print("  Lower disc r=1 at z=[0, 1], upper disc r=1 at z=[1, 2].")
    print("  Cohort by face-touch at z=1. The shared interior horizontal face is")
    print("  a single arc-bounded TShape; both wedge slabs see the same triangulation.")
    lower = PolyPrism(
        polygons=_disc(0, 0, 1.0, n=48),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="lower",
        structured=True,
        identify_arcs=True,
    )
    upper = PolyPrism(
        polygons=_disc(0, 0, 1.0, n=48),
        buffers={1.0: 0.0, 2.0: 0.0},
        physical_name="upper",
        structured=True,
        identify_arcs=True,
    )
    base = PolyPrism(
        polygons=_disc(0, 0, 1.0, n=48),
        buffers={-1.0: 0.0, 0.0: 0.0},
        physical_name="base",
        identify_arcs=True,
    )
    cap = PolyPrism(
        polygons=_disc(0, 0, 1.0, n=48),
        buffers={2.0: 0.0, 3.0: 0.0},
        physical_name="cap",
        identify_arcs=True,
    )
    out = out_dir / "stacked_discs.msh"
    generate_mesh(
        [lower, upper, base, cap],
        dim=3,
        output_mesh=out,
        default_characteristic_length=0.3,
        resolution_specs={
            "lower": [StructuredExtrusionResolutionSpec(n_layers=2)],
            "upper": [StructuredExtrusionResolutionSpec(n_layers=2)],
        },
    )
    _report(out)
    return out


def scene_annulus_on_disc(out_dir: Path) -> Path:
    """Annulus on top of a disc: tests concentric arcs at the cohort z-plane."""
    print("\nScene 3: annulus_on_disc")
    print("  Disc r=2 at z=[0, 1]. Annulus r_out=2, r_in=0.8 at z=[1, 2].")
    print("  The cohort top face at z=1 is split into 'inner disc' + 'annulus' by")
    print("  the inner-hole arc. Both arcs (outer + inner) are shared between the")
    print("  lower wedge top and the upper wedge bot.")
    disc = PolyPrism(
        polygons=_disc(0, 0, 2.0, n=48),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="lower_disc",
        structured=True,
        identify_arcs=True,
    )
    annulus = PolyPrism(
        polygons=_annulus(0, 0, r_out=2.0, r_in=0.8, n=48),
        buffers={1.0: 0.0, 2.0: 0.0},
        physical_name="upper_annulus",
        structured=True,
        identify_arcs=True,
    )
    base = PolyPrism(
        polygons=_disc(0, 0, 2.0, n=48),
        buffers={-1.0: 0.0, 0.0: 0.0},
        physical_name="base",
        identify_arcs=True,
    )
    cap = PolyPrism(
        polygons=_annulus(0, 0, r_out=2.0, r_in=0.8, n=48),
        buffers={2.0: 0.0, 3.0: 0.0},
        physical_name="cap",
        identify_arcs=True,
    )
    out = out_dir / "annulus_disc_stack.msh"
    generate_mesh(
        [disc, annulus, base, cap],
        dim=3,
        output_mesh=out,
        default_characteristic_length=0.3,
        resolution_specs={
            "lower_disc": [StructuredExtrusionResolutionSpec(n_layers=2)],
            "upper_annulus": [StructuredExtrusionResolutionSpec(n_layers=2)],
        },
    )
    _report(out)
    return out


def main() -> None:
    """Build all three demo scenes and write them under a temp directory."""
    import tempfile

    out_dir = Path(tempfile.gettempdir()) / "curves_demo"
    out_dir.mkdir(exist_ok=True)
    print(f"Writing meshes to {out_dir}\n")

    scenes = [scene_single_disc, scene_stacked_discs, scene_annulus_on_disc]
    for scene in scenes:
        try:
            scene(out_dir)
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")

    print(f"\nOpen any of these with: gmsh {out_dir}/<name>.msh")


if __name__ == "__main__":
    main()
