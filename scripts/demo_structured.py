# ruff: noqa: S108, D103
"""Demo of the clean structured-polyprism pipeline.

Builds example scenes that exercise what the rewrite supports on
``feat/structured-clean`` today, generates meshes, and prints
element-count summaries.

Run::

    .venv/bin/python scripts/demo_structured.py [output_dir]

Output: ``.msh`` files under ``output_dir`` (defaults to
``/tmp/structured_demo``), plus a console report.

Scenes
------

1. ``simple_slab`` — A 2x2 structured polyprism at z=[0, 1] with
   n_layers=2. No neighbours. The slab volume is meshed as wedge
   prisms. The single-piece end-to-end case.

2. ``slab_with_top_cap`` — A 4x4 structured slab at z=[0, 1] with
   n_layers=2 plus a non-structured ``cap`` at z=[1, 2] covering the
   left half (x=[0, 2]). The slab's top face is partitioned into 2
   pieces by the planner; the multi-piece routing introduced in
   Phase 5(a) handles boundary-node correspondence via Layer B's
   ``output_edges`` map. Both pieces produce wedge prisms; the cap
   above produces tets.

7. ``overlapping_top_neighbours`` — Phase 5(b): two non-structured
   neighbours at z=[1, 2] with overlapping xy footprints sitting on
   the slab top. BOP splits the slab's top OCC face into 3 sub-faces;
   Phase 5(b) routes each bottom triangle to the right sub-face by
   centroid XY and builds wedges correctly.

What this does NOT yet exercise
-------------------------------

- Mid-height-cut lateral faces (Phase 5(c), now rejected at plan stage).
- Arc-bearing structured prisms (Phase 6+).
"""
from __future__ import annotations

import sys
from pathlib import Path

import meshio
from shapely.geometry import Polygon

from meshwell.orchestrator import generate_mesh
from meshwell.polyprism import PolyPrism
from meshwell.structured import StructuredExtrusionResolutionSpec


def _square(x: float, y: float, w: float, h: float) -> Polygon:
    return Polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])


def _summarize_mesh(path: Path) -> None:
    m = meshio.read(path)
    print(f"\n  Mesh file: {path} ({path.stat().st_size:,} bytes)")
    print(f"  Total points: {len(m.points)}")
    print("  Cell blocks:")
    for cb in m.cells:
        print(f"    {cb.type:12s} count={len(cb.data)}")
    print("  Physical groups:")
    for name, (tag, dim) in sorted(m.field_data.items()):
        print(f"    {name:20s} dim={dim}  tag={tag}")


def scene_simple_slab(out_dir: Path) -> Path:
    """Single structured polyprism, no neighbours."""
    print("\n" + "=" * 70)
    print("Scene 1: simple_slab")
    print("=" * 70)
    print(
        "  A 2x2 structured polyprism at z=[0, 1] with n_layers=2.\n"
        "  No neighbours -> face_partition has 1 piece.\n"
        "  Expected: 2 wedge layers, no other 3D cells."
    )

    slab = PolyPrism(
        polygons=_square(0, 0, 2, 2),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
        physical_name="slab",
    )

    out_msh = out_dir / "simple_slab.msh"
    generate_mesh([slab], dim=3, output_mesh=out_msh, default_characteristic_length=0.5)
    _summarize_mesh(out_msh)
    return out_msh


def scene_slab_with_top_cap(out_dir: Path) -> Path:
    """Structured slab + non-structured cap on top half.

    Exercises Phase-4 multi-piece routing + Phase-5(a) edge-correspondence
    boundary node lookup.
    """
    print("\n" + "=" * 70)
    print("Scene 2: slab_with_top_cap")
    print("=" * 70)
    print(
        "  4x4 structured slab at z=[0, 1], n_layers=2.\n"
        "  Non-structured cap at z=[1, 2] covering x=[0, 2] (left half).\n"
        "  -> face_partition has 2 pieces (covered + uncovered halves).\n"
        "  Expected: wedge prisms in both slab pieces, tets in the cap."
    )

    slab = PolyPrism(
        polygons=_square(0, 0, 4, 4),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
        physical_name="slab",
    )
    cap = PolyPrism(
        polygons=_square(0, 0, 2, 4),
        buffers={1.0: 0.0, 2.0: 0.0},
        physical_name="cap",
    )

    out_msh = out_dir / "slab_with_top_cap.msh"
    generate_mesh(
        [slab, cap], dim=3, output_mesh=out_msh, default_characteristic_length=0.5
    )
    _summarize_mesh(out_msh)
    return out_msh


def scene_embedded_slab(out_dir: Path) -> Path:
    """Structured slab fully embedded inside a larger unstructured cladding box.

    Cladding has lower priority (mesh_order=2.0); slab wins. Cladding gets
    fragmented into: top cap (z=[1,2]), bottom cap (z=[-1,0]), and a
    side ring (xy outside the slab footprint). Slab has neighbours on all
    6 lateral surfaces.

    Exercises: neighbour cuts on bottom plane + top plane (both planes
    have neighbour); structured-vs-unstructured priority resolution.
    """
    print("\n" + "=" * 70)
    print("Scene 3: embedded_slab")
    print("=" * 70)
    print(
        "  Structured slab z=[0, 1], xy=[0, 4]x[0, 4], n_layers=2,\n"
        "  embedded inside unstructured cladding z=[-1, 2], xy=[-2, 6]x[-2, 6].\n"
        "  -> slab has neighbours on bottom + top + all 4 lateral sides.\n"
        "  Expected: wedges in slab, tets in cladding (split into 6 pieces by BOP)."
    )

    slab = PolyPrism(
        polygons=_square(0, 0, 4, 4),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
        physical_name="slab",
        mesh_order=1.0,
    )
    cladding = PolyPrism(
        polygons=_square(-2, -2, 8, 8),
        buffers={-1.0: 0.0, 2.0: 0.0},
        physical_name="cladding",
        mesh_order=2.0,
    )

    out_msh = out_dir / "embedded_slab.msh"
    generate_mesh(
        [slab, cladding], dim=3, output_mesh=out_msh, default_characteristic_length=1.0
    )
    _summarize_mesh(out_msh)
    return out_msh


def scene_stacked_structured(out_dir: Path) -> Path:
    """Two structured slabs stacked z=[0,1] and z=[1,2], sharing the z=1 plane.

    Each slab has its own n_layers. They share OCC faces at z=1 via BOP.
    Both should mesh as wedges, conformally joined at the interface.

    Exercises: structured-on-structured stacking; per-slab phantom
    construction with shared interface face.
    """
    print("\n" + "=" * 70)
    print("Scene 4: stacked_structured")
    print("=" * 70)
    print(
        "  Lower structured slab z=[0, 1], n_layers=2.\n"
        "  Upper structured slab z=[1, 2], n_layers=3.\n"
        "  Same xy=[0, 2]x[0, 2] footprint, sharing the z=1 plane.\n"
        "  Expected: wedges in both slabs, conformal across z=1 interface."
    )

    lower = PolyPrism(
        polygons=_square(0, 0, 2, 2),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
        physical_name="lower",
    )
    upper = PolyPrism(
        polygons=_square(0, 0, 2, 2),
        buffers={1.0: 0.0, 2.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[3])],
        physical_name="upper",
    )

    out_msh = out_dir / "stacked_structured.msh"
    generate_mesh(
        [lower, upper], dim=3, output_mesh=out_msh, default_characteristic_length=0.5
    )
    _summarize_mesh(out_msh)
    return out_msh


def scene_side_by_side(out_dir: Path) -> Path:
    """Structured + unstructured side by side, sharing a lateral face.

    Both occupy z=[0, 1]. Structured at xy=[0, 2]x[0, 2]; unstructured at
    xy=[2, 4]x[0, 2]. They share the lateral face at x=2.

    Exercises: lateral face shared between structured slab and an
    unstructured neighbour at the same z range; conformal interface on
    the shared lateral.
    """
    print("\n" + "=" * 70)
    print("Scene 5: side_by_side")
    print("=" * 70)
    print(
        "  Structured slab z=[0, 1], xy=[0, 2]x[0, 2], n_layers=2.\n"
        "  Unstructured neighbour z=[0, 1], xy=[2, 4]x[0, 2].\n"
        "  -> shared lateral face at x=2.\n"
        "  Expected: wedges in slab, tets in neighbour, conformal at x=2."
    )

    slab = PolyPrism(
        polygons=_square(0, 0, 2, 2),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
        physical_name="slab",
    )
    side = PolyPrism(
        polygons=_square(2, 0, 2, 2),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="side",
    )

    out_msh = out_dir / "side_by_side.msh"
    generate_mesh(
        [slab, side], dim=3, output_mesh=out_msh, default_characteristic_length=0.5
    )
    _summarize_mesh(out_msh)
    return out_msh


def scene_midheight_cut_rejected(out_dir: Path) -> Path | None:
    """Mid-height cut by a neighbour should be rejected at plan stage.

    Structured slab z=[0, 1]; unstructured neighbour z=[0.3, 1.5] that
    intrudes into the slab xy. The neighbour's zmin=0.3 is STRICTLY
    inside the slab's z-extent (0 < 0.3 < 1), which would cause BOP to
    introduce a vertex on the slab's lateral face at z=0.3 — a
    mid-height cut the structured pipeline can't form a conformal wedge
    grid through.

    Phase 5(c) detects this at plan stage and raises
    StructuredMidHeightCutError with a remediation message.
    """
    print("\n" + "=" * 70)
    print("Scene 6: midheight_cut_rejected (expected to raise)")
    print("=" * 70)
    print(
        "  Structured slab z=[0, 1] + neighbour z=[0.3, 1.5] intruding\n"
        "  laterally. Neighbour zmin=0.3 is strictly inside slab z-extent.\n"
        "  Expected: StructuredMidHeightCutError with remediation guidance."
    )

    from meshwell.structured.spec import StructuredMidHeightCutError

    slab = PolyPrism(
        polygons=_square(0, 0, 4, 4),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
        physical_name="slab",
    )
    intruder = PolyPrism(
        polygons=_square(3, 1, 2, 2),  # overlaps slab in xy
        buffers={0.3: 0.0, 1.5: 0.0},  # zmin=0.3 strictly inside [0, 1]
        physical_name="intruder",
    )

    out_msh = out_dir / "midheight_cut_rejected.msh"
    try:
        generate_mesh(
            [slab, intruder],
            dim=3,
            output_mesh=out_msh,
            default_characteristic_length=0.5,
        )
        print("  !! ERROR: expected StructuredMidHeightCutError, got success!")
        return out_msh
    except StructuredMidHeightCutError as e:
        print(f"  OK - raised as expected:\n    {str(e)[:200]}")
        return None


def scene_overlapping_top_neighbours(out_dir: Path) -> Path:
    """Phase 5(b): slab with two overlapping non-structured top neighbours.

    Slab at z=[0, 1]. Neighbour A at z=[1, 2] covers x=[0, 3]; neighbour B
    at z=[1, 2] covers x=[1, 4]. Their xy union covers the full slab top,
    so Phase 1's planner makes 1 piece. BOP splits the slab's top OCC face
    into 3 sub-faces (A-only, AB-overlap, B-only). Phase 5(b) routes per
    centroid XY and builds wedges correctly.

    Exercises: _stamp_top_face_mesh_multi + multi-top OCC volume lookup.
    """
    print("\n" + "=" * 70)
    print("Scene 7: overlapping_top_neighbours (Phase 5(b))")
    print("=" * 70)
    print(
        "  Structured slab z=[0, 1], xy=[0, 4]x[0, 4], n_layers=2.\n"
        "  Neighbour A at z=[1, 2], xy=[0, 3]x[0, 4].\n"
        "  Neighbour B at z=[1, 2], xy=[1, 4]x[0, 4].  (overlap: x=[1, 3])\n"
        "  -> BOP splits slab top into 3 sub-faces.\n"
        "  Expected: wedges in slab, tets in A and B."
    )

    slab = PolyPrism(
        polygons=_square(0, 0, 4, 4),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
        physical_name="slab",
    )
    a = PolyPrism(
        polygons=_square(0, 0, 3, 4),
        buffers={1.0: 0.0, 2.0: 0.0},
        physical_name="a",
    )
    b = PolyPrism(
        polygons=_square(1, 0, 3, 4),
        buffers={1.0: 0.0, 2.0: 0.0},
        physical_name="b",
    )

    out_msh = out_dir / "overlapping_top_neighbours.msh"
    generate_mesh(
        [slab, a, b], dim=3, output_mesh=out_msh, default_characteristic_length=0.5
    )
    _summarize_mesh(out_msh)
    return out_msh


def main() -> int:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/structured_demo")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")

    scenes_to_run = [
        ("simple_slab", scene_simple_slab),
        ("slab_with_top_cap", scene_slab_with_top_cap),
        ("embedded_slab", scene_embedded_slab),
        ("stacked_structured", scene_stacked_structured),
        ("side_by_side", scene_side_by_side),
        ("midheight_cut_rejected", scene_midheight_cut_rejected),
        ("overlapping_top_neighbours", scene_overlapping_top_neighbours),
    ]
    results: list[tuple[str, str, Path | None]] = []
    for name, fn in scenes_to_run:
        try:
            path = fn(out_dir)
            results.append((name, "OK", path))
        except Exception as e:
            print(f"\n  !! Scene {name} FAILED: {type(e).__name__}: {e}")
            results.append((name, "FAIL", None))

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    for name, status, path in results:
        if status == "OK":
            print(f"  {status} {name}: {path}")
        else:
            print(f"  {status} {name}: see error above")
    n_ok = sum(1 for _, s, _ in results if s == "OK")
    print(f"\n{n_ok}/{len(results)} scenes succeeded.")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
