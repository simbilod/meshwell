#!/usr/bin/env python3
"""Inspect Phase 3 cohort envelope geometry for a chosen scene.

Dumps:
  - plan.slabs, arrangement edges, face_partition, face_partition_edges
  - env.outline_xy_to_corner_id, env.horizontal_edges, env.lateral_faces
  - env.bottom_sub_faces / env.top_sub_faces (bbox + tolerance)
  - cohort envelope solid via TopExp_Explorer (face/edge/vertex counts)

Exports (optional, --export-step <dir>):
  - <dir>/cohort_envelope.step          : the assembled cohort solid
  - <dir>/piece_<i>_bot.brep            : per-piece bot sub-face
  - <dir>/piece_<i>_top.brep            : per-piece top sub-face
  - <dir>/cad_occ.xao                   : the post-BOP XAO compound

Optionally launches gmsh GUI on the XAO (--gui).

Scenes (--scene <name>):
  cap            4x4 slab + 2x4 cap on top of left half
  disc-cap       disc r=1 + cap covering upper half
  annulus-cap    annulus r=[0.4,1] + cap covering upper half
  overlap-2-caps slab + 2 overlapping top neighbours
  simple-slab    single 2x2 slab, n_layers=2 (lateral conformality test)
  custom         build from a JSON config (--config path)

Usage:
  python scripts/inspect_phase3_geometry.py --scene annulus-cap
  python scripts/inspect_phase3_geometry.py --scene disc-cap --export-step /tmp/disc
  python scripts/inspect_phase3_geometry.py --scene annulus-cap --gui
"""

from __future__ import annotations

import argparse
import os
import sys
from math import cos, pi, sin
from pathlib import Path

from shapely.geometry import Polygon

# Force Phase 3 on for inspection.
os.environ["_MESHWELL_FORCE_PHASE3"] = "1"
from unittest.mock import patch

_phase3_patcher = patch("meshwell.structured.phantom._USE_DISCRETE_COHORT_MESH", True)
_phase3_patcher.start()

from meshwell.polyprism import PolyPrism  # noqa: E402
from meshwell.structured import StructuredExtrusionResolutionSpec  # noqa: E402
from meshwell.structured.cohort_envelope import (  # noqa: E402
    assemble_cohort_envelope_solid,
    build_cohort_envelope,
)
from meshwell.structured.plan import build_plan  # noqa: E402


def _disc(cx=0.0, cy=0.0, r=1.0, n=48):
    return Polygon(
        [(cx + r * cos(2 * pi * i / n), cy + r * sin(2 * pi * i / n)) for i in range(n)]
    )


def _annulus(r_outer=1.0, r_inner=0.4, n=48):
    outer = [
        (r_outer * cos(2 * pi * i / n), r_outer * sin(2 * pi * i / n)) for i in range(n)
    ]
    inner = [
        (r_inner * cos(2 * pi * i / n), r_inner * sin(2 * pi * i / n)) for i in range(n)
    ]
    return Polygon(outer, [inner])


def _box(x0, y0, x1, y1):
    return Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


SCENES = {
    "cap": [
        PolyPrism(
            polygons=_box(0, 0, 4, 4),
            buffers={0.0: 0.0, 1.0: 0.0},
            structured=True,
            resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
            physical_name="slab",
        ),
        PolyPrism(
            polygons=_box(0, 0, 2, 4),
            buffers={1.0: 0.0, 2.0: 0.0},
            physical_name="cap",
        ),
    ],
    "disc-cap": [
        PolyPrism(
            polygons=_disc(n=48),
            buffers={0.0: 0.0, 1.0: 0.0},
            structured=True,
            resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
            identify_arcs=True,
            min_arc_points=4,
            arc_tolerance=1e-3,
            physical_name="disc",
            mesh_order=1.0,
        ),
        PolyPrism(
            polygons=_box(-2, 0, 2, 2),
            buffers={1.0: 0.0, 2.0: 0.0},
            physical_name="cap",
            mesh_order=2.0,
        ),
    ],
    "annulus-cap": [
        PolyPrism(
            polygons=_annulus(),
            buffers={0.0: 0.0, 1.0: 0.0},
            structured=True,
            resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
            identify_arcs=True,
            min_arc_points=4,
            arc_tolerance=1e-3,
            physical_name="ann",
            mesh_order=1.0,
        ),
        PolyPrism(
            polygons=_box(-2, 0, 2, 2),
            buffers={1.0: 0.0, 2.0: 0.0},
            physical_name="cap",
            mesh_order=2.0,
        ),
    ],
    "overlap-2-caps": [
        PolyPrism(
            polygons=_box(0, 0, 4, 4),
            buffers={0.0: 0.0, 1.0: 0.0},
            structured=True,
            resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
            physical_name="slab",
        ),
        PolyPrism(
            polygons=_box(0, 0, 3, 4),
            buffers={1.0: 0.0, 2.0: 0.0},
            physical_name="a",
        ),
        PolyPrism(
            polygons=_box(1, 0, 3, 4),
            buffers={1.0: 0.0, 2.0: 0.0},
            physical_name="b",
        ),
    ],
    "simple-slab": [
        PolyPrism(
            polygons=_box(0, 0, 2, 2),
            buffers={0.0: 0.0, 1.0: 0.0},
            structured=True,
            resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
            physical_name="slab",
        ),
    ],
    # Two stacked structured slabs sharing a 4x4 footprint, with an
    # unstructured cap above layer2 covering x in [0,2]. Builds ONE
    # cohort spanning z in [0, 2]. Per-cohort arrangement unification
    # means BOTH layers share the same 2-piece partition at x=2 —
    # the cap's chord projects through the entire cohort. Internal
    # horizontal interface at z=1 between layer1 top and layer2 bot
    # gets stamped as a discrete 2D entity at mesh time.
    "stacked-cap-features": [
        PolyPrism(
            polygons=_box(0, 0, 4, 4),
            buffers={0.0: 0.0, 1.0: 0.0},
            structured=True,
            resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
            physical_name="layer1",
            mesh_order=2.0,
        ),
        PolyPrism(
            polygons=_box(0, 0, 4, 4),
            buffers={1.0: 0.0, 2.0: 0.0},
            structured=True,
            resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
            physical_name="layer2",
            mesh_order=2.0,
        ),
        PolyPrism(
            polygons=_box(0, 0, 2, 4),
            buffers={2.0: 0.0, 3.0: 0.0},
            physical_name="cap",
        ),
    ],
    # Three stacked structured slabs sharing a 4x4 footprint, with a
    # cap ABOVE layer3 (chord at x=2) AND a "pad" UNDER layer1 (chord
    # at x=1). ONE cohort spanning z in [0, 3]. Per-cohort arrangement
    # unification: ALL three layers share the same 3-piece partition
    # at x in {1, 2}. Two internal horizontal interfaces at z=1 and
    # z=2; six internal vertical interfaces (two chord lines x stacked
    # over three z-ranges). Good stress test for Phase 3 discrete
    # entity stamping across multiple z-ranges + multiple piece
    # boundaries.
    "stacked-3-layer-features": [
        PolyPrism(
            polygons=_box(0, 0, 4, 4),
            buffers={0.0: 0.0, 1.0: 0.0},
            structured=True,
            resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
            physical_name="layer1",
            mesh_order=2.0,
        ),
        PolyPrism(
            polygons=_box(0, 0, 4, 4),
            buffers={1.0: 0.0, 2.0: 0.0},
            structured=True,
            resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
            physical_name="layer2",
            mesh_order=2.0,
        ),
        PolyPrism(
            polygons=_box(0, 0, 4, 4),
            buffers={2.0: 0.0, 3.0: 0.0},
            structured=True,
            resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
            physical_name="layer3",
            mesh_order=2.0,
        ),
        # Pad below layer1 (touches z=0)
        PolyPrism(
            polygons=_box(0, 0, 1, 4),
            buffers={-1.0: 0.0, 0.0: 0.0},
            physical_name="pad",
        ),
        # Cap above layer3 (touches z=3)
        PolyPrism(
            polygons=_box(0, 0, 2, 4),
            buffers={3.0: 0.0, 4.0: 0.0},
            physical_name="cap",
        ),
    ],
    # Photonics-like cross-section: stacked structured layers with a
    # circular waveguide at the middle z-range, plus an unstructured
    # cladding above and a keep=False sim-box footprint to fix the
    # cohort's XY extent.
    #
    #   z in [4, 5]: 'air'  — unstructured cladding, keep=True
    #   z in [3, 4]: 'top_oxide' — structured (full footprint)
    #   z in [2, 3]: 'waveguide' (disc r=1) + 'side_oxide' (frame)
    #                — TWO structured slabs at the same z-interval, one
    #                  with arcs, one filling the rest. Both keep=True.
    #   z in [1, 2]: 'bottom_oxide' — structured (full footprint)
    #   z in [0, 1]: 'substrate' — structured (full footprint)
    #
    # All 5 structured layers share the same 6x4 XY footprint via the
    # waveguide/side_oxide co-located pair, so they form ONE cohort.
    # The disc waveguide is the only piece with arc geometry; the
    # planner's arrangement at z in [2, 3] subdivides the cohort into
    # 'disc' and 'around-disc' regions, and that subdivision propagates
    # to every other slab in the cohort (per-cohort arrangement
    # unification — every layer sees a disc-shaped piece + an annular
    # 'around-disc' piece, even though disc only physically exists in
    # the waveguide layer).
    "photonics-cross-section": [
        PolyPrism(
            polygons=_box(-3, -2, 3, 2),
            buffers={0.0: 0.0, 1.0: 0.0},
            structured=True,
            resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
            physical_name="substrate",
            mesh_order=5.0,
        ),
        PolyPrism(
            polygons=_box(-3, -2, 3, 2),
            buffers={1.0: 0.0, 2.0: 0.0},
            structured=True,
            resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
            physical_name="bottom_oxide",
            mesh_order=5.0,
        ),
        # Disc waveguide with arcs (the only arc-bearing slab).
        PolyPrism(
            polygons=_disc(cx=0.0, cy=0.0, r=1.0, n=48),
            buffers={2.0: 0.0, 3.0: 0.0},
            structured=True,
            resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
            identify_arcs=True,
            min_arc_points=4,
            arc_tolerance=1e-3,
            physical_name="waveguide",
            mesh_order=1.0,
        ),
        # Filler oxide at the same z as the waveguide (the disc carves
        # it). Lower priority so disc wins.
        PolyPrism(
            polygons=_box(-3, -2, 3, 2),
            buffers={2.0: 0.0, 3.0: 0.0},
            structured=True,
            resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
            physical_name="side_oxide",
            mesh_order=5.0,
        ),
        PolyPrism(
            polygons=_box(-3, -2, 3, 2),
            buffers={3.0: 0.0, 4.0: 0.0},
            structured=True,
            resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
            physical_name="top_oxide",
            mesh_order=5.0,
        ),
        # Unstructured air on top.
        PolyPrism(
            polygons=_box(-3, -2, 3, 2),
            buffers={4.0: 0.0, 5.0: 0.0},
            physical_name="air",
            mesh_order=5.0,
        ),
    ],
}


def _dump_plan(plan):
    print("=== plan.slabs ===")
    for i, s in enumerate(plan.slabs):
        print(
            f"  slab[{i}]: src={s.source_index} name={s.physical_name}"
            f" z=[{s.zlo},{s.zhi}] cohort={s.component_index}"
            f" pieces={len(s.face_partition)}"
            f" identify_arcs={s.identify_arcs}"
        )
        for j, piece in enumerate(s.face_partition):
            n_int = len(piece.interiors)
            tag = f" [hole x{n_int}]" if n_int else ""
            print(f"    piece[{j}]: bounds={piece.bounds} area={piece.area:.3f}{tag}")
        if s.face_partition_edges is not None:
            for j, edges in enumerate(s.face_partition_edges):
                print(f"    piece[{j}] edges: {edges}")

    print()
    print("=== plan.arrangements ===")
    for cidx, arr in plan.arrangements.items():
        print(f"  cohort {cidx}: {len(arr.edges)} edges, {len(arr.faces)} faces")
        for e in arr.edges:
            arc = " [ARC]" if e.circle else ""
            n_v = len(e.vertices)
            head = e.vertices[0]
            tail = e.vertices[-1]
            print(f"    edge {e.edge_id}: v0={head} vN={tail} n_vertices={n_v}{arc}")
        for f in arr.faces:
            print(
                f"    face {f.face_id}: boundary={f.boundary} polygon_bounds={f.polygon.bounds}"
            )


def _dump_env(env):
    print()
    print("=== env.outline_xy_to_corner_id ===")
    for xy, cid in sorted(env.outline_xy_to_corner_id.items(), key=lambda kv: kv[1]):
        print(f"  cid={cid}: xy=({xy[0]:.6f}, {xy[1]:.6f})")

    print()
    print(
        f"=== env.horizontal_edges ({len(env.horizontal_edges)} entries, keyed by (z, edge_id)) ==="
    )
    by_z: dict[float, list[int]] = {}
    for z, eid in env.horizontal_edges.keys():  # noqa: SIM118
        by_z.setdefault(z, []).append(eid)
    for z in sorted(by_z):
        print(f"  z={z}: edges {sorted(by_z[z])}")

    print()
    print(f"=== env.lateral_faces ({len(env.lateral_faces)} entries) ===")
    for (sidx, eid), faces in sorted(env.lateral_faces.items()):
        print(f"  (slab={sidx}, edge={eid}): {len(faces)} sub-face(s)")

    print()
    print(
        f"=== env.bottom_sub_faces ({len(env.bottom_sub_faces)}) / "
        f"env.top_sub_faces ({len(env.top_sub_faces)}) ==="
    )
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib

    def _fk_sort(kv):
        return (kv[0].slab_index, kv[0].side, kv[0].piece_index)

    for fk, face in sorted(env.bottom_sub_faces.items(), key=_fk_sort):
        bbox = Bnd_Box()
        BRepBndLib.Add_s(face, bbox)
        if not bbox.IsVoid():
            xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
            print(
                f"  bot  {fk}: bbox=({xmin:.3f},{ymin:.3f},{zmin:.3f})-({xmax:.3f},{ymax:.3f},{zmax:.3f})"
            )
        else:
            print(f"  bot  {fk}: void bbox")
    for fk, face in sorted(env.top_sub_faces.items(), key=_fk_sort):
        bbox = Bnd_Box()
        BRepBndLib.Add_s(face, bbox)
        if not bbox.IsVoid():
            xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
            print(
                f"  top  {fk}: bbox=({xmin:.3f},{ymin:.3f},{zmin:.3f})-({xmax:.3f},{ymax:.3f},{zmax:.3f})"
            )
        else:
            print(f"  top  {fk}: void bbox")

    print()
    print(
        f"=== env.multi_piece_shares_edges_by_slab: {env.multi_piece_shares_edges_by_slab}"
    )


def _dump_solid(solid):
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_SHELL, TopAbs_VERTEX
    from OCP.TopExp import TopExp_Explorer

    def _count(shape, kind):
        n = 0
        exp = TopExp_Explorer(shape, kind)
        while exp.More():
            n += 1
            exp.Next()
        return n

    print()
    print(
        f"=== cohort solid: shells={_count(solid, TopAbs_SHELL)} "
        f"faces={_count(solid, TopAbs_FACE)} "
        f"edges={_count(solid, TopAbs_EDGE)} "
        f"vertices={_count(solid, TopAbs_VERTEX)}"
    )


def _export_step_brep(env, solid, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    from OCP.BRepTools import BRepTools
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer

    w = STEPControl_Writer()
    w.Transfer(solid, STEPControl_AsIs)
    w.Write(str(outdir / "cohort_envelope.step"))
    print(f"  wrote {outdir/'cohort_envelope.step'}")

    for fk, face in env.bottom_sub_faces.items():
        path = outdir / f"piece_{fk.piece_index}_bot_slab{fk.slab_index}.brep"
        BRepTools.Write_s(face, str(path))
        print(f"  wrote {path}")
    for fk, face in env.top_sub_faces.items():
        path = outdir / f"piece_{fk.piece_index}_top_slab{fk.slab_index}.brep"
        BRepTools.Write_s(face, str(path))
        print(f"  wrote {path}")


def _run_cad_occ_and_export_xao(entities, outdir: Path):
    """Run the full cad_occ pipeline and dump the XAO compound."""
    from meshwell.orchestrator import generate_mesh

    outdir.mkdir(parents=True, exist_ok=True)
    out_msh = outdir / "cad_occ.msh"
    try:
        generate_mesh(
            entities,
            dim=3,
            output_mesh=str(out_msh),
            default_characteristic_length=0.4,
        )
        print(f"  wrote {out_msh}")
    except Exception as exc:
        print(f"  generate_mesh failed: {exc}")

    # Look for any .xao that orchestrator might have written into a tmpdir.
    # (cad_occ writes to a tempdir; copy if discoverable.)
    return out_msh


def _launch_gmsh_gui(msh_path: Path):
    import gmsh

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.open(str(msh_path))
    gmsh.fltk.run()
    gmsh.finalize()


def main(argv):
    """Parse args and run the inspection pipeline for one scene."""
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "--scene",
        default="annulus-cap",
        choices=sorted(SCENES.keys()),
        help="which scene to build",
    )
    parser.add_argument(
        "--export-step",
        default=None,
        help="directory to write STEP/BREP/XAO files",
    )
    parser.add_argument(
        "--run-cad-occ",
        action="store_true",
        help="also run the full generate_mesh pipeline (writes .msh)",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="launch gmsh GUI on the resulting mesh (requires --run-cad-occ)",
    )
    args = parser.parse_args(argv)

    entities = SCENES[args.scene]
    print(f"Scene: {args.scene} ({len(entities)} entities)")
    print()

    plan = build_plan(entities)
    _dump_plan(plan)

    if not plan.arrangements:
        print("(no arrangements — empty plan, exiting)")
        return 0

    for cidx in sorted(plan.arrangements.keys()):
        print()
        print(f"### Cohort {cidx} ###")
        env = build_cohort_envelope(plan, component_index=cidx)
        _dump_env(env)
        try:
            solid = assemble_cohort_envelope_solid(env)
            _dump_solid(solid)
        except Exception as exc:
            print(f"  assemble_cohort_envelope_solid failed: {exc}")
            continue

        if args.export_step:
            outdir = Path(args.export_step) / f"cohort_{cidx}"
            _export_step_brep(env, solid, outdir)

    if args.run_cad_occ:
        import tempfile

        outdir = Path(args.export_step or tempfile.mkdtemp(prefix="inspect_phase3_"))
        msh = _run_cad_occ_and_export_xao(entities, outdir)
        if args.gui and msh.exists():
            _launch_gmsh_gui(msh)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
