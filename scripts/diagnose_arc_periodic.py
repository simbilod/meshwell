"""Curve-level diagnostic for structured-slab periodic failures.

For each slab in the scene, dump:
  - Bottom and top OCC face tags + bboxes
  - Each bounding curve: tag, bbox, length, whether ``setTransfiniteCurve``
    was applied (and to how many nodes), and after ``mesh.generate(2)``
    runs, the actual 1D node count.
  - A pairwise bottom->top match by xy projection: which top curve does
    each bottom curve correspond to under z-translation, and do they have
    matching node counts?

Run:
    python diagnose_arc_periodic.py isolated
    python diagnose_arc_periodic.py multi
    python diagnose_arc_periodic.py wires

The goal is to identify whether mismatches at ``setPeriodic`` time are
caused by:
  (a) bottom/top OCC faces having a different *number* of bounding curves
      (i.e., the BOP fragmentation produced asymmetric topology),
  (b) the bounding curves are mirror-symmetric in count but a vertical-seam
      curve has no transfinite hint on one side while its twin does,
  (c) arc curves on bottom and top get tessellated by gmsh with different
      sizes (size field reading the curve differently), or
  (d) curves are TShape-shared between bottom and top (impossible since
      they're at different z, so this should never happen, but check).
"""
from __future__ import annotations

import math
import sys
from collections import defaultdict
from pathlib import Path

import gmsh
from shapely.geometry import Polygon

from meshwell import structured_polyprism as sp
from meshwell.orchestrator import generate_mesh
from meshwell.polyprism import PolyPrism

# ---------------------------------------------------------------------------
# Scene builders
# ---------------------------------------------------------------------------


def _annulus(cx: float, cy: float, ro: float, ri: float, n: int = 48) -> Polygon:
    outer = [
        (
            cx + ro * math.cos(2 * math.pi * i / n),
            cy + ro * math.sin(2 * math.pi * i / n),
        )
        for i in range(n)
    ]
    inner = [
        (
            cx + ri * math.cos(2 * math.pi * i / n),
            cy + ri * math.sin(2 * math.pi * i / n),
        )
        for i in range(n)
    ]
    return Polygon(outer, holes=[inner])


def _square(x0, x1, y0, y1) -> Polygon:
    return Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


def scene_isolated_ring() -> list:
    return [
        PolyPrism(
            polygons=_annulus(0.0, 0.0, 1.2, 0.6, n=48),
            buffers={0.0: 0.0, 0.6: 0.0},
            n_layers=[4],
            physical_name="ring",
            mesh_order=1,
            identify_arcs=True,
        )
    ]


def scene_multi_ring() -> list:
    return [
        PolyPrism(
            polygons=_annulus(4.0 + 6.0 * k, 0.0, 1.2, 0.6, n=48),
            buffers={0.0: 0.0, 0.6: 0.0},
            n_layers=[4],
            physical_name=f"ring_{k}",
            mesh_order=1,
            identify_arcs=True,
        )
        for k in range(3)
    ]


def scene_wires() -> list:
    """Rectangular wire grid, no arcs. Reproduces the 6-vs-9 failure."""
    ents: list = []
    for ix in range(3):
        for iy in range(2):
            x0 = 1.0 + 4.0 * ix
            y0 = 1.0 + 1.5 * iy
            ents.append(
                PolyPrism(
                    polygons=_square(x0, x0 + 1.2, y0, y0 + 1.0),
                    buffers={0.4: 0.0, 1.0: 0.0},
                    n_layers=[4],
                    physical_name=f"wire_{ix}_{iy}",
                    mesh_order=2,
                )
            )
    return ents


def scene_wires_with_neighbours() -> list:
    """Wires (structured) + cladding below + encapsulant above.

    Reproduces the bench failure mode where the encapsulant's bottom and
    the cladding's top share planes with the wire tops/bottoms respectively.
    cad_occ fragments these touching faces -- our hypothesis is that the
    fragmentation breaks bottom/top symmetry on at least one slab.
    """
    ents: list = [
        PolyPrism(
            polygons=_square(-1, 21, -1, 4),
            buffers={0.0: 0.0, 0.4: 0.0},
            physical_name="cladding",
            mesh_order=20,
        ),
        PolyPrism(
            polygons=_square(-1, 21, -1, 4),
            buffers={1.0: 0.0, 1.5: 0.0},
            physical_name="encapsulant",
            mesh_order=12,
        ),
    ]
    for ix in range(3):
        for iy in range(2):
            x0 = 1.0 + 4.0 * ix
            y0 = 1.0 + 1.5 * iy
            ents.append(
                PolyPrism(
                    polygons=_square(x0, x0 + 1.2, y0, y0 + 1.0),
                    buffers={0.4: 0.0, 1.0: 0.0},
                    n_layers=[4],
                    physical_name=f"wire_{ix}_{iy}",
                    mesh_order=2,
                )
            )
    return ents


def scene_rings_with_neighbours() -> list:
    """Arc rings + cladding below + encapsulant above (the actual bench
    failure pattern for struct_ring_*).
    """
    ents: list = [
        PolyPrism(
            polygons=_square(-2, 22, -3, 3),
            buffers={0.0: 0.0, 0.4: 0.0},
            physical_name="cladding",
            mesh_order=20,
        ),
        PolyPrism(
            polygons=_square(-2, 22, -3, 3),
            buffers={0.6: 0.0, 1.1: 0.0},
            physical_name="encapsulant",
            mesh_order=12,
        ),
    ]
    for k in range(3):
        ents.append(
            PolyPrism(
                polygons=_annulus(4.0 + 6.0 * k, 0.0, 1.2, 0.6, n=48),
                buffers={0.4: 0.0, 0.6: 0.0},
                n_layers=[4],
                physical_name=f"ring_{k}",
                mesh_order=1,
                identify_arcs=True,
            )
        )
    return ents


def scene_bench() -> list:
    """Full bench scene -- the one that triggered the failures."""

    def _disk(cx, cy, r, n=24):
        return Polygon(
            [
                (
                    cx + r * math.cos(2 * math.pi * i / n),
                    cy + r * math.sin(2 * math.pi * i / n),
                )
                for i in range(n)
            ]
        )

    ents: list = [
        PolyPrism(
            polygons=_square(-1, 41, -1, 17),
            buffers={0.0: 0.0, 0.4: 0.0},
            physical_name="cladding",
            mesh_order=20,
        ),
        PolyPrism(
            polygons=_square(-1, 41, 8.0, 17.0),
            buffers={0.4: 0.0, 1.0: 0.0},
            physical_name="filler_top",
            mesh_order=15,
        ),
        PolyPrism(
            polygons=_square(-1, 41, -1, 17),
            buffers={1.0: 0.0, 1.5: 0.0},
            physical_name="encapsulant",
            mesh_order=12,
        ),
    ]
    for ix in range(6):
        for iy in range(4):
            x0 = 1.0 + 4.0 * ix
            y0 = 1.0 + 1.5 * iy
            ents.append(
                PolyPrism(
                    polygons=_square(x0, x0 + 1.2, y0, y0 + 1.0),
                    buffers={0.4: 0.0, 1.0: 0.0},
                    n_layers=[4],
                    physical_name=f"wire_{ix}_{iy}",
                    mesh_order=2,
                )
            )
    for k in range(4):
        x0 = 1.0 + 6.0 * k
        ents.append(
            PolyPrism(
                polygons=_square(x0, x0 + 1.5, 14.0, 15.5),
                buffers={0.4: 0.0, 0.6: 0.0, 0.8: 0.0, 1.0: 0.0},
                n_layers=[2, 3, 4],
                physical_name=f"stack_{k}",
                mesh_order=3,
            )
        )
    for k in range(3):
        ents.append(
            PolyPrism(
                polygons=_annulus(4.0 + 6.0 * k, 11.5, 1.2, 0.6, n=48),
                buffers={0.4: 0.0, 1.0: 0.0},
                n_layers=[4],
                physical_name=f"struct_ring_{k}",
                mesh_order=2,
                identify_arcs=True,
            )
        )
    for k in range(3):
        ents.append(
            PolyPrism(
                polygons=_annulus(4.0 + 6.0 * k, 9.5, 0.9, 0.4, n=48),
                buffers={0.4: 0.0, 1.0: 0.0},
                physical_name=f"ring_{k}",
                mesh_order=4,
                identify_arcs=True,
            )
        )
    for k in range(8):
        ents.append(
            PolyPrism(
                polygons=_disk(2.5 + 4.0 * k, 12.0, 0.5, n=24),
                buffers={0.0: 0.0, 1.5: 0.0},
                physical_name=f"pillar_{k}",
                mesh_order=18,
            )
        )
    return ents


SCENES = {
    "isolated": scene_isolated_ring,
    "multi": scene_multi_ring,
    "wires": scene_wires,
    "wires_neigh": scene_wires_with_neighbours,
    "rings_neigh": scene_rings_with_neighbours,
    "bench": scene_bench,
}


# ---------------------------------------------------------------------------
# Instrumentation
# ---------------------------------------------------------------------------

# (curve_tag, n_nodes) recorded by every setTransfiniteCurve call.
TRANSFINITE_CURVES: dict[int, int] = {}
CAPTURED_SLABS: list = []


def _install_patches() -> None:
    """Capture slabs list and transfinite hints; instrument generate(2)."""
    orig_apply = sp.apply_structured_slabs

    def patched_apply(mm, slabs):
        CAPTURED_SLABS.clear()
        CAPTURED_SLABS.extend(slabs)
        return orig_apply(mm, slabs)

    sp.apply_structured_slabs = patched_apply

    orig_set_tc = gmsh.model.mesh.setTransfiniteCurve

    def patched_set_tc(tag, n_nodes, *a, **kw):
        TRANSFINITE_CURVES[int(tag)] = int(n_nodes)
        return orig_set_tc(tag, n_nodes, *a, **kw)

    gmsh.model.mesh.setTransfiniteCurve = patched_set_tc

    orig_generate = gmsh.model.mesh.generate

    def patched_generate(dim):
        if dim == 2:
            _dump_pre_mesh()
        try:
            result = orig_generate(dim)
        except Exception as exc:
            print(f"\n>>> mesh.generate({dim}) raised: {exc}")
            if dim == 2:
                _dump_post_mesh()
            raise
        if dim == 2:
            _dump_post_mesh()
        return result

    gmsh.model.mesh.generate = patched_generate


# ---------------------------------------------------------------------------
# Dump routines
# ---------------------------------------------------------------------------


def _find_faces_at_z(target_z: float, tol: float = 1e-3) -> list[int]:
    """Return tags of horizontal 2D OCC faces sitting at z=target_z."""
    out = []
    for dim, ftag in gmsh.model.occ.getEntities(2):
        if dim != 2:
            continue
        try:
            bb = gmsh.model.occ.getBoundingBox(2, ftag)
        except Exception:
            continue
        if abs(bb[2] - bb[5]) > tol:
            continue  # not horizontal
        z_face = 0.5 * (bb[2] + bb[5])
        if abs(z_face - target_z) > tol:
            continue
        out.append(ftag)
    return out


def _face_curves(ftag: int) -> list[int]:
    try:
        bnd = gmsh.model.getBoundary([(2, ftag)], oriented=False, recursive=False)
    except Exception:
        return []
    return [t for d, t in bnd if d == 1]


def _curve_bbox(ctag: int):
    try:
        return gmsh.model.occ.getBoundingBox(1, ctag)
    except Exception:
        return None


def _curve_xy_signature(ctag: int, tol: float = 1e-6) -> tuple:
    """xy-projected bbox key for matching bottom curve to top curve."""
    bb = _curve_bbox(ctag)
    if bb is None:
        return ()
    return (
        round(bb[0] / tol),
        round(bb[1] / tol),
        round(bb[3] / tol),
        round(bb[4] / tol),
    )


def _dump_pre_mesh() -> None:
    """Called right before mesh.generate(2). OCC topology is final;
    1D node counts don't exist yet, but transfinite hints are recorded.
    """
    print("\n" + "=" * 80)
    print("PRE-MESH state (curves + transfinite hints; no node counts yet)")
    print("=" * 80)
    for slab in CAPTURED_SLABS:
        bot_faces = _find_faces_at_z(slab.zlo)
        top_faces = _find_faces_at_z(slab.zhi)
        # Restrict to faces likely owned by this slab (xy intersects footprint).
        bot_faces = _filter_faces_to_footprint(bot_faces, slab)
        top_faces = _filter_faces_to_footprint(top_faces, slab)
        print(
            f"\nSLAB {slab.physical_name} z=[{slab.zlo}, {slab.zhi}] "
            f"n_layers={slab.n_layers}"
        )
        print(f"  bottom face tags ({len(bot_faces)}): {bot_faces}")
        print(f"  top    face tags ({len(top_faces)}): {top_faces}")
        bot_curves = _aggregate_curves(bot_faces)
        top_curves = _aggregate_curves(top_faces)
        print(f"  bottom bounding curves ({len(bot_curves)}): {sorted(bot_curves)}")
        print(f"  top    bounding curves ({len(top_curves)}): {sorted(top_curves)}")
        # transfinite-hint coverage
        bot_tf = sum(1 for c in bot_curves if c in TRANSFINITE_CURVES)
        top_tf = sum(1 for c in top_curves if c in TRANSFINITE_CURVES)
        print(
            f"  transfinite hints applied: bottom {bot_tf}/{len(bot_curves)}, "
            f"top {top_tf}/{len(top_curves)}"
        )
        # Vertical-seam curves (bounded by both bottom and top faces)
        shared = bot_curves & top_curves
        print(
            f"  curves SHARED between bottom and top faces "
            f"(TShape sharing): {len(shared)}: {sorted(shared)}"
        )
        # xy-projected matching
        bot_sig = {c: _curve_xy_signature(c) for c in bot_curves}
        top_sig = {c: _curve_xy_signature(c) for c in top_curves}
        from collections import Counter as _C

        bot_sigs = _C(bot_sig.values())
        top_sigs = _C(top_sig.values())
        matched = sum((bot_sigs & top_sigs).values())
        unmatched_bot = sum((bot_sigs - top_sigs).values())
        unmatched_top = sum((top_sigs - bot_sigs).values())
        print(
            f"  xy-projection match: {matched} matched, "
            f"{unmatched_bot} bottom-only, {unmatched_top} top-only"
        )
        if unmatched_bot or unmatched_top:
            for c, s in sorted(bot_sig.items()):
                if top_sigs.get(s, 0) == 0:
                    bb = _curve_bbox(c)
                    print(f"    bottom-only curve {c}: bbox={bb}")
            for c, s in sorted(top_sig.items()):
                if bot_sigs.get(s, 0) == 0:
                    bb = _curve_bbox(c)
                    print(f"    top-only    curve {c}: bbox={bb}")


def _filter_faces_to_footprint(faces: list[int], slab) -> list[int]:
    """Keep faces whose xy-bbox overlaps slab.footprint substantially.

    Use intersection area (not bbox-center containment) so that annular
    footprints whose bbox center sits inside the hole still match.
    """
    from shapely.geometry import box as sh_box

    fp = slab.footprint
    fp_area = fp.area
    if fp_area <= 0:
        return list(faces)
    kept = []
    for f in faces:
        bb = gmsh.model.occ.getBoundingBox(2, f)
        face_box = sh_box(bb[0], bb[1], bb[3], bb[4])
        face_area = face_box.area
        if face_area <= 0:
            continue
        try:
            inter = fp.intersection(face_box).area
        except Exception:
            kept.append(f)
            continue
        # Coverage relative to the face bbox -- mirrors
        # _find_all_occ_faces_for_slab's 50% rule.
        if inter / face_area >= 0.5:
            kept.append(f)
    return kept


def _aggregate_curves(faces: list[int]) -> set[int]:
    out: set[int] = set()
    for f in faces:
        out.update(_face_curves(f))
    return out


def _dump_post_mesh() -> None:
    """Called after (or upon failure of) mesh.generate(2). Node counts
    on 1D curves are now populated.
    """
    print("\n" + "=" * 80)
    print("POST-MESH state (1D node counts per curve)")
    print("=" * 80)
    for slab in CAPTURED_SLABS:
        bot_faces = _filter_faces_to_footprint(_find_faces_at_z(slab.zlo), slab)
        top_faces = _filter_faces_to_footprint(_find_faces_at_z(slab.zhi), slab)
        bot_curves = _aggregate_curves(bot_faces)
        top_curves = _aggregate_curves(top_faces)
        print(f"\nSLAB {slab.physical_name} z=[{slab.zlo}, {slab.zhi}]")
        bot_nodes = _curve_node_counts(bot_curves)
        top_nodes = _curve_node_counts(top_curves)
        bot_total = sum(bot_nodes.values())
        top_total = sum(top_nodes.values())
        print(
            f"  bottom total nodes across {len(bot_curves)} curves: "
            f"{bot_total} {dict(sorted(bot_nodes.items()))}"
        )
        print(
            f"  top    total nodes across {len(top_curves)} curves: "
            f"{top_total} {dict(sorted(top_nodes.items()))}"
        )
        if bot_total != top_total:
            print(f"  >>> NODE COUNT MISMATCH: bottom {bot_total} vs top {top_total}")
        # Pair by xy signature; flag node-count mismatches.
        bot_sig = {c: _curve_xy_signature(c) for c in bot_curves}
        top_by_sig = defaultdict(list)
        for c in top_curves:
            top_by_sig[_curve_xy_signature(c)].append(c)
        for bc, sig in sorted(bot_sig.items()):
            tcands = top_by_sig.get(sig, [])
            if not tcands:
                print(f"  curve {bc} ({bot_nodes[bc]} nodes): NO top twin")
                continue
            tc = tcands[0]
            mark = "" if bot_nodes[bc] == top_nodes[tc] else "  <-- MISMATCH"
            print(
                f"  curve {bc} (bot, {bot_nodes[bc]} nodes) <-> "
                f"curve {tc} (top, {top_nodes[tc]} nodes){mark}"
            )


def _curve_node_counts(curves: set[int]) -> dict[int, int]:
    out: dict[int, int] = {}
    for c in curves:
        try:
            tags, _coords, _params = gmsh.model.mesh.getNodes(
                1, c, includeBoundary=True
            )
        except Exception:
            out[c] = -1
            continue
        out[c] = len(tags)
    return out


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def main(scene_name: str) -> None:
    if scene_name not in SCENES:
        print(f"Unknown scene: {scene_name!r}. Choices: {list(SCENES)}")
        sys.exit(2)
    entities = SCENES[scene_name]()
    print(f"Scene '{scene_name}': {len(entities)} entities")
    _install_patches()
    # Also: dump pre-CAD partition lengths per slab to see whether the
    # cascade itself is asymmetric, or whether asymmetry is introduced
    # by BOP fragmentation against external neighbours.
    orig_resolve = sp.resolve_structured_slabs

    def _patched_resolve(entities_list):
        slabs = orig_resolve(entities_list)
        print("\n" + "=" * 80)
        print("PRE-CAD partition state (per slab, from resolve_structured_slabs)")
        print("=" * 80)
        for s in slabs:
            fp = getattr(s, "face_partition", None)
            n = 0 if fp is None else len(fp)
            print(
                f"  slab {s.physical_name} z=[{s.zlo}, {s.zhi}] "
                f"footprint_area={s.footprint.area:.3f} "
                f"face_partition pieces={n}"
            )
        return slabs

    sp.resolve_structured_slabs = _patched_resolve
    out = Path(f"diagnose_{scene_name}.msh")
    try:
        generate_mesh(
            entities=entities, dim=3, output_mesh=out, default_characteristic_length=0.3
        )
    except Exception as exc:
        print(f"\n>>> generate_mesh raised: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "isolated")
