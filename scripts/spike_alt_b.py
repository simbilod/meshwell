"""Spike runner for Alt B: freeze cohort lateral mesh before generate(2).

Monkey-patches the orchestrator to swap apply_lateral_transfinite_hints
for alt_b_pre_2d. Runs the complex stress scene and compares wedge /
group counts vs the transfinite baseline and the manual-spike approach.

Usage:
    PYTHONPATH=/home/simbil/Github/meshwell_structured python scripts/spike_alt_b.py
"""
from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon

import meshio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _circle(cx, cy, r, n=48):
    a = np.linspace(0, 2 * np.pi, n + 1)[:-1]
    return Polygon([(cx + r * np.cos(t), cy + r * np.sin(t)) for t in a])


def _annulus(cx, cy, r_out, r_in, n=48):
    return Polygon(
        _circle(cx, cy, r_out, n).exterior.coords,
        holes=[_circle(cx, cy, r_in, n).exterior.coords],
    )


def _rect(x1, y1, x2, y2):
    return Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])


def build_complex_scene():
    """Return the 10-entity complex stress scene used by the test."""
    from meshwell.polyprism import PolyPrism

    SQUARE_A = _rect(-5, -5, 5, 5)
    CIRCLE_A = _circle(0, 8, 2)
    RECT_HOLE_A = Polygon(
        _rect(-9, -9, -3, -3).exterior.coords,
        holes=[_rect(-7, -7, -5, -5).exterior.coords],
    )
    CIRCLE_B = _circle(0, 0, 3)
    ANNULUS_B = _annulus(0, 8, 2.5, 1.2)
    HEX_C = Polygon(
        [(2 * np.cos(a), 2 * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 7)[:-1]]
    )
    VOID_C = _circle(0, 0, 0.5)
    BIG_BASE = _rect(-15, -15, 15, 15)
    HOLE_BASE = _circle(0, 0, 1.0)
    BIG_CAP = _rect(-15, -15, 15, 15)
    CAP_ARCH = _circle(3, 3, 2)

    ARC = dict(identify_arcs=True)
    return [
        PolyPrism(
            SQUARE_A,
            {0.0: 0.0, 1.0: 0.0},
            physical_name="A_square",
            structured=True,
            mesh_order=3.0,
            **ARC,
        ),
        PolyPrism(
            CIRCLE_A,
            {0.0: 0.0, 1.0: 0.0},
            physical_name="A_circle",
            structured=True,
            mesh_order=3.0,
            **ARC,
        ),
        PolyPrism(
            RECT_HOLE_A,
            {0.0: 0.0, 1.0: 0.0},
            physical_name="A_recth",
            structured=True,
            mesh_order=3.0,
            **ARC,
        ),
        PolyPrism(
            CIRCLE_B,
            {1.0: 0.0, 2.0: 0.0},
            physical_name="B_circle",
            structured=True,
            mesh_order=3.0,
            **ARC,
        ),
        PolyPrism(
            ANNULUS_B,
            {1.0: 0.0, 2.0: 0.0},
            physical_name="B_annulus",
            structured=True,
            mesh_order=3.0,
            **ARC,
        ),
        PolyPrism(
            HEX_C,
            {2.0: 0.0, 3.0: 0.0},
            physical_name="C_hex",
            structured=True,
            mesh_order=3.0,
            **ARC,
        ),
        PolyPrism(
            VOID_C,
            {2.0: 0.0, 3.0: 0.0},
            physical_name="C_void",
            structured=True,
            mesh_order=1.0,
            mesh_bool=False,
            **ARC,
        ),
        PolyPrism(
            Polygon(BIG_BASE.exterior.coords, holes=[HOLE_BASE.exterior.coords]),
            {-2.0: 0.0, 0.0: 0.0},
            physical_name="base",
            mesh_order=5.0,
            **ARC,
        ),
        PolyPrism(
            BIG_CAP, {3.0: 0.0, 5.0: 0.0}, physical_name="cap", mesh_order=5.0, **ARC
        ),
        PolyPrism(
            CAP_ARCH,
            {3.0: 0.0, 5.0: 0.0},
            physical_name="cap_arch",
            mesh_order=2.0,
            **ARC,
        ),
    ]


def specs():
    """Resolution specs for the structured slabs (n_layers=2 each)."""
    from meshwell.resolution import StructuredExtrusionResolutionSpec

    return {
        n: [StructuredExtrusionResolutionSpec(n_layers=2)]
        for n in (
            "A_square",
            "A_circle",
            "A_recth",
            "B_circle",
            "B_annulus",
            "C_hex",
        )
    }


def run(label: str, mode: str):
    """Mesh and report. mode is one of: baseline, manual, alt_b."""
    import meshwell.orchestrator as orch
    import meshwell.structured.wedge as wedge_mod
    from meshwell.structured.wedge_alt_b_spike import alt_b_pre_2d
    from meshwell.structured.wedge_manual_spike import (
        construct_lateral_quads,
        manual_pre_2d_validate,
    )

    orig_pre_2d = wedge_mod.apply_lateral_transfinite_hints
    orig_stamp = wedge_mod.stamp_wedges

    try:
        if mode == "manual":
            wedge_mod.apply_lateral_transfinite_hints = manual_pre_2d_validate
            orch.apply_lateral_transfinite_hints = manual_pre_2d_validate

            def stamp_with_lateral(
                slab_meta,
                face_tag_by_key,
                sub_solid_tag_by_key,
                resolution_specs=None,
                point_tolerance=1e-3,
            ):
                construct_lateral_quads(
                    slab_meta=slab_meta,
                    face_tag_by_key=face_tag_by_key,
                    resolution_specs=resolution_specs,
                )
                return orig_stamp(
                    slab_meta,
                    face_tag_by_key,
                    sub_solid_tag_by_key,
                    resolution_specs=resolution_specs,
                    point_tolerance=point_tolerance,
                )

            wedge_mod.stamp_wedges = stamp_with_lateral
            orch.stamp_wedges = stamp_with_lateral

        elif mode == "alt_b":

            def alt_b_pre_2d_adapter(slab_meta, face_tag_by_key, resolution_specs=None):
                alt_b_pre_2d(
                    slab_meta=slab_meta,
                    face_tag_by_key=face_tag_by_key,
                    resolution_specs=resolution_specs,
                )

            wedge_mod.apply_lateral_transfinite_hints = alt_b_pre_2d_adapter
            orch.apply_lateral_transfinite_hints = alt_b_pre_2d_adapter
            # stamp_wedges stays as production version

        from meshwell.orchestrator import generate_mesh

        ents = build_complex_scene()
        with tempfile.TemporaryDirectory() as td:
            msh_path = Path(td) / "out.msh"
            t0 = time.perf_counter()
            generate_mesh(
                ents,
                dim=3,
                output_mesh=msh_path,
                default_characteristic_length=0.8,
                resolution_specs=specs(),
            )
            elapsed = time.perf_counter() - t0

            m = meshio.read(msh_path)
            wedges = sum(cb.data.shape[0] for cb in m.cells if cb.type == "wedge")
            tets = sum(cb.data.shape[0] for cb in m.cells if cb.type == "tetra")
            quads = sum(cb.data.shape[0] for cb in m.cells if cb.type == "quad")
            tris = sum(cb.data.shape[0] for cb in m.cells if cb.type == "triangle")
            groups = sorted(m.cell_sets.keys())

            print(
                f"[{label}] wedges={wedges} tets={tets} quads={quads} "
                f"tris={tris} groups={len(groups)} elapsed={elapsed:.2f}s"
            )
            return {
                "wedges": wedges,
                "tets": tets,
                "quads": quads,
                "tris": tris,
                "groups": groups,
                "elapsed": elapsed,
            }
    finally:
        wedge_mod.apply_lateral_transfinite_hints = orig_pre_2d
        orch.apply_lateral_transfinite_hints = orig_pre_2d
        wedge_mod.stamp_wedges = orig_stamp
        orch.stamp_wedges = orig_stamp


if __name__ == "__main__":
    print("\n=== BASELINE (transfinite) ===")
    base = run("baseline", mode="baseline")
    print("\n=== MANUAL SPIKE (clear-and-rebuild) ===")
    manual = run("manual", mode="manual")
    print("\n=== ALT B SPIKE (freeze-before-2D) ===")
    alt_b = run("alt_b", mode="alt_b")

    print("\n=== Comparison ===")
    print(f"{'metric':<10} {'baseline':>10} {'manual':>10} {'alt_b':>10}")
    for k in ("wedges", "tets", "quads", "tris"):
        print(f"{k:<10} {base[k]:>10} {manual[k]:>10} {alt_b[k]:>10}")
    print(
        f"{'elapsed_s':<10} {base['elapsed']:>10.2f} "
        f"{manual['elapsed']:>10.2f} {alt_b['elapsed']:>10.2f}"
    )
    same_groups = set(base["groups"]) == set(manual["groups"]) == set(alt_b["groups"])
    print(f"Group sets equal across all three: {same_groups}")
