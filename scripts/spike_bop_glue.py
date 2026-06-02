r"""Spike for Option C: force BOP to share TShapes via Glue mode.

Hypothesis: if BOP always produces shared TShapes for geometrically
coincident faces, then:
  (a) the AABB fallback in occ_xao_writer becomes dead code, AND
  (b) Option A (gmsh adjacency lookup) becomes viable as a full
      replacement.

Test:
  baseline   — current pipeline (BOPAlgo_GlueOff)
  glue_shift — patched pipeline using BOPAlgo_GlueShift
  glue_full  — patched pipeline using BOPAlgo_GlueFull

For each variant, count:
  - AABB fallback invocations (each time the TShape-identity path
    returned empty but AABB matching had to rescue)
  - Physical group count + correctness vs baseline
  - Wall time end-to-end

Usage:
    PYTHONPATH=/home/simbil/Github/meshwell_structured python \
        scripts/spike_bop_glue.py
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

from OCP.BOPAlgo import BOPAlgo_Builder, BOPAlgo_GlueEnum

# ---------- scene ----------


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
    """Return the 10-entity complex stress scene."""
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


# ---------- monkey-patches ----------


# Save originals so we can restore between runs
_ORIG_BUILDER_INIT = BOPAlgo_Builder.__init__


def install_glue_patch(glue_mode):
    """Patch BOPAlgo_Builder so every instance has SetGlue(glue_mode) applied.

    Pass ``None`` to restore default behaviour.
    """
    import meshwell.cad_occ as cad_occ_mod

    if glue_mode is None:
        cad_occ_mod.BOPAlgo_Builder = BOPAlgo_Builder
        return

    class _GluedBuilder(BOPAlgo_Builder):
        def __init__(self):
            super().__init__()
            self.SetGlue(glue_mode)

    cad_occ_mod.BOPAlgo_Builder = _GluedBuilder


_FALLBACK_COUNTER = {"count": 0, "details": []}


def install_fallback_counter():
    """Patch occ_xao_writer to count AABB fallback invocations."""
    import meshwell.occ_xao_writer as xao_mod

    # Reset counter
    _FALLBACK_COUNTER["count"] = 0
    _FALLBACK_COUNTER["details"].clear()

    # Wrap _aabbs_close to count its True returns from inside the fallback
    original_aabbs_close = xao_mod._aabbs_close
    # We need a different approach — instrument _compute_physical_groups
    # directly by monkey-patching it.
    original_compute = xao_mod._compute_physical_groups

    def patched_compute(entities, *args, **kwargs):
        # Just delegate to original but intercept calls to _aabbs_close
        # by tracking when the fallback block triggers
        # Simpler: hook _shape_aabb itself isn't useful. Instead, look at
        # the _compute output by counting interface groups whose face set
        # would have been empty without fallback.
        # Cheaper: instrument by re-implementing the fallback branch as
        # opt-in via env var. For the spike, just count calls to _aabbs_close.
        return original_compute(entities, *args, **kwargs)

    # Wrap _aabbs_close to count whenever it returns True
    def counting_aabbs_close(b1, b2, tol):
        result = original_aabbs_close(b1, b2, tol)
        if result:
            _FALLBACK_COUNTER["count"] += 1
        return result

    xao_mod._aabbs_close = counting_aabbs_close


# ---------- runner ----------


def run_variant(_label: str, glue_mode):
    """Build + mesh the complex scene. Return metrics."""
    install_fallback_counter()
    install_glue_patch(glue_mode)

    # Lazy import after patch so the module sees the patched BOPAlgo_Builder
    # (note: cad_occ.py imports BOPAlgo_Builder at module level, so the
    # monkey-patch must rebind the symbol on the cad_occ module — done above)
    import importlib

    import meshwell.cad_occ

    importlib.reload(meshwell.cad_occ)
    install_glue_patch(
        glue_mode
    )  # re-apply after reload (BOPAlgo_Builder is rebound by reload)

    # Re-import dependents that snapshot cad_occ symbols
    import meshwell.orchestrator as orch

    importlib.reload(orch)

    from meshwell.orchestrator import generate_mesh

    ents = build_complex_scene()
    with tempfile.TemporaryDirectory() as td:
        msh = Path(td) / "out.msh"
        t0 = time.perf_counter()
        try:
            generate_mesh(
                ents,
                dim=3,
                output_mesh=msh,
                default_characteristic_length=0.8,
                resolution_specs=specs(),
            )
            elapsed = time.perf_counter() - t0
            m = meshio.read(msh)
            wedges = sum(cb.data.shape[0] for cb in m.cells if cb.type == "wedge")
            tets = sum(cb.data.shape[0] for cb in m.cells if cb.type == "tetra")
            quads = sum(cb.data.shape[0] for cb in m.cells if cb.type == "quad")
            groups = sorted(m.cell_sets.keys())
            return {
                "ok": True,
                "elapsed": elapsed,
                "fallback_aabb_matches": _FALLBACK_COUNTER["count"],
                "wedges": wedges,
                "tets": tets,
                "quads": quads,
                "groups": groups,
                "error": None,
            }
        except Exception as e:
            return {
                "ok": False,
                "elapsed": time.perf_counter() - t0,
                "fallback_aabb_matches": _FALLBACK_COUNTER["count"],
                "wedges": 0,
                "tets": 0,
                "quads": 0,
                "groups": [],
                "error": f"{type(e).__name__}: {str(e)[:200]}",
            }


def main():
    """Run the three glue mode variants and report metrics."""
    print("Probing BOP glue modes...\n")

    variants = [
        ("baseline (GlueOff)", None),
        ("GlueShift", BOPAlgo_GlueEnum.BOPAlgo_GlueShift),
        ("GlueFull", BOPAlgo_GlueEnum.BOPAlgo_GlueFull),
    ]

    results = {}
    for label, mode in variants:
        print(f"=== {label} ===", flush=True)
        results[label] = run_variant(label, mode)
        r = results[label]
        if r["ok"]:
            print(
                f"  OK   elapsed={r['elapsed']:.2f}s  "
                f"aabb_matches={r['fallback_aabb_matches']}  "
                f"wedges={r['wedges']} tets={r['tets']} quads={r['quads']} "
                f"groups={len(r['groups'])}"
            )
        else:
            print(
                f"  FAIL elapsed={r['elapsed']:.2f}s  "
                f"aabb_matches={r['fallback_aabb_matches']}  "
                f"error={r['error']}"
            )
        print()

    print("\n=== Summary ===\n")
    print(f"{'variant':<25s} {'ok':<4s} {'time':>8s} {'aabb_hit':>10s} {'groups':>8s}")
    for label, _mode in variants:
        r = results[label]
        ok = "✓" if r["ok"] else "✗"
        print(
            f"{label:<25s} {ok:<4s} {r['elapsed']:>7.2f}s "
            f"{r['fallback_aabb_matches']:>10d} {len(r['groups']):>8d}"
        )

    baseline_groups = set(results["baseline (GlueOff)"]["groups"])
    for label, _mode in variants[1:]:
        if not results[label]["ok"]:
            continue
        cg = set(results[label]["groups"])
        only_b = baseline_groups - cg
        only_c = cg - baseline_groups
        print(
            f"\n{label} vs baseline:  "
            f"only_in_baseline={len(only_b)}  only_in_{label}={len(only_c)}"
        )
        if only_b:
            print(f"  only in baseline: {sorted(only_b)[:10]}")
        if only_c:
            print(f"  only in {label}: {sorted(only_c)[:10]}")


if __name__ == "__main__":
    main()
