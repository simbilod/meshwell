r"""Spike for AABB-fallback replacement: gmsh-side adjacency lookup.

Compare:
  baseline — physical groups as written by occ_xao_writer
             (TShape identity match + AABB fallback)
  adj      — physical groups computed from gmsh's adjacency graph
             after merge: for each pair of volumes, intersect their
             boundary face tag sets

Runs the 10-entity complex stress scene through both paths, compares
group set + per-group face tag counts. Reports timings.

Scope:
  - kept-vs-kept volumes only
  - 3D entities producing 2D interface groups
  - skip keep=False helpers, embedded lower-dim entities, synthetic
    annotators (deferred — TShape-based logic handles those today)

Usage:
    PYTHONPATH=/home/simbil/Github/meshwell_structured python \
        scripts/spike_aabb_replacement.py
"""
from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gmsh

from meshwell.cad_occ import cad_occ
from meshwell.occ_xao_writer import write_xao
from meshwell.polyprism import PolyPrism
from meshwell.structured.pipeline import structured_post_pass, structured_pre_pass

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


# ---------- baseline: existing writer ----------


def snapshot_baseline_groups(xao_path: Path) -> dict[tuple[int, str], set[int]]:
    """Open the XAO in a fresh gmsh model, snapshot the physical groups."""
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("baseline")
        gmsh.merge(str(xao_path))
        gmsh.model.occ.synchronize()
        groups: dict[tuple[int, str], set[int]] = {}
        for dim, tag in gmsh.model.getPhysicalGroups():
            name = gmsh.model.getPhysicalName(dim, tag)
            entities = set(gmsh.model.getEntitiesForPhysicalGroup(dim, tag))
            groups[(dim, name)] = entities
        return groups
    finally:
        gmsh.finalize()


# ---------- candidate: gmsh adjacency ----------


def _strip_synthetic(name: str) -> str | None:
    """Drop __cohort_* synthetic names; return None for pure-synthetic."""
    if name.startswith("__cohort_"):
        return None
    return name


def compute_adj_groups(xao_path: Path) -> dict[tuple[int, str], set[int]]:
    """Replicate the interface/boundary groups using gmsh adjacency.

    For each 3D volume tag, look up its boundary face tags via
    ``getAdjacencies``. For each pair of volumes, intersect their
    boundary sets to find shared faces. Use the existing
    volume-level physical group names (which the XAO writer sets) to
    label the resulting interfaces ``A___B`` / ``A___None``.
    """
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("adj")
        gmsh.merge(str(xao_path))
        gmsh.model.occ.synchronize()

        # vol_tag -> set of user-visible names
        vol_names: dict[int, set[str]] = {}
        for dim, tag in gmsh.model.getPhysicalGroups():
            if dim != 3:
                continue
            name = gmsh.model.getPhysicalName(dim, tag)
            real = _strip_synthetic(name)
            if real is None:
                continue
            for vol in gmsh.model.getEntitiesForPhysicalGroup(dim, tag):
                vol_names.setdefault(int(vol), set()).add(real)

        # vol_tag -> set of boundary face tags
        vol_faces: dict[int, set[int]] = {}
        for _dim, tag in gmsh.model.getEntities(3):
            _up, faces = gmsh.model.getAdjacencies(3, tag)
            vol_faces[int(tag)] = {int(f) for f in faces}

        # interface groups: for each pair of named volumes, intersect face sets
        adj_groups: dict[tuple[int, str], set[int]] = {}
        vols = sorted(vol_faces.keys())
        for i, v1 in enumerate(vols):
            n1 = vol_names.get(v1, set())
            if not n1:
                continue
            for v2 in vols[i + 1 :]:
                n2 = vol_names.get(v2, set())
                if not n2:
                    continue
                shared = vol_faces[v1] & vol_faces[v2]
                if not shared:
                    continue
                for name1 in n1:
                    for name2 in n2:
                        if name1 == name2:
                            continue
                        key = (2, f"{name1}___{name2}")
                        adj_groups.setdefault(key, set()).update(shared)

        # boundary groups: per-volume faces NOT in any interface
        all_interface_faces: set[int] = set()
        for faces in adj_groups.values():
            all_interface_faces.update(faces)
        for vol, name_set in vol_names.items():
            exterior = vol_faces[vol] - all_interface_faces
            if not exterior:
                continue
            for name in name_set:
                key = (2, f"{name}___None")
                adj_groups.setdefault(key, set()).update(exterior)

        return adj_groups
    finally:
        gmsh.finalize()


# ---------- comparison ----------


def compare(
    label: str,
    baseline: dict[tuple[int, str], set[int]],
    candidate: dict[tuple[int, str], set[int]],
) -> None:
    """Print side-by-side comparison of 2D interface + boundary groups only."""

    # Filter to 2D interface + boundary groups (dim 2, has "___" delimiter).
    def filt(g):
        return {k: v for k, v in g.items() if k[0] == 2 and "___" in k[1]}

    b2d = filt(baseline)
    c2d = filt(candidate)

    print(f"\n=== {label}: 2D interface/boundary groups ===")
    print(f"  baseline groups:  {len(b2d)}")
    print(f"  candidate groups: {len(c2d)}")

    only_b = set(b2d) - set(c2d)
    only_c = set(c2d) - set(b2d)
    common = set(b2d) & set(c2d)

    print(f"  in baseline only: {len(only_b)}")
    print(f"  in candidate only: {len(only_c)}")
    print(f"  in both:          {len(common)}")

    if only_b:
        print("\n  only-in-baseline groups:")
        for k in sorted(only_b):
            print(f"    {k}: {len(b2d[k])} faces")

    if only_c:
        print("\n  only-in-candidate groups:")
        for k in sorted(only_c):
            print(f"    {k}: {len(c2d[k])} faces")

    print("\n  per-group face-tag set comparison (intersection):")
    diff_count = 0
    for k in sorted(common):
        b_set = b2d[k]
        c_set = c2d[k]
        if b_set == c_set:
            status = "✓"
        else:
            status = "✗"
            diff_count += 1
        n_overlap = len(b_set & c_set)
        n_b_only = len(b_set - c_set)
        n_c_only = len(c_set - b_set)
        print(
            f"    {status} {k[1]:<35s} baseline={len(b_set):>3d}  "
            f"candidate={len(c_set):>3d}  shared={n_overlap}  "
            f"b_only={n_b_only}  c_only={n_c_only}"
        )

    print(f"\n  groups with differing face-tag sets: {diff_count}/{len(common)}")


# ---------- main ----------


def main():
    """Run the comparison and print results."""
    ents = build_complex_scene()

    print("Building OCC entities + writing XAO...", flush=True)
    state = structured_pre_pass(ents, point_tolerance=1e-3)
    occ_entities = cad_occ(state.entities_out, prepared=True)
    final = structured_post_pass(occ_entities, state)

    with tempfile.TemporaryDirectory() as td:
        xao_path = Path(td) / "scene.xao"
        write_xao(final, xao_path)
        print(f"  XAO: {xao_path.stat().st_size} bytes")

        # baseline
        print("\nSnapshot baseline (XAO writer's groups)...", flush=True)
        t0 = time.perf_counter()
        baseline = snapshot_baseline_groups(xao_path)
        t_baseline = time.perf_counter() - t0
        print(f"  load + read: {t_baseline:.3f}s, {len(baseline)} groups total")

        # candidate
        print("\nCompute candidate (gmsh adjacency)...", flush=True)
        t0 = time.perf_counter()
        adj = compute_adj_groups(xao_path)
        t_adj = time.perf_counter() - t0
        print(f"  load + adjacency: {t_adj:.3f}s, {len(adj)} groups total")

    compare("complex scene", baseline, adj)


if __name__ == "__main__":
    main()
