"""Minimal repro: structured arc-bearing slab + partial top/bottom intersection.

Captures the gap described in the deep-dive (2026-05-13): the face_partition
machinery already honors ``identify_arcs`` for **isolated** arc slabs and works
for **rectangular** partial-cover cases, but the combination -- arc-bearing
slab whose top is partially covered by a neighbour -- exercises
``_make_occ_wire_from_vertices`` with a *mixed* exterior (arc-vertices +
splitter-cut straight vertices). The heuristic arc-detector in
``GeometryEntity.decompose_vertices`` was never designed for mixed input.

Drives the provenance-preserving fix planned in
``docs/superpowers/plans/2026-05-13-arc-provenance-face-partition.md``.
"""
from __future__ import annotations

import math
import pathlib

from shapely.geometry import Polygon

import meshwell.mesh as meshwell_mesh
from meshwell.cad_occ import cad_occ
from meshwell.occ_xao_writer import (
    write_structured_slabs_sidecar,
    write_xao,
)
from meshwell.polyprism import PolyPrism
from meshwell.polysurface import PolySurface


def _disc(cx: float, cy: float, r: float, n: int = 48) -> Polygon:
    return Polygon(
        [
            (
                cx + r * math.cos(2 * math.pi * k / n),
                cy + r * math.sin(2 * math.pi * k / n),
            )
            for k in range(n)
        ]
    )


def _mesh(entities, tmp_path: pathlib.Path, name: str, default_h: float = 0.25):
    occ_entities, slabs = cad_occ(entities, point_tolerance=1e-3, return_slabs=True)
    xao = tmp_path / f"{name}.xao"
    write_xao(occ_entities, str(xao))
    write_structured_slabs_sidecar(str(xao), slabs)
    return meshwell_mesh.mesh(
        input_file=str(xao),
        output_file=str(tmp_path / f"{name}.msh"),
        dim=3,
        default_characteristic_length=default_h,
        gmsh_version=4.1,
        verbosity=0,
        n_threads=1,
    )


def _physical_names_3d(mesh):
    out = set()
    for name, per_block in mesh.cell_sets.items():
        for cb_idx, idx in enumerate(per_block):
            if idx is None or len(idx) == 0:
                continue
            if mesh.cells[cb_idx].type in ("tetra", "wedge", "hexahedron"):
                out.add(name)
                break
    return out


def test_arc_disc_with_partial_top_cover(tmp_path):
    """Structured disc with identify_arcs=True, partially covered on top by a PolySurface.

    The partition splits the disc into pieces whose exterior
    mixes arc vertices and a straight cut. With heuristic arc detection the
    cut gets mis-classified.
    """
    disc_poly = _disc(0.0, 0.0, 1.0, n=48)
    disc = PolyPrism(
        polygons=disc_poly,
        buffers={0.0: 0.0, 1.0: 0.0},
        n_layers=[3],
        physical_name="disc",
        identify_arcs=True,
        arc_tolerance=0.01,
        mesh_order=10,
    )
    hot_spot = PolySurface(
        polygons=Polygon([(-0.3, -0.3), (0.3, -0.3), (0.3, 0.3), (-0.3, 0.3)]),
        translation=(0.0, 0.0, 1.0),  # at disc.zhi
        physical_name="hot_spot",
        mesh_order=2,
    )
    result = _mesh([hot_spot, disc], tmp_path, "arc_partial_top")
    assert "disc" in _physical_names_3d(result)


def test_arc_annulus_with_partial_bottom_cover(tmp_path):
    """Annular footprint (inner+outer arcs) + interior PolySurface at z=zlo covering part of the annulus."""
    outer = [
        (1.0 * math.cos(2 * math.pi * k / 48), 1.0 * math.sin(2 * math.pi * k / 48))
        for k in range(48)
    ]
    inner = [
        (0.4 * math.cos(2 * math.pi * k / 48), 0.4 * math.sin(2 * math.pi * k / 48))
        for k in range(48)
    ]
    annulus = Polygon(outer, holes=[inner])
    ring = PolyPrism(
        polygons=annulus,
        buffers={0.0: 0.0, 1.0: 0.0},
        n_layers=[3],
        physical_name="ring",
        identify_arcs=True,
        arc_tolerance=0.01,
        mesh_order=10,
    )
    # A rectangle straddling the annulus material at z=0.
    hot_spot = PolySurface(
        polygons=Polygon([(0.5, -0.2), (0.9, -0.2), (0.9, 0.2), (0.5, 0.2)]),
        translation=(0.0, 0.0, 0.0),
        physical_name="hot_spot",
        mesh_order=2,
    )
    result = _mesh([hot_spot, ring], tmp_path, "arc_partial_bottom_annulus")
    assert "ring" in _physical_names_3d(result)


if __name__ == "__main__":
    import sys
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        try:
            test_arc_disc_with_partial_top_cover(pathlib.Path(td))
            print("disc+partial-top: PASS")
        except Exception as e:
            print(f"disc+partial-top: FAIL -- {type(e).__name__}: {e}")
            import traceback

            traceback.print_exc(file=sys.stdout)
    with tempfile.TemporaryDirectory() as td:
        try:
            test_arc_annulus_with_partial_bottom_cover(pathlib.Path(td))
            print("annulus+partial-bottom: PASS")
        except Exception as e:
            print(f"annulus+partial-bottom: FAIL -- {type(e).__name__}: {e}")
            import traceback

            traceback.print_exc(file=sys.stdout)
