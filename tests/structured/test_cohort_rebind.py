"""Regression: cohort / unstructured-neighbour interface faces must be shared, not duplicated.

BOP builds brand-new edges for coincident edge-on-edge pairs (e.g. the outer
domain-wall edges of a cohort bot face and the substrate top face). Cohorts
keep their pre-BOP compound, so without re-binding the neighbour referenced
the new edges and the interface face was emitted twice (meshed twice, then
node-deduplicated).
"""
from __future__ import annotations

import itertools

import gmsh
import pytest
from shapely.geometry import box

from meshwell.orchestrator import cad
from meshwell.polyprism import PolyPrism


def _coincident_unshared_surfaces() -> list[tuple[int, int]]:
    """Pairs of distinct surfaces with identical bounding boxes."""
    boxes = {
        s: tuple(round(v, 6) for v in gmsh.model.getBoundingBox(2, s))
        for _, s in gmsh.model.getEntities(2)
    }
    return [(a, b) for a, b in itertools.combinations(boxes, 2) if boxes[a] == boxes[b]]


@pytest.mark.parametrize("perturbation", [0.0, 1e-5])
def test_cohort_bot_face_shared_with_unstructured_substrate(tmp_path, perturbation):
    domain = box(0, 0, 100, 40)
    # Two structured slabs tiling the domain -> the cohort bot face is split
    # while the substrate top is one face: its wall edges get BOP-merged.
    left = PolyPrism(
        box(0, 0, 60, 40), {1.0: 0.0, 2.0: 0.0}, physical_name="left", structured=True
    )
    right = PolyPrism(
        box(60, 0, 100, 40),
        {1.0: 0.0, 2.0: 0.0},
        physical_name="right",
        structured=True,
    )
    substrate = PolyPrism(domain, {0.0: 0.0, 1.0: 0.0}, physical_name="substrate")
    xao = tmp_path / "rebind.xao"
    cad([left, right, substrate], output_file=xao, perturbation=perturbation)

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(str(xao))
        assert _coincident_unshared_surfaces() == []
        # Every interface face at z=1 must bound exactly two volumes.
        for _, s in gmsh.model.getEntities(2):
            b = gmsh.model.getBoundingBox(2, s)
            if abs(b[2] - 1.0) < 1e-6 and abs(b[5] - 1.0) < 1e-6:
                up, _ = gmsh.model.getAdjacencies(2, s)
                assert len(up) == 2, f"interface surface {s} bounds {list(up)}"
    finally:
        gmsh.finalize()
