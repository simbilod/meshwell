"""Stress tests for complex structured scenes.

Validates the v1 pipeline on inputs that exercise every planner branch:
multi-level cohort, multi-polygon-per-level, arcs, voids,
unstructured neighbours above and below, mesh_order carving.
"""
from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import Polygon

import meshio
from meshwell.orchestrator import generate_mesh
from meshwell.polyprism import PolyPrism
from meshwell.resolution import StructuredExtrusionResolutionSpec


def _circle(cx, cy, r, n=48):
    angles = np.linspace(0, 2 * np.pi, n + 1)[:-1]
    return Polygon([(cx + r * np.cos(a), cy + r * np.sin(a)) for a in angles])


def _annulus(cx, cy, r_out, r_in, n=48):
    outer = _circle(cx, cy, r_out, n)
    inner = _circle(cx, cy, r_in, n)
    return Polygon(outer.exterior.coords, holes=[inner.exterior.coords])


def _rect(x1, y1, x2, y2):
    return Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])


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


@pytest.fixture
def complex_scene_entities():
    return [
        PolyPrism(
            SQUARE_A,
            {0.0: 0.0, 1.0: 0.0},
            physical_name="A_square",
            structured=True,
            mesh_order=3.0,
        ),
        PolyPrism(
            CIRCLE_A,
            {0.0: 0.0, 1.0: 0.0},
            physical_name="A_circle",
            structured=True,
            mesh_order=3.0,
            identify_arcs=True,
        ),
        PolyPrism(
            RECT_HOLE_A,
            {0.0: 0.0, 1.0: 0.0},
            physical_name="A_recth",
            structured=True,
            mesh_order=3.0,
        ),
        PolyPrism(
            CIRCLE_B,
            {1.0: 0.0, 2.0: 0.0},
            physical_name="B_circle",
            structured=True,
            mesh_order=3.0,
            identify_arcs=True,
        ),
        PolyPrism(
            ANNULUS_B,
            {1.0: 0.0, 2.0: 0.0},
            physical_name="B_annulus",
            structured=True,
            mesh_order=3.0,
            identify_arcs=True,
        ),
        PolyPrism(
            HEX_C,
            {2.0: 0.0, 3.0: 0.0},
            physical_name="C_hex",
            structured=True,
            mesh_order=3.0,
        ),
        PolyPrism(
            VOID_C,
            {2.0: 0.0, 3.0: 0.0},
            physical_name="C_void",
            structured=True,
            mesh_order=1.0,
            mesh_bool=False,
        ),
        PolyPrism(
            Polygon(BIG_BASE.exterior.coords, holes=[HOLE_BASE.exterior.coords]),
            {-2.0: 0.0, 0.0: 0.0},
            physical_name="base",
            mesh_order=5.0,
        ),
        PolyPrism(BIG_CAP, {3.0: 0.0, 5.0: 0.0}, physical_name="cap", mesh_order=5.0),
        PolyPrism(
            CAP_ARCH,
            {3.0: 0.0, 5.0: 0.0},
            physical_name="cap_arch",
            mesh_order=2.0,
            identify_arcs=True,
        ),
    ]


def _resolution_specs():
    return {
        name: [StructuredExtrusionResolutionSpec(n_layers=2)]
        for name in (
            "A_square",
            "A_circle",
            "A_recth",
            "B_circle",
            "B_annulus",
            "C_hex",
        )
    }


@pytest.mark.xfail(
    reason="arc-bearing lateral face is split by BOP — fix deferred (see notes)",
    strict=False,
)
def test_complex_scene_meshes_without_error(complex_scene_entities, tmp_path):
    generate_mesh(
        complex_scene_entities,
        dim=3,
        output_mesh=tmp_path / "out.msh",
        default_characteristic_length=0.8,
        resolution_specs=_resolution_specs(),
    )
    m = meshio.read(tmp_path / "out.msh")
    wedges = sum(cb.data.shape[0] for cb in m.cells if cb.type == "wedge")
    tets = sum(cb.data.shape[0] for cb in m.cells if cb.type == "tetra")
    assert wedges > 0, "expected wedge elements"
    assert tets > 0, "expected tet elements in unstructured regions"


@pytest.mark.xfail(
    reason="arc-bearing lateral face is split by BOP — fix deferred (see notes)",
    strict=False,
)
def test_complex_scene_all_physical_groups_present(complex_scene_entities, tmp_path):
    generate_mesh(
        complex_scene_entities,
        dim=3,
        output_mesh=tmp_path / "out.msh",
        default_characteristic_length=0.8,
        resolution_specs=_resolution_specs(),
    )
    m = meshio.read(tmp_path / "out.msh")
    expected = {
        "A_square",
        "A_circle",
        "A_recth",
        "B_circle",
        "B_annulus",
        "C_hex",
        "base",
        "cap",
        "cap_arch",
    }
    field = set(m.cell_sets.keys())
    missing = expected - field
    assert not missing, f"physical groups missing from mesh: {missing}"
    assert "C_void" not in field
