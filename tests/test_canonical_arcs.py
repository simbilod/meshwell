"""Tests for pipeline-level arc identification + the canonical geometry pipeline."""
import numpy as np
import pytest
from shapely.geometry import Polygon


def _ring(n: int, radius: float, phase: float = 0.0) -> Polygon:
    pts = [
        (
            radius * np.cos(2 * np.pi * k / n + phase),
            radius * np.sin(2 * np.pi * k / n + phase),
        )
        for k in range(n)
    ]
    return Polygon(pts)


def test_apply_arc_params_stamps_entities():
    from meshwell.cad_common import apply_arc_params
    from meshwell.polyprism import PolyPrism

    p = PolyPrism(
        polygons=_ring(16, 5.0),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="p",
        mesh_order=1,
    )
    assert p.identify_arcs is False  # GeometryEntity class default
    apply_arc_params([p], identify_arcs=True, min_arc_points=7, arc_tolerance=2e-3)
    assert p.identify_arcs is True
    assert p.min_arc_points == 7
    assert p.arc_tolerance == 2e-3


def test_apply_arc_params_skips_non_extrude_prism():
    from meshwell.cad_common import apply_arc_params
    from meshwell.polyprism import PolyPrism

    p = PolyPrism(
        polygons=_ring(16, 5.0),
        buffers={0.0: 0.1, 1.0: 0.0},
        physical_name="p",
        mesh_order=1,
    )
    apply_arc_params([p], identify_arcs=True)
    assert p.identify_arcs is False  # arcs unsupported with z-varying buffers


def test_polyprism_no_longer_accepts_arc_kwargs():
    from meshwell.polyprism import PolyPrism

    with pytest.raises(TypeError):
        PolyPrism(
            polygons=_ring(16, 5.0),
            buffers={0.0: 0.0, 1.0: 0.0},
            physical_name="p",
            mesh_order=1,
            identify_arcs=True,
        )


def test_from_dict_ignores_legacy_arc_keys():
    from meshwell.polyprism import PolyPrism

    p = PolyPrism(
        polygons=_ring(16, 5.0),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="p",
        mesh_order=1,
    )
    d = p.to_dict()
    assert "identify_arcs" not in d
    d["identify_arcs"] = True  # legacy serialized scenes carry these keys
    d["min_arc_points"] = 5
    d["arc_tolerance"] = 1e-3
    p2 = PolyPrism.from_dict(d)
    assert p2.identify_arcs is False
