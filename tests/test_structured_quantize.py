"""quantize_key is the single quantization implementation."""
import inspect

from meshwell.structured.types import Arrangement, quantize_key


def test_quantize_matches_vertex_registry_convention():
    assert quantize_key(0.0004, -0.0004, 1.0, 1e-3) == (0, 0, 1000)
    assert quantize_key(0.0006, 0.0, 0.0, 1e-3) == (1, 0, 0)


def test_vertex_registry_delegates():
    from meshwell.structured.build import VertexRegistry

    src = inspect.getsource(VertexRegistry._key)
    assert "quantize_key" in src


def test_arrangement_carries_point_tolerance():
    assert "point_tolerance" in Arrangement.__dataclass_fields__


def test_polygon_point_tol_heuristic_deleted():
    import meshwell.structured.decompose as d

    assert not hasattr(d, "_polygon_point_tol")
