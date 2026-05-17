"""Unit tests for _resolve_tol helper."""
import gmsh
import pytest

from meshwell.structured.validator import _resolve_tol


@pytest.fixture
def gmsh_unit_cube_meshed():
    """A meshed unit cube — gives us real elements to query getElementQualities on."""
    gmsh.initialize()
    gmsh.model.add("tol_test")
    gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
    gmsh.model.occ.synchronize()
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.5)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.5)
    gmsh.model.mesh.generate(3)
    yield
    gmsh.finalize()


def test_resolve_tol_returns_explicit_value():
    assert _resolve_tol(1e-8) == 1e-8


def test_resolve_tol_derives_from_mesh(gmsh_unit_cube_meshed):  # noqa: ARG001
    tol = _resolve_tol(None)
    # Cube edge ~0.5; tol should be ~5e-7, definitely between 1e-9 and 1e-3.
    assert 1e-12 < tol < 1e-3


def test_resolve_tol_zero_falls_back_to_safe_default():
    # No gmsh session — guard against AttributeError or NaN.
    gmsh.initialize()
    gmsh.model.add("empty")
    tol = _resolve_tol(None)
    assert tol > 0
    gmsh.finalize()
