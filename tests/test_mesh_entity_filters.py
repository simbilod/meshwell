"""filter_tags_by_target_dimension must return the right dimension's tags."""
import gmsh
import pytest

from meshwell._mesh_entity import _MeshEntity, entity_name_set


def _make_entity(model, dim, tags, boundaries):
    """Build a bare _MeshEntity, bypassing full __init__ plumbing.

    ``dim`` and ``tags`` are read-only properties backed by ``_explicit_dim``
    and ``dimtags``, so we set those rather than the properties directly.
    ``boundaries`` is an instance attribute (a plain list of ints).
    """
    ent = _MeshEntity.__new__(_MeshEntity)
    ent.model = model
    ent._explicit_dim = dim
    ent.dimtags = [(dim, t) for t in tags]
    ent.boundaries = boundaries
    return ent


@pytest.fixture
def box_entity():
    if not gmsh.isInitialized():
        gmsh.initialize()
    gmsh.model.add("filter_dim_test")
    tag = gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
    gmsh.model.occ.synchronize()
    boundaries = [abs(t) for d, t in gmsh.model.getBoundary([(3, tag)], oriented=False)]
    ent = _make_entity(gmsh.model, 3, [tag], boundaries)
    yield ent
    gmsh.model.remove()


def test_same_dim_returns_tags(box_entity):
    assert box_entity.filter_tags_by_target_dimension(3) == box_entity.tags


def test_one_below_returns_boundaries(box_entity):
    assert sorted(box_entity.filter_tags_by_target_dimension(2)) == sorted(
        box_entity.boundaries
    )


def test_curves_from_volume(box_entity):
    curves = box_entity.filter_tags_by_target_dimension(1)
    assert len(set(curves)) == 12  # a box has 12 edges


def test_points_from_volume_returns_points_not_curves(box_entity):
    points = box_entity.filter_tags_by_target_dimension(0)
    expected = {
        t
        for d, t in gmsh.model.getBoundary(
            [(3, box_entity.tags[0])], combined=False, oriented=False, recursive=True
        )
        if d == 0
    }
    assert set(points) == expected
    assert len(expected) == 8  # a box has 8 corners


def test_points_from_surface_returns_points(box_entity):
    surf = box_entity.boundaries[0]
    boundaries = [
        abs(t) for d, t in gmsh.model.getBoundary([(2, surf)], oriented=False)
    ]
    ent = _make_entity(gmsh.model, 2, [surf], boundaries)
    points = ent.filter_tags_by_target_dimension(0)
    assert len(set(points)) == 4  # a face has 4 corners, not its 4 curves


def test_target_above_dim_warns_and_returns_empty(box_entity):
    ent = _make_entity(box_entity.model, 1, [1], [])
    with pytest.warns(UserWarning, match="exceeds entity dimension"):
        assert ent.filter_tags_by_target_dimension(3) == []


def test_entity_name_set_from_str():
    assert entity_name_set("metal") == {"metal"}


def test_entity_name_set_from_tuple():
    assert entity_name_set(("metal", "conductor")) == {"metal", "conductor"}


def test_no_substring_matching():
    # "metal" must NOT be treated as matching "metal2"
    assert "metal2" not in entity_name_set("metal")
    assert not entity_name_set("metal") & entity_name_set("metal2")
