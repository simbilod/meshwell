"""Unit tests for ``Mesh._append_interface_group``.

This helper is the shared body extracted from the two near-duplicate
interface-recovery branches inside ``Mesh._recover_labels_from_cad``
(the ``parts[0] == physical_name`` and ``parts[1] == physical_name``
branches).  Driving the full ``_recover_labels_from_cad`` path only
exercises the coupled real ``_MeshEntity`` (whose ``dim`` property is
derived from ``_explicit_dim``), so we unit-test the helper directly
with a lightweight stub that lets us pin the two behaviours the WP3
Task-6 refactor cares about:

1. the ``_explicit_dim``-raising formula must never *lower* an
   already-raised ``_explicit_dim`` (the pre-refactor second branch used
   ``max(entities.dim, i_dim + 1)`` which discards it); and
2. boundary routing to ``mesh_edge_name_interfaces`` happens only when a
   ``suffix`` equal to ``boundary_delimiter`` is supplied -- the second
   branch passes ``suffix=None`` and must therefore keep every tag in
   ``interfaces`` (see the suffix-asymmetry investigation in the task
   report).
"""

from meshwell.mesh import Mesh


class _StubEntity:
    """Minimal stand-in for ``_MeshEntity``.

    Unlike the real entity, ``dim`` and ``_explicit_dim`` are independent
    attributes so a test can construct the otherwise-impossible-through-
    the-property state ``dim == -1`` while ``_explicit_dim`` is already
    raised -- exactly the state the buggy ``max(entities.dim, ...)``
    formula would silently discard.
    """

    def __init__(self, dim, explicit_dim):
        self.dim = dim
        self._explicit_dim = explicit_dim
        self.interfaces = []
        self.boundaries = []
        self.mesh_edge_name_interfaces = []


class _StubMesh:
    """Stub ``self`` exposing only the ``get_physical_dimtags`` the helper needs."""

    def __init__(self, dimtags):
        self._dimtags = dimtags

    def get_physical_dimtags(self, physical_name):  # noqa: ARG002
        return self._dimtags


def _call(mesh, entities, group_name="A___B", boundary_delimiter="None", **kw):
    # Invoke the unbound helper against the stub ``self``.
    return Mesh._append_interface_group(
        mesh, entities, group_name, boundary_delimiter, **kw
    )


def test_does_not_lower_previously_raised_explicit_dim():
    """A raised ``_explicit_dim`` must survive a lower-dim interface.

    Entity has ``_explicit_dim == 2`` but ``dim`` still reports ``-1``
    (decoupled by the stub).  A dim-0 interface (``i_dim + 1 == 1``) must
    NOT drop ``_explicit_dim`` to 1.  The pre-refactor second branch used
    ``max(entities.dim, i_dim + 1) == max(-1, 1) == 1`` and would.
    """
    entity = _StubEntity(dim=-1, explicit_dim=2)
    mesh = _StubMesh([(0, 42)])

    _call(mesh, entity, group_name="C___A")  # branch-2 style: suffix=None

    assert entity._explicit_dim == 2  # preserved, not lowered to 1
    assert entity.interfaces == [42]
    assert entity.boundaries == [42]
    assert entity.mesh_edge_name_interfaces == []


def test_raises_explicit_dim_from_none():
    """With no prior ``_explicit_dim`` (None), it is set to ``i_dim + 1``."""
    entity = _StubEntity(dim=-1, explicit_dim=None)
    mesh = _StubMesh([(1, 7)])

    _call(mesh, entity, group_name="C___A")

    assert entity._explicit_dim == 2  # max(0, 1 + 1)


def test_boundary_suffix_routes_to_mesh_edge():
    """A ``suffix == boundary_delimiter`` tag lands in mesh_edge_name_interfaces."""
    entity = _StubEntity(dim=-1, explicit_dim=None)
    mesh = _StubMesh([(1, 9)])

    _call(mesh, entity, group_name="A___None", suffix="None")

    assert entity.mesh_edge_name_interfaces == [9]
    assert entity.interfaces == []
    assert entity.boundaries == [9]


def test_no_suffix_never_routes_to_mesh_edge():
    """Second-branch calls (suffix=None) keep tags in ``interfaces``.

    Even when ``parts[0]`` would literally equal ``boundary_delimiter``
    -- only possible if a user names a real entity "None" -- the tag is a
    genuine interface, not an exterior boundary, so it must stay in
    ``interfaces``.  Passing ``suffix=None`` guarantees that.
    """
    entity = _StubEntity(dim=-1, explicit_dim=None)
    mesh = _StubMesh([(1, 5)])

    _call(mesh, entity, group_name="None___A")  # suffix omitted -> None

    assert entity.interfaces == [5]
    assert entity.mesh_edge_name_interfaces == []


def test_empty_dimtags_is_a_noop():
    entity = _StubEntity(dim=-1, explicit_dim=None)
    mesh = _StubMesh([])

    _call(mesh, entity)

    assert entity._explicit_dim is None
    assert entity.interfaces == []
    assert entity.boundaries == []
