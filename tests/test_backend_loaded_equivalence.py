"""Cross-backend equivalence of *loaded* gmsh state.

Each scene in this file is fed to both ``cad_gmsh`` and ``cad_occ``, then
loaded into a fresh gmsh session, and the resulting model state is
snapshotted and compared. Unlike
:mod:`tests.test_backend_cross_compare` -- which compares the meshed
output -- this file checks the pre-meshing CAD state:

- entity count per dimension is identical;
- the set of physical group names is identical (interface names
  normalized for order);
- per physical group, the entity count and the total geometric mass
  (volume / area / length) match within ``rel_tol``;
- per top-dim physical group, the boundary-face signature
  (sorted (n_faces, total_area) tuples across the group's entities)
  matches.

The scenes deliberately progress from trivial to topology-heavy:
abutting prisms, three-way junctions, nested prisms that overlap on a
single face, arcs on 2D and 3D bodies, internal embedded surfaces,
helper-only cuts (keep=False), and entities carrying multiple
physical names.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pytest
import shapely
from shapely.geometry import LineString

import gmsh
from meshwell.cad_gmsh import cad_gmsh
from meshwell.cad_occ import cad_occ
from meshwell.interface_tag import InterfaceTag
from meshwell.model import ModelManager
from meshwell.occ_xao_writer import write_xao
from meshwell.polyprism import PolyPrism
from meshwell.polysurface import PolySurface

# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


_IFACE_DELIM = "___"


def _normalize_group_name(name: str) -> str:
    """``A___B`` and ``B___A`` collapse to the same canonical key."""
    if _IFACE_DELIM in name:
        return _IFACE_DELIM.join(sorted(name.split(_IFACE_DELIM)))
    return name


@dataclass
class _GroupInfo:
    n_entities: int = 0
    total_mass: float = 0.0
    # Sorted tuple of (n_boundary_faces, total_boundary_area) per entity
    # in the group. Compared as a multiset across the group's entities.
    boundary_signature: tuple[tuple[int, float], ...] = ()


@dataclass
class _ModelSnapshot:
    entity_count_by_dim: dict[int, int] = field(default_factory=dict)
    # Canonical-name -> dim -> _GroupInfo. Two physical groups can share
    # a name across different dims (e.g. a 2D iface group and a 3D
    # entity with the same physical_name); dim-keying separates them.
    groups: dict[str, dict[int, _GroupInfo]] = field(default_factory=dict)


def _snapshot_active_model() -> _ModelSnapshot:
    """Capture entity counts, physical groups, and per-group mass / boundary."""
    gmsh.model.occ.synchronize()
    snap = _ModelSnapshot()
    for d in (0, 1, 2, 3):
        snap.entity_count_by_dim[d] = len(gmsh.model.getEntities(d))

    top_dim = max(
        (d for d in (3, 2, 1) if snap.entity_count_by_dim.get(d, 0) > 0),
        default=0,
    )

    for dim, ptag in gmsh.model.getPhysicalGroups():
        name = gmsh.model.getPhysicalName(dim, ptag)
        if not name:
            continue
        canonical = _normalize_group_name(name)
        ent_tags = list(gmsh.model.getEntitiesForPhysicalGroup(dim, ptag))

        total_mass = 0.0
        bdy_sig: list[tuple[int, float]] = []
        for t in ent_tags:
            try:
                mass = gmsh.model.occ.getMass(dim, int(t))
            except Exception:
                mass = 0.0
            total_mass += float(mass)

            if dim == top_dim and dim >= 2:
                bdy_dimtags = gmsh.model.getBoundary(
                    [(dim, int(t))],
                    combined=False,
                    oriented=False,
                    recursive=False,
                )
                bdy_area = 0.0
                for bd, bt in bdy_dimtags:
                    with contextlib.suppress(Exception):
                        bdy_area += float(gmsh.model.occ.getMass(bd, int(bt)))
                bdy_sig.append((len(bdy_dimtags), bdy_area))

        info = _GroupInfo(
            n_entities=len(ent_tags),
            total_mass=total_mass,
            boundary_signature=tuple(sorted(bdy_sig)),
        )
        snap.groups.setdefault(canonical, {})[dim] = info

    return snap


# ---------------------------------------------------------------------------
# Backend drivers (each leaves the snapshot's data in Python state and
# finalizes gmsh, so the two backends never share a model)
# ---------------------------------------------------------------------------


def _capture_gmsh_backend(entities) -> _ModelSnapshot:
    _, mm = cad_gmsh(entities)
    try:
        return _snapshot_active_model()
    finally:
        mm.finalize()


def _capture_occ_backend(entities, tmp_path: Path) -> _ModelSnapshot:
    labeled = cad_occ(entities)
    xao_path = tmp_path / "occ.xao"
    write_xao(labeled, str(xao_path))

    mm = ModelManager(filename="occ_load")
    try:
        mm.load_from_xao(xao_path)
        return _snapshot_active_model()
    finally:
        mm.finalize()


def _capture_both(
    factory: Callable[[], list], tmp_path: Path
) -> tuple[_ModelSnapshot, _ModelSnapshot]:
    s_gmsh = _capture_gmsh_backend(factory())
    s_occ = _capture_occ_backend(factory(), tmp_path)
    return s_gmsh, s_occ


# ---------------------------------------------------------------------------
# Comparator
# ---------------------------------------------------------------------------


def _close(a: float, b: float, rel_tol: float, abs_tol: float = 1e-9) -> bool:
    denom = max(abs(a), abs(b))
    if denom <= abs_tol:
        return abs(a - b) <= abs_tol
    return abs(a - b) / denom <= rel_tol


def _assert_loaded_equivalent(
    s_gmsh: _ModelSnapshot,
    s_occ: _ModelSnapshot,
    *,
    rel_tol: float = 1e-3,
    ignore_groups: set[str] = frozenset(),
    ignore_entity_count_dims: set[int] = frozenset(),
    mass_only_groups: set[str] = frozenset(),
    check_boundary_signature: bool = True,
) -> None:
    """Assert two loaded-model snapshots are equivalent.

    ``ignore_entity_count_dims`` opts out of per-dim entity count
    comparison for specific dims (1D edges in particular can differ in
    count without affecting per-group mass when one backend introduces
    seam edges).

    ``mass_only_groups`` opts out of per-group entity-count and
    boundary-signature checks for the listed groups -- only total mass
    is compared. Useful where one backend splits one logical face into
    several fragments and the other keeps it whole (the area is still
    invariant).
    """
    # Per-dim entity counts.
    for d in (0, 1, 2, 3):
        if d in ignore_entity_count_dims:
            continue
        g = s_gmsh.entity_count_by_dim.get(d, 0)
        o = s_occ.entity_count_by_dim.get(d, 0)
        assert g == o, f"Entity count differs at dim={d}: gmsh={g} occ={o}"

    # Physical group key sets.
    g_keys = {
        (name, dim)
        for name, by_dim in s_gmsh.groups.items()
        if name not in ignore_groups
        for dim in by_dim
    }
    o_keys = {
        (name, dim)
        for name, by_dim in s_occ.groups.items()
        if name not in ignore_groups
        for dim in by_dim
    }
    assert g_keys == o_keys, (
        f"Group/dim sets differ.\n"
        f"  gmsh-only: {g_keys - o_keys}\n"
        f"  occ-only:  {o_keys - g_keys}"
    )

    # Per-group entity count + mass + boundary signature.
    for name, dim in sorted(g_keys):
        gi = s_gmsh.groups[name][dim]
        oi = s_occ.groups[name][dim]
        assert _close(gi.total_mass, oi.total_mass, rel_tol), (
            f"Total mass for group {name!r} (dim={dim}) differs: "
            f"gmsh={gi.total_mass:.6g} occ={oi.total_mass:.6g} "
            f"(rel_tol={rel_tol:.1e})"
        )
        if name in mass_only_groups:
            continue
        assert gi.n_entities == oi.n_entities, (
            f"Entity count for group {name!r} (dim={dim}) differs: "
            f"gmsh={gi.n_entities} occ={oi.n_entities}"
        )
        if not check_boundary_signature:
            continue
        if not gi.boundary_signature and not oi.boundary_signature:
            continue
        assert len(gi.boundary_signature) == len(oi.boundary_signature), (
            f"Boundary signature length differs for group {name!r}: "
            f"gmsh={gi.boundary_signature} occ={oi.boundary_signature}"
        )
        for (gn, ga), (on, oa) in zip(gi.boundary_signature, oi.boundary_signature):
            # Face-count per entity must match exactly; areas within rel_tol.
            assert gn == on, (
                f"Per-entity boundary face count differs in group {name!r}: "
                f"gmsh entity has {gn} bdy faces, occ has {on}; "
                f"full signatures: gmsh={gi.boundary_signature} occ={oi.boundary_signature}"
            )
            assert _close(ga, oa, rel_tol), (
                f"Per-entity boundary area differs in group {name!r}: "
                f"gmsh={ga:.6g} occ={oa:.6g} (rel_tol={rel_tol:.1e})"
            )


# ---------------------------------------------------------------------------
# Scene library
# ---------------------------------------------------------------------------


_NO_TAPER = {0.0: 0.0, 1.0: 0.0}


def _scene_two_abutting_prisms() -> list:
    A = shapely.Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
    B = shapely.Polygon([(5, 0), (10, 0), (10, 5), (5, 5)])
    return [
        PolyPrism(polygons=A, buffers=_NO_TAPER, physical_name="A", mesh_order=1),
        PolyPrism(polygons=B, buffers=_NO_TAPER, physical_name="B", mesh_order=2),
    ]


def _scene_three_in_a_row() -> list:
    A = shapely.Polygon([(0, 0), (2, 0), (2, 5), (0, 5)])
    B = shapely.Polygon([(2, 0), (5, 0), (5, 5), (2, 5)])
    C = shapely.Polygon([(5, 0), (10, 0), (10, 5), (5, 5)])
    return [
        PolyPrism(polygons=A, buffers=_NO_TAPER, physical_name="A", mesh_order=1),
        PolyPrism(polygons=B, buffers=_NO_TAPER, physical_name="B", mesh_order=2),
        PolyPrism(polygons=C, buffers=_NO_TAPER, physical_name="C", mesh_order=3),
    ]


def _scene_nested_prism_sharing_one_face() -> list:
    """Small high-mesh_order prism sits inside a large one, flush against +x face.

    The outer prism keeps everything except the carved region; the
    inner prism shares exactly one face with the outer at x=10. Both
    backends must produce: outer body, inner body, the shared face as
    ``inner___outer``, and the rest of the outer's boundary as
    ``outer___None``. Inner's other 5 faces must show up as
    ``inner___None``.
    """
    outer = shapely.Polygon([(0, 0), (10, 0), (10, 5), (0, 5)])
    inner = shapely.Polygon([(7, 1), (10, 1), (10, 4), (7, 4)])
    return [
        PolyPrism(
            polygons=outer, buffers=_NO_TAPER, physical_name="outer", mesh_order=2
        ),
        PolyPrism(
            polygons=inner, buffers=_NO_TAPER, physical_name="inner", mesh_order=1
        ),
    ]


def _scene_nested_with_donut_outer() -> list:
    """Outer prism has a hole; inner prism plugs it (every outer face is shared)."""
    outer = shapely.Polygon(
        [(0, 0), (10, 0), (10, 10), (0, 10)],
        holes=[[(4, 4), (6, 4), (6, 6), (4, 6)]],
    )
    inner = shapely.Polygon([(4, 4), (6, 4), (6, 6), (4, 6)])
    return [
        PolyPrism(polygons=outer, buffers=_NO_TAPER, physical_name="O", mesh_order=2),
        PolyPrism(polygons=inner, buffers=_NO_TAPER, physical_name="I", mesh_order=1),
    ]


def _arc_polygon(n_arc_pts: int = 24, radius: float = 5.0) -> shapely.Polygon:
    """Quarter-disc with a many-vertex circular arc."""
    theta = np.linspace(0.0, np.pi / 2, n_arc_pts)
    arc_pts = [(radius * np.cos(t), radius * np.sin(t)) for t in theta]
    # Close back to origin through the axes.
    return shapely.Polygon([(0.0, 0.0), *arc_pts, (0.0, 0.0)])


def _scene_arc_polysurface() -> list:
    """2D arc-bearing surface with ``identify_arcs=True``."""
    return [
        PolySurface(
            polygons=_arc_polygon(),
            physical_name="quarter_disc",
            mesh_order=1,
            identify_arcs=True,
            min_arc_points=4,
            arc_tolerance=1e-3,
        ),
    ]


def _scene_arc_polyprism_with_inner_box() -> list:
    """Extruded arc-bearing prism with a small interior box that shares one face.

    The inner box is positioned so its +x face coincides with the outer
    prism's flat right edge (x=5 line, from origin out to radius 5).
    Both backends must tag inner___outer for the shared face and emit
    the curved outer boundary as a single ``outer___None`` group whose
    mass equals the lateral arc-cylinder area minus the shared face.
    """
    outer = _arc_polygon(n_arc_pts=24, radius=5.0)
    inner = shapely.Polygon([(3, 0), (5, 0), (5, 2), (3, 2)])
    return [
        PolyPrism(
            polygons=outer,
            buffers=_NO_TAPER,
            physical_name="outer",
            mesh_order=2,
            identify_arcs=True,
            min_arc_points=4,
            arc_tolerance=1e-3,
        ),
        PolyPrism(
            polygons=inner,
            buffers=_NO_TAPER,
            physical_name="inner",
            mesh_order=1,
        ),
    ]


def _scene_embedded_internal_surface() -> list:
    """A 2D PolySurface embedded inside a 3D PolyPrism at z=0.5.

    The internal surface carves the prism into two stacked volumes
    sharing a tagged interior face.
    """
    vol_poly = shapely.Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
    surf_poly = shapely.Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
    return [
        PolyPrism(
            polygons=vol_poly,
            buffers={0.0: 0.0, 1.0: 0.0},
            physical_name="bulk",
            mesh_order=2,
        ),
        PolySurface(
            polygons=surf_poly,
            physical_name="midplane",
            mesh_order=1,
            translation=(0.0, 0.0, 0.5),
        ),
    ]


def _scene_keep_false_helper_corner_overlap() -> list:
    """Helper carves a corner of a kept prism; helper is not in the final model."""
    kept = shapely.Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
    helper = shapely.Polygon([(3, 3), (6, 3), (6, 6), (3, 6)])
    return [
        PolyPrism(polygons=kept, buffers=_NO_TAPER, physical_name="kept", mesh_order=1),
        PolyPrism(
            polygons=helper,
            buffers=_NO_TAPER,
            physical_name="helper",
            mesh_order=2,
            mesh_bool=False,
        ),
    ]


def _scene_interface_tag_between_prisms() -> list:
    """Two abutting prisms plus an explicit InterfaceTag at their shared face."""
    A = shapely.Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
    B = shapely.Polygon([(5, 0), (10, 0), (10, 5), (5, 5)])
    return [
        PolyPrism(polygons=A, buffers=_NO_TAPER, physical_name="A", mesh_order=1),
        PolyPrism(polygons=B, buffers=_NO_TAPER, physical_name="B", mesh_order=2),
        InterfaceTag(
            linestrings=LineString([(5, 0), (5, 5)]),
            zmin=0.0,
            zmax=1.0,
            physical_name="iface",
            mesh_order=3,
        ),
    ]


def _scene_multi_physical_names() -> list:
    """Single body carrying a tuple of physical names.

    Tagging must emit the entity under every name in the tuple, and
    interfaces must be tagged for every (name1, name2) pair.
    """
    A = shapely.Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
    B = shapely.Polygon([(5, 0), (10, 0), (10, 5), (5, 5)])
    return [
        PolyPrism(
            polygons=A,
            buffers=_NO_TAPER,
            physical_name=("A", "left"),
            mesh_order=1,
        ),
        PolyPrism(
            polygons=B,
            buffers=_NO_TAPER,
            physical_name="B",
            mesh_order=2,
        ),
    ]


# ---------------------------------------------------------------------------
# Parametrized test
# ---------------------------------------------------------------------------


# Each entry: (id, factory, comparator kwargs, xfail_reason).
# Kwargs can opt out of strict checks where a backend has a known
# topological artifact (e.g. extra edges from BRepAlgoAPI_Cut seams on
# 2D PolySurfaces; arcs from identify_arcs+OCC bridge yield different
# vertex/edge counts than the gmsh-side polygon-from-arcs construction).
# ``xfail_reason`` flags a real cross-backend divergence the test
# correctly catches but that needs a separate fix.
_SCENES: list[tuple[str, Callable[[], list], dict, str | None]] = [
    ("two_abutting_prisms", _scene_two_abutting_prisms, {}, None),
    ("three_in_a_row", _scene_three_in_a_row, {}, None),
    (
        "nested_prism_sharing_one_face",
        _scene_nested_prism_sharing_one_face,
        {},
        None,
    ),
    (
        "donut_with_inner_plug",
        _scene_nested_with_donut_outer,
        {},
        # cad_occ produces a topologically-broken outer solid for a
        # shapely Polygon with a hole when the hole is plugged by a
        # neighbour body: ``gmsh.model.occ.getMass`` reports the outer
        # volume as 104 (vs the correct 96) and the outer's exterior
        # surface area as 248 (vs 232). The 4 hole side walls leak
        # into ``O___None`` even though they're also correctly tagged
        # as ``O___I``. Tracked separately -- the test pins the bug.
        "cad_occ donut topology bug: hole walls leak into exterior, "
        "outer volume over-counted by the plug's volume",
    ),
    (
        "arc_polysurface",
        _scene_arc_polysurface,
        # Lower dim counts (vertices, edges) can legitimately differ
        # between the two arc-identifier paths -- only the surface area
        # is invariant. Disable strict per-entity boundary-area check
        # since arc curve decomposition into edges can differ.
        {
            "ignore_entity_count_dims": {0, 1},
            "check_boundary_signature": False,
        },
        None,
    ),
    (
        "arc_polyprism_with_inner_box",
        _scene_arc_polyprism_with_inner_box,
        {
            "ignore_entity_count_dims": {0, 1},
            "check_boundary_signature": False,
        },
        None,
    ),
    (
        "embedded_internal_surface",
        _scene_embedded_internal_surface,
        # The embedded-surface interface naming (``midplane___bulk``)
        # is not yet symmetric across backends; existing cross-compare
        # test ignores it too. Body masses must still match.
        {
            "ignore_groups": {"bulk___midplane", "midplane"},
        },
        None,
    ),
    (
        "keep_false_helper_corner_overlap",
        _scene_keep_false_helper_corner_overlap,
        # The helper itself is keep=False; cad_gmsh removes it from the
        # model entirely while cad_occ omits it from the BREP. Compare
        # only what survives: kept body + its interfaces.
        {"ignore_groups": {"helper"}},
        None,
    ),
    (
        "interface_tag_between_prisms",
        _scene_interface_tag_between_prisms,
        # InterfaceTag emits one logical face that gmsh splits into 3
        # face entities (each fragment of the AB shared face) while
        # cad_occ keeps it as a single face. The total iface area is
        # invariant; the n_entities + per-entity boundary structure
        # legitimately differs. The same applies to A___B / B___A
        # which sits on the same face. Edge counts also differ for
        # the same reason.
        {
            "ignore_entity_count_dims": {1, 2},
            "mass_only_groups": {"iface", "A___B"},
        },
        None,
    ),
    ("multi_physical_names", _scene_multi_physical_names, {}, None),
]


def _parametrize_scene(sid, factory, kwargs, xfail_reason):
    marks = (
        [pytest.mark.xfail(reason=xfail_reason, strict=True)] if xfail_reason else []
    )
    return pytest.param(factory, kwargs, id=sid, marks=marks)


@pytest.mark.parametrize(
    ("factory", "kwargs"),
    [_parametrize_scene(sid, fac, kw, xf) for sid, fac, kw, xf in _SCENES],
)
def test_loaded_state_matches(factory, kwargs, tmp_path):
    s_gmsh, s_occ = _capture_both(factory, tmp_path)
    _assert_loaded_equivalent(s_gmsh, s_occ, **kwargs)
