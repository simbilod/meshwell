"""TransfiniteResolutionSpec: structured 2D grids on cohort regions.

Every model below uses one lower-priority cladding box overlapping the
cohorts; the cohorts carve it (see ``test_e2e_cohort_carve.py``).
Expected counts are ``round(length / spacing)`` per curve, raised until
opposite sides of every region match (finest wins).
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pytest
import shapely
import shapely.affinity

from meshwell.orchestrator import generate_mesh
from meshwell.polyprism import PolyPrism
from meshwell.resolution import (
    RecombineResolutionSpec,
    StructuredExtrusionResolutionSpec,
    TransfiniteResolutionSpec,
)
from meshwell.structured.exceptions import (
    StructuredError,
    StructuredTransfiniteShapeError,
)
from meshwell.utils import deserialize

_VOLUME_TYPES = {"tetra", "pyramid", "wedge", "hexahedron"}


def _prism(poly, z0, z1, name, structured=False, mesh_order=None):
    return PolyPrism(
        polygons=poly,
        buffers={z0: 0.0, z1: 0.0},
        physical_name=name,
        structured=structured,
        mesh_order=mesh_order,
    )


def _cohort(poly, z0, z1, name):
    return _prism(poly, z0, z1, name, structured=True, mesh_order=1)


def _clad(*cohort_polys, z1=1.0):
    dom = shapely.union_all(cohort_polys).envelope.buffer(1.0, join_style="mitre")
    return _prism(dom, -0.5, z1, "clad")


def _specs(spacings, recombine=True, n_layers=2, **kw):
    """``{name: (long, short)}`` -> resolution specs."""
    out = {}
    for name, (long_axis, short_axis) in spacings.items():
        specs = [
            StructuredExtrusionResolutionSpec(n_layers=n_layers),
            TransfiniteResolutionSpec(long_axis=long_axis, short_axis=short_axis, **kw),
        ]
        if recombine:
            specs.append(RecombineResolutionSpec())
        out[name] = specs
    return out


def _mesh(entities, resolution_specs, **kw):
    return generate_mesh(
        entities=entities,
        dim=3,
        output_mesh=None,
        default_characteristic_length=0.5,
        resolution_specs=resolution_specs,
        **kw,
    )


def _volume_cells(mesh, name) -> Counter:
    out = Counter()
    for block, ids in zip(mesh.cells, mesh.cell_sets[name], strict=True):
        if block.type in _VOLUME_TYPES and ids is not None and len(ids):
            out[block.type] += len(ids)
    return out


def _sector(r0, r1, a0, a1, n=64) -> shapely.Polygon:
    t = np.radians(np.linspace(a0, a1, n))
    outer = np.c_[r1 * np.cos(t), r1 * np.sin(t)]
    inner = np.c_[r0 * np.cos(t), r0 * np.sin(t)][::-1]
    return shapely.Polygon(np.vstack([outer, inner]))


CORE = shapely.box(0, 0, 4, 1)


# ---------------------------------------------------------------------------
# Single region
# ---------------------------------------------------------------------------


def test_rectangle_recombined_is_all_hexes():
    mesh = _mesh(
        [_cohort(CORE, 0, 0.4, "core"), _clad(CORE)],
        _specs({"core": (0.5, 0.25)}),
    )
    assert _volume_cells(mesh, "core") == {"hexahedron": 8 * 4 * 2}


def test_rectangle_without_recombine_is_structured_wedges():
    mesh = _mesh(
        [_cohort(CORE, 0, 0.4, "core"), _clad(CORE)],
        _specs({"core": (0.5, 0.25)}, recombine=False),
    )
    assert _volume_cells(mesh, "core") == {"wedge": 2 * 8 * 4 * 2}


def test_rotated_rectangle_classifies_sides():
    core = shapely.affinity.rotate(CORE, 30, origin=(0, 0))
    mesh = _mesh(
        [_cohort(core, 0, 0.4, "core"), _clad(core)],
        _specs({"core": (0.5, 0.1)}),
    )
    assert _volume_cells(mesh, "core") == {"hexahedron": 8 * 10 * 2}


def test_t_junction_side_chains_curves():
    """An unstructured neighbour splits one side into three curves."""
    nb = shapely.box(1, 1, 2, 2)
    mesh = _mesh(
        [
            _cohort(CORE, 0, 0.4, "core"),
            _prism(nb, 0, 0.4, "nb", mesh_order=2),
            _clad(CORE, nb),
        ],
        _specs({"core": (0.5, 0.25)}),
    )
    assert _volume_cells(mesh, "core") == {"hexahedron": 8 * 4 * 2}
    assert _volume_cells(mesh, "nb")


def test_curved_sector():
    """30-degree annular sector: outer arc 8, inner arc 7 -> 8; radial 5."""
    arc = _sector(4.0, 4.5, 45, 75)
    mesh = _mesh(
        [_cohort(arc, 0, 0.4, "arc"), _clad(arc)],
        _specs({"arc": (0.3, 0.1)}),
        identify_arcs=True,
    )
    assert _volume_cells(mesh, "arc") == {"hexahedron": 8 * 5 * 2}


# ---------------------------------------------------------------------------
# Coupled regions
# ---------------------------------------------------------------------------


def test_adjacent_physicals_finest_wins():
    """The shared side takes b's finer count, which propagates across a."""
    a, b = shapely.box(0, 0, 2, 1), shapely.box(2, 0, 4, 1)
    mesh = _mesh(
        [_cohort(a, 0, 0.4, "a"), _cohort(b, 0, 0.4, "b"), _clad(a, b)],
        _specs({"a": (0.5, 0.25), "b": (0.25, 0.1)}),
    )
    assert _volume_cells(mesh, "a") == {"hexahedron": 4 * 10 * 2}
    assert _volume_cells(mesh, "b") == {"hexahedron": 8 * 10 * 2}


def test_stacked_slabs_finest_in_column():
    """A half-length upper slab splits the lower one into two columns."""
    hi = shapely.box(0, 0, 2, 1)
    mesh = _mesh(
        [_cohort(CORE, 0, 0.4, "lo"), _cohort(hi, 0.4, 0.8, "hi"), _clad(CORE)],
        _specs({"lo": (0.5, 0.25), "hi": (0.25, 0.125)}),
    )
    # Column under hi: 8 x 8; the other column: 4 x 8 (shared side).
    assert _volume_cells(mesh, "lo") == {"hexahedron": (8 * 8 + 4 * 8) * 2}
    assert _volume_cells(mesh, "hi") == {"hexahedron": 8 * 8 * 2}


def test_stepped_cohort_periodic_root_wins():
    """The overhang's side is a periodic copy of the lower column's side.

    lo [0,4] x [0,1.5] (0.4 / 0.1), hi [2,6] x [0,1.5] (0.3 / 0.2):
    columns [0,2] 5 x 15 and [2,4] 7 x 15; the overhang [4,6] would be
    7 x 8 alone, but its x=4 side chains down to the lower column's 15.
    """
    lo, hi = shapely.box(0, 0, 4, 1.5), shapely.box(2, 0, 6, 1.5)
    specs = _specs({"lo": (0.4, 0.1)}) | _specs({"hi": (0.3, 0.2)}, n_layers=3)
    mesh = _mesh(
        [_cohort(lo, 0, 0.4, "lo"), _cohort(hi, 0.4, 0.8, "hi"), _clad(lo, hi)],
        specs,
    )
    assert _volume_cells(mesh, "lo") == {"hexahedron": (5 * 15 + 7 * 15) * 2}
    assert _volume_cells(mesh, "hi") == {"hexahedron": (7 * 15 + 7 * 15) * 3}


def test_l_split_into_two_rectangles_passes():
    a, b = shapely.box(0, 0, 2, 1), shapely.box(0, 1, 1, 2)
    mesh = _mesh(
        [_cohort(a, 0, 0.4, "a"), _cohort(b, 0, 0.4, "b"), _clad(a, b)],
        _specs({"a": (0.25, 0.25), "b": (0.25, 0.25)}),
    )
    assert _volume_cells(mesh, "a") == {"hexahedron": 8 * 4 * 2}
    assert _volume_cells(mesh, "b") == {"hexahedron": 4 * 4 * 2}


# ---------------------------------------------------------------------------
# Rejected shapes and specs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("poly", "reason"),
    [
        (shapely.box(0, 0, 2, 2).difference(shapely.box(0.5, 0.5, 1.5, 1.5)), "loop"),
        (shapely.box(0, 0, 2, 2).difference(shapely.box(1, 1, 2, 2)), "6 corners"),
        (shapely.Polygon([(0, 0), (2, 0), (2, 1.5), (1.5, 2), (0, 2)]), "5 corners"),
    ],
    ids=["island", "l_notch", "chamfer"],
)
def test_non_four_sided_region_raises(poly, reason):
    with pytest.raises(StructuredTransfiniteShapeError, match=reason) as exc:
        _mesh(
            [_cohort(poly, 0, 0.4, "core"), _clad(poly)], _specs({"core": (0.5, 0.5)})
        )
    assert exc.value.physical_name == "core"


def test_mass_filter_skips_region():
    """A region outside [min_mass, max_mass] is left unstructured (no raise)."""
    poly = shapely.box(0, 0, 2, 2).difference(shapely.box(1, 1, 2, 2))
    mesh = _mesh(
        [_cohort(poly, 0, 0.4, "core"), _clad(poly)],
        _specs({"core": (0.5, 0.5)}, recombine=False, max_mass=1.0),
    )
    assert set(_volume_cells(mesh, "core")) == {"wedge"}


def test_spec_on_unstructured_physical_raises():
    specs = _specs({"core": (0.5, 0.25)})
    specs["clad"] = [TransfiniteResolutionSpec(long_axis=0.5, short_axis=0.5)]
    with pytest.raises(StructuredError, match=r"not .*cohort"):
        _mesh([_cohort(CORE, 0, 0.4, "core"), _clad(CORE)], specs)


def test_spec_without_cohorts_raises():
    with pytest.raises(StructuredError):
        _mesh(
            [_prism(CORE, 0, 0.4, "core")],
            {"core": [TransfiniteResolutionSpec(long_axis=0.5, short_axis=0.5)]},
        )


def test_duplicate_spec_raises():
    specs = _specs({"core": (0.5, 0.25)})
    specs["core"].append(TransfiniteResolutionSpec(long_axis=0.5, short_axis=0.5))
    with pytest.raises(StructuredError, match="TransfiniteResolutionSpec entries"):
        _mesh([_cohort(CORE, 0, 0.4, "core"), _clad(CORE)], specs)


# ---------------------------------------------------------------------------
# Spec object
# ---------------------------------------------------------------------------


def test_refine_scales_spacings():
    spec = TransfiniteResolutionSpec(long_axis=0.4, short_axis=0.1)
    fine = spec.refine(0.5)
    assert math.isclose(fine.long_axis, 0.2)
    assert math.isclose(fine.short_axis, 0.05)
    assert spec.long_axis == 0.4


def test_serialization_roundtrip():
    spec = TransfiniteResolutionSpec(
        long_axis=0.4, short_axis=0.1, corner_angle_tol=20.0, min_mass=0.5
    )
    d = spec.to_dict()
    assert d["resolution_type"] == "TransfiniteResolutionSpec"
    back = deserialize(d)
    assert isinstance(back, TransfiniteResolutionSpec)
    assert back == spec


@pytest.mark.parametrize("kw", [{"long_axis": 0.0}, {"short_axis": -1.0}])
def test_non_positive_spacing_rejected(kw):
    with pytest.raises(ValueError, match="greater than 0"):
        TransfiniteResolutionSpec(**({"long_axis": 0.5, "short_axis": 0.5} | kw))
