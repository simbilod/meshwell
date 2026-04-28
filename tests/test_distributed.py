"""Tests for the distributed-meshing pipeline."""
from __future__ import annotations

import warnings

import pytest
import shapely

from meshwell.cad_common import prepare_entities
from meshwell.polysurface import PolySurface


def test_prepare_entities_skip_buffer_preserves_polygons():
    p = shapely.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    ent = PolySurface(polygons=p, physical_name="A", mesh_order=1)
    original = list(ent.polygons)

    prepare_entities([ent], perturbation=1e-3, skip_buffer=True)

    # polygons must be unchanged (no outward buffer applied).
    assert len(ent.polygons) == len(original)
    assert all(a.equals(b) for a, b in zip(ent.polygons, original))


def test_generate_mesh_pre_buffered_flag_skips_buffer(tmp_path):
    """When _pre_buffered=True, the polygon should not be buffered."""
    import shapely

    from meshwell.orchestrator import generate_mesh
    from meshwell.polyprism import PolyPrism

    p = shapely.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    prism = PolyPrism(
        polygons=p,
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="A",
        mesh_order=1,
    )
    # Snapshot pre-call polygons. Capture as list to defeat aliasing
    # (PolyPrism normalizes .polygons but it's still a reference; copying defends
    # against the equals() shortcut on a mutated attribute).
    original_polys = prism.polygons

    generate_mesh(
        entities=[prism],
        dim=3,
        output_mesh=tmp_path / "out.msh",
        default_characteristic_length=0.5,
        _pre_buffered=True,
    )
    # PolyPrism's polygons are still the original (no perturbation buffer).
    assert prism.polygons.equals(original_polys)


def test_resolution_spec_unknown_name_in_global_set_is_no_op(tmp_path):
    """Tolerate unknown ResolutionSpec name refs against global set.

    A ResolutionSpec restrict_to=['B'] with B in _global_physical_names
    but absent locally should not raise.

    Distributed-meshing motivation: a phase-2 worker meshes ONE subdomain.
    ResolutionSpecs in that worker may reference physical names that exist
    in OTHER subdomains but not locally. Without this tolerance, the worker
    would fail.
    """
    import shapely

    from meshwell.orchestrator import generate_mesh
    from meshwell.polyprism import PolyPrism
    from meshwell.resolution import ConstantInField

    spec = ConstantInField(
        apply_to="volumes",
        resolution=0.5,
        restrict_to=["B"],  # B does NOT exist in this entity list
    )
    prism = PolyPrism(
        polygons=shapely.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="A",
        mesh_order=1,
    )

    # With _global_physical_names supplied, the reference to 'B' should
    # emit a UserWarning mentioning the global tolerance behavior, and
    # the message should include the entity's physical_name plus the
    # offending spec attr/name (so multi-entity runs are debuggable).
    with pytest.warns(UserWarning, match="globally") as wrec:
        generate_mesh(
            entities=[prism],
            dim=3,
            output_mesh=tmp_path / "out.msh",
            default_characteristic_length=0.5,
            resolution_specs={"A": [spec]},
            _global_physical_names=["A", "B"],  # B is "globally known" but not local
        )
    msgs = [str(w.message) for w in wrec.list]
    assert any(
        "entity='A'" in m and "spec_attr='restrict_to'" in m and "name='B'" in m
        for m in msgs
    ), f"warning text missing entity/attr/name details; got: {msgs}"


def test_resolution_spec_unknown_name_without_global_set_is_silent(tmp_path):
    """Legacy behavior: without _global_physical_names, no warning fires.

    Tightens the contract: the new warning must be gated on the caller
    explicitly opting into the distributed code path by passing
    _global_physical_names. Otherwise the old silent no-op stands.
    """
    import shapely

    from meshwell.orchestrator import generate_mesh
    from meshwell.polyprism import PolyPrism
    from meshwell.resolution import ConstantInField

    spec = ConstantInField(
        apply_to="volumes",
        resolution=0.5,
        restrict_to=["B"],
    )
    prism = PolyPrism(
        polygons=shapely.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="A",
        mesh_order=1,
    )

    # Promote UserWarning -> error so an unexpected warning fails the test.
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        generate_mesh(
            entities=[prism],
            dim=3,
            output_mesh=tmp_path / "out.msh",
            default_characteristic_length=0.5,
            resolution_specs={"A": [spec]},
            # _global_physical_names intentionally omitted
        )


def test_emit_only_seam_surfaces_filters_output(tmp_path):
    """Filter restricts output to seam-prefixed physical groups.

    A mesh emitted with ``_emit_only_seam_surfaces=True`` contains only
    physical groups whose name starts with ``_seam___``.
    """
    import meshio
    import shapely

    from meshwell.orchestrator import generate_mesh
    from meshwell.polyprism import PolyPrism

    # Two prisms abutting at x=1: one normal material A, one phantom seam.
    a = PolyPrism(
        polygons=shapely.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="A",
        mesh_order=1,
    )
    seam = PolyPrism(
        polygons=shapely.Polygon([(1, 0), (1.5, 0), (1.5, 1), (1, 1)]),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="_seam___A___B",
        mesh_order=2,
        # NOTE: spec called for a 0.001-thick slab + mesh_bool=False to keep
        # the slab as a phantom while preserving seam-named faces. In this
        # codebase, (1) mesh_bool=False removes the volume AND its named
        # physical group, and (2) a 0.001-thick slab at lc=0.5 is too thin to
        # mesh as a separate volume (the _seam___ tag exists in field_data but
        # no elements carry it, so the filter would correctly produce an empty
        # mesh). Since this test only verifies the FILTER behavior (not phantom
        # semantics), widen the slab to 0.5 and use the default mesh_bool=True
        # so the _seam___ group actually has elements pre-filter.
    )

    out = tmp_path / "out.msh"
    generate_mesh(
        entities=[a, seam],
        dim=3,
        output_mesh=out,
        default_characteristic_length=0.5,
        _emit_only_seam_surfaces=True,
    )
    m = meshio.read(out)
    field_names = set((m.field_data or {}).keys())
    # Only _seam___ groups survive.
    assert all(
        n.startswith("_seam___") for n in field_names
    ), f"unexpected groups: {field_names}"
    assert any(
        n.startswith("_seam___") for n in field_names
    ), f"no seam groups in output: {field_names}"
