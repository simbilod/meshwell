"""Tests for the distributed-meshing pipeline."""
from __future__ import annotations

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

    # No exception means success.
    generate_mesh(
        entities=[prism],
        dim=3,
        output_mesh=tmp_path / "out.msh",
        default_characteristic_length=0.5,
        resolution_specs={"A": [spec]},
        _global_physical_names=["A", "B"],  # B is "globally known" but not local
    )
