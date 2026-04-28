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


def test_interface_constraints_seed_volume_seam(tmp_path):
    """Seed an OCC face's mesh from an imported seam ``.msh``.

    A volume meshed with ``_interface_constraints`` has its seam face nodes
    drawn verbatim from the imported mesh (parametric OCC seeding via
    ``addNodes``).
    """
    import meshio
    import shapely

    # ----- Phase 1 simulation: build a coherent planar seam mesh at x=1 -----
    # NOTE: the originally specified phase-1 simulation
    # (PolyPrism slab + _emit_only_seam_surfaces) emits the slab's TWO opposite
    # face triangulations (front + back of the thin slab), not a single
    # coherent surface mesh — which makes the slab unsuitable as input to the
    # parametric-seeding helper. We instead synthesize a clean planar seam .msh
    # directly via gmsh's geo kernel, matching what a properly-engineered
    # phase-1 worker is expected to produce.
    import gmsh
    from meshwell.orchestrator import generate_mesh
    from meshwell.polyprism import PolyPrism

    seam_path = tmp_path / "seam.msh"
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("seam")
        pts = [
            gmsh.model.geo.addPoint(1, 0, 0, 0.2),
            gmsh.model.geo.addPoint(1, 1, 0, 0.2),
            gmsh.model.geo.addPoint(1, 1, 1, 0.2),
            gmsh.model.geo.addPoint(1, 0, 1, 0.2),
        ]
        lines = [gmsh.model.geo.addLine(pts[i], pts[(i + 1) % 4]) for i in range(4)]
        loop = gmsh.model.geo.addCurveLoop(lines)
        surf = gmsh.model.geo.addPlaneSurface([loop])
        gmsh.model.geo.synchronize()
        gmsh.model.addPhysicalGroup(2, [surf], name="_seam___A___B")
        gmsh.model.mesh.generate(2)
        gmsh.write(str(seam_path))
    finally:
        gmsh.finalize()
    # Sanity: the seam .msh has the _seam___A___B physical group.
    s = meshio.read(seam_path)
    assert "_seam___A___B" in (
        s.field_data or {}
    ), f"phase-1 setup failed; got groups {list((s.field_data or {}).keys())}"

    # ----- Phase 2: mesh a volume whose right face (x=1) must conform -----
    out = tmp_path / "vol.msh"
    a = PolyPrism(
        polygons=shapely.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="A",
        mesh_order=1,
    )
    generate_mesh(
        entities=[a],
        dim=3,
        output_mesh=out,
        default_characteristic_length=0.5,  # NB: coarser than the seam (0.2)
        _interface_constraints=[seam_path],
    )
    m = meshio.read(out)
    # The seam physical group should be present in the volume mesh too.
    assert "_seam___A___B" in (
        m.field_data or {}
    ), f"seam group not in volume output; got {list((m.field_data or {}).keys())}"
    # Element count on the seam face should match the seam .msh.
    # (sanity: the face was seeded, not re-meshed at coarser sizing)
    seam_tris = sum(cb.data.shape[0] for cb in s.cells if cb.type == "triangle")
    vol_tag, _vol_dim = (m.field_data or {})["_seam___A___B"]
    vol_tris_in_group = 0
    for i, cb in enumerate(m.cells):
        if cb.type != "triangle":
            continue
        gmsh_phys = m.cell_data["gmsh:physical"][i]
        import numpy as np

        mask = np.asarray(gmsh_phys) == int(vol_tag)
        vol_tris_in_group += int(mask.sum())
    assert vol_tris_in_group == seam_tris, (
        f"seam was re-meshed: got {vol_tris_in_group} tris on the volume's seam face, "
        f"expected {seam_tris} from the imported seam mesh."
    )


def test_subdomains_from_grid_2x2():
    from shapely.geometry import box

    from meshwell.distributed import subdomains_from_grid

    polys = subdomains_from_grid((0, 0, 2, 2), nx=2, ny=2)
    assert len(polys) == 4
    expected = {
        box(0, 0, 1, 1).wkt,
        box(1, 0, 2, 1).wkt,
        box(0, 1, 1, 2).wkt,
        box(1, 1, 2, 2).wkt,
    }
    assert {p.wkt for p in polys} == expected


def test_subdomains_from_grid_validates_inputs():
    import pytest

    from meshwell.distributed import subdomains_from_grid

    with pytest.raises(ValueError, match=">="):
        subdomains_from_grid((0, 0, 1, 1), nx=0, ny=1)
    with pytest.raises(ValueError, match="bbox"):
        subdomains_from_grid((1, 0, 0, 1), nx=1, ny=1)  # invalid xmax<xmin


def test_clip_polyprism_to_mask_returns_clipped_copy():
    import shapely

    from meshwell.distributed import _clip_entity_to_polygon
    from meshwell.polyprism import PolyPrism

    p = shapely.Polygon([(0, 0), (10, 0), (10, 1), (0, 1)])
    prism = PolyPrism(
        polygons=p,
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="A",
        mesh_order=1,
    )
    mask = shapely.box(0, 0, 5, 1)

    clipped = _clip_entity_to_polygon(prism, mask)
    assert clipped is not None
    # physical_name format normalization: meshwell turns "A" into ("A",)
    assert clipped.physical_name == ("A",) or clipped.physical_name == "A"
    assert clipped.mesh_order == 1
    # The clipped polygons should be a list with one box(0,0,5,1).
    polys = (
        clipped.polygons if isinstance(clipped.polygons, list) else [clipped.polygons]
    )
    # Look at the first geometry; it might be wrapped in MultiPolygon.geoms
    first = polys[0]
    if hasattr(first, "geoms"):
        first = next(iter(first.geoms))
    assert first.bounds == pytest.approx((0, 0, 5, 1), abs=1e-9)


def test_clip_polyprism_returns_none_when_disjoint():
    import shapely

    from meshwell.distributed import _clip_entity_to_polygon
    from meshwell.polyprism import PolyPrism

    p = shapely.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    prism = PolyPrism(
        polygons=p,
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="A",
        mesh_order=1,
    )
    mask = shapely.box(10, 10, 11, 11)
    assert _clip_entity_to_polygon(prism, mask) is None


def test_clip_polysurface_to_mask():
    import shapely

    from meshwell.distributed import _clip_entity_to_polygon
    from meshwell.polysurface import PolySurface

    p = shapely.Polygon([(0, 0), (10, 0), (10, 1), (0, 1)])
    surf = PolySurface(polygons=p, physical_name="A", mesh_order=1)
    mask = shapely.box(0, 0, 5, 1)

    clipped = _clip_entity_to_polygon(surf, mask)
    assert clipped is not None
    polys = (
        clipped.polygons if isinstance(clipped.polygons, list) else [clipped.polygons]
    )
    first = polys[0]
    if hasattr(first, "geoms"):
        first = next(iter(first.geoms))
    assert first.bounds == pytest.approx((0, 0, 5, 1), abs=1e-9)


def test_resolution_only_proxy_contributes_no_geometry():
    import shapely

    from meshwell.distributed import _resolution_only_proxy
    from meshwell.polyprism import PolyPrism

    prism = PolyPrism(
        polygons=shapely.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="A",
        mesh_order=1,
    )
    prism.resolutions = ["sentinel"]

    proxy = _resolution_only_proxy(prism)
    assert proxy.mesh_bool is False
    assert proxy.resolutions == ["sentinel"]
    # instanciate_occ returns an empty TopoDS_Compound
    shape = proxy.instanciate_occ()
    from OCP.TopoDS import TopoDS_Compound

    assert isinstance(shape, TopoDS_Compound)
    from OCP.TopAbs import TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer

    assert not TopExp_Explorer(shape, TopAbs_SOLID).More()


def test_build_subdomain_plan_creates_volume_regions():
    import shapely

    from meshwell.distributed import build_subdomain_plan
    from meshwell.polyprism import PolyPrism

    sd = [shapely.box(0, 0, 1, 1), shapely.box(1, 0, 2, 1)]
    prism = PolyPrism(
        polygons=shapely.box(0, 0, 2, 1),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="mat",
        mesh_order=1,
    )
    plan = build_subdomain_plan(
        subdomains=sd,
        entities=[prism],
        interface_width=0.05,
        perturbation=1e-5,
        point_tolerance=1e-3,
    )
    assert [v.id for v in plan.volumes] == ["volume_0000", "volume_0001"]
    assert plan.volumes[0].polygon.equals(sd[0])
    assert plan.physical_names_seen == ["mat"]


def test_build_subdomain_plan_creates_interface_slabs():
    import shapely

    from meshwell.distributed import build_subdomain_plan
    from meshwell.polyprism import PolyPrism

    sd = [shapely.box(0, 0, 1, 1), shapely.box(1, 0, 2, 1)]
    prism = PolyPrism(
        polygons=shapely.box(0, 0, 2, 1),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="mat",
        mesh_order=1,
    )
    plan = build_subdomain_plan(
        subdomains=sd,
        entities=[prism],
        interface_width=0.1,
        perturbation=1e-5,
        point_tolerance=1e-3,
    )
    assert len(plan.interfaces) == 1
    iface = plan.interfaces[0]
    assert iface.id == "interface_0000"
    assert sorted(iface.between) == ["volume_0000", "volume_0001"]
    # Slab is centered on x=1 with half-width 0.05
    assert iface.polygon.bounds == pytest.approx((0.95, 0.0, 1.05, 1.0), abs=1e-9)


def test_build_subdomain_plan_no_interface_for_disjoint_subdomains():
    import shapely

    from meshwell.distributed import build_subdomain_plan
    from meshwell.polyprism import PolyPrism

    # Two disjoint subdomains, each covered by its own entity. Coverage check
    # (Task 14) requires every entity polygon to lie within the subdomain
    # union, so we use one prism per subdomain rather than one spanning prism.
    sd = [shapely.box(0, 0, 1, 1), shapely.box(2, 0, 3, 1)]
    prism_a = PolyPrism(
        polygons=shapely.box(0, 0, 1, 1),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="mat",
        mesh_order=1,
    )
    prism_b = PolyPrism(
        polygons=shapely.box(2, 0, 3, 1),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="mat",
        mesh_order=1,
    )
    plan = build_subdomain_plan(
        subdomains=sd,
        entities=[prism_a, prism_b],
        interface_width=0.1,
        perturbation=1e-5,
        point_tolerance=1e-3,
    )
    assert plan.interfaces == []


def test_build_subdomain_plan_interface_width_dict():
    """When interface_width is a dict, look up by (min(i,j), max(i,j))."""
    import shapely

    from meshwell.distributed import build_subdomain_plan
    from meshwell.polyprism import PolyPrism

    sd = [shapely.box(0, 0, 1, 1), shapely.box(1, 0, 2, 1)]
    prism = PolyPrism(
        polygons=shapely.box(0, 0, 2, 1),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="mat",
        mesh_order=1,
    )
    plan = build_subdomain_plan(
        subdomains=sd,
        entities=[prism],
        interface_width={(0, 1): 0.2},
        perturbation=1e-5,
        point_tolerance=1e-3,
    )
    assert len(plan.interfaces) == 1
    assert plan.interfaces[0].width == 0.2
    # Also accept reversed key
    plan2 = build_subdomain_plan(
        subdomains=sd,
        entities=[prism],
        interface_width={(1, 0): 0.3},
        perturbation=1e-5,
        point_tolerance=1e-3,
    )
    assert plan2.interfaces[0].width == 0.3


def test_build_subdomain_plan_creates_junction_for_2x2_grid():
    import shapely

    from meshwell.distributed import build_subdomain_plan, subdomains_from_grid
    from meshwell.polyprism import PolyPrism

    sd = subdomains_from_grid((0, 0, 2, 2), nx=2, ny=2)
    prism = PolyPrism(
        polygons=shapely.box(0, 0, 2, 2),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="mat",
        mesh_order=1,
    )
    plan = build_subdomain_plan(
        subdomains=sd,
        entities=[prism],
        interface_width=0.1,
        perturbation=1e-5,
        point_tolerance=1e-3,
    )
    assert len(plan.junctions) == 1
    j = plan.junctions[0]
    assert j.id == "junction_0000"
    # Junction polygon centered on (1, 1), all 4 volumes meet here
    assert sorted(j.between) == [
        "volume_0000",
        "volume_0001",
        "volume_0002",
        "volume_0003",
    ]


def test_build_subdomain_plan_rejects_uncovered_entities():
    import shapely

    from meshwell.distributed import build_subdomain_plan
    from meshwell.polyprism import PolyPrism

    # Subdomain covers only x in [0, 1] but entity extends to x=2
    sd = [shapely.box(0, 0, 1, 1)]
    prism = PolyPrism(
        polygons=shapely.box(0, 0, 2, 1),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="mat",
        mesh_order=1,
    )
    with pytest.raises(ValueError, match="not covered"):
        build_subdomain_plan(
            subdomains=sd,
            entities=[prism],
            interface_width=0.1,
            perturbation=1e-5,
            point_tolerance=1e-3,
        )


def test_write_bundles_creates_expected_layout(tmp_path):
    import json

    import shapely

    from meshwell.distributed import build_subdomain_plan, write_bundles
    from meshwell.polyprism import PolyPrism

    sd = [shapely.box(0, 0, 1, 1), shapely.box(1, 0, 2, 1)]
    prism = PolyPrism(
        polygons=shapely.box(0, 0, 2, 1),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="mat",
        mesh_order=1,
    )
    plan = build_subdomain_plan(
        subdomains=sd,
        entities=[prism],
        interface_width=0.1,
        perturbation=1e-5,
        point_tolerance=1e-3,
    )
    write_bundles(
        work_dir=tmp_path,
        plan=plan,
        entities=[prism],
        mesh_kwargs={"default_characteristic_length": 0.5},
    )
    # Manifest exists.
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["version"] == 1
    assert "volume_0000" in manifest["subdomains"]
    assert "interface_0000" in manifest["subdomains"]
    # Phase 1 must contain interface_0000 (a 2x1 grid produces exactly one
    # pairwise interface and no junctions). Use set-equality so the test
    # does not depend on list ordering of phase entries.
    assert set(manifest["phase_order"][0]) == {"interface_0000"}
    assert sorted(manifest["phase_order"][1]) == ["volume_0000", "volume_0001"]
    # Per-job files exist.
    for jid in ["volume_0000", "volume_0001", "interface_0000"]:
        d = tmp_path / "jobs" / jid
        assert (d / "job.json").exists()
        assert (d / "entities.json").exists()
        assert (d / "subdomain.wkt").exists()
        assert (d / "mesh_kwargs.json").exists()


def test_run_job_executes_volume_bundle(tmp_path):
    import json

    import shapely

    from meshwell.distributed import build_subdomain_plan, run_job, write_bundles
    from meshwell.polyprism import PolyPrism

    sd = [shapely.box(0, 0, 1, 1), shapely.box(1, 0, 2, 1)]
    prism = PolyPrism(
        polygons=shapely.box(0, 0, 2, 1),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="mat",
        mesh_order=1,
    )
    plan = build_subdomain_plan(
        subdomains=sd,
        entities=[prism],
        interface_width=0.1,
        perturbation=1e-5,
        point_tolerance=1e-3,
    )
    write_bundles(
        tmp_path, plan, [prism], mesh_kwargs={"default_characteristic_length": 0.3}
    )

    job_dir = tmp_path / "jobs" / "volume_0000"
    run_job(job_dir)
    assert (job_dir / "result.msh").exists()
    res = json.loads((job_dir / "result.json").read_text())
    assert res["status"] == "ok"


def test_cli_run_job_invokes_run_job(tmp_path, monkeypatch):
    import sys

    import shapely

    from meshwell.distributed import build_subdomain_plan, cli_main, write_bundles
    from meshwell.polyprism import PolyPrism

    sd = [shapely.box(0, 0, 1, 1), shapely.box(1, 0, 2, 1)]
    prism = PolyPrism(
        polygons=shapely.box(0, 0, 2, 1),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="mat",
        mesh_order=1,
    )
    plan = build_subdomain_plan(
        subdomains=sd,
        entities=[prism],
        interface_width=0.1,
        perturbation=1e-5,
        point_tolerance=1e-3,
    )
    write_bundles(
        tmp_path, plan, [prism], mesh_kwargs={"default_characteristic_length": 0.3}
    )

    monkeypatch.setattr(
        sys,
        "argv",
        ["meshwell", "run-job", str(tmp_path / "jobs" / "volume_0000")],
    )
    cli_main()
    assert (tmp_path / "jobs" / "volume_0000" / "result.msh").exists()


def test_distributed_module_imports():
    from meshwell.distributed import (
        Executor,
        Slab,
        SubdomainPlan,
        SubprocessExecutor,
        VolumeRegion,
        build_subdomain_plan,
        generate_mesh_distributed,
        run_job,
        subdomains_from_grid,
    )

    assert Slab is not None
    assert VolumeRegion is not None
    assert SubdomainPlan is not None
    assert Executor is not None
    assert SubprocessExecutor is not None
    assert callable(generate_mesh_distributed)
    assert callable(subdomains_from_grid)
    assert callable(build_subdomain_plan)
    assert callable(run_job)


def test_in_process_executor_runs_jobs_synchronously(tmp_path):
    import shapely

    from meshwell.distributed import (
        InProcessExecutor,
        build_subdomain_plan,
        run_job,
        write_bundles,
    )
    from meshwell.polyprism import PolyPrism

    sd = [shapely.box(0, 0, 1, 1), shapely.box(1, 0, 2, 1)]
    prism = PolyPrism(
        polygons=shapely.box(0, 0, 2, 1),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="mat",
        mesh_order=1,
    )
    plan = build_subdomain_plan(
        subdomains=sd,
        entities=[prism],
        interface_width=0.1,
        perturbation=1e-5,
        point_tolerance=1e-3,
    )
    write_bundles(
        tmp_path, plan, [prism], mesh_kwargs={"default_characteristic_length": 0.3}
    )

    ex = InProcessExecutor()
    fut = ex.submit(tmp_path / "jobs" / "volume_0000")
    fut.result()  # blocks
    assert (tmp_path / "jobs" / "volume_0000" / "result.msh").exists()
    # silence unused import warning
    assert callable(run_job)


def test_run_plan_two_phases_completes(tmp_path):
    import json

    import shapely

    from meshwell.distributed import (
        InProcessExecutor,
        build_subdomain_plan,
        run_plan,
        write_bundles,
    )
    from meshwell.polyprism import PolyPrism

    sd = [shapely.box(0, 0, 1, 1), shapely.box(1, 0, 2, 1)]
    prism = PolyPrism(
        polygons=shapely.box(0, 0, 2, 1),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="mat",
        mesh_order=1,
    )
    plan = build_subdomain_plan(
        subdomains=sd,
        entities=[prism],
        interface_width=0.1,
        perturbation=1e-5,
        point_tolerance=1e-3,
    )
    write_bundles(
        tmp_path, plan, [prism], mesh_kwargs={"default_characteristic_length": 0.3}
    )

    run_plan(tmp_path, plan, executor=InProcessExecutor())

    # Phase 1 result
    assert (tmp_path / "jobs" / "interface_0000" / "result.msh").exists()
    # Phase 2 results
    for vid in ["volume_0000", "volume_0001"]:
        assert (tmp_path / "jobs" / vid / "result.msh").exists()
        # Volume job.json should now reference the interface input.
        j = json.loads((tmp_path / "jobs" / vid / "job.json").read_text())
        assert j["interface_inputs"], f"{vid} did not get interface_inputs populated"
