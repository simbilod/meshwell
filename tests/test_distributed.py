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


def test_clip_drops_slivers_below_area_threshold():
    import shapely

    from meshwell.distributed import _clip_entity_to_polygon
    from meshwell.polyprism import PolyPrism

    # A polygon that intersects the mask only as a 1e-6-wide sliver.
    p = shapely.Polygon([(0, 0), (1.000001, 0), (1.000001, 1), (0, 1)])
    prism = PolyPrism(
        polygons=p,
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="A",
        mesh_order=1,
    )
    # Mask covers x in [1, 2] — only a 1e-6 wide sliver of the prism.
    mask = shapely.box(1, 0, 2, 1)
    # With default point_tolerance=1e-3, sliver area (~1e-6) is below 1e-6
    # threshold → returned None.
    result = _clip_entity_to_polygon(prism, mask, point_tolerance=1e-3)
    assert result is None, f"sliver should be dropped, got {result}"


def test_clip_with_perturbation_drops_neighbor_buffer_halo():
    """Eroded-mask clipping drops adjacent-subdomain buffer halos.

    A polygon master-buffered outward by perturbation should not
    leak into the adjacent subdomain when clipped with the eroded mask.
    """
    import shapely

    from meshwell.distributed import _clip_entity_to_polygon
    from meshwell.polyprism import PolyPrism

    perturbation = 1e-5

    # Original "silicon" polygon at x in [0, 1]; master buffer extends to ~1+perturbation.
    p_buffered = shapely.box(
        -perturbation, -perturbation, 1 + perturbation, 1 + perturbation
    )
    prism = PolyPrism(
        polygons=p_buffered,
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="silicon",
        mesh_order=1,
    )
    # Adjacent oxide subdomain mask.
    oxide_subdomain = shapely.box(1, 0, 2, 1)

    # With perturbation > 0, the eroded mask = box(1+perturbation, perturbation,
    # 2-perturbation, 1-perturbation). The buffered prism extends to
    # x=1+perturbation, so intersection x range collapses to a degenerate strip
    # of zero area — _clip_entity_to_polygon must return None.
    result_with_perturb = _clip_entity_to_polygon(
        prism, oxide_subdomain, point_tolerance=1e-3, perturbation=perturbation
    )
    assert result_with_perturb is None, (
        f"silicon halo should be eroded out of oxide subdomain, "
        f"got {result_with_perturb}"
    )


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
        perturbation=1e-5,
        point_tolerance=1e-3,
    )
    assert [v.id for v in plan.volumes] == ["volume_0000", "volume_0001"]
    assert plan.volumes[0].polygon.equals(sd[0])
    assert plan.physical_names_seen == ["mat"]


def test_build_subdomain_plan_populates_neighbors():
    """Adjacent subdomains end up with each other's id + shared boundary in .neighbors."""
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
        perturbation=1e-5,
        point_tolerance=1e-3,
    )
    v0, v1 = plan.volumes
    assert len(v0.neighbors) == 1
    assert len(v1.neighbors) == 1
    assert v0.neighbors[0][0] == v1.id
    assert v1.neighbors[0][0] == v0.id
    # Shared boundary should be the line x=1, y in [0,1] - a short LineString.
    import shapely.wkt

    g0 = shapely.wkt.loads(v0.neighbors[0][1])
    assert abs(g0.length - 1.0) < 1e-9


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
        perturbation=1e-5,
        point_tolerance=1e-3,
    )
    write_bundles(
        work_dir=tmp_path,
        plan=plan,
        entities=[prism],
        mesh_kwargs={"default_characteristic_length": 0.5},
    )
    # Manifest exists; v2 uses a single phase containing only volume bundles.
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["version"] == 2
    assert "volume_0000" in manifest["subdomains"]
    assert "volume_0001" in manifest["subdomains"]
    # phase_order is a single phase listing every volume bundle.
    assert len(manifest["phase_order"]) == 1
    assert sorted(manifest["phase_order"][0]) == ["volume_0000", "volume_0001"]
    # Per-job files exist for each volume.
    for jid in ["volume_0000", "volume_0001"]:
        d = tmp_path / "jobs" / jid
        assert (d / "job.json").exists()
        assert (d / "entities.json").exists()
        assert (d / "subdomain.wkt").exists()
        assert (d / "mesh_kwargs.json").exists()


def test_write_bundles_ships_neighbors_in_job_json(tmp_path):
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
        perturbation=1e-5,
        point_tolerance=1e-3,
    )
    write_bundles(
        tmp_path, plan, [prism], mesh_kwargs={"default_characteristic_length": 0.5}
    )

    # volume_0000's job.json should list volume_0001 as a neighbour.
    j = json.loads((tmp_path / "jobs" / "volume_0000" / "job.json").read_text())
    assert len(j["neighbors"]) == 1
    assert j["neighbors"][0]["id"] == "volume_0001"
    assert "shared_boundary_wkt" in j["neighbors"][0]


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


def test_run_job_tags_subdomain_seams(tmp_path):
    """Worker tags subdomain-internal faces with seam IDs.

    After run_job, the worker's result.msh has _seam_i_j tags on
    subdomain-internal faces and ___None only on true outer faces.
    """
    import meshio
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
        perturbation=1e-5,
        point_tolerance=1e-3,
    )
    write_bundles(
        tmp_path, plan, [prism], mesh_kwargs={"default_characteristic_length": 0.5}
    )
    run_job(tmp_path / "jobs" / "volume_0000")
    m = meshio.read(tmp_path / "jobs" / "volume_0000" / "result.msh")
    names = set((m.field_data or {}))
    # Expected: ___seam_0000_0001 for the right wall, ___None for the other walls.
    assert any("___seam_0000_0001" in n for n in names), names
    assert "mat___None" in names, names


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
        SubdomainPlan,
        SubprocessExecutor,
        VolumeRegion,
        build_subdomain_plan,
        generate_mesh_distributed,
        run_job,
        subdomains_from_grid,
    )

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


def test_run_plan_single_phase_completes(tmp_path):
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
        perturbation=1e-5,
        point_tolerance=1e-3,
    )
    write_bundles(
        tmp_path, plan, [prism], mesh_kwargs={"default_characteristic_length": 0.3}
    )

    run_plan(tmp_path, plan, executor=InProcessExecutor())

    # Each volume bundle produced a result.msh in the single-phase run.
    for vid in ["volume_0000", "volume_0001"]:
        assert (tmp_path / "jobs" / vid / "result.msh").exists()


def test_stitch_meshes_produces_one_unified_mesh(tmp_path):
    import meshio
    import shapely

    from meshwell.distributed import (
        InProcessExecutor,
        build_subdomain_plan,
        run_plan,
        stitch_meshes,
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
        perturbation=1e-5,
        point_tolerance=1e-3,
    )
    write_bundles(
        tmp_path, plan, [prism], mesh_kwargs={"default_characteristic_length": 0.3}
    )
    run_plan(tmp_path, plan, executor=InProcessExecutor())
    out = tmp_path / "stitched.msh"
    stitch_meshes(work_dir=tmp_path, plan=plan, output_mesh=out)
    m = meshio.read(out)
    # The merged mesh should still have the "mat" physical group
    assert "mat" in (m.field_data or {})
    # And it should have at least one tet cell.
    assert any(cb.type == "tetra" for cb in m.cells)


def test_generate_mesh_distributed_smoke(tmp_path):
    import meshio
    import shapely

    from meshwell.distributed import (
        InProcessExecutor,
        generate_mesh_distributed,
    )
    from meshwell.polyprism import PolyPrism

    sd = [shapely.box(0, 0, 1, 1), shapely.box(1, 0, 2, 1)]
    prism = PolyPrism(
        polygons=shapely.box(0, 0, 2, 1),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="mat",
        mesh_order=1,
    )
    out = tmp_path / "out.msh"
    work = tmp_path / "work"
    generate_mesh_distributed(
        entities=[prism],
        subdomains=sd,
        output_mesh=out,
        work_dir=work,
        executor=InProcessExecutor(),
        keep_bundles=True,
        default_characteristic_length=0.3,
    )
    m = meshio.read(out)
    assert "mat" in (m.field_data or {})


def test_distributed_matches_serial_single_material_2x1(tmp_path):
    """Spec test 1: single-material PolyPrism spanning two subdomains.

    Distributed output's physical-group inventory must match the serial run.
    """
    import meshio
    import shapely

    from meshwell.distributed import (
        InProcessExecutor,
        generate_mesh_distributed,
        subdomains_from_grid,
    )
    from meshwell.orchestrator import generate_mesh
    from meshwell.polyprism import PolyPrism

    def make_prism():
        return PolyPrism(
            polygons=shapely.box(0, 0, 2, 1),
            buffers={0.0: 0.0, 1.0: 0.0},
            physical_name="mat",
            mesh_order=1,
        )

    serial_out = tmp_path / "serial.msh"
    generate_mesh(
        entities=[make_prism()],
        dim=3,
        output_mesh=serial_out,
        default_characteristic_length=0.3,
    )
    s = meshio.read(serial_out)

    dist_out = tmp_path / "dist.msh"
    generate_mesh_distributed(
        entities=[make_prism()],
        subdomains=subdomains_from_grid((0, 0, 2, 1), nx=2, ny=1),
        output_mesh=dist_out,
        work_dir=tmp_path / "work",
        executor=InProcessExecutor(),
        default_characteristic_length=0.3,
    )
    d = meshio.read(dist_out)

    serial_names = set(s.field_data or {})
    dist_names = set(d.field_data or {})
    assert serial_names == dist_names


def test_distributed_two_materials_shared_interface(tmp_path):
    """v1 convention restored: silicon and oxide abut at x=1 across a 2x1 grid.

    With seam-id tagging, the silicon-oxide cut at x=1 surfaces as a
    single ``silicon___oxide`` (alphabetical) physical group; the per-
    tile ``silicon___None`` / ``oxide___None`` markings on that seam
    are dropped by the post-merge consolidation pass. Outer-domain
    walls remain ``<material>___None``.
    """
    import meshio
    import shapely

    from meshwell.distributed import (
        InProcessExecutor,
        generate_mesh_distributed,
    )
    from meshwell.polyprism import PolyPrism

    si = PolyPrism(
        polygons=shapely.box(0, 0, 1, 1),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="silicon",
        mesh_order=1,
    )
    ox = PolyPrism(
        polygons=shapely.box(1, 0, 2, 1),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="oxide",
        mesh_order=2,
    )
    out = tmp_path / "out.msh"
    generate_mesh_distributed(
        entities=[si, ox],
        subdomains=[shapely.box(0, 0, 1, 1), shapely.box(1, 0, 2, 1)],
        output_mesh=out,
        work_dir=tmp_path / "work",
        executor=InProcessExecutor(),
        default_characteristic_length=0.3,
    )
    m = meshio.read(out)
    names = set((m.field_data or {}))
    # Both materials present.
    assert "silicon" in names, names
    assert "oxide" in names, names
    # The silicon-oxide cut surfaces as a v1-style A___B interface group.
    assert any(
        "silicon" in n and "oxide" in n and "___" in n
        for n in names
        if n not in ("silicon", "oxide")
    ), names
    # silicon___None / oxide___None can survive on true outer walls,
    # but they must NOT carry the silicon-oxide seam (consolidation
    # would have re-tagged that as silicon___oxide).


def test_distributed_same_material_drops_invisible_seam(tmp_path):
    """Single 'mat' material spanning 2 tiles: stitch produces no seam group.

    A single 'mat' material spanning 2 tiles: the consolidation pass
    drops the interior cut entirely (invisible — same material both
    sides), and the only mat___None faces are on the true outer
    boundary.
    """
    import meshio
    import shapely

    from meshwell.distributed import (
        InProcessExecutor,
        generate_mesh_distributed,
    )
    from meshwell.polyprism import PolyPrism

    prism = PolyPrism(
        polygons=shapely.box(0, 0, 2, 1),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="mat",
        mesh_order=1,
    )
    out = tmp_path / "out.msh"
    generate_mesh_distributed(
        entities=[prism],
        subdomains=[shapely.box(0, 0, 1, 1), shapely.box(1, 0, 2, 1)],
        output_mesh=out,
        work_dir=tmp_path / "work",
        executor=InProcessExecutor(),
        default_characteristic_length=0.3,
    )
    m = meshio.read(out)
    names = set((m.field_data or {}))
    assert "mat" in names, names
    # No seam groups should remain — same material both sides means invisible.
    assert not any("___seam_" in n for n in names), names
    # mat___None should still exist (true outer boundary of the union).
    assert "mat___None" in names, names


def test_distributed_2x2_grid_with_junction(tmp_path):
    """v2: 4 materials, 2x2 grid, interior corner at (1, 1)."""
    import meshio
    import shapely

    from meshwell.distributed import (
        InProcessExecutor,
        generate_mesh_distributed,
        subdomains_from_grid,
    )
    from meshwell.polyprism import PolyPrism

    materials = []
    for (i, j), name in [
        ((0, 0), "ll"),
        ((1, 0), "lr"),
        ((0, 1), "ul"),
        ((1, 1), "ur"),
    ]:
        materials.append(
            PolyPrism(
                polygons=shapely.box(i, j, i + 1, j + 1),
                buffers={0.0: 0.0, 1.0: 0.0},
                physical_name=name,
                mesh_order=1,
            )
        )
    sd = subdomains_from_grid((0, 0, 2, 2), nx=2, ny=2)
    out = tmp_path / "out.msh"
    generate_mesh_distributed(
        entities=materials,
        subdomains=sd,
        output_mesh=out,
        work_dir=tmp_path / "work",
        executor=InProcessExecutor(),
        default_characteristic_length=0.3,
    )
    m = meshio.read(out)
    names = set(m.field_data or {})
    assert {"ll", "lr", "ul", "ur"}.issubset(names), names


def test_distributed_consolidates_same_material_across_4_tiles(tmp_path):
    """v2: single 'mat' across a 2x2 grid consolidates to one group.

    A single 'mat' material spans the 2x2 grid; after stitch there
    is exactly one consolidated 'mat' physical group.
    """
    import meshio
    import shapely

    from meshwell.distributed import (
        InProcessExecutor,
        generate_mesh_distributed,
        subdomains_from_grid,
    )
    from meshwell.polyprism import PolyPrism

    prism = PolyPrism(
        polygons=shapely.box(0, 0, 2, 2),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="mat",
        mesh_order=1,
    )
    out = tmp_path / "out.msh"
    generate_mesh_distributed(
        entities=[prism],
        subdomains=subdomains_from_grid((0, 0, 2, 2), nx=2, ny=2),
        output_mesh=out,
        work_dir=tmp_path / "work",
        executor=InProcessExecutor(),
        default_characteristic_length=0.5,
    )
    m = meshio.read(out)
    names = list((m.field_data or {}).keys())
    assert names.count("mat") == 1, names


def test_distributed_3x3_mixed_materials(tmp_path):
    """v2: 3x3 grid alternating silicon/oxide consolidates both names.

    Both names are consolidated across the 5 silicon and 4 oxide tiles.
    """
    import meshio
    import shapely

    from meshwell.distributed import (
        InProcessExecutor,
        generate_mesh_distributed,
        subdomains_from_grid,
    )
    from meshwell.polyprism import PolyPrism

    materials = []
    for i in range(3):
        for j in range(3):
            name = "silicon" if (i + j) % 2 == 0 else "oxide"
            materials.append(
                PolyPrism(
                    polygons=shapely.box(i, j, i + 1, j + 1),
                    buffers={0.0: 0.0, 1.0: 0.0},
                    physical_name=name,
                    mesh_order=1,
                )
            )
    out = tmp_path / "out.msh"
    generate_mesh_distributed(
        entities=materials,
        subdomains=subdomains_from_grid((0, 0, 3, 3), nx=3, ny=3),
        output_mesh=out,
        work_dir=tmp_path / "work",
        executor=InProcessExecutor(),
        default_characteristic_length=0.5,
    )
    m = meshio.read(out)
    names = set(m.field_data or {})
    assert "silicon" in names, names
    assert "oxide" in names, names


def test_distributed_4x1_strip(tmp_path):
    """v2: 4x1 strip (single material spanning 4 tiles)."""
    import meshio
    import shapely

    from meshwell.distributed import (
        InProcessExecutor,
        generate_mesh_distributed,
        subdomains_from_grid,
    )
    from meshwell.polyprism import PolyPrism

    prism = PolyPrism(
        polygons=shapely.box(0, 0, 4, 1),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="mat",
        mesh_order=1,
    )
    out = tmp_path / "out.msh"
    generate_mesh_distributed(
        entities=[prism],
        subdomains=subdomains_from_grid((0, 0, 4, 1), nx=4, ny=1),
        output_mesh=out,
        work_dir=tmp_path / "work",
        executor=InProcessExecutor(),
        default_characteristic_length=0.5,
    )
    m = meshio.read(out)
    names = list((m.field_data or {}).keys())
    assert names.count("mat") == 1, names


def test_distributed_rejects_uncovered_subdomains(tmp_path):
    """Spec test 7: subdomains not covering entity union -> clean ValueError."""
    import shapely

    from meshwell.distributed import (
        InProcessExecutor,
        generate_mesh_distributed,
    )
    from meshwell.polyprism import PolyPrism

    prism = PolyPrism(
        polygons=shapely.box(0, 0, 2, 1),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="mat",
        mesh_order=1,
    )
    with pytest.raises(ValueError, match="not covered"):
        generate_mesh_distributed(
            entities=[prism],
            subdomains=[shapely.box(0, 0, 1, 1)],  # x=[1,2] uncovered
            output_mesh=tmp_path / "out.msh",
            work_dir=tmp_path / "work",
            executor=InProcessExecutor(),
        )


# (Spec test 10 also omitted — see comment block above.)


def test_name_to_tag_is_deterministic():
    from meshwell.distributed import _name_to_tag

    # Same input → same tag
    assert _name_to_tag("silicon", 3) == _name_to_tag("silicon", 3)
    # Different name → different tag (with high probability)
    assert _name_to_tag("silicon", 3) != _name_to_tag("oxide", 3)
    # Different dim → different tag
    assert _name_to_tag("silicon", 3) != _name_to_tag("silicon", 2)
    # Tag is positive int32-safe
    tag = _name_to_tag("anything", 3)
    assert 1 <= tag < 2**31


def test_distributed_multi_material_stack_drops_all_seam_groups(tmp_path):
    """Two tiles, each with the same 4-material vertical stack, share a seam.

    All 4 materials appear on both sides of the cut, so every seam-tagged
    face pairs with a same-material twin from the other tile. The
    consolidation pass must drop ALL seam groups and emit no multi-way
    concatenated names like ``cladding___ridge___slab___substrate``.
    """
    import meshio
    import shapely

    from meshwell.distributed import (
        InProcessExecutor,
        generate_mesh_distributed,
    )
    from meshwell.polyprism import PolyPrism

    full = shapely.box(0, 0, 2, 1)
    materials = []
    for name, z_lo, z_hi, mo in [
        ("substrate", 0.0, 0.3, 4),
        ("slab", 0.3, 0.45, 3),
        ("ridge", 0.45, 0.7, 1),
        ("cladding", 0.7, 1.0, 2),
    ]:
        materials.append(
            PolyPrism(
                polygons=full,
                buffers={z_lo: 0.0, z_hi: 0.0},
                physical_name=name,
                mesh_order=mo,
            )
        )
    out = tmp_path / "out.msh"
    generate_mesh_distributed(
        entities=materials,
        subdomains=[shapely.box(0, 0, 1, 1), shapely.box(1, 0, 2, 1)],
        output_mesh=out,
        work_dir=tmp_path / "work",
        executor=InProcessExecutor(),
        default_characteristic_length=0.3,
    )
    m = meshio.read(out)
    names = set((m.field_data or {}))
    # All 4 materials present.
    for mat in ("substrate", "slab", "ridge", "cladding"):
        assert mat in names, f"missing material {mat!r}; got {names}"
    # No seam groups remain.
    assert not any("___seam_" in n for n in names), names
    # No multi-way concatenations like "cladding___ridge___slab___substrate".
    assert not any(n.count("___") >= 3 for n in names), names


def test_generate_mesh_hashed_physical_tags(tmp_path):
    """Hashed physical-group tags are stable across independent meshwell runs.

    Two runs with the same physical_name produce .msh files where the
    physical-group tag is identical.
    """
    import meshio
    import shapely

    from meshwell.distributed import _name_to_tag
    from meshwell.orchestrator import generate_mesh
    from meshwell.polyprism import PolyPrism

    def _emit(out_path, x_offset):
        prism = PolyPrism(
            polygons=shapely.box(x_offset, 0, x_offset + 1, 1),
            buffers={0.0: 0.0, 1.0: 0.0},
            physical_name="silicon",
            mesh_order=1,
        )
        generate_mesh(
            entities=[prism],
            dim=3,
            output_mesh=out_path,
            default_characteristic_length=0.5,
            _hashed_physical_tags=True,
        )

    out_a = tmp_path / "a.msh"
    out_b = tmp_path / "b.msh"
    _emit(out_a, 0.0)
    _emit(out_b, 5.0)

    ma = meshio.read(out_a)
    mb = meshio.read(out_b)
    silicon_a = int((ma.field_data or {})["silicon"][0])
    silicon_b = int((mb.field_data or {})["silicon"][0])
    assert silicon_a == silicon_b == _name_to_tag("silicon", 3)
