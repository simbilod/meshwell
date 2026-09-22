"""Tests for Streamlined Entity Subgroup Copying (`CopyGroup` / `CopyInstance` via `.msh` stamping)."""

from __future__ import annotations

import tempfile
from collections import Counter
from pathlib import Path

import gmsh
import numpy as np
import shapely
from scipy.spatial import KDTree

from meshwell import (
    CopyGroup,
    CopyInstance,
    generate_mesh,
)
from meshwell.copy_group import make_periodic_pitch_pre_2d_hook
from meshwell.polyprism import PolyPrism

LAYERS = [("pad_bot", 0.0, 0.3), ("via", 0.3, 0.7), ("pad_top", 0.7, 1.0)]


def _build_via_donor_msh(
    msh_path: Path,
    cl: float = 0.35,
    periodic_pitch: tuple[float, float, float] | None = None,
) -> None:
    """Generate a standalone 3-layer via-stack donor `.msh` file."""
    ents: list[PolyPrism] = []
    for idx, (name, z0, z1) in enumerate(LAYERS, start=1):
        w = 1.0 if name != "via" else 0.4
        off = 0.0 if name != "via" else 0.3
        ents.append(
            PolyPrism(
                polygons=shapely.box(off, off, off + w, off + w),
                buffers={z0: 0.0, z1: 0.0},
                physical_name=name,
                mesh_order=idx,
                point_tolerance=0.0,
            )
        )
    hook = make_periodic_pitch_pre_2d_hook(periodic_pitch) if periodic_pitch else None
    generate_mesh(
        entities=ents,
        dim=3,
        default_characteristic_length=cl,
        output_mesh=msh_path,
        pre_2d_hook=hook,
        save_all=True,
        verbosity=0,
    )


def _assert_global_conformality(
    expected_volume: float,
) -> dict[str, tuple[int, np.ndarray]]:
    """Verify global tet mesh conformality and return per-physical-group `(ntets, coords)`."""
    for dim, gtag in gmsh.model.getPhysicalGroups():
        gname = gmsh.model.getPhysicalName(dim, gtag)
        assert not gname.startswith("__copy|"), f"Leaked synthetic group: {gname}"

    allt, allc, _ = gmsh.model.mesh.getNodes()
    pos = {
        int(t): np.asarray(allc[3 * k : 3 * k + 3], dtype=float)
        for k, t in enumerate(allt)
    }
    rows = []
    for _, v in gmsh.model.getEntities(3):
        en = gmsh.model.mesh.getElements(3, v)[2]
        if en:
            rows.append(np.asarray(en[0], dtype=int).reshape(-1, 4))
    T = np.vstack(rows)
    p = np.array([[pos[int(n)] for n in r] for r in T])
    vol = (
        np.einsum(
            "ij,ij->i",
            np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]),
            p[:, 3] - p[:, 0],
        )
        / 6.0
    )
    assert int((vol <= 0.0).sum()) == 0
    assert abs(float(np.abs(vol).sum()) - expected_volume) < 1e-5

    fc: Counter = Counter()
    for t in T:
        for f in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)):
            fc[tuple(sorted(t[list(f)]))] += 1
    inc = Counter(fc.values())
    assert set(inc.keys()) <= {1, 2}, f"Non-manifold face incidence: {dict(inc)}"

    coin = len(KDTree(np.asarray(allc, dtype=float).reshape(-1, 3)).query_pairs(r=1e-9))
    assert coin == 0, f"Found {coin} coincident duplicate node pairs"

    group_info: dict[str, tuple[int, np.ndarray]] = {}
    for dim, gtag in gmsh.model.getPhysicalGroups(3):
        gname = gmsh.model.getPhysicalName(dim, gtag)
        vtags = gmsh.model.getEntitiesForPhysicalGroup(dim, gtag)
        tet_cnt = 0
        pts_set = set()
        for v in vtags:
            en = gmsh.model.mesh.getElements(3, int(v))[2]
            if en:
                tets = np.asarray(en[0], dtype=int).reshape(-1, 4)
                tet_cnt += len(tets)
                for n in tets.ravel():
                    pts_set.add(tuple(pos[int(n)]))
        group_info[gname] = (tet_cnt, np.array(sorted(pts_set), dtype=float))
    return group_info


def test_copy_group_serialization() -> None:
    inst = CopyInstance(
        members={"pad_bot": "pad_bot_i1", "via": "via_i1"},
        translation=(2.0, 1.0, 0.0),
        rotation_axis=(0.0, 0.0, 1.0),
        rotation_angle_deg=37.0,
        rotation_origin=(0.5, 0.5, 0.0),
    )
    grp = CopyGroup(
        name="via_stack",
        msh_path=Path("donor_via.msh"),
        instances=[inst],
        role_tolerance=1e-5,
    )
    assert CopyGroup.from_dict(grp.to_dict()) == grp


def test_multi_entity_translated_and_rotated_subgroup(tmp_path: Path) -> None:
    """Test 3-slab via stack `.msh` stamped at 0 deg, 37 deg, and 90 deg inside background filler."""
    donor_msh = tmp_path / "via_donor.msh"
    _build_via_donor_msh(donor_msh, cl=0.35)

    origin = (0.5, 0.5, 0.0)
    specs = [
        ("i0", (0.0, 0.0, 0.0), 0.0),
        ("i1", (2.5, 0.0, 0.0), 37.0),
        ("i2", (5.0, 0.0, 0.0), 90.0),
    ]
    ents: list[PolyPrism] = []
    order = 1
    for tag, trans, angle in specs:
        for name, z0, z1 in LAYERS:
            w = 1.0 if name != "via" else 0.4
            off = 0.0 if name != "via" else 0.3
            ents.append(
                PolyPrism(
                    polygons=shapely.box(off, off, off + w, off + w),
                    buffers={z0: 0.0, z1: 0.0},
                    physical_name=f"{name}_{tag}",
                    mesh_order=order,
                    point_tolerance=0.0,
                    translation=trans,
                    rotation_axis=(0.0, 0.0, 1.0) if angle != 0.0 else None,
                    rotation_point=origin if angle != 0.0 else None,
                    rotation_angle=angle,
                )
            )
            order += 1

    ents.append(
        PolyPrism(
            polygons=shapely.box(-1.5, -1.5, 7.5, 2.5),
            buffers={-0.5: 0.0, 1.5: 0.0},
            physical_name="bg",
            mesh_order=100,
        )
    )

    instances = [
        CopyInstance(
            members={name: f"{name}_{tag}" for name, _, _ in LAYERS},
            translation=trans,
            rotation_axis=(0.0, 0.0, 1.0),
            rotation_angle_deg=angle,
            rotation_origin=origin,
        )
        for tag, trans, angle in specs
    ]
    cg = CopyGroup(name="via_stack", msh_path=donor_msh, instances=instances)

    generate_mesh(
        entities=ents,
        dim=3,
        default_characteristic_length=0.35,
        copy_groups=[cg],
        verbosity=0,
    )
    expected_vol = (7.5 - (-1.5)) * (2.5 - (-1.5)) * (1.5 - (-0.5))
    info = _assert_global_conformality(expected_vol)

    for inst, (tag, _trans, _angle) in zip(instances[1:], specs[1:]):
        for name, _, _ in LAYERS:
            nt0, pts0 = info[f"{name}_i0"]
            ntk, ptsk = info[f"{name}_{tag}"]
            assert ntk == nt0
            pts_mapped = np.array(
                sorted(
                    tuple(np.round(inst.inverse_transform_point(p), 10)) for p in ptsk
                )
            )
            pts0_sorted = np.array(sorted(tuple(np.round(p, 10)) for p in pts0))
            assert np.max(np.abs(pts_mapped - pts0_sorted)) < 1e-9


def test_abutting_subgroup_instances_zero_gap(tmp_path: Path) -> None:
    """Test 3 abutting via-stack instances (`gap = 0`, `DX = [0.0, 1.0, 2.0]`)."""
    donor_msh = tmp_path / "via_periodic_donor.msh"
    _build_via_donor_msh(donor_msh, cl=0.35, periodic_pitch=(1.0, 0.0, 0.0))

    dx_list = [0.0, 1.0, 2.0]
    ents: list[PolyPrism] = []
    order = 1
    for k, dx in enumerate(dx_list):
        for name, z0, z1 in LAYERS:
            w = 1.0 if name != "via" else 0.4
            off = 0.0 if name != "via" else 0.3
            ents.append(
                PolyPrism(
                    polygons=shapely.box(dx + off, off, dx + off + w, off + w),
                    buffers={z0: 0.0, z1: 0.0},
                    physical_name=f"{name}_i{k}",
                    mesh_order=order,
                )
            )
            order += 1

    ents.append(
        PolyPrism(
            polygons=shapely.box(-1.0, -1.0, 4.0, 2.0),
            buffers={-0.5: 0.0, 1.5: 0.0},
            physical_name="bg",
            mesh_order=100,
        )
    )

    cg = CopyGroup(
        name="abutting_via",
        msh_path=donor_msh,
        instances=[
            CopyInstance(
                members={name: f"{name}_i{k}" for name, _, _ in LAYERS},
                translation=(dx, 0.0, 0.0),
            )
            for k, dx in enumerate(dx_list)
        ],
    )

    generate_mesh(
        entities=ents,
        dim=3,
        default_characteristic_length=0.35,
        copy_groups=[cg],
        verbosity=0,
    )
    expected_vol = (4.0 - (-1.0)) * (2.0 - (-1.0)) * (1.5 - (-0.5))
    info = _assert_global_conformality(expected_vol)

    for k, dx in enumerate(dx_list[1:], start=1):
        for name, _, _ in LAYERS:
            nt0, pts0 = info[f"{name}_i0"]
            ntk, ptsk = info[f"{name}_i{k}"]
            assert ntk == nt0
            shifted = np.array(
                sorted(tuple(np.round(p - np.array([dx, 0.0, 0.0]), 10)) for p in ptsk)
            )
            ref = np.array(sorted(tuple(np.round(p, 10)) for p in pts0))
            assert np.max(np.abs(shifted - ref)) == 0.0


def test_recursive_msh_hierarchy_stamping(tmp_path: Path) -> None:
    """Test 2-level `.msh` composition (`via.msh` -> stamped into `tile.msh` -> stamped into `chip.msh`)."""
    via_msh = tmp_path / "via.msh"
    tile_msh = tmp_path / "tile.msh"
    _build_via_donor_msh(via_msh, cl=0.35)

    via_dx = [0.0, 1.2]
    tile_tx = [0.0, 4.0]

    # Level 1: Build `tile.msh` by stamping `via.msh` onto `v0` and `v1` inside `tile_filler`
    tile_ents: list[PolyPrism] = []
    order = 1
    for v_idx, vx in enumerate(via_dx):
        for name, z0, z1 in LAYERS:
            w = 1.0 if name != "via" else 0.4
            off = 0.0 if name != "via" else 0.3
            tile_ents.append(
                PolyPrism(
                    polygons=shapely.box(vx + off, off, vx + off + w, off + w),
                    buffers={z0: 0.0, z1: 0.0},
                    physical_name=f"{name}_v{v_idx}",
                    mesh_order=order,
                    point_tolerance=0.0,
                )
            )
            order += 1
    tile_ents.append(
        PolyPrism(
            polygons=shapely.box(-0.2, -0.2, 2.4, 1.2),
            buffers={-0.2: 0.0, 1.2: 0.0},
            physical_name="tile_filler",
            mesh_order=50,
            point_tolerance=0.0,
        )
    )
    via_cg = CopyGroup(
        name="via_in_tile",
        msh_path=via_msh,
        instances=[
            CopyInstance(
                members={name: f"{name}_v{v_idx}" for name, _, _ in LAYERS},
                translation=(vx, 0.0, 0.0),
            )
            for v_idx, vx in enumerate(via_dx)
        ],
    )
    generate_mesh(
        entities=tile_ents,
        dim=3,
        default_characteristic_length=0.35,
        copy_groups=[via_cg],
        output_mesh=tile_msh,
        save_all=True,
        verbosity=0,
    )
    assert tile_msh.exists()

    # Level 2: Build `chip` scene and stamp `tile.msh` into `tile_0` and `tile_1`
    chip_ents: list[PolyPrism] = []
    order = 1
    for t_idx, tx in enumerate(tile_tx):
        for v_idx, vx in enumerate(via_dx):
            for name, z0, z1 in LAYERS:
                w = 1.0 if name != "via" else 0.4
                off = 0.0 if name != "via" else 0.3
                chip_ents.append(
                    PolyPrism(
                        polygons=shapely.box(
                            tx + vx + off, off, tx + vx + off + w, off + w
                        ),
                        buffers={z0: 0.0, z1: 0.0},
                        physical_name=f"{name}_t{t_idx}_v{v_idx}",
                        mesh_order=order,
                        point_tolerance=0.0,
                    )
                )
                order += 1
    for t_idx, tx in enumerate(tile_tx):
        chip_ents.append(
            PolyPrism(
                polygons=shapely.box(tx - 0.2, -0.2, tx + 2.4, 1.2),
                buffers={-0.2: 0.0, 1.2: 0.0},
                physical_name=f"tile_filler_t{t_idx}",
                mesh_order=50 + t_idx,
                point_tolerance=0.0,
            )
        )
    chip_ents.append(
        PolyPrism(
            polygons=shapely.box(-1.0, -1.0, 7.5, 2.2),
            buffers={-1.0: 0.0, 2.0: 0.0},
            physical_name="chip_bg",
            mesh_order=100,
        )
    )

    tile_cg = CopyGroup(
        name="tile_in_chip",
        msh_path=tile_msh,
        instances=[
            CopyInstance(
                members={
                    **{
                        f"{name}_v{v_idx}": f"{name}_t{t_idx}_v{v_idx}"
                        for v_idx in range(len(via_dx))
                        for name, _, _ in LAYERS
                    },
                    "tile_filler": f"tile_filler_t{t_idx}",
                },
                translation=(tx, 0.0, 0.0),
            )
            for t_idx, tx in enumerate(tile_tx)
        ],
    )
    generate_mesh(
        entities=chip_ents,
        dim=3,
        default_characteristic_length=0.35,
        copy_groups=[tile_cg],
        verbosity=0,
    )
    expected_vol = (7.5 - (-1.0)) * (2.2 - (-1.0)) * (2.0 - (-1.0))
    info = _assert_global_conformality(expected_vol)
    assert info["tile_filler_t0"][0] == info["tile_filler_t1"][0]


def test_surface_only_hollow_cavity_stamping(tmp_path: Path) -> None:
    """Test `surface_only=True`: stamps only the outer 2D shell and leaves the 3D interior hollow."""
    donor_msh = tmp_path / "via_shell_donor.msh"
    _build_via_donor_msh(donor_msh, cl=0.35)

    origin = (0.5, 0.5, 0.0)
    specs = [
        ("i0", (0.0, 0.0, 0.0), 0.0),
        ("i1", (2.5, 0.0, 0.0), 37.0),
    ]
    ents: list[PolyPrism] = []
    order = 1
    for tag, trans, angle in specs:
        for name, z0, z1 in LAYERS:
            w = 1.0 if name != "via" else 0.4
            off = 0.0 if name != "via" else 0.3
            ents.append(
                PolyPrism(
                    polygons=shapely.box(off, off, off + w, off + w),
                    buffers={z0: 0.0, z1: 0.0},
                    physical_name=f"{name}_{tag}",
                    mesh_order=order,
                    point_tolerance=0.0,
                    translation=trans,
                    rotation_axis=(0.0, 0.0, 1.0) if angle != 0.0 else None,
                    rotation_point=origin if angle != 0.0 else None,
                    rotation_angle=angle,
                )
            )
            order += 1

    ents.append(
        PolyPrism(
            polygons=shapely.box(-1.5, -1.5, 5.0, 2.5),
            buffers={-0.5: 0.0, 1.5: 0.0},
            physical_name="bg",
            mesh_order=100,
        )
    )

    instances = [
        CopyInstance(
            members={name: f"{name}_{tag}" for name, _, _ in LAYERS},
            translation=trans,
            rotation_axis=(0.0, 0.0, 1.0),
            rotation_angle_deg=angle,
            rotation_origin=origin,
        )
        for tag, trans, angle in specs
    ]
    cg = CopyGroup(
        name="via_hollow_shell",
        msh_path=donor_msh,
        instances=instances,
        surface_only=True,
    )

    generate_mesh(
        entities=ents,
        dim=3,
        default_characteristic_length=0.35,
        copy_groups=[cg],
        verbosity=0,
    )

    # Outer box volume minus 2 hollow via stacks (each stack = 0.3 + 0.4*0.4*0.4 + 0.3 = 0.664)
    via_stack_vol = 1.0 * 1.0 * 0.3 + 0.4 * 0.4 * 0.4 + 1.0 * 1.0 * 0.3
    expected_vol = (5.0 - (-1.5)) * (2.5 - (-1.5)) * (
        1.5 - (-0.5)
    ) - 2.0 * via_stack_vol
    info_3d = _assert_global_conformality(expected_vol)

    # Only "bg" should remain as a 3D physical group (cavity interiors are hollow)
    assert set(info_3d.keys()) == {"bg"}

    # Verify 2D shell interface groups (`{name}_i0___bg` vs `{name}_i1___bg`) are bit-identical
    allt, allc, _ = gmsh.model.mesh.getNodes()
    pos = {
        int(t): np.asarray(allc[3 * k : 3 * k + 3], dtype=float)
        for k, t in enumerate(allt)
    }
    surf_groups: dict[str, tuple[int, np.ndarray]] = {}
    for dim, gtag in gmsh.model.getPhysicalGroups(2):
        gname = gmsh.model.getPhysicalName(dim, gtag)
        ftags = gmsh.model.getEntitiesForPhysicalGroup(dim, gtag)
        tri_cnt = 0
        pts_set = set()
        for f in ftags:
            en = gmsh.model.mesh.getElements(2, int(f))[2]
            if en:
                tris = np.asarray(en[0], dtype=int).reshape(-1, 3)
                tri_cnt += len(tris)
                for n in tris.ravel():
                    pts_set.add(tuple(pos[int(n)]))
        surf_groups[gname] = (tri_cnt, np.array(sorted(pts_set), dtype=float))

    inst1 = instances[1]
    for name, _, _ in LAYERS:
        nt0, pts0 = surf_groups[f"{name}_i0___bg"]
        nt1, pts1 = surf_groups[f"{name}_i1___bg"]
        assert nt0 > 0
        assert nt1 == nt0
        pts1_mapped = np.array(
            sorted(tuple(np.round(inst1.inverse_transform_point(p), 10)) for p in pts1)
        )
        pts0_sorted = np.array(sorted(tuple(np.round(p, 10)) for p in pts0))
        assert np.max(np.abs(pts1_mapped - pts0_sorted)) < 1e-9


if __name__ == "__main__":
    for test_name, test_fn in [
        ("test_copy_group_serialization", test_copy_group_serialization),
        (
            "test_multi_entity_translated_and_rotated_subgroup",
            lambda: test_multi_entity_translated_and_rotated_subgroup(
                Path(tempfile.mkdtemp())
            ),
        ),
        (
            "test_abutting_subgroup_instances_zero_gap",
            lambda: test_abutting_subgroup_instances_zero_gap(Path(tempfile.mkdtemp())),
        ),
        (
            "test_recursive_msh_hierarchy_stamping",
            lambda: test_recursive_msh_hierarchy_stamping(Path(tempfile.mkdtemp())),
        ),
        (
            "test_surface_only_hollow_cavity_stamping",
            lambda: test_surface_only_hollow_cavity_stamping(Path(tempfile.mkdtemp())),
        ),
    ]:
        print(f"Running {test_name} ...", flush=True)
        test_fn()
        print(f"  PASSED: {test_name}", flush=True)
