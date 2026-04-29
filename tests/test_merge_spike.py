"""Spike: probe NxM merge strategies for independently-meshed .msh files.

Each test builds N x M independent box meshes (one material per tile) at
the same characteristic length, then exercises one or more merge
strategies, and reports:

- node count (raw vs after dedup)
- cell count
- physical groups preserved by name
- whether the seam is conformal (no internal duplicate nodes after dedup)

The goal is to establish, EMPIRICALLY, which merge strategy supports
which tile topology — independent of the seam-mesh-then-seed approach
the current ``meshwell.distributed`` pipeline takes (which only works
for 2-tile strips).

Strategies probed:
- ``gmsh_merge`` — gmsh.merge per file + Mesh.removeDuplicateNodes
  with a tolerance
- ``meshio_concat`` — read each .msh with meshio, concatenate
  points/cells, dedup nodes by quantized coordinate, build a unified
  field_data table by name (the strategy meshwell.distributed already
  uses for stitch_meshes)

Each test prints a summary and asserts only the minimal physical-tag-
preservation invariants. Conformity is measured but not asserted
(some strategies are non-conformal on shared faces by design).
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import meshio
import numpy as np
import pytest

import gmsh

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def fresh_gmsh():
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    yield
    if gmsh.is_initialized():
        gmsh.finalize()


def _build_box_msh(
    path: Path,
    bbox: tuple[float, float, float, float, float, float],
    name: str,
    lc: float,
) -> None:
    """Mesh one OCC box and write to .msh with a single physical volume group."""
    gmsh.model.add(f"box_{name}")
    xmin, ymin, zmin, xmax, ymax, zmax = bbox
    gmsh.model.occ.addBox(xmin, ymin, zmin, xmax - xmin, ymax - ymin, zmax - zmin)
    gmsh.model.occ.synchronize()
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc)
    vols = [t for d, t in gmsh.model.getEntities(3)]
    gmsh.model.addPhysicalGroup(3, vols, name=name)
    gmsh.model.mesh.generate(3)
    gmsh.write(str(path))
    gmsh.model.remove()


def _write_grid_meshes(
    work_dir: Path,
    nx: int,
    ny: int,
    nz: int = 1,
    extent: tuple[float, float, float] = (1.0, 1.0, 1.0),
    lc: float = 0.5,
    name_prefix: str = "tile",
) -> list[tuple[Path, str]]:
    """Build nx*ny*nz adjacent unit boxes; return [(path, name), ...]."""
    work_dir.mkdir(parents=True, exist_ok=True)
    lx, ly, lz = extent
    paths = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                bbox = (
                    i * lx,
                    j * ly,
                    k * lz,
                    (i + 1) * lx,
                    (j + 1) * ly,
                    (k + 1) * lz,
                )
                name = f"{name_prefix}_{i}_{j}_{k}"
                path = work_dir / f"{name}.msh"
                _build_box_msh(path, bbox, name=name, lc=lc)
                paths.append((path, name))
    return paths


# --------------------------------------------------------------------------
# Merge strategies
# --------------------------------------------------------------------------


def _gmsh_merge_strategy(paths: Sequence[Path], dedup_tol: float) -> dict[str, object]:
    """Strategy: gmsh.merge each file + removeDuplicateNodes with tolerance.

    Returns a dict with measured metrics. Output mesh is left in the
    active gmsh model (caller may write it).
    """
    gmsh.model.add("merged_gmsh")
    for p in paths:
        gmsh.merge(str(p))
    pre_node_count = len(gmsh.model.mesh.getNodes()[0])
    gmsh.option.setNumber("Geometry.Tolerance", dedup_tol)
    gmsh.model.mesh.removeDuplicateNodes()
    post_node_count = len(gmsh.model.mesh.getNodes()[0])

    pgs = {}
    for dim in (0, 1, 2, 3):
        for d, tag in gmsh.model.getPhysicalGroups(dim):
            name = gmsh.model.getPhysicalName(d, tag)
            pgs.setdefault(name, []).append((d, tag))

    return {
        "pre_dedup_nodes": pre_node_count,
        "post_dedup_nodes": post_node_count,
        "deduped": pre_node_count - post_node_count,
        "physical_groups": pgs,
        "physical_group_names": sorted(pgs),
    }


def _meshio_concat_strategy(
    paths: Sequence[Path], dedup_tol: float
) -> dict[str, object]:
    """Strategy: meshio concat + KD-tree-style coordinate dedup + name merge.

    Mirrors ``meshwell.distributed.stitch_meshes`` but operates on an
    arbitrary list of .msh files.
    """
    files = [(p.stem, meshio.read(p)) for p in paths]

    # 1. Concatenate points; track per-file offsets.
    all_pts = []
    pt_offsets = []
    cur = 0
    for _, m in files:
        all_pts.append(m.points)
        pt_offsets.append(cur)
        cur += m.points.shape[0]
    pts = np.vstack(all_pts) if all_pts else np.zeros((0, 3))

    # 2. Build a unified name -> (new_tag, dim) table.
    name_dim_to_new_tag: dict[tuple[str, int], int] = {}
    next_tag_per_dim: dict[int, int] = {}
    for _, m in files:
        for name, arr in (m.field_data or {}).items():
            tag, dim = int(arr[0]), int(arr[1])
            key = (name, dim)
            if key not in name_dim_to_new_tag:
                next_tag_per_dim.setdefault(dim, 0)
                next_tag_per_dim[dim] += 1
                name_dim_to_new_tag[key] = next_tag_per_dim[dim]

    field_data = {
        name: np.array([new_tag, dim])
        for (name, dim), new_tag in name_dim_to_new_tag.items()
    }

    # 3. Concatenate cells; remap (dim, old_tag) -> new_tag per file.
    type_to_dim = {
        "vertex": 0,
        "line": 1,
        "triangle": 2,
        "quad": 2,
        "tetra": 3,
        "hexahedron": 3,
        "wedge": 3,
        "pyramid": 3,
    }

    cell_blocks: list[meshio.CellBlock] = []
    cell_data_phys: list[np.ndarray] = []

    for file_idx, (_, m) in enumerate(files):
        offset = pt_offsets[file_idx]
        per_file: dict[tuple[int, int], int] = {}
        for name, arr in (m.field_data or {}).items():
            tag, dim = int(arr[0]), int(arr[1])
            per_file[(dim, tag)] = name_dim_to_new_tag[(name, dim)]

        gphys = m.cell_data.get("gmsh:physical") if m.cell_data else None
        for block_idx, block in enumerate(m.cells):
            new_data = block.data + offset
            cell_blocks.append(meshio.CellBlock(block.type, new_data))
            dim = type_to_dim.get(block.type, -1)
            old = (
                gphys[block_idx]
                if gphys is not None and block_idx < len(gphys)
                else None
            )
            if old is None:
                cell_data_phys.append(np.zeros(len(new_data), dtype=np.int32))
            else:
                cell_data_phys.append(
                    np.array(
                        [per_file.get((dim, int(t)), 0) for t in old],
                        dtype=np.int32,
                    )
                )

    # 4. Dedup nodes by quantized coordinate.
    quantized = np.round(pts / max(dedup_tol, 1e-12)).astype(np.int64)
    canonical: dict[tuple, int] = {}
    remap = np.empty(pts.shape[0], dtype=np.int64)
    new_pts: list = []
    for i in range(pts.shape[0]):
        k = tuple(quantized[i])
        if k not in canonical:
            canonical[k] = len(new_pts)
            new_pts.append(pts[i])
        remap[i] = canonical[k]
    deduped_pts = np.array(new_pts)

    new_blocks = [meshio.CellBlock(b.type, remap[b.data]) for b in cell_blocks]

    return {
        "pre_dedup_nodes": pts.shape[0],
        "post_dedup_nodes": deduped_pts.shape[0],
        "deduped": pts.shape[0] - deduped_pts.shape[0],
        "physical_group_names": sorted(field_data),
        "physical_groups": field_data,
        "mesh": meshio.Mesh(
            points=deduped_pts,
            cells=new_blocks,
            cell_data={"gmsh:physical": cell_data_phys},
            field_data=field_data,
        ),
    }


# --------------------------------------------------------------------------
# Conformity probe: count internal duplicate nodes after dedup.
# A "shared face" is identified by its xmin/xmax/ymin/ymax/zmin/zmax constraint.
# --------------------------------------------------------------------------


def _count_nodes_on_plane(pts: np.ndarray, axis: int, value: float, tol: float) -> int:
    return int(np.sum(np.abs(pts[:, axis] - value) < tol))


# --------------------------------------------------------------------------
# Spike tests
# --------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason="documented finding: gmsh.merge collides physical-group tag IDs "
    "across files; only the FIRST file's name survives for any given (dim, tag). "
    "gmsh_merge_strategy is fundamentally broken for cross-file physical-tag "
    "preservation — meshio_concat_strategy is the correct path. This xfail "
    "pins the broken behavior so a future gmsh fix flips it red.",
)
def test_2x1_single_material_via_gmsh_merge(tmp_path):
    """2x1: two adjacent unit boxes, single physical group per tile."""
    lc = 0.5
    paths = _write_grid_meshes(tmp_path, nx=2, ny=1, lc=lc)
    res = _gmsh_merge_strategy([p for p, _ in paths], dedup_tol=lc / 100)
    print("\n2x1 gmsh_merge:", {k: v for k, v in res.items() if k != "physical_groups"})
    assert {"tile_0_0_0", "tile_1_0_0"} <= set(res["physical_group_names"])
    assert res["deduped"] > 0, "expected coincident nodes at x=1 to dedup"


def test_2x1_single_material_via_meshio_concat(tmp_path):
    lc = 0.5
    paths = _write_grid_meshes(tmp_path, nx=2, ny=1, lc=lc)
    res = _meshio_concat_strategy([p for p, _ in paths], dedup_tol=lc / 100)
    print(
        "\n2x1 meshio_concat:",
        {k: v for k, v in res.items() if k not in {"physical_groups", "mesh"}},
    )
    assert {"tile_0_0_0", "tile_1_0_0"} <= set(res["physical_group_names"])
    assert res["deduped"] > 0


@pytest.mark.xfail(strict=True, reason="see test_2x1_single_material_via_gmsh_merge")
def test_2x2_single_material_via_gmsh_merge(tmp_path):
    """2x2: four tiles meeting at an interior corner (the v1-failing case)."""
    lc = 0.5
    paths = _write_grid_meshes(tmp_path, nx=2, ny=2, lc=lc)
    res = _gmsh_merge_strategy([p for p, _ in paths], dedup_tol=lc / 100)
    print("\n2x2 gmsh_merge:", {k: v for k, v in res.items() if k != "physical_groups"})
    expected_names = {f"tile_{i}_{j}_0" for i in range(2) for j in range(2)}
    assert expected_names <= set(res["physical_group_names"])
    assert res["deduped"] > 0


def test_2x2_single_material_via_meshio_concat(tmp_path):
    lc = 0.5
    paths = _write_grid_meshes(tmp_path, nx=2, ny=2, lc=lc)
    res = _meshio_concat_strategy([p for p, _ in paths], dedup_tol=lc / 100)
    print(
        "\n2x2 meshio_concat:",
        {k: v for k, v in res.items() if k not in {"physical_groups", "mesh"}},
    )
    expected_names = {f"tile_{i}_{j}_0" for i in range(2) for j in range(2)}
    assert expected_names <= set(res["physical_group_names"])
    assert res["deduped"] > 0


@pytest.mark.xfail(strict=True, reason="see test_2x1_single_material_via_gmsh_merge")
def test_3x3_single_material_via_gmsh_merge(tmp_path):
    lc = 0.5
    paths = _write_grid_meshes(tmp_path, nx=3, ny=3, lc=lc)
    res = _gmsh_merge_strategy([p for p, _ in paths], dedup_tol=lc / 100)
    print("\n3x3 gmsh_merge:", {k: v for k, v in res.items() if k != "physical_groups"})
    expected_names = {f"tile_{i}_{j}_0" for i in range(3) for j in range(3)}
    assert expected_names <= set(res["physical_group_names"])


def test_3x3_single_material_via_meshio_concat(tmp_path):
    lc = 0.5
    paths = _write_grid_meshes(tmp_path, nx=3, ny=3, lc=lc)
    res = _meshio_concat_strategy([p for p, _ in paths], dedup_tol=lc / 100)
    print(
        "\n3x3 meshio_concat:",
        {k: v for k, v in res.items() if k not in {"physical_groups", "mesh"}},
    )
    expected_names = {f"tile_{i}_{j}_0" for i in range(3) for j in range(3)}
    assert expected_names <= set(res["physical_group_names"])


def test_2x1_seam_conformity_count(tmp_path):
    """How many nodes get deduped at the shared face for 2x1 with matching lc?

    Reports raw vs deduped vs nodes-on-the-shared-plane. The shared
    face is x=1; we count how many nodes from the raw concat lie there
    and how many remain after dedup.
    """
    lc = 0.5
    paths = _write_grid_meshes(tmp_path, nx=2, ny=1, lc=lc)
    # Read each, count nodes on x=1.
    counts = {}
    pts_all = []
    for p, name in paths:
        m = meshio.read(p)
        on_plane = _count_nodes_on_plane(m.points, axis=0, value=1.0, tol=lc / 100)
        counts[name] = (m.points.shape[0], on_plane)
        pts_all.append(m.points)
    raw = sum(c[0] for c in counts.values())
    raw_on_plane = sum(c[1] for c in counts.values())

    res = _meshio_concat_strategy([p for p, _ in paths], dedup_tol=lc / 100)
    print(
        f"\n2x1 conformity: per-file (nodes, on x=1)={counts}; "
        f"raw={raw}, raw_on_plane={raw_on_plane}, "
        f"deduped={res['deduped']}, post_dedup={res['post_dedup_nodes']}"
    )
    # Conformity check: every node-on-plane in tile 0 should have a coincident
    # twin in tile 1, so dedup should remove exactly one of each pair.
    # If gmsh produced deterministic curve+face nodes on the shared plane,
    # raw_on_plane should be 2 * (nodes shared).
    expected_dedup = raw_on_plane // 2
    print(
        f"  expected dedup if every plane node had a twin: {expected_dedup}; "
        f"actual: {res['deduped']}"
    )


def test_2x2_seam_conformity_count(tmp_path):
    """Same conformity probe for 2x2."""
    lc = 0.5
    paths = _write_grid_meshes(tmp_path, nx=2, ny=2, lc=lc)
    counts = {}
    for p, name in paths:
        m = meshio.read(p)
        on_x1 = _count_nodes_on_plane(m.points, axis=0, value=1.0, tol=lc / 100)
        on_y1 = _count_nodes_on_plane(m.points, axis=1, value=1.0, tol=lc / 100)
        counts[name] = (m.points.shape[0], on_x1, on_y1)
    res = _meshio_concat_strategy([p for p, _ in paths], dedup_tol=lc / 100)
    print(
        f"\n2x2 conformity: per-file (nodes, on x=1, on y=1)={counts}; "
        f"raw_total={sum(c[0] for c in counts.values())}, "
        f"deduped={res['deduped']}, post_dedup={res['post_dedup_nodes']}"
    )


# --------------------------------------------------------------------------
# Multi-material per tile: each tile has TWO sub-volumes (e.g., bottom
# half + top half). Tests whether physical-group preservation works when
# names repeat across tiles AND multiple groups exist per file.
# --------------------------------------------------------------------------


def _build_two_material_tile_msh(
    path: Path,
    bbox: tuple[float, float, float, float, float, float],
    mat_lo: str,
    mat_hi: str,
    split_z: float,
    lc: float,
) -> None:
    """Mesh two stacked OCC boxes (bottom + top) sharing a face at z=split_z."""
    gmsh.model.add(f"two_mat_{mat_lo}_{mat_hi}")
    xmin, ymin, zmin, xmax, ymax, zmax = bbox
    bot = gmsh.model.occ.addBox(
        xmin, ymin, zmin, xmax - xmin, ymax - ymin, split_z - zmin
    )
    top = gmsh.model.occ.addBox(
        xmin, ymin, split_z, xmax - xmin, ymax - ymin, zmax - split_z
    )
    # Fragment so the shared face is a single shared topological entity.
    gmsh.model.occ.fragment([(3, bot), (3, top)], [])
    gmsh.model.occ.synchronize()
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc)
    # Tag volumes by their z-centroid.
    for d, t in gmsh.model.getEntities(3):
        bb = gmsh.model.getBoundingBox(d, t)
        zc = 0.5 * (bb[2] + bb[5])
        name = mat_lo if zc < split_z else mat_hi
        gmsh.model.addPhysicalGroup(3, [t], name=name)
    gmsh.model.mesh.generate(3)
    gmsh.write(str(path))
    gmsh.model.remove()


def test_2x1_two_materials_per_tile_meshio_concat(tmp_path):
    """2 tiles, each holds 'silicon' (bottom) + 'oxide' (top); merge keeps both.

    After merge, we should see exactly 2 physical groups: 'silicon' and
    'oxide', each spanning both tiles.
    """
    lc = 0.4
    paths = []
    for i in range(2):
        p = tmp_path / f"tile_{i}.msh"
        _build_two_material_tile_msh(
            p,
            bbox=(i * 1.0, 0, 0, (i + 1) * 1.0, 1, 1),
            mat_lo="silicon",
            mat_hi="oxide",
            split_z=0.5,
            lc=lc,
        )
        paths.append(p)
    res = _meshio_concat_strategy(paths, dedup_tol=lc / 100)
    print(
        "\n2x1 two-mat meshio_concat:",
        {k: v for k, v in res.items() if k not in {"physical_groups", "mesh"}},
    )
    names = set(res["physical_group_names"])
    assert "silicon" in names, names
    assert "oxide" in names, names
    # The merged mesh should NOT have separate per-tile silicon entries.
    assert len([n for n in names if "silicon" in n]) == 1
    assert len([n for n in names if "oxide" in n]) == 1
    # Each tile contributes ~half its nodes to the seam at x=1.
    assert res["deduped"] > 0


def test_2x2_two_materials_per_tile_meshio_concat(tmp_path):
    """4-tile 2x2 grid, two materials per tile; merge keeps all names.

    Verifies merge handles both inter-tile seams (x=1, y=1) AND
    intra-tile material seams (z=0.5) without losing any group names.
    """
    lc = 0.4
    paths = []
    for i in range(2):
        for j in range(2):
            p = tmp_path / f"tile_{i}_{j}.msh"
            _build_two_material_tile_msh(
                p,
                bbox=(i * 1.0, j * 1.0, 0, (i + 1) * 1.0, (j + 1) * 1.0, 1),
                mat_lo="silicon",
                mat_hi="oxide",
                split_z=0.5,
                lc=lc,
            )
            paths.append(p)
    res = _meshio_concat_strategy(paths, dedup_tol=lc / 100)
    print(
        "\n2x2 two-mat meshio_concat:",
        {k: v for k, v in res.items() if k not in {"physical_groups", "mesh"}},
    )
    names = set(res["physical_group_names"])
    assert "silicon" in names, names
    assert "oxide" in names, names
    # Should consolidate to exactly 2 named groups (after name-merge).
    assert len(names) == 2, names


def test_2x1_mismatched_lc_breaks_conformity(tmp_path):
    """Two adjacent tiles with DIFFERENT lc do not produce a conformal seam.

    Most boundary nodes will not have coincident twins; only corners
    survive dedup.
    """
    paths = []
    for i, lc in enumerate([0.5, 0.3]):
        p = tmp_path / f"tile_{i}.msh"
        _build_box_msh(p, bbox=(i, 0, 0, i + 1, 1, 1), name=f"tile_{i}", lc=lc)
        paths.append(p)
    res = _meshio_concat_strategy(paths, dedup_tol=0.001)
    # Count nodes on shared plane x=1 in raw (concat-only).
    raw_pts = np.vstack([meshio.read(p).points for p in paths])
    on_plane_raw = _count_nodes_on_plane(raw_pts, axis=0, value=1.0, tol=0.001)
    print(
        f"\n2x1 mismatched lc: raw_nodes={raw_pts.shape[0]}, "
        f"on_x=1 (raw)={on_plane_raw}, deduped={res['deduped']}, "
        f"post_dedup={res['post_dedup_nodes']}"
    )
    # If conformity broke, dedup count is much smaller than on_plane_raw / 2
    # (the corners always coincide; the interior boundary nodes won't).
    print(
        f"  expected dedup if fully conformal: {on_plane_raw // 2}, "
        f"actual: {res['deduped']}"
    )


def test_2x1_tile_meshed_with_internal_polygon_material(tmp_path):
    """Each tile contains a circular sub-region (different material).

    Probes whether conformity at the inter-tile seam survives when the
    intra-tile geometry has features that differ from the cuboidal shell.
    """
    lc = 0.3
    paths = []
    for i in range(2):
        p = tmp_path / f"tile_{i}.msh"
        gmsh.model.add(f"complex_tile_{i}")
        # Outer box.
        gmsh.model.occ.addBox(i, 0, 0, 1, 1, 1)
        # Inner cylinder, axis-aligned along z, centered in the tile.
        cx = i + 0.5
        gmsh.model.occ.addCylinder(cx, 0.5, 0.0, 0, 0, 1, 0.2)
        gmsh.model.occ.fragment([(3, t) for d, t in gmsh.model.occ.getEntities(3)], [])
        gmsh.model.occ.synchronize()
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc)
        # Tag by volume mass: the cylinder is small, the shell is large.
        for d, t in gmsh.model.getEntities(3):
            mass = gmsh.model.occ.getMass(d, t)
            name = "core" if mass < 0.5 else "shell"  # cylinder ~0.126, shell ~0.874
            gmsh.model.addPhysicalGroup(3, [t], name=name)
        gmsh.model.mesh.generate(3)
        gmsh.write(str(p))
        gmsh.model.remove()
        paths.append(p)

    res = _meshio_concat_strategy(paths, dedup_tol=lc / 100)
    raw_pts = np.vstack([meshio.read(p).points for p in paths])
    on_plane_raw = _count_nodes_on_plane(raw_pts, axis=0, value=1.0, tol=lc / 100)
    print(
        f"\n2x1 internal-polygon: per-tile cells differ; raw nodes={raw_pts.shape[0]}, "
        f"on x=1 (raw)={on_plane_raw}, deduped={res['deduped']}, "
        f"post_dedup={res['post_dedup_nodes']}, "
        f"physical_groups={sorted(res['physical_group_names'])}"
    )
    assert {"core", "shell"} == set(res["physical_group_names"])
    # Conformity at x=1 only requires the SHELL face mesh to match — the
    # core doesn't touch x=1. If the box face mesh is deterministic, dedup
    # should equal on_plane_raw / 2.
    print(
        f"  expected dedup if shared face fully conformal: {on_plane_raw // 2}; "
        f"actual: {res['deduped']}"
    )
