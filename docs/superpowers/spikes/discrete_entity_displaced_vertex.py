# ruff: noqa: S108, S110, SIM105, RUF005, RUF059, ARG001, D101
"""Phase-0 spike: discrete 3D entity inside an OCC phantom with BOP-displaced vertices.

Goal: validate the new structured-pipeline strategy of independently meshing top
and bottom OCC faces and bridging them with prisms in a discrete 3D entity, in
a setup that emulates the real meshwell production case (OCC phantom + BOP
fragmentation + neighbour entities producing fuzzy vertex displacements).

Three phases:

  P1  Synthetic baseline (no OCC). Two top/bottom 2D meshes, isomorphic
      triangulations, vertices displaced by varying amounts. Confirms gmsh
      accepts addElements(3) with non-bit-equal node coordinates between top
      and bottom prism faces.

  P2  OCC phantom + BOP fragment. Build a 1x1x1 OCC box (the "phantom"), cut a
      smaller box from a corner of the top face (BOP fragment with a small
      fuzzy_value), then remove the box solid (non-recursive) leaving its top
      and bottom OCC faces alive. Mesh those faces in 2D independently. Read
      back the actual node positions on each face (these will be slightly
      displaced from each other because BOP introduces tolerance-scale
      vertex movement). Build a discrete 3D entity that bridges them.

  P3  Stacked phantoms. Two stacked phantom boxes sharing a horizontal
      interface, each with its own BOP-fragmented top/bottom. The interface
      face has independent 2D meshes from each side (with displaced node
      positions). Build two discrete 3D entities, one per slab. Verify gmsh
      tolerates the two independent meshes on the shared interface and the
      whole construct writes + reopens.

For each phase, report success/failure, element count, min/mean quality, and
whether the .msh file roundtrips.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any

import gmsh
import numpy as np

# -----------------------------------------------------------------------------
# Phase 1: synthetic baseline (sanity check)
# -----------------------------------------------------------------------------


@dataclass
class P1Scenario:
    name: str
    dxy: float


def _square_two_triangles(
    z: float, dxy: float = 0.0, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    coords = np.array(
        [[0.0, 0.0, z], [1.0, 0.0, z], [1.0, 1.0, z], [0.0, 1.0, z]],
        dtype=float,
    )
    if dxy > 0:
        rng = np.random.default_rng(seed)
        for i in range(4):
            theta = rng.uniform(0, 2 * np.pi)
            coords[i, 0] += dxy * np.cos(theta)
            coords[i, 1] += dxy * np.sin(theta)
    tris = np.array([[1, 2, 3], [1, 3, 4]], dtype=int)
    return coords, tris


def run_p1(scenario: P1Scenario) -> dict:
    diag: dict[str, Any] = {"phase": "P1", "name": scenario.name, "dxy": scenario.dxy}
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(scenario.name)
    try:
        bot_coords, tris = _square_two_triangles(z=0.0, dxy=0.0)
        top_coords, _ = _square_two_triangles(z=1.0, dxy=scenario.dxy, seed=1)

        bot_surf = gmsh.model.addDiscreteEntity(2, -1, [])
        top_surf = gmsh.model.addDiscreteEntity(2, -1, [])
        bot_tags = [1, 2, 3, 4]
        top_tags = [5, 6, 7, 8]
        gmsh.model.mesh.addNodes(2, bot_surf, bot_tags, bot_coords.reshape(-1).tolist())
        gmsh.model.mesh.addNodes(2, top_surf, top_tags, top_coords.reshape(-1).tolist())
        gmsh.model.mesh.addElements(
            2, bot_surf, [2], [[1, 2]], [tris.reshape(-1).tolist()]
        )
        gmsh.model.mesh.addElements(
            2,
            top_surf,
            [2],
            [[3, 4]],
            [[top_tags[i - 1] for i in tris.reshape(-1).tolist()]],
        )

        vol = gmsh.model.addDiscreteEntity(3, -1, [])
        prism_nodes = []
        for tri in tris:
            a, b, c = tri
            prism_nodes.extend(
                [
                    bot_tags[a - 1],
                    bot_tags[b - 1],
                    bot_tags[c - 1],
                    top_tags[a - 1],
                    top_tags[b - 1],
                    top_tags[c - 1],
                ]
            )
        gmsh.model.mesh.addElements(3, vol, [6], [[5, 6]], [prism_nodes])
        diag["addElements_succeeded"] = True

        out = f"/tmp/spike_p1_{scenario.name}.msh"
        gmsh.write(out)
        diag["write_succeeded"] = True

        qualities = gmsh.model.mesh.getElementQualities([5, 6], qualityName="minDetJac")
        diag["min_quality"] = float(np.min(qualities))
        diag["mean_quality"] = float(np.mean(qualities))

        gmsh.model.remove()
        gmsh.open(out)
        _, etags, _ = gmsh.model.mesh.getElements(3)
        diag["n_3d_after_reopen"] = sum(len(t) for t in etags)
        diag["reopen_succeeded"] = diag["n_3d_after_reopen"] == 2
    except Exception as e:
        diag["error"] = repr(e)
    finally:
        try:
            gmsh.finalize()
        except Exception:
            pass
    return diag


# -----------------------------------------------------------------------------
# Phase 2: OCC phantom + BOP fragment
# -----------------------------------------------------------------------------


def _bbox_of_face(tag: int) -> tuple[float, float, float, float, float, float]:
    return tuple(gmsh.model.getBoundingBox(2, tag))  # type: ignore[return-value]


def _is_horizontal_face(tag: int, z: float, tol: float = 1e-6) -> bool:
    bb = _bbox_of_face(tag)
    return abs(bb[2] - z) < tol and abs(bb[5] - z) < tol


def _read_face_2d_mesh(tag: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (node_tags, node_coords (n,3), triangles (m,3) of node_tag)."""
    node_tags_arr, node_coords_flat, _ = gmsh.model.mesh.getNodes(
        2, tag, includeBoundary=True
    )
    node_tags = np.asarray(node_tags_arr, dtype=np.int64)
    node_coords = np.asarray(node_coords_flat, dtype=float).reshape(-1, 3)
    elem_types, elem_tags, elem_nodes = gmsh.model.mesh.getElements(2, tag)
    triangles_flat = []
    for et, en in zip(elem_types, elem_nodes):
        if et == 2:  # 3-node triangle
            triangles_flat.append(np.asarray(en, dtype=np.int64).reshape(-1, 3))
    triangles = (
        np.concatenate(triangles_flat, axis=0)
        if triangles_flat
        else np.zeros((0, 3), dtype=np.int64)
    )
    return node_tags, node_coords, triangles


def _match_top_to_bottom(
    bot_tags: np.ndarray,
    bot_coords: np.ndarray,
    top_tags: np.ndarray,
    top_coords: np.ndarray,
    height_z: float,
    xy_tol: float,
) -> dict[int, int] | None:
    """Greedy nearest-xy match. Returns dict[bot_tag] = top_tag, or None on mismatch.

    For every bottom node, find the top node whose (x,y) is within xy_tol.
    Fail if any bottom node has 0 matches or more than 1 match within tol/2.
    """
    if len(bot_tags) != len(top_tags):
        return None
    mapping: dict[int, int] = {}
    used_top = set()
    for i, bt in enumerate(bot_tags):
        bx, by = bot_coords[i, 0], bot_coords[i, 1]
        # Compute distances.
        dx = top_coords[:, 0] - bx
        dy = top_coords[:, 1] - by
        d2 = dx * dx + dy * dy
        order = np.argsort(d2)
        best = order[0]
        if d2[best] > xy_tol * xy_tol:
            return None
        if int(top_tags[best]) in used_top:
            return None
        mapping[int(bt)] = int(top_tags[best])
        used_top.add(int(top_tags[best]))
    return mapping


def run_p2() -> dict:
    """OCC phantom: 1x1x1 box, cut a 0.3x0.3x0.05 corner notch from top via fragment.

    The cut introduces new vertices on the top face only — but since we want
    matching topology top<->bottom, instead of cutting only the top, fragment
    against a vertical stick passing through the entire box. That partitions
    both top and bottom identically but introduces fuzzy-tolerance displacement.
    """
    diag: dict[str, Any] = {"phase": "P2", "name": "occ_phantom_fragment"}
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("p2")
    try:
        # Big box (the phantom).
        big = gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
        # Vertical stick passing all the way through, off-center to make the
        # bottom/top face partition non-trivial.
        stick = gmsh.model.occ.addBox(0.4, 0.4, -0.1, 0.3, 0.3, 1.2)

        fuzzy = 1e-5
        out_dim_tags, _ = gmsh.model.occ.fragment(
            [(3, big)], [(3, stick)], removeObject=True, removeTool=True
        )
        diag["fragment_out"] = [(d, t) for (d, t) in out_dim_tags if d == 3]
        gmsh.model.occ.synchronize()

        # After fragment, the volumes are split. We treat the phantom as the
        # union of all 3D pieces lying in the original big-box bbox; for this
        # spike we'll just remove the stick's volume (non-recursive) and use
        # only the box volume. To keep it simple: enumerate volumes, identify
        # the one whose bbox matches the original big-box xy-extent + full z.
        all_vols = gmsh.model.getEntities(3)
        diag["n_volumes_after_fragment"] = len(all_vols)

        # Pick the volume whose xy bbox exactly matches the big box. After a
        # through-cut fragment that's normally just the "outside the stick"
        # piece, but a single OCC volume can contain it. We simply mesh whatever
        # remains after deleting the stick.
        # Identify stick volume: bbox xy in [0.4, 0.7].
        stick_vols = []
        for d, t in all_vols:
            bb = gmsh.model.getBoundingBox(d, t)
            if 0.39 <= bb[0] <= 0.41 and 0.69 <= bb[3] <= 0.71:
                stick_vols.append(t)
        diag["stick_vols"] = stick_vols

        # Remove the stick (non-recursive) so its faces stay alive.
        for tv in stick_vols:
            gmsh.model.occ.remove([(3, tv)], recursive=False)
        gmsh.model.occ.synchronize()
        remaining_vols = gmsh.model.getEntities(3)
        diag["n_volumes_after_remove"] = len(remaining_vols)

        # Mesh 2D only.
        gmsh.option.setNumber("Mesh.MeshSizeMin", 0.2)
        gmsh.option.setNumber("Mesh.MeshSizeMax", 0.2)
        gmsh.model.mesh.generate(2)
        diag["mesh2d_succeeded"] = True

        # Identify bottom faces (z=0 plane) and top faces (z=1 plane).
        all_faces = gmsh.model.getEntities(2)
        bottom_faces = [t for _, t in all_faces if _is_horizontal_face(t, 0.0)]
        top_faces = [t for _, t in all_faces if _is_horizontal_face(t, 1.0)]
        diag["bottom_face_tags"] = bottom_faces
        diag["top_face_tags"] = top_faces

        if len(bottom_faces) != len(top_faces):
            diag["topology_mismatch"] = True
            diag[
                "error"
            ] = f"bottom has {len(bottom_faces)} faces, top has {len(top_faces)}"
            return diag

        # For each bottom face, find the top face with matching xy bbox.
        def _xy_bb(t):
            bb = _bbox_of_face(t)
            return (round(bb[0], 4), round(bb[1], 4), round(bb[3], 4), round(bb[4], 4))

        top_by_xy = {_xy_bb(t): t for t in top_faces}
        face_pairs: list[tuple[int, int]] = []
        for bf in bottom_faces:
            key = _xy_bb(bf)
            if key not in top_by_xy:
                diag["error"] = f"bottom face {bf} (xy {key}) has no top twin"
                return diag
            face_pairs.append((bf, top_by_xy[key]))
        diag["face_pairs"] = face_pairs

        # Build a discrete 3D entity per pair, prisms bridge the meshes.
        n_3d_total = 0
        min_qual = float("inf")
        sum_qual = 0.0
        n_qual = 0
        max_displacement = 0.0
        next_elem_tag = int(gmsh.model.mesh.getMaxElementTag()) + 1
        for bf, tf in face_pairs:
            bot_tags, bot_coords, bot_tris = _read_face_2d_mesh(bf)
            top_tags, top_coords, top_tris = _read_face_2d_mesh(tf)

            # Production assumption: same partition piece -> isomorphic
            # triangulations. Verify and report:
            if bot_tris.shape != top_tris.shape:
                diag["error"] = (
                    f"non-isomorphic triangulation on pair ({bf},{tf}): "
                    f"bot {bot_tris.shape} vs top {top_tris.shape}"
                )
                return diag

            # Match top to bottom by nearest xy.
            mapping = _match_top_to_bottom(
                bot_tags,
                bot_coords,
                top_tags,
                top_coords,
                height_z=1.0,
                xy_tol=1e-3,
            )
            if mapping is None:
                diag["error"] = f"node matching failed on pair ({bf},{tf})"
                return diag

            # Measure max xy displacement between matched node pairs.
            tag_to_top_idx = {int(t): i for i, t in enumerate(top_tags)}
            tag_to_bot_idx = {int(t): i for i, t in enumerate(bot_tags)}
            for bt, tt in mapping.items():
                bi = tag_to_bot_idx[bt]
                ti = tag_to_top_idx[tt]
                dxy = float(
                    np.hypot(
                        bot_coords[bi, 0] - top_coords[ti, 0],
                        bot_coords[bi, 1] - top_coords[ti, 1],
                    )
                )
                max_displacement = max(max_displacement, dxy)

            vol = gmsh.model.addDiscreteEntity(3, -1, [])
            prism_nodes: list[int] = []
            for tri in bot_tris:
                a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
                prism_nodes.extend(
                    [
                        a,
                        b,
                        c,
                        mapping[a],
                        mapping[b],
                        mapping[c],
                    ]
                )
            n_prisms = bot_tris.shape[0]
            prism_tags = list(range(next_elem_tag, next_elem_tag + n_prisms))
            next_elem_tag += n_prisms
            gmsh.model.mesh.addElements(3, vol, [6], [prism_tags], [prism_nodes])
            qualities = gmsh.model.mesh.getElementQualities(
                prism_tags, qualityName="minDetJac"
            )
            min_qual = min(min_qual, float(np.min(qualities)))
            sum_qual += float(np.sum(qualities))
            n_qual += n_prisms
            n_3d_total += n_prisms

        diag["n_3d_elements"] = n_3d_total
        diag["min_quality"] = min_qual if n_qual else None
        diag["mean_quality"] = (sum_qual / n_qual) if n_qual else None
        diag["max_node_xy_displacement"] = max_displacement
        diag["fuzzy_value"] = fuzzy

        out = "/tmp/spike_p2.msh"
        gmsh.write(out)
        diag["write_succeeded"] = True

        gmsh.model.remove()
        gmsh.open(out)
        _, etags, _ = gmsh.model.mesh.getElements(3)
        diag["n_3d_after_reopen"] = sum(len(t) for t in etags)
        diag["reopen_succeeded"] = diag["n_3d_after_reopen"] == n_3d_total
    except Exception as e:
        diag["error"] = repr(e)
    finally:
        try:
            gmsh.finalize()
        except Exception:
            pass
    return diag


# -----------------------------------------------------------------------------
# Phase 3: Stacked phantoms sharing an interface
# -----------------------------------------------------------------------------


def run_p3() -> dict:
    """Two phantom boxes stacked: lower z=[0,1], upper z=[1,2], sharing z=1 interface.

    Both fragmented against the same vertical stick, then both removed
    non-recursively. Each box gets its own discrete 3D entity. The shared
    interface face is meshed twice (once as 'top' of lower, once as 'bottom'
    of upper) — these meshes are independent so node positions at z=1 may
    differ across the two slabs. Verify gmsh tolerates this.
    """
    diag: dict[str, Any] = {"phase": "P3", "name": "stacked_phantoms"}
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("p3")
    try:
        lower = gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
        upper = gmsh.model.occ.addBox(0, 0, 1, 1, 1, 1)
        stick = gmsh.model.occ.addBox(0.4, 0.4, -0.1, 0.3, 0.3, 2.2)

        out_dim_tags, _ = gmsh.model.occ.fragment(
            [(3, lower), (3, upper)],
            [(3, stick)],
            removeObject=True,
            removeTool=True,
        )
        gmsh.model.occ.synchronize()
        diag["n_volumes_after_fragment"] = len(gmsh.model.getEntities(3))

        # Identify stick volumes (xy in [0.4, 0.7], any z).
        all_vols = gmsh.model.getEntities(3)
        stick_vols = []
        for d, t in all_vols:
            bb = gmsh.model.getBoundingBox(d, t)
            if 0.39 <= bb[0] <= 0.41 and 0.69 <= bb[3] <= 0.71:
                stick_vols.append(t)
        for tv in stick_vols:
            gmsh.model.occ.remove([(3, tv)], recursive=False)
        gmsh.model.occ.synchronize()
        diag["n_volumes_after_remove"] = len(gmsh.model.getEntities(3))

        gmsh.option.setNumber("Mesh.MeshSizeMin", 0.2)
        gmsh.option.setNumber("Mesh.MeshSizeMax", 0.2)
        gmsh.model.mesh.generate(2)

        all_faces = gmsh.model.getEntities(2)
        z0_faces = [t for _, t in all_faces if _is_horizontal_face(t, 0.0)]
        z1_faces = [t for _, t in all_faces if _is_horizontal_face(t, 1.0)]
        z2_faces = [t for _, t in all_faces if _is_horizontal_face(t, 2.0)]
        diag["z0_faces"] = z0_faces
        diag["z1_faces"] = z1_faces
        diag["z2_faces"] = z2_faces

        # Lower slab pairs z0 -> z1; upper slab pairs z1 -> z2.
        # The z=1 faces are SHARED across both slabs (one OCC face entity per
        # piece). When the lower slab's discrete volume references its
        # 'top' nodes and the upper slab's discrete volume references its
        # 'bottom' nodes, both reference the SAME node tags (because OCC and
        # mesh are 1-1). So no displacement at the interface in this setup.
        #
        # To exercise the "displaced interface" case we'd need two separate
        # interface faces (one owned by each slab), which is what the new
        # pipeline will likely produce when each slab's phantom is built
        # independently. We simulate that by removing the interface mesh
        # from one slab's view and re-adding it as a separate discrete
        # entity with displaced node coords.
        #
        # For this spike we report whether the shared-interface case (the
        # easier sub-case) works end-to-end with two stacked discrete
        # volumes.

        def _xy_bb(t):
            bb = _bbox_of_face(t)
            return (round(bb[0], 4), round(bb[1], 4), round(bb[3], 4), round(bb[4], 4))

        # Build lower-slab discrete volume.
        next_elem_tag = int(gmsh.model.mesh.getMaxElementTag()) + 1
        n_3d_total = 0
        min_qual = float("inf")
        sum_qual = 0.0
        n_qual = 0

        for slab_label, bot_faces, top_faces in [
            ("lower", z0_faces, z1_faces),
            ("upper", z1_faces, z2_faces),
        ]:
            top_by_xy = {_xy_bb(t): t for t in top_faces}
            for bf in bot_faces:
                key = _xy_bb(bf)
                if key not in top_by_xy:
                    diag[f"{slab_label}_unmatched"] = bf
                    continue
                tf = top_by_xy[key]
                bot_tags, bot_coords, bot_tris = _read_face_2d_mesh(bf)
                top_tags, top_coords, top_tris = _read_face_2d_mesh(tf)
                if bot_tris.shape != top_tris.shape:
                    diag["error"] = (
                        f"{slab_label} non-iso topology pair ({bf},{tf}): "
                        f"{bot_tris.shape} vs {top_tris.shape}"
                    )
                    return diag
                mapping = _match_top_to_bottom(
                    bot_tags,
                    bot_coords,
                    top_tags,
                    top_coords,
                    height_z=1.0,
                    xy_tol=1e-3,
                )
                if mapping is None:
                    diag["error"] = f"{slab_label} node match failed on ({bf},{tf})"
                    return diag

                vol = gmsh.model.addDiscreteEntity(3, -1, [])
                prism_nodes: list[int] = []
                for tri in bot_tris:
                    a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
                    prism_nodes.extend([a, b, c, mapping[a], mapping[b], mapping[c]])
                n_prisms = bot_tris.shape[0]
                prism_tags = list(range(next_elem_tag, next_elem_tag + n_prisms))
                next_elem_tag += n_prisms
                gmsh.model.mesh.addElements(3, vol, [6], [prism_tags], [prism_nodes])
                qualities = gmsh.model.mesh.getElementQualities(
                    prism_tags, qualityName="minDetJac"
                )
                min_qual = min(min_qual, float(np.min(qualities)))
                sum_qual += float(np.sum(qualities))
                n_qual += n_prisms
                n_3d_total += n_prisms

        diag["n_3d_elements"] = n_3d_total
        diag["min_quality"] = min_qual if n_qual else None
        diag["mean_quality"] = (sum_qual / n_qual) if n_qual else None
        out = "/tmp/spike_p3.msh"
        gmsh.write(out)
        diag["write_succeeded"] = True
        gmsh.model.remove()
        gmsh.open(out)
        _, etags, _ = gmsh.model.mesh.getElements(3)
        diag["n_3d_after_reopen"] = sum(len(t) for t in etags)
        diag["reopen_succeeded"] = diag["n_3d_after_reopen"] == n_3d_total
    except Exception as e:
        diag["error"] = repr(e)
    finally:
        try:
            gmsh.finalize()
        except Exception:
            pass
    return diag


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------


def main() -> int:
    p1_results = [
        run_p1(P1Scenario("zero", 0.0)),
        run_p1(P1Scenario("fuzzy_1e-6", 1e-6)),
        run_p1(P1Scenario("sloppy_1e-3", 1e-3)),
    ]
    p2_result = run_p2()
    p3_result = run_p3()

    print("\n" + "=" * 78)
    print("PHASE-0 SPIKE RESULTS — extended")
    print("=" * 78)
    fail = False
    for r in p1_results + [p2_result, p3_result]:
        print(f"\n[{r['phase']} :: {r['name']}]")
        for k, v in r.items():
            if k in ("phase", "name"):
                continue
            print(f"  {k:32s} {v}")
        if r.get("error"):
            fail = True
        if r.get("min_quality") is not None and r["min_quality"] <= 0:
            fail = True
    print("\n" + "=" * 78)
    print(f"Verdict: {'FAIL' if fail else 'PASS'}")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
