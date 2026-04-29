"""Mesh class definition."""
from __future__ import annotations

import contextlib
import tempfile
from collections.abc import Sequence
from os import cpu_count
from pathlib import Path

import meshio
import numpy as np

import gmsh
from meshwell._mesh_entity import _MeshEntity
from meshwell.model import ModelManager


def _normalize_algo(alg: int | Sequence[int]) -> tuple[int, ...]:
    """Coerce a single algorithm id or a sequence of fallbacks into a tuple."""
    if isinstance(alg, Sequence) and not isinstance(alg, (str, bytes)):
        seq = tuple(alg)
        if not seq:
            raise ValueError("algorithm sequence must not be empty")
        return seq
    return (int(alg),)


def _pair_algos(
    algos_2d: tuple[int, ...], algos_3d: tuple[int, ...]
) -> list[tuple[int, int]]:
    """Pair 2D/3D fallback sequences position-wise, padding with the last value."""
    n = max(len(algos_2d), len(algos_3d))
    return [
        (algos_2d[min(i, len(algos_2d) - 1)], algos_3d[min(i, len(algos_3d) - 1)])
        for i in range(n)
    ]


def _filter_msh_to_seam_groups(msh_path):
    """Rewrite msh_path keeping only physical groups whose name starts with '_seam___'.

    Reads via meshio, drops cells / cell_data / field_data not associated with seam
    groups, writes back to msh_path. Used by phase-1 workers in the distributed
    pipeline to emit only the seam surface mesh.

    Notes:
        meshio's ``field_data`` maps physical-group name -> ``[tag_id, dim]``.
        Gmsh tag ids are scoped per-dimension, so the filter keeps a (dim,
        tag_id) pair set rather than a flat tag-id set (different physical
        groups in different dimensions can share a tag id).

        We must write with ``file_format="gmsh22"``: meshio 5.3 .msh inference
        picks gmsh4 which silently drops ``field_data`` on round-trip.
    """
    # Hard-coded type -> topological dim. Covers everything meshwell emits
    # (triangles for surface groups, tetras for volume groups, plus higher-
    # order variants). Unknown types are skipped rather than guessed.
    _TYPE_DIM = {
        "vertex": 0,
        "line": 1,
        "line3": 1,
        "triangle": 2,
        "triangle6": 2,
        "triangle7": 2,
        "quad": 2,
        "quad8": 2,
        "quad9": 2,
        "tetra": 3,
        "tetra10": 3,
        "hexahedron": 3,
        "hexahedron20": 3,
        "hexahedron27": 3,
        "wedge": 3,
        "wedge15": 3,
        "wedge18": 3,
        "pyramid": 3,
        "pyramid13": 3,
        "pyramid14": 3,
    }

    msh_path = Path(msh_path)
    m = meshio.read(msh_path)
    keep_field_data = {
        name: np.asarray(tag_dim)
        for name, tag_dim in (m.field_data or {}).items()
        if name.startswith("_seam___")
    }
    if not keep_field_data:
        # Nothing to keep — write a minimal empty mesh.
        empty = meshio.Mesh(points=m.points[:0], cells=[])
        meshio.write(msh_path, empty, file_format="gmsh22")
        return

    # (dim, tag_id) pairs we want to keep. tag ids are per-dimension in gmsh.
    keep_pairs = {(int(arr[1]), int(arr[0])) for arr in keep_field_data.values()}

    new_cells = []
    new_cell_data = {k: [] for k in (m.cell_data or {})}
    gmsh_phys_arr = (m.cell_data or {}).get("gmsh:physical")
    for i, cellblock in enumerate(m.cells):
        if gmsh_phys_arr is None or i >= len(gmsh_phys_arr):
            continue
        block_dim = _TYPE_DIM.get(cellblock.type)
        if block_dim is None:
            # Unknown cell type — skip rather than guess.
            continue
        gmsh_phys = gmsh_phys_arr[i]
        mask = np.array(
            [(block_dim, int(t)) in keep_pairs for t in gmsh_phys],
            dtype=bool,
        )
        if not mask.any():
            continue
        idx = np.where(mask)[0]
        new_cells.append(meshio.CellBlock(cellblock.type, cellblock.data[idx]))
        for k in new_cell_data:
            new_cell_data[k].append(np.asarray(m.cell_data[k][i])[idx])

    out = meshio.Mesh(
        points=m.points,
        cells=new_cells,
        cell_data=new_cell_data,
        field_data=keep_field_data,
    )
    # Force gmsh22 format: meshio's gmsh4 writer drops field_data on .msh
    # files in our version (5.3.5), but gmsh22 round-trips it correctly.
    meshio.write(msh_path, out, file_format="gmsh22")


def _seed_occ_face_from_seam(seam_path, point_tolerance: float = 1e-6) -> None:
    """Seed the matching OCC face's mesh from an interface .msh.

    Implements the parametric-OCC-seeding recipe validated by
    ``tests/test_distributed_spike.py::test_occ_face_parametric_seed_from_discrete``.

    Steps:
        1. Read seam mesh into a SCRATCH gmsh model; capture the seam's seam
           physical name plus per-triangle node tags & xyz from the dim-2
           ``_seam___...`` group.
        2. Switch back to the active model and locate the OCC face:
             - PRIMARY: by seam's physical group name (any 2-D physical group
               in the active model with the matching ``_seam___...`` name).
             - FALLBACK: by collapsed-bbox + plane match (handles thin
               phantom-slab seams whose bbox has one near-zero axis).
        3. For each OCC corner point of the matched face: ``addNodes(0, ...)``.
        4. For each OCC boundary edge: parametrize seam-boundary nodes,
           ``addNodes(1, edge, ..., parametricCoord=t)`` then
           ``addElementsByType(edge, 1, ...)``.
        5. For the OCC face: parametrize interior seam nodes,
           ``addNodes(2, face, ..., parametricCoord=uv)`` then
           ``addElementsByType(face, 2, ...)``.
        6. Add the seam's physical name to the OCC face as a 2-D group so it
           survives into the final output.

    Caller MUST set ``Mesh.MeshOnlyEmpty=1`` and call ``mesh.generate(dim)``
    only once (no precursor ``generate(N)`` for ``N < dim``) — sequential
    ``generate()`` calls wipe seeded higher-dim elements.

    DO NOT use ``gmsh.merge`` or ``gmsh.model.mesh.embed`` to integrate the
    seam: neither preserves the imported triangulation when the host kernel
    is OCC. See ``tests/test_distributed_spike.py`` for the failed attempts.

    Args:
        seam_path: Path to a ``.msh`` containing one or more ``_seam___...``
            physical groups (typically produced by a phase-1 worker run with
            ``_emit_only_seam_surfaces=True``).
        point_tolerance: Coordinate tolerance for matching seam nodes to OCC
            entities (corner points, edges, plane). Default 1e-6.
    """
    seam_path = Path(seam_path)

    # ---- Step 1: read seam into scratch model, capture data ----
    try:
        active_model = gmsh.model.getCurrent()
    except Exception:
        active_model = None

    scratch_name = f"_seam_capture_{seam_path.stem}"
    gmsh.model.add(scratch_name)
    gmsh.model.setCurrent(scratch_name)
    try:
        gmsh.merge(str(seam_path))

        # Collect dim-2 physical groups whose name starts with _seam___
        seam_phys_groups = []  # list of (name, tag, [surface_entity_tags])
        for dim, ptag in gmsh.model.getPhysicalGroups(2):
            try:
                name = gmsh.model.getPhysicalName(dim, ptag)
            except Exception:
                name = ""
            if name and name.startswith("_seam___"):
                ents = list(gmsh.model.getEntitiesForPhysicalGroup(dim, ptag))
                if ents:
                    seam_phys_groups.append((name, ptag, ents))

        if not seam_phys_groups:
            # Nothing to seed.
            return

        # Capture node coordinates and triangles for each seam group.
        captured = []  # list of dicts
        for seam_name, _ptag, ent_tags in seam_phys_groups:
            tri_node_tag_triples = []  # list of (n0, n1, n2)
            node_tag_to_xyz = {}
            for ent_tag in ent_tags:
                etypes, _etags, enodes = gmsh.model.mesh.getElements(2, ent_tag)
                for etype, enode_block in zip(etypes, enodes):
                    if etype != 2:  # only first-order triangles
                        continue
                    arr = list(enode_block)
                    tri_node_tag_triples.extend(
                        (int(arr[i]), int(arr[i + 1]), int(arr[i + 2]))
                        for i in range(0, len(arr), 3)
                    )
                # Also gather all node coords for this surface (incl. boundary).
                ntags, ncoords, _ = gmsh.model.mesh.getNodes(
                    2, ent_tag, includeBoundary=True
                )
                for j, t in enumerate(ntags):
                    node_tag_to_xyz[int(t)] = (
                        float(ncoords[3 * j]),
                        float(ncoords[3 * j + 1]),
                        float(ncoords[3 * j + 2]),
                    )

            if not tri_node_tag_triples:
                continue

            # Compute bbox of the seam.
            xs = [p[0] for p in node_tag_to_xyz.values()]
            ys = [p[1] for p in node_tag_to_xyz.values()]
            zs = [p[2] for p in node_tag_to_xyz.values()]
            bbox = (min(xs), min(ys), min(zs), max(xs), max(ys), max(zs))
            captured.append(
                {
                    "name": seam_name,
                    "node_tag_to_xyz": node_tag_to_xyz,
                    "triangles": tri_node_tag_triples,
                    "bbox": bbox,
                }
            )
    finally:
        gmsh.model.remove()
        if active_model is not None:
            gmsh.model.setCurrent(active_model)

    if not captured:
        return

    # ---- Step 2 onwards: for each captured seam, locate and seed an OCC face. ----
    for cap in captured:
        _seed_one_seam_into_active_model(cap, point_tolerance=point_tolerance)


def _seed_one_seam_into_active_model(cap: dict, point_tolerance: float) -> None:
    """Seed a single captured seam's data into the active OCC model.

    See ``_seed_occ_face_from_seam`` for the recipe. ``cap`` is a dict with
    keys ``name``, ``node_tag_to_xyz``, ``triangles``, ``bbox``.
    """
    seam_name = cap["name"]
    seam_node_tag_to_xyz = dict(cap["node_tag_to_xyz"])
    seam_triangles = list(cap["triangles"])
    seam_bbox = cap["bbox"]

    # Determine the seam's "thin axis" (if any) and midplane coordinate.
    spans = (
        seam_bbox[3] - seam_bbox[0],
        seam_bbox[4] - seam_bbox[1],
        seam_bbox[5] - seam_bbox[2],
    )
    thin_axis = min(range(3), key=lambda i: spans[i])
    midplane = 0.5 * (seam_bbox[thin_axis] + seam_bbox[thin_axis + 3])
    # Tolerance for axis-aligned plane match: the larger of the configured
    # point_tolerance and half the seam slab's thin span (so a thin phantom
    # slab matches the OCC face on its midplane). Multiplied by 100 to absorb
    # the small outward offset gmsh applies to OCC face bounding boxes when
    # ``Geometry.OCCBoundsUseStl`` is enabled.
    plane_tol = max(
        point_tolerance * 100, 0.5 * spans[thin_axis] + point_tolerance * 100
    )

    # Project seam nodes onto the midplane of the thin axis. This is a no-op
    # for already-flat seams (spans[thin_axis] == 0) and collapses thin-slab
    # phantoms (spans[thin_axis] tiny but non-zero) onto a single plane so they
    # can seed a single OCC face.
    if spans[thin_axis] > 0:
        for t in list(seam_node_tag_to_xyz):
            x, y, z = seam_node_tag_to_xyz[t]
            arr = [x, y, z]
            arr[thin_axis] = midplane
            seam_node_tag_to_xyz[t] = (arr[0], arr[1], arr[2])

    # ---- Locate matching OCC face. ----
    occ_face = None

    # PRIMARY: by physical-group name in the active model.
    for dim, ptag in gmsh.model.getPhysicalGroups(2):
        try:
            name = gmsh.model.getPhysicalName(dim, ptag)
        except Exception:
            name = ""
        if name == seam_name:
            ents = gmsh.model.getEntitiesForPhysicalGroup(dim, ptag)
            if ents:
                occ_face = int(ents[0])
                break

    # FALLBACK: by axis-aligned plane + bbox match.
    if occ_face is None:
        # Build the projected seam bbox (collapsed in thin axis to midplane).
        proj_bbox = list(seam_bbox)
        proj_bbox[thin_axis] = midplane
        proj_bbox[thin_axis + 3] = midplane

        best_face = None
        best_score = float("inf")
        for d, t in gmsh.model.getEntities(2):
            bb = gmsh.model.getBoundingBox(d, t)
            # Must lie on the seam's thin-axis midplane.
            if abs(bb[thin_axis] - midplane) > plane_tol:
                continue
            if abs(bb[thin_axis + 3] - midplane) > plane_tol:
                continue
            # Score = max gap between OCC face bbox and projected seam bbox
            # over the in-plane axes. Lower = better.
            score = 0.0
            for ax in range(3):
                if ax == thin_axis:
                    continue
                score = max(
                    score,
                    abs(bb[ax] - proj_bbox[ax]),
                    abs(bb[ax + 3] - proj_bbox[ax + 3]),
                )
            if score < best_score:
                best_score = score
                best_face = int(t)
        if best_face is not None:
            occ_face = best_face

    if occ_face is None:
        # Nothing to seed against; skip silently. The user will get an under-
        # constrained mesh but no error — same as the no-op pre-distributed path.
        return

    # ---- Get OCC face boundary edges + corner points. ----
    edges_dimtags = gmsh.model.getBoundary(
        [(2, occ_face)], oriented=False, recursive=False
    )
    occ_edges = [int(t) for d, t in edges_dimtags if d == 1]
    pts_dimtags = gmsh.model.getBoundary(
        [(2, occ_face)], oriented=False, recursive=True
    )
    occ_corner_pts = sorted({int(t) for d, t in pts_dimtags if d == 0})

    # ---- Compute starting node tag (avoid collisions). ----
    try:
        existing_tags, _, _ = gmsh.model.mesh.getNodes()
        base_tag = int(max(existing_tags)) + 1 if len(existing_tags) else 1
    except Exception:
        base_tag = 1
    base_tag = max(base_tag, max(seam_node_tag_to_xyz) + 1)
    next_node_tag = [base_tag + 1000]

    def _alloc():
        v = next_node_tag[0]
        next_node_tag[0] += 1
        return v

    # ---- Step 3: corner points. ----
    # Map seam node tag -> OCC node tag. Also dedupe by xyz so nodes that
    # coincide post-projection share one OCC tag.
    seam_to_occ_node_tag: dict[int, int] = {}
    occ_corner_node_tag: dict[int, int] = {}

    def _key(xyz):
        return (
            round(xyz[0] / point_tolerance) * point_tolerance,
            round(xyz[1] / point_tolerance) * point_tolerance,
            round(xyz[2] / point_tolerance) * point_tolerance,
        )

    xyz_to_occ_node_tag: dict[tuple, int] = {}

    for pt_tag in occ_corner_pts:
        bb = gmsh.model.getBoundingBox(0, pt_tag)
        x_pt, y_pt, z_pt = bb[0], bb[1], bb[2]
        new_tag = _alloc()
        occ_corner_node_tag[pt_tag] = new_tag
        xyz_to_occ_node_tag[_key((x_pt, y_pt, z_pt))] = new_tag
        gmsh.model.mesh.addNodes(0, pt_tag, [new_tag], [x_pt, y_pt, z_pt])
        # Map ALL seam nodes coincident with this corner (within point_tolerance
        # in the in-plane axes) to the new OCC tag. Handles thin-slab seams whose
        # mirror corner pairs all collapse to the OCC corner after projection.
        for stag, (sx, sy, sz) in seam_node_tag_to_xyz.items():
            if (
                abs(sx - x_pt) < point_tolerance * 100
                and abs(sy - y_pt) < point_tolerance * 100
                and abs(sz - z_pt) < point_tolerance * 100
            ):
                seam_to_occ_node_tag[stag] = new_tag

    # ---- Step 4: classify each OCC edge & seed boundary nodes. ----
    # Classify a seam node as "on edge E" if it lies on E (within tolerance),
    # using gmsh.model.isInside / closest-point — but cheaper here: parametrize
    # candidate seam nodes onto the edge and accept those whose 3D distance
    # back to the seam node is small.
    def _seam_nodes_on_edge(edge_tag):
        """Return seam nodes lying on edge_tag as a list of (tag, xyz, t).

        Uses ``getClosestPoint`` to detect on-curve nodes by projection
        distance, then ``getParametrization`` to compute the parametric
        coordinate (the parametric returned by ``getClosestPoint`` is
        unreliable in our gmsh build).
        """
        result = []
        if not seam_node_tag_to_xyz:
            return result
        flat = []
        tag_order = []
        for stag, xyz in seam_node_tag_to_xyz.items():
            if stag in seam_to_occ_node_tag:
                # Already assigned (corner) — skip; corners are handled separately.
                continue
            flat.extend(xyz)
            tag_order.append((stag, xyz))
        if not tag_order:
            return result
        try:
            proj_xyz, _ = gmsh.model.getClosestPoint(1, edge_tag, flat)
        except Exception:
            return result
        # Filter to on-edge candidates by distance.
        on_edge_flat = []
        on_edge_tags = []
        for i, (stag, xyz) in enumerate(tag_order):
            px, py, pz = (
                float(proj_xyz[3 * i]),
                float(proj_xyz[3 * i + 1]),
                float(proj_xyz[3 * i + 2]),
            )
            if (
                abs(px - xyz[0]) < point_tolerance * 100
                and abs(py - xyz[1]) < point_tolerance * 100
                and abs(pz - xyz[2]) < point_tolerance * 100
            ):
                on_edge_flat.extend(xyz)
                on_edge_tags.append((stag, xyz))
        if not on_edge_tags:
            return result
        try:
            params = gmsh.model.getParametrization(1, edge_tag, on_edge_flat)
        except Exception:
            return result
        for i, (stag, xyz) in enumerate(on_edge_tags):
            result.append((stag, xyz, float(params[i])))
        return result

    edge_chains: dict[int, list[int]] = {}
    for edge_tag in occ_edges:
        # Endpoints (corners on this edge).
        edge_endpoints = gmsh.model.getBoundary(
            [(1, edge_tag)], oriented=False, recursive=False
        )
        endpoint_pt_tags = [int(t) for d, t in edge_endpoints if d == 0]
        endpoint_node_tags = [occ_corner_node_tag.get(p) for p in endpoint_pt_tags]
        if any(n is None for n in endpoint_node_tags):
            edge_chains[edge_tag] = []
            continue
        endpoint_xyz = []
        for p in endpoint_pt_tags:
            bb = gmsh.model.getBoundingBox(0, p)
            endpoint_xyz.extend([bb[0], bb[1], bb[2]])
        try:
            endpoint_t = gmsh.model.getParametrization(1, edge_tag, endpoint_xyz)
        except Exception:
            endpoint_t = [0.0, 1.0]
        if endpoint_t[0] < endpoint_t[1]:
            start_node, end_node = endpoint_node_tags
        else:
            end_node, start_node = endpoint_node_tags

        interior = _seam_nodes_on_edge(edge_tag)
        # Sort by parametric coordinate.
        interior.sort(key=lambda r: r[2])

        # Dedupe interior by xyz key — mirror seam nodes (e.g. on the front
        # and back of a thin slab projected to the same midplane) collapse to
        # a single OCC tag. Use first occurrence per key.
        seen_keys: set = set()
        unique_interior: list = []
        for stag, xyz, t in interior:
            k = _key(xyz)
            if k in seen_keys:
                # Map this seam node to the already-allocated tag for k.
                if k in xyz_to_occ_node_tag:
                    seam_to_occ_node_tag[stag] = xyz_to_occ_node_tag[k]
                continue
            seen_keys.add(k)
            unique_interior.append((stag, xyz, t, k))

        nodes_to_add_tags = []
        nodes_to_add_xyz = []
        nodes_to_add_params = []
        ordered_tags = []
        for stag, xyz, t, k in unique_interior:
            if k in xyz_to_occ_node_tag:
                # Already seeded (e.g. coincident with a corner pt).
                new_tag = xyz_to_occ_node_tag[k]
                seam_to_occ_node_tag[stag] = new_tag
                ordered_tags.append(new_tag)
            else:
                new_tag = _alloc()
                xyz_to_occ_node_tag[k] = new_tag
                seam_to_occ_node_tag[stag] = new_tag
                ordered_tags.append(new_tag)
                nodes_to_add_tags.append(new_tag)
                nodes_to_add_xyz.extend(xyz)
                nodes_to_add_params.append(t)

        if nodes_to_add_tags:
            # Re-evaluate xyz from the parametric coord on the OCC curve so the
            # seeded node lies exactly on the curve (the seam's xyz may be off
            # by the OCC-vs-source-kernel offset, which would otherwise leave
            # the line elements slightly off-curve and confuse 1D mesh
            # recovery during neighbouring-surface meshing).
            true_xyz = []
            for t_val in nodes_to_add_params:
                pt = gmsh.model.getValue(1, edge_tag, [t_val])
                true_xyz.extend([float(pt[0]), float(pt[1]), float(pt[2])])
            gmsh.model.mesh.addNodes(
                1,
                edge_tag,
                nodes_to_add_tags,
                true_xyz,
                nodes_to_add_params,
            )

        # Build line element chain. Dedupe consecutive duplicates (corners
        # may re-appear if interior xyz coincides with start/end).
        chain = [start_node, *ordered_tags, end_node]
        deduped_chain = [chain[0]]
        for c in chain[1:]:
            if c != deduped_chain[-1]:
                deduped_chain.append(c)
        edge_chains[edge_tag] = deduped_chain
        line_node_tags = []
        for i in range(len(deduped_chain) - 1):
            line_node_tags.extend([deduped_chain[i], deduped_chain[i + 1]])
        n_segments = len(deduped_chain) - 1
        if n_segments > 0:
            line_elem_tags = [_alloc() for _ in range(n_segments)]
            gmsh.model.mesh.addElementsByType(
                edge_tag, 1, line_elem_tags, line_node_tags
            )

    # ---- Step 5: interior face nodes + triangles. ----
    interior_seam = [
        (stag, xyz)
        for stag, xyz in seam_node_tag_to_xyz.items()
        if stag not in seam_to_occ_node_tag
    ]

    if interior_seam:
        # Dedupe interior nodes by xyz so mirrors share one tag.
        flat_xyz_unique = []
        unique_keys = []
        unique_tags = []
        for stag, xyz in interior_seam:
            k = _key(xyz)
            if k in xyz_to_occ_node_tag:
                new_tag = xyz_to_occ_node_tag[k]
                seam_to_occ_node_tag[stag] = new_tag
                continue
            new_tag = _alloc()
            xyz_to_occ_node_tag[k] = new_tag
            seam_to_occ_node_tag[stag] = new_tag
            unique_tags.append(new_tag)
            unique_keys.append(k)
            flat_xyz_unique.extend(xyz)

        if unique_tags:
            params_uv = gmsh.model.getParametrization(2, occ_face, flat_xyz_unique)
            # Re-evaluate xyz from (u, v) so the seeded node lies exactly on
            # the OCC face (see note for edges above).
            true_xyz = []
            for i in range(len(unique_tags)):
                u, v = float(params_uv[2 * i]), float(params_uv[2 * i + 1])
                pt = gmsh.model.getValue(2, occ_face, [u, v])
                true_xyz.extend([float(pt[0]), float(pt[1]), float(pt[2])])
            gmsh.model.mesh.addNodes(2, occ_face, unique_tags, true_xyz, params_uv)

    # ---- Triangles. Filter degenerates from mirror collapse. ----
    occ_triangle_node_tags = []
    n_triangles_emitted = 0
    for tri in seam_triangles:
        try:
            mapped = [seam_to_occ_node_tag[int(s)] for s in tri]
        except KeyError:
            continue
        # Skip degenerate triangles (any two equal node tags).
        if mapped[0] == mapped[1] or mapped[1] == mapped[2] or mapped[0] == mapped[2]:
            continue
        occ_triangle_node_tags.extend(mapped)
        n_triangles_emitted += 1
    if n_triangles_emitted > 0:
        tri_elem_tags = [_alloc() for _ in range(n_triangles_emitted)]
        gmsh.model.mesh.addElementsByType(
            occ_face, 2, tri_elem_tags, occ_triangle_node_tags
        )

    # ---- Step 6: register the seam name as a 2-D physical group on the OCC
    # face, so it survives glue and shows up in the output.
    #
    # Subtlety: gmsh allows an entity to be in MULTIPLE physical groups, but
    # meshio's MSH4 reader assigns only the FIRST physical group's tag to
    # each cell. Since the auto-created boundary group (e.g. ``A___None``) was
    # registered during CAD load — BEFORE this helper runs — naively adding a
    # new ``_seam___...`` group would leave face elements tagged with the
    # boundary group's id, not the seam id. To ensure seam-tagged elements
    # appear under the seam name in the output, we REMOVE this face from
    # every existing physical group it currently belongs to, then add it to a
    # fresh seam group. This means the seam face is no longer part of the
    # generic boundary group — which is the correct semantic anyway, since
    # seam faces are interface faces, not external boundaries.
    existing_groups_for_face = list(gmsh.model.getPhysicalGroupsForEntity(2, occ_face))
    for old_ptag in existing_groups_for_face:
        try:
            old_ents = list(gmsh.model.getEntitiesForPhysicalGroup(2, int(old_ptag)))
        except Exception:  # noqa: S112  pre-existing group lookup is best-effort
            continue
        try:
            old_name = gmsh.model.getPhysicalName(2, int(old_ptag))
        except Exception:
            old_name = ""
        # Remove the existing group, then re-add it with the seam face
        # excluded (preserving its name and other entities).
        gmsh.model.removePhysicalGroups([(2, int(old_ptag))])
        kept = [int(e) for e in old_ents if int(e) != int(occ_face)]
        if kept:
            new_ptag = gmsh.model.addPhysicalGroup(2, kept)
            if old_name:
                gmsh.model.setPhysicalName(2, new_ptag, old_name)

    # Now add the seam face to its seam-named group (reusing an existing
    # group with the same name if one happens to be present).
    existing_seam_ptag = None
    for d, ptag in gmsh.model.getPhysicalGroups(2):
        try:
            n = gmsh.model.getPhysicalName(d, ptag)
        except Exception:
            n = ""
        if n == seam_name:
            existing_seam_ptag = ptag
            break
    if existing_seam_ptag is not None:
        # Augment the existing group with this face.
        existing_ents = list(
            gmsh.model.getEntitiesForPhysicalGroup(2, int(existing_seam_ptag))
        )
        existing_ents = [int(e) for e in existing_ents]
        if int(occ_face) not in existing_ents:
            existing_ents.append(int(occ_face))
        gmsh.model.removePhysicalGroups([(2, int(existing_seam_ptag))])
        new_ptag = gmsh.model.addPhysicalGroup(2, existing_ents)
        gmsh.model.setPhysicalName(2, new_ptag, seam_name)
    else:
        new_ptag = gmsh.model.addPhysicalGroup(2, [int(occ_face)])
        gmsh.model.setPhysicalName(2, new_ptag, seam_name)


class Mesh:
    """Mesh class for generating meshes from cad models."""

    def __init__(
        self,
        n_threads: int = cpu_count(),
        filename: str = "temp",
        model: ModelManager | None = None,
        point_tolerance: float | None = None,
    ):
        """Initialize mesh generator.

        Args:
            n_threads: Number of threads for processing
            filename: Base filename for the model
            model: Optional Model instance to use (creates new if None)
            point_tolerance: Optional point tolerance for the model

        """
        # Use provided model or create new one
        if model is None:
            self.model_manager = ModelManager(
                n_threads=n_threads,
                filename=filename,
                point_tolerance=point_tolerance,
            )
            self._owns_model = True
        else:
            self.model_manager = model
            self._owns_model = False

    def _initialize_model(self, input_file: Path | None = None) -> None:
        """Initialize GMSH model and optionally load .xao file."""
        self.model_manager.ensure_initialized("temp")

        if input_file is not None:
            input_file = Path(input_file)
            gmsh.merge(str(input_file.with_suffix(".xao")))

    def _initialize_mesh_settings(
        self,
        verbosity: int,
        default_characteristic_length: float,
        global_2D_algorithm: int | Sequence[int],
        global_3D_algorithm: int | Sequence[int],
        gmsh_version: float | None,
        mesh_element_order: int = 1,
    ) -> None:
        """Initialize basic mesh settings.

        If either algorithm argument is a sequence, the first value is applied
        here and process_mesh handles fallback to later ones.
        """
        gmsh.option.setNumber("General.Terminal", verbosity)
        gmsh.option.setNumber(
            "Mesh.CharacteristicLengthMax", default_characteristic_length
        )
        gmsh.option.setNumber("Mesh.Algorithm", _normalize_algo(global_2D_algorithm)[0])
        gmsh.option.setNumber(
            "Mesh.Algorithm3D", _normalize_algo(global_3D_algorithm)[0]
        )
        gmsh.option.setNumber("Mesh.ElementOrder", mesh_element_order)
        if gmsh_version is not None:
            gmsh.option.setNumber("Mesh.MshFileVersion", gmsh_version)
        self.model_manager.sync_model()

    def _apply_periodic_boundaries(
        self, periodic_entities: list[tuple[str, str]]
    ) -> None:
        """Apply periodic boundary conditions."""
        mapping = {
            self.model_manager.model.getPhysicalName(dimtag[0], dimtag[1]): dimtag
            for dimtag in self.model_manager.model.getPhysicalGroups()
        }

        for label1, label2 in periodic_entities:
            if label1 not in mapping or label2 not in mapping:
                continue

            self._set_periodic_pair(mapping, label1, label2)

    def _set_periodic_pair(self, mapping: dict, label1: str, label2: str) -> None:
        """Set up periodic boundary pair."""
        tags1 = self.model_manager.model.getEntitiesForPhysicalGroup(*mapping[label1])
        tags2 = self.model_manager.model.getEntitiesForPhysicalGroup(*mapping[label2])

        vector1 = self.model_manager.model.occ.getCenterOfMass(
            mapping[label1][0], tags1[0]
        )
        vector2 = self.model_manager.model.occ.getCenterOfMass(
            mapping[label1][0], tags2[0]
        )
        vector = np.subtract(vector1, vector2)

        self.model_manager.model.mesh.setPeriodic(
            mapping[label1][0],
            tags1,
            tags2,
            [1, 0, 0, vector[0], 0, 1, 0, vector[1], 0, 0, 1, vector[2], 0, 0, 0, 1],
        )

    def _apply_mesh_refinement(
        self,
        background_remeshing_file: Path | None | None,
        boundary_delimiter: str,
        resolution_specs: dict,
        interface_delimiter: str = "___",
        _global_physical_names: list[str] | None = None,
    ) -> None:
        """Apply mesh refinement settings.

        TODO: enable simultaneous background mesh and entity-based refinement
        """
        if background_remeshing_file is None:
            self._apply_entity_refinement(
                boundary_delimiter,
                resolution_specs,
                interface_delimiter,
                _global_physical_names=_global_physical_names,
            )
        else:
            self._apply_background_refinement()

    def get_top_physical_names(self) -> list[str]:
        """Get all physical names of dimension dim from the GMSH model.

        Returns:
            List of physical names as strings

        """
        return self.model_manager.get_top_physical_names()

    def get_all_physical_names(self) -> list[str]:
        """Get all physical names from the GMSH model.

        Returns:
            List of physical names as strings

        """
        return self.model_manager.get_physical_names()

    def get_physical_dimtags(self, physical_name: str) -> list[tuple[int, int]]:
        """Get the dimtags associated with a physical group name.

        Args:
            physical_name: Name of the physical group

        Returns:
            List of (dim, tag) tuples for entities in the physical group

        """
        return self.model_manager.get_physical_dimtags(physical_name)

    def _restore_structured_sweeps(self, blueprint: dict) -> None:
        """Analyze structured sweeps."""
        import logging

        logger = logging.getLogger(__name__)

        top_names = self.get_top_physical_names()
        for p_name in top_names:
            if p_name in blueprint and blueprint[p_name].get("mesh_structured", False):
                logger.warning(
                    f"Physical group '{p_name}' requested mesh_structured=True. "
                    "Note: Native OpenCASCADE structured sweeping via 'removeAllDuplicates' "
                    "cannot guarantee conformality in hybrid Gmsh meshes without stripping ExtrudeParams. "
                    "Proceeding with unstructured or recombined fallback."
                )

    def _recover_labels_from_cad(
        self,
        resolution_specs: dict,
        interface_delimiter: str = "___",
        boundary_delimiter: str = "None",
    ) -> tuple[list, dict]:
        """Recover labeled entities from loaded CAD model.

        Args:
            resolution_specs: Dictionary mapping physical names to resolution specifications
            blueprint: mapping between entity and extrusion type
            interface_delimiter: String used to separate names in an interface
            boundary_delimiter: String used to identify boundary entities

        Returns:
            Tuple of (final_entity_list, final_entity_dict)
        """
        final_entity_list = []
        final_entity_dict = {}

        # We address entities by "named" physicals (not default):
        top_physical_names = self.get_top_physical_names()
        all_physical_names = self.get_all_physical_names()

        # Collect all base names from interfaces to handle removed entities (voids)
        base_names = set(all_physical_names)
        for other_p_name in all_physical_names:
            parts = other_p_name.split(interface_delimiter)
            if len(parts) == 2:
                if parts[0]:
                    base_names.add(parts[0])
                if parts[1] and parts[1] != boundary_delimiter:
                    base_names.add(parts[1])

        for index, physical_name in enumerate(sorted(base_names)):
            resolutions = resolution_specs.get(physical_name, [])

            if not resolutions and physical_name not in top_physical_names:
                continue

            entities = _MeshEntity(
                index=index,
                physical_name=physical_name,
                model=self.model_manager.model,
                dimtags=self.get_physical_dimtags(physical_name=physical_name),
                resolutions=resolutions,
            )
            entities.update_boundaries()

            # Recover interfaces and boundaries from physical groups
            # We look for groups named "A___B" or "B___A" where A is physical_name
            for other_p_name in all_physical_names:
                parts = other_p_name.split(interface_delimiter)
                if len(parts) == 2:
                    if parts[0] == physical_name:
                        suffix = parts[1]
                        dimtags = self.get_physical_dimtags(other_p_name)
                        if dimtags:
                            for i_dim, i_tag in dimtags:
                                if i_dim < entities.dim or entities.dim == -1:
                                    if entities.dim == -1:
                                        entities._explicit_dim = max(
                                            entities._explicit_dim or 0, i_dim + 1
                                        )

                                    if suffix == boundary_delimiter:
                                        entities.mesh_edge_name_interfaces.append(i_tag)
                                    else:
                                        entities.interfaces.append(i_tag)

                                    # Always treat interfaces as boundaries for refinement
                                    entities.boundaries.append(i_tag)

                    elif parts[1] == physical_name:
                        dimtags = self.get_physical_dimtags(other_p_name)
                        if dimtags:
                            for i_dim, i_tag in dimtags:
                                if i_dim < entities.dim or entities.dim == -1:
                                    if entities.dim == -1:
                                        entities._explicit_dim = max(
                                            entities.dim, i_dim + 1
                                        )

                                    entities.interfaces.append(i_tag)

                                    # Always treat interfaces as boundaries for refinement
                                    entities.boundaries.append(i_tag)

            final_entity_list.append(entities)
            final_entity_dict[physical_name] = entities

        return final_entity_list, final_entity_dict

    def _apply_entity_refinement(
        self,
        boundary_delimiter: str,
        resolution_specs: dict,
        interface_delimiter: str = "___",
        _global_physical_names: list[str] | None = None,
    ) -> None:
        """Apply mesh refinement based on entity information.

        Args:
            boundary_delimiter: String used to identify boundary entities
            resolution_specs: Resolution specifications
            interface_delimiter: String used to separate names in an interface
            _global_physical_names: Optional list of physical names that exist
                across the wider distributed model. ResolutionSpec name refs
                (``restrict_to`` / ``sharing`` / ``not_sharing``) that match a
                name in this set but not in the local entity dict are silently
                no-op'd with a warning, instead of being treated as a typo.

        """
        from collections import defaultdict

        # Recover labeled entities from loaded CAD model
        final_entity_list, final_entity_dict = self._recover_labels_from_cad(
            resolution_specs,
            interface_delimiter=interface_delimiter,
            boundary_delimiter=boundary_delimiter,
        )

        # Build reverse indices for performance
        tag_to_entity_names = defaultdict(set)
        for name, entity in final_entity_dict.items():
            # Include tags for all dimensions that this entity covers
            for d in range(entity.dim + 1):
                tags = entity.filter_tags_by_target_dimension(d)
                for tag in tags:
                    tag_to_entity_names[(d, tag)].add(name)

        # Collect all refinement fields
        refinement_field_indices = []

        # Handle Global Specs (key is None)
        if None in resolution_specs:
            for spec in resolution_specs[None]:
                # Apply globally (empty dict means no mass filtering, restrict_to_tags=None means global)
                field_index = spec.apply(
                    self.model_manager.model, {}, restrict_to_tags=None
                )
                refinement_field_indices.append(field_index)

        # Collect constant fields for batching
        constant_collector = defaultdict(lambda: defaultdict(list))

        for entity in final_entity_list:
            refinement_field_indices.extend(
                entity.add_refinement_fields_to_model(
                    final_entity_dict,
                    boundary_delimiter,
                    constant_collector=constant_collector,
                    tag_to_entity_names=tag_to_entity_names,
                    global_physical_names=_global_physical_names,
                )
            )

        # Process constant fields in batches
        for resolution, entity_types in constant_collector.items():
            matheval_field_index = self.model_manager.model.mesh.field.add("MathEval")
            self.model_manager.model.mesh.field.setString(
                matheval_field_index, "F", f"{resolution}"
            )

            restrict_field_index = self.model_manager.model.mesh.field.add("Restrict")
            self.model_manager.model.mesh.field.setNumber(
                restrict_field_index, "InField", matheval_field_index
            )

            for entity_str, tags in entity_types.items():
                self.model_manager.model.mesh.field.setNumbers(
                    restrict_field_index,
                    entity_str,
                    list(set(tags)),
                )
            refinement_field_indices.append(restrict_field_index)

        # If we have refinement fields, create a minimum field
        if refinement_field_indices:
            # Use the smallest element size overall
            min_field_index = self.model_manager.model.mesh.field.add("Min")
            self.model_manager.model.mesh.field.setNumbers(
                min_field_index, "FieldsList", refinement_field_indices
            )
            self.model_manager.model.mesh.field.setAsBackgroundMesh(min_field_index)

        # Turn off default meshing options
        # NOTE: this disable is unconditional even when refinement_field_indices
        # is empty — preserved for backward compatibility. The distributed
        # _interface_constraints flow re-enables MeshSizeFromPoints and
        # MeshSizeExtendFromBoundary in process_mesh because seeded OCC faces
        # need them to be honored. If you ever move these inside the
        # `if refinement_field_indices:` block, also remove the override at
        # process_mesh's _interface_constraints branch.
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)

    def _apply_background_refinement(self) -> None:
        """Apply mesh refinement based on background mesh."""
        # Create background field from post-processing view
        bg_field = self.model_manager.model.mesh.field.add("PostView")
        self.model_manager.model.mesh.field.setNumber(bg_field, "ViewIndex", 0)
        gmsh.model.mesh.field.setAsBackgroundMesh(bg_field)

        # Turn off default meshing options
        # NOTE: see the matching note in _apply_entity_refinement above —
        # the distributed _interface_constraints flow re-enables these
        # options in process_mesh because seeded OCC faces need them.
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)

    def process_mesh(
        self,
        dim: int,
        global_3D_algorithm: int,
        global_scaling: float,
        verbosity: int,
        optimization_flags: tuple[tuple[str, int]] | None,
        _interface_constraints: list | None = None,
        point_tolerance: float = 1e-6,
    ) -> meshio.Mesh:
        """Generate mesh and return meshio object (no file I/O).

        The caller is expected to have already set Mesh.Algorithm /
        Mesh.Algorithm3D on the gmsh option state (see ``_initialize_mesh_settings``).
        Retry-on-failure logic lives in ``process_geometry`` because a raised
        ``mesh.generate()`` leaves the gmsh runtime in a "busy" state that only
        a full finalize+reinit will clear.

        If ``_interface_constraints`` is non-empty, each path is loaded and
        used to seed a matching OCC face's mesh via the parametric
        ``addNodes`` recipe. ``Mesh.MeshOnlyEmpty=1`` is set so the final
        ``generate(dim)`` call leaves seeded entities untouched and only
        meshes the unseeded rest. NOTE: we deliberately call ``generate(dim)``
        ONCE — sequential ``generate(N)`` calls for ``N < dim`` wipe seeded
        higher-dim elements (validated by the spike test).
        """
        gmsh.option.setNumber("Mesh.ScalingFactor", global_scaling)
        gmsh.option.setNumber("Mesh.AngleToleranceFacetOverlap", 1e-5)

        if global_3D_algorithm == 1 and verbosity:
            gmsh.logger.start()

        if _interface_constraints:
            # Set MeshOnlyEmpty BEFORE seeding so the meshing pass treats every
            # newly-seeded entity (with line/triangle elements present) as
            # already-meshed and skips remeshing it.
            gmsh.option.setNumber("Mesh.MeshOnlyEmpty", 1)
            # Re-enable point/boundary sizing heuristics: ``_apply_entity_refinement``
            # unconditionally disables them (even when no resolution_specs are
            # set), but with them disabled gmsh fails to surface-mesh OCC faces
            # neighbouring our seeded face — it ends up RE-meshing the seeded
            # face at a much coarser resolution despite ``MeshOnlyEmpty=1``,
            # silently dropping seeded triangles. Re-enabling MeshSizeFromPoints
            # gives gmsh enough sizing context to leave seeded faces intact.
            gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 1)
            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
            for path in _interface_constraints:
                _seed_occ_face_from_seam(Path(path), point_tolerance=point_tolerance)

        self.model_manager.model.mesh.generate(dim)

        if optimization_flags:
            for optimization_flag, niter in optimization_flags:
                self.model_manager.model.mesh.optimize(optimization_flag, niter=niter)

        # Return mesh object without writing to file
        with contextlib.redirect_stdout(
            None
        ), tempfile.TemporaryDirectory() as tmpdirname:
            temp_mesh_path = f"{tmpdirname}/mesh.msh"
            gmsh.write(temp_mesh_path)
            return meshio.read(temp_mesh_path)

    def save_to_file(self, output_file: Path) -> None:
        """Save current mesh to file.

        Args:
            output_file: Output mesh file path

        """
        self.model_manager.save_to_mesh(output_file)

    def to_msh(self, output_file: Path, format: str = "msh") -> None:
        """Save current mesh to .msh file.

        Args:
            output_file: Output file path (will be suffixed with .format)
            format: File format to use in gmsh

        """
        self.model_manager.save_to_mesh(output_file, format)

    def to_meshio(self) -> meshio.Mesh:
        """Convert current mesh to meshio.Mesh object.

        Returns:
            meshio.Mesh: Current mesh as meshio object

        """
        with contextlib.redirect_stdout(
            None
        ), tempfile.TemporaryDirectory() as tmpdirname:
            temp_mesh_path = f"{tmpdirname}/mesh.msh"
            gmsh.write(temp_mesh_path)
            return meshio.read(temp_mesh_path)

    def load_xao_file(self, input_file: Path) -> None:
        """Load CAD geometry from .xao file.

        Args:
            input_file: Input .xao file path

        """
        self.model_manager.load_from_xao(input_file)

    def process_geometry(
        self,
        dim: int,
        default_characteristic_length: float,
        background_remeshing_file: Path | None = None,
        global_scaling: float = 1.0,
        global_2D_algorithm: int | Sequence[int] = 6,
        global_3D_algorithm: int | Sequence[int] = 1,
        mesh_element_order: int = 1,
        verbosity: int | None = 0,
        periodic_entities: list[tuple[str, str]] | None = None,  # noqa: ARG002
        optimization_flags: tuple[tuple[str, int]] | None = None,
        boundary_delimiter: str = "None",
        resolution_specs: dict = (),
        gmsh_version: float | None = None,
        interface_delimiter: str = "___",
        _global_physical_names: list[str] | None = None,
        _interface_constraints: list | None = None,
        point_tolerance: float = 1e-6,
    ) -> meshio.Mesh:
        """Process loaded geometry into mesh (no file I/O).

        Args:
            dim: Dimension of mesh to generate
            default_characteristic_length: Default mesh size
            background_remeshing_file: Optional background mesh file for refinement
            global_scaling: Global scaling factor
            global_2D_algorithm: GMSH 2D meshing algorithm, or a sequence of
                algorithms to try in order if earlier attempts fail.
            global_3D_algorithm: GMSH 3D meshing algorithm, or a sequence of
                algorithms to try in order if earlier attempts fail.
            mesh_element_order: Element order
            verbosity: GMSH verbosity level
            periodic_entities: List of periodic boundary pairs
            optimization_flags: Mesh optimization flags
            boundary_delimiter: Delimiter for boundary names
            resolution_specs: Mesh resolution specifications
            gmsh_version: GMSH version
            blueprint: mapping between entity and extrusion type
            interface_delimiter: String used to separate names in an interface
            _global_physical_names: Optional list of physical names known across
                the wider distributed model (see :func:`generate_mesh`).
            _interface_constraints: Optional list of seam ``.msh`` paths for
                phase-2 OCC face seeding (see :func:`_seed_occ_face_from_seam`).
            point_tolerance: Coordinate tolerance forwarded to
                :func:`_seed_occ_face_from_seam` for OCC entity matching.

        Returns:
            meshio.Mesh: Generated mesh object

        """
        self._initialize_model()

        attempts = _pair_algos(
            _normalize_algo(global_2D_algorithm),
            _normalize_algo(global_3D_algorithm),
        )

        def _run_once(algo2d: int, algo3d: int) -> meshio.Mesh:
            self._initialize_mesh_settings(
                verbosity=verbosity,
                default_characteristic_length=default_characteristic_length,
                global_2D_algorithm=algo2d,
                global_3D_algorithm=algo3d,
                gmsh_version=gmsh_version,
                mesh_element_order=mesh_element_order,
            )
            self._apply_mesh_refinement(
                background_remeshing_file=background_remeshing_file,
                boundary_delimiter=boundary_delimiter,
                resolution_specs=resolution_specs,
                interface_delimiter=interface_delimiter,
                _global_physical_names=_global_physical_names,
            )
            return self.process_mesh(
                dim=dim,
                global_3D_algorithm=algo3d,
                global_scaling=global_scaling,
                verbosity=verbosity,
                optimization_flags=optimization_flags,
                _interface_constraints=_interface_constraints,
                point_tolerance=point_tolerance,
            )

        if len(attempts) == 1:
            algo2d, algo3d = attempts[0]
            return _run_once(algo2d, algo3d)

        # Multi-attempt: persist CAD before the first try so we can restore
        # after a failed generate() leaves gmsh in an unrecoverable "busy"
        # state (neither mesh.clear() nor re-setting options releases it).
        with tempfile.TemporaryDirectory() as tmp:
            cad_checkpoint = Path(tmp) / "cad_checkpoint.xao"
            self.model_manager.save_to_xao(cad_checkpoint)

            for attempt_idx, (algo2d, algo3d) in enumerate(attempts):
                try:
                    return _run_once(algo2d, algo3d)
                except Exception as exc:
                    if attempt_idx == len(attempts) - 1:
                        raise
                    print(
                        f"mesh attempt {attempt_idx + 1}/{len(attempts)} "
                        f"(2D={algo2d}, 3D={algo3d}) failed: {exc}. "
                        f"Retrying with next algorithm.",
                        flush=True,
                    )
                    # Full gmsh reset: finalize + reinit + reload CAD.
                    self.model_manager.finalize()
                    self.model_manager.load_from_xao(cad_checkpoint)

        raise RuntimeError("unreachable: retry loop exited without returning")


def mesh(
    dim: int,
    default_characteristic_length: float,
    input_file: Path | None = None,
    output_file: Path | None = None,
    resolution_specs: dict | None = None,
    background_remeshing_file: Path | None = None,
    global_scaling: float = 1.0,
    global_2D_algorithm: int | Sequence[int] = 6,
    global_3D_algorithm: int | Sequence[int] = 1,
    mesh_element_order: int = 1,
    verbosity: int | None = 0,
    periodic_entities: list[tuple[str, str]] | None = None,
    optimization_flags: tuple[tuple[str, int]] | None = None,
    boundary_delimiter: str = "None",
    n_threads: int = cpu_count(),
    filename: str = "temp",
    model: ModelManager | None = None,
    point_tolerance: float | None = None,
    gmsh_version: float | None = None,
    interface_delimiter: str = "___",
    _global_physical_names: list[str] | None = None,
    _emit_only_seam_surfaces: bool = False,
    _interface_constraints: list | None = None,
) -> meshio.Mesh | None:
    """Utility function that wraps the Mesh class for easier usage.

    Args:
        dim: Dimension of mesh to generate
        default_characteristic_length: Default mesh size
        input_file: Path to input .xao file
        output_file: Path for output mesh file
        resolution_specs: Mesh resolution specifications
        background_remeshing_file: Optional background mesh file for refinement
        global_scaling: Global scaling factor
        global_2D_algorithm: GMSH 2D meshing algorithm, or a sequence of
            algorithms tried in order with fallback on failure.
        global_3D_algorithm: GMSH 3D meshing algorithm, or a sequence of
            algorithms tried in order with fallback on failure.
        mesh_element_order: Element order
        verbosity: GMSH verbosity level
        periodic_entities: List of periodic boundary pairs
        optimization_flags: Mesh optimization flags
        boundary_delimiter: Delimiter for boundary names
        n_threads: Number of threads to use
        filename: Temporary filename for GMSH model
        model: Optional Model instance to use (creates new if None)
        gmsh_version: GMSH MSH file version (e.g. 2.2 or 4.1)
        point_tolerance: used to set GMSH global variables. Should be similar to used in CAD.
        interface_delimiter: String used to separate names in an interface
        _global_physical_names: Optional list of physical names known across the
            wider distributed model (see :func:`generate_mesh`).
        _emit_only_seam_surfaces: If True, post-process ``output_file`` so that
            only physical groups whose name starts with ``_seam___`` survive.
            Used by phase-1 workers in the distributed-meshing pipeline to emit
            just the seam-surface mesh; everything else is treated as scratch.
            No effect when ``output_file`` is None. Implies a rewrite to gmsh22
            format (meshio's gmsh4 writer drops field_data on round-trip).
        _interface_constraints: Optional list of paths to seam ``.msh`` files
            (typically produced by phase-1 workers). For each path, the matching
            OCC face's mesh is seeded from the seam triangulation via the
            parametric ``addNodes`` recipe (see :func:`_seed_occ_face_from_seam`),
            and ``Mesh.MeshOnlyEmpty=1`` is set so the rest of the mesh is built
            around the seeded faces. Used by phase-2 workers in the distributed
            pipeline to enforce conformality across subdomain boundaries.

    Returns:
        Optional[meshio.Mesh]: Generated mesh object

    """
    mesh_generator = Mesh(
        n_threads=n_threads,
        filename=filename,
        model=model,
        point_tolerance=point_tolerance,
    )

    if resolution_specs is None:
        resolution_specs = {}

    try:
        # Load geometry from file if provided
        if input_file is not None:
            mesh_generator.load_xao_file(input_file)

        # Process geometry into mesh
        mesh_obj = mesh_generator.process_geometry(
            dim=dim,
            background_remeshing_file=background_remeshing_file,
            default_characteristic_length=default_characteristic_length,
            global_scaling=global_scaling,
            global_2D_algorithm=global_2D_algorithm,
            global_3D_algorithm=global_3D_algorithm,
            mesh_element_order=mesh_element_order,
            verbosity=verbosity,
            periodic_entities=periodic_entities,
            optimization_flags=optimization_flags,
            boundary_delimiter=boundary_delimiter,
            resolution_specs=resolution_specs,
            gmsh_version=gmsh_version,
            interface_delimiter=interface_delimiter,
            _global_physical_names=_global_physical_names,
            _interface_constraints=_interface_constraints,
            point_tolerance=point_tolerance if point_tolerance is not None else 1e-6,
        )

        # Save to file if output file provided
        if output_file is not None:
            mesh_generator.save_to_file(output_file)
            if _emit_only_seam_surfaces:
                _filter_msh_to_seam_groups(output_file)
    finally:
        # Finalize if we created the model -- even on failure, so gmsh
        # state doesn't leak into subsequent test runs / callers.
        if model is None:
            mesh_generator.model_manager.finalize()

    return mesh_obj
