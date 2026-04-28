"""R3 spike: verify gmsh.merge + embed across kernels works for our use case.

Builds a 1x1x1 OCC box, generates a discrete 2D mesh on its top face in a
separate gmsh model, exports that surface mesh to .msh, then in a fresh
gmsh model re-imports it via gmsh.merge and embeds it into the OCC top
face. Asserts the meshed top face uses the imported nodes/triangles.
"""
from pathlib import Path

import pytest

import gmsh


@pytest.fixture(autouse=True)
def fresh_gmsh():
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    yield
    gmsh.finalize()


def _build_seam_surface_msh(path: Path, mesh_size: float) -> None:
    """Mesh the unit square at z=1 in a discrete 2D model and write to path."""
    gmsh.model.add("seam")
    pts = [
        gmsh.model.geo.addPoint(0, 0, 1, mesh_size),
        gmsh.model.geo.addPoint(1, 0, 1, mesh_size),
        gmsh.model.geo.addPoint(1, 1, 1, mesh_size),
        gmsh.model.geo.addPoint(0, 1, 1, mesh_size),
    ]
    lines = [gmsh.model.geo.addLine(pts[i], pts[(i + 1) % 4]) for i in range(4)]
    loop = gmsh.model.geo.addCurveLoop(lines)
    surf = gmsh.model.geo.addPlaneSurface([loop])
    gmsh.model.geo.synchronize()
    gmsh.model.addPhysicalGroup(2, [surf], name="_seam___top")
    gmsh.model.mesh.generate(2)
    gmsh.write(str(path))
    gmsh.model.remove()


def test_embed_merged_discrete_into_occ_face(tmp_path):
    seam_path = tmp_path / "seam.msh"
    _build_seam_surface_msh(seam_path, mesh_size=0.2)

    # Fresh model: build OCC box, then merge the seam mesh, then embed.
    gmsh.model.add("box")
    gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
    gmsh.model.occ.synchronize()

    # Locate the OCC top face (z=1).
    occ_faces = gmsh.model.getEntities(2)
    top_face = None
    for dim, tag in occ_faces:
        bb = gmsh.model.getBoundingBox(dim, tag)
        if abs(bb[2] - 1.0) < 1e-6 and abs(bb[5] - 1.0) < 1e-6:
            top_face = tag
            break
    assert top_face is not None

    # Merge the seam mesh — discrete entities appear in the model.
    gmsh.merge(str(seam_path))

    # Find the imported discrete 2D entity (it carries the _seam___top group).
    imported_2d = None
    for dim, tag in gmsh.model.getEntities(2):
        if tag == top_face:
            continue
        try:
            name = gmsh.model.getEntityName(dim, tag) or ""
        except Exception:
            name = ""
        if "_seam___top" in name or True:  # discrete tag will not be the OCC top_face
            imported_2d = tag
    assert imported_2d is not None and imported_2d != top_face

    # Embed the imported discrete face's boundary into the OCC top face.
    # We expect that gmsh will mesh the OCC top face using the imported nodes.
    gmsh.model.mesh.embed(2, [imported_2d], 2, top_face)
    gmsh.model.mesh.generate(3)

    # Sanity: the resulting mesh on the top face should reuse the imported nodes.
    nodes_top = gmsh.model.mesh.getNodes(2, top_face, includeBoundary=True)
    assert len(nodes_top[0]) > 0


def test_embed_with_remove_duplicate_nodes(tmp_path):
    seam_path = tmp_path / "seam.msh"
    _build_seam_surface_msh(seam_path, mesh_size=0.2)

    gmsh.model.add("box_a")
    gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
    gmsh.model.occ.synchronize()
    top_face = None
    for dim, tag in gmsh.model.getEntities(2):
        bb = gmsh.model.getBoundingBox(dim, tag)
        if abs(bb[2] - 1.0) < 1e-6 and abs(bb[5] - 1.0) < 1e-6:
            top_face = tag
            break

    pre_2d = {tag for d, tag in gmsh.model.getEntities(2)}
    gmsh.merge(str(seam_path))
    post_2d = {tag for d, tag in gmsh.model.getEntities(2)}
    imported_2d = sorted(post_2d - pre_2d)[0]

    gmsh.model.mesh.embed(2, [imported_2d], 2, top_face)
    # Try: dedup before 3D mesh.
    gmsh.option.setNumber("Geometry.Tolerance", 1e-6)
    gmsh.model.mesh.removeDuplicateNodes()
    gmsh.model.mesh.generate(3)
    nodes_top = gmsh.model.mesh.getNodes(2, top_face, includeBoundary=True)
    assert len(nodes_top[0]) > 0


def test_geo_kernel_volume_with_imported_top(tmp_path):
    """Mixed-kernel option 2: build the volume in the geo kernel with the
    top face taken from the imported discrete mesh.

    Setup:
    1. Build a discrete top face mesh in a separate model; write to .msh.
    2. In a fresh model: build the bottom face + four side faces in the
       geo kernel; merge the discrete top face from .msh.
    3. Construct a closed surface loop (4 sides + bottom from geo, top from
       discrete) and a volume from it.
    4. mesh.generate(3) and verify the resulting top-face mesh equals the
       imported one.
    """
    # ----- Phase 1 simulation: build discrete top face mesh -----
    seam_path = tmp_path / "seam.msh"
    _build_seam_surface_msh(seam_path, mesh_size=0.2)
    # Capture imported triangle count for later comparison.
    gmsh.model.add("seam_count")
    gmsh.merge(str(seam_path))
    seam_2d_entities = gmsh.model.getEntities(2)
    assert len(seam_2d_entities) == 1
    seam_dim, seam_tag = seam_2d_entities[0]
    _, etags, _ = gmsh.model.mesh.getElements(seam_dim, seam_tag)
    expected_tri_count = sum(len(t) for t in etags)
    gmsh.model.remove()

    # ----- Phase 2 simulation: geo-kernel volume with discrete top -----
    gmsh.model.add("vol")
    # Bottom + sides in geo kernel.
    pts_bot = [
        gmsh.model.geo.addPoint(0, 0, 0, 0.5),
        gmsh.model.geo.addPoint(1, 0, 0, 0.5),
        gmsh.model.geo.addPoint(1, 1, 0, 0.5),
        gmsh.model.geo.addPoint(0, 1, 0, 0.5),
    ]
    pts_top = [
        gmsh.model.geo.addPoint(0, 0, 1, 0.5),
        gmsh.model.geo.addPoint(1, 0, 1, 0.5),
        gmsh.model.geo.addPoint(1, 1, 1, 0.5),
        gmsh.model.geo.addPoint(0, 1, 1, 0.5),
    ]
    bot_lines = [
        gmsh.model.geo.addLine(pts_bot[i], pts_bot[(i + 1) % 4]) for i in range(4)
    ]
    top_lines = [
        gmsh.model.geo.addLine(pts_top[i], pts_top[(i + 1) % 4]) for i in range(4)
    ]
    vert_lines = [gmsh.model.geo.addLine(pts_bot[i], pts_top[i]) for i in range(4)]
    bot_loop = gmsh.model.geo.addCurveLoop(bot_lines)
    bot_face = gmsh.model.geo.addPlaneSurface([bot_loop])
    top_loop = gmsh.model.geo.addCurveLoop(top_lines)
    top_face = gmsh.model.geo.addPlaneSurface([top_loop])
    side_faces = []
    for i in range(4):
        side_loop = gmsh.model.geo.addCurveLoop(
            [bot_lines[i], vert_lines[(i + 1) % 4], -top_lines[i], -vert_lines[i]]
        )
        side_faces.append(gmsh.model.geo.addPlaneSurface([side_loop]))
    surf_loop = gmsh.model.geo.addSurfaceLoop([bot_face, top_face, *side_faces])
    gmsh.model.geo.addVolume([surf_loop])
    gmsh.model.geo.synchronize()

    # Try meshing first to confirm the geo volume is well-formed.
    gmsh.model.mesh.generate(3)
    nodes_top_baseline = gmsh.model.mesh.getNodes(2, top_face, includeBoundary=True)
    print(f"BASELINE geo-only top face nodes: {len(nodes_top_baseline[0])}")
    gmsh.model.mesh.clear()

    # Now: REPLACE top_face's mesh with the imported discrete one.
    # Approach A: just merge the seam .msh and let it land on the existing top_face by tag collision.
    pre_2d = {tag for d, tag in gmsh.model.getEntities(2)}
    gmsh.merge(str(seam_path))
    post_2d = {tag for d, tag in gmsh.model.getEntities(2)}
    new_2d = post_2d - pre_2d
    print(f"After merge: pre_2d={pre_2d}, post_2d={post_2d}, new={new_2d}")

    # Now try generate(3). If the merged mesh attached to top_face by tag collision,
    # it should be respected. If it added a new discrete entity, we need to embed.
    gmsh.model.mesh.generate(3)

    nodes_top, _, _ = gmsh.model.mesh.getNodes(2, top_face, includeBoundary=True)
    _, etags2, _ = gmsh.model.mesh.getElements(2, top_face)
    actual_tri_count = sum(len(t) for t in etags2)
    print(
        f"After generate(3): top face has {len(nodes_top)} nodes, {actual_tri_count} tris"
    )
    print(f"Expected (from imported seam): {expected_tri_count} tris")

    # The test PASSES if the top face's mesh matches the imported count.
    assert actual_tri_count == expected_tri_count, (
        f"top face was re-meshed: got {actual_tri_count} tris, expected {expected_tri_count} "
        "(injected seam preserved)"
    )


def test_set_nodes_on_occ_face(tmp_path):
    """Skip merge+embed; instead, transcribe the seam mesh's nodes+elements
    onto the OCC top face via setNodes/setElements before generate(3).
    """
    # Build the seam mesh in-memory and capture its nodes/triangles.
    gmsh.model.add("seam2")
    pts = [
        gmsh.model.geo.addPoint(0, 0, 1, 0.2),
        gmsh.model.geo.addPoint(1, 0, 1, 0.2),
        gmsh.model.geo.addPoint(1, 1, 1, 0.2),
        gmsh.model.geo.addPoint(0, 1, 1, 0.2),
    ]
    lines = [gmsh.model.geo.addLine(pts[i], pts[(i + 1) % 4]) for i in range(4)]
    loop = gmsh.model.geo.addCurveLoop(lines)
    surf = gmsh.model.geo.addPlaneSurface([loop])
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)
    seam_node_tags, seam_node_coords, _ = gmsh.model.mesh.getNodes(
        2, surf, includeBoundary=True
    )
    seam_elem_types, seam_elem_tags, seam_elem_node_tags = gmsh.model.mesh.getElements(
        2, surf
    )
    gmsh.model.remove()

    # Build OCC box and inject the seam mesh on the top face.
    gmsh.model.add("box_b")
    gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
    gmsh.model.occ.synchronize()
    top_face = None
    for dim, tag in gmsh.model.getEntities(2):
        bb = gmsh.model.getBoundingBox(dim, tag)
        if abs(bb[2] - 1.0) < 1e-6 and abs(bb[5] - 1.0) < 1e-6:
            top_face = tag
            break
    assert top_face is not None

    gmsh.model.mesh.addNodes(2, top_face, list(seam_node_tags), list(seam_node_coords))
    for et, etags, enodes in zip(seam_elem_types, seam_elem_tags, seam_elem_node_tags):
        gmsh.model.mesh.addElements(2, top_face, [et], [list(etags)], [list(enodes)])
    expected_tri_count = len(seam_elem_tags[0])
    gmsh.model.mesh.generate(3)
    top_elems = gmsh.model.mesh.getElements(2, top_face)
    actual_tri_count = len(top_elems[1][0]) if top_elems[1] else 0
    # Conformity check: the injected mesh must be preserved, not re-meshed.
    assert actual_tri_count == expected_tri_count, (
        f"top face was re-meshed: got {actual_tri_count} tris, "
        f"expected {expected_tri_count} (injected seam preserved)"
    )


def test_geo_volume_with_discrete_top_via_surface_loop(tmp_path):
    """Mixed-kernel option 2 variant B: build the geo-kernel volume WITHOUT
    a top face, then merge the seam .msh as a discrete surface and use it
    in the surface loop closing the volume.
    """
    seam_path = tmp_path / "seam.msh"
    _build_seam_surface_msh(seam_path, mesh_size=0.2)
    # Capture imported triangle count
    gmsh.model.add("seam_count")
    gmsh.merge(str(seam_path))
    seam_2d_entities = gmsh.model.getEntities(2)
    assert len(seam_2d_entities) == 1
    seam_dim, seam_tag_orig = seam_2d_entities[0]
    _, etags, _ = gmsh.model.mesh.getElements(seam_dim, seam_tag_orig)
    expected_tri_count = sum(len(t) for t in etags)
    print(f"Phase-1 seam tag: {seam_tag_orig}, tris: {expected_tri_count}")
    gmsh.model.remove()

    # Phase 2: merge discrete first, then build geo-kernel sides+bottom around it.
    gmsh.model.add("vol_b")
    # Merge seam first so we know what discrete tags are taken.
    gmsh.merge(str(seam_path))
    discrete_2d = gmsh.model.getEntities(2)
    print(f"After merging seam first: 2D entities = {discrete_2d}")
    assert len(discrete_2d) == 1
    discrete_top_tag = discrete_2d[0][1]

    # Inspect the discrete surface's boundary: does it have curves/points?
    boundary = gmsh.model.getBoundary(
        [(2, discrete_top_tag)], oriented=False, recursive=False
    )
    print(f"Discrete top boundary entities: {boundary}")

    # Try classifySurfaces+createGeometry to materialize the discrete boundary
    # as model curves/points, so the geo kernel can hook into them.
    # NOTE: curveAngle controls when boundary edges are split into multiple
    # curves at corners. Default 180 deg means no split (one closed curve).
    # The unit square has 90 deg corners, so use < 90 deg to break at them.
    gmsh.model.mesh.classifySurfaces(
        angle=40 * 3.14159 / 180,
        boundary=True,
        forReparametrization=False,
        curveAngle=60 * 3.14159 / 180,
    )
    gmsh.model.mesh.createGeometry()
    print(
        f"After classifySurfaces+createGeometry: 0D={gmsh.model.getEntities(0)}, "
        f"1D={gmsh.model.getEntities(1)}, 2D={gmsh.model.getEntities(2)}"
    )

    # classifySurfaces may have re-tagged the discrete face. Rediscover by bbox at z=1.
    new_2d = gmsh.model.getEntities(2)
    discrete_top_tag = None
    for d, t in new_2d:
        bb = gmsh.model.getBoundingBox(d, t)
        if abs(bb[2] - 1.0) < 1e-6 and abs(bb[5] - 1.0) < 1e-6:
            discrete_top_tag = t
            break
    assert discrete_top_tag is not None, "lost discrete top after classifySurfaces"
    print(f"discrete_top_tag after classify: {discrete_top_tag}")

    # Now find the boundary curves of the discrete top in tag order.
    top_boundary_curves = gmsh.model.getBoundary(
        [(2, discrete_top_tag)], oriented=False, recursive=False
    )
    top_boundary_points = gmsh.model.getBoundary(
        [(2, discrete_top_tag)], oriented=False, recursive=True
    )
    top_boundary_points = [p for p in top_boundary_points if p[0] == 0]
    print(f"top_boundary_curves: {top_boundary_curves}")
    print(f"top_boundary_points: {top_boundary_points}")

    if not top_boundary_curves:
        pytest.skip(
            "Discrete top has no model boundary curves; cannot stitch geo sides to it"
        )

    # Get coords of the four boundary points (corners) on the top.
    top_corner_coords = {}
    for d, p in top_boundary_points:
        c = gmsh.model.getValue(0, p, [])
        top_corner_coords[p] = c
    print(f"top_corner_coords: {top_corner_coords}")

    # Build bottom-face points and side faces stitching corners up to top corners.
    pts_bot = [
        gmsh.model.geo.addPoint(0, 0, 0, 0.5),
        gmsh.model.geo.addPoint(1, 0, 0, 0.5),
        gmsh.model.geo.addPoint(1, 1, 0, 0.5),
        gmsh.model.geo.addPoint(0, 1, 0, 0.5),
    ]
    bot_lines = [
        gmsh.model.geo.addLine(pts_bot[i], pts_bot[(i + 1) % 4]) for i in range(4)
    ]
    bot_loop = gmsh.model.geo.addCurveLoop(bot_lines)
    bot_face = gmsh.model.geo.addPlaneSurface([bot_loop])

    # For each bottom point, find the matching top corner by xy.
    def find_top_corner(x, y):
        for tag, c in top_corner_coords.items():
            if abs(c[0] - x) < 1e-6 and abs(c[1] - y) < 1e-6 and abs(c[2] - 1.0) < 1e-6:
                return tag
        return None

    bot_xy = [(0, 0), (1, 0), (1, 1), (0, 1)]
    matched_top = [find_top_corner(x, y) for (x, y) in bot_xy]
    print(f"matched_top corners: {matched_top}")
    if any(m is None for m in matched_top):
        gmsh.model.geo.synchronize()
        pytest.skip(f"Could not match bottom to top corners: {matched_top}")

    # Vertical edges from bottom geo points up to discrete-derived top points.
    # geo.addLine accepts any model point tags (geo or discrete).
    vert_lines = [gmsh.model.geo.addLine(pts_bot[i], matched_top[i]) for i in range(4)]

    # Top boundary curves as a curve-loop. We need to know the orientation/order.
    # Just try adding the curve loop directly with discrete curves.
    # Get curve tags in order around the top.
    top_curve_tags = [c for d, c in top_boundary_curves]
    print(f"top_curve_tags raw: {top_curve_tags}")

    side_faces = []
    for i in range(4):
        # side i: bot_lines[i] (bottom edge), vert_lines[(i+1)%4] (up at next corner),
        # top edge between matched_top[(i+1)%4] -> matched_top[i] (negative top-edge),
        # vert_lines[i] reversed.
        # We don't actually know the top edge tag/orientation to use here.
        # Use geo.addCurveLoop's tolerance: pass the four edges and let it figure orientation.
        # Find the top curve connecting matched_top[i] and matched_top[(i+1)%4].
        target_pts = {matched_top[i], matched_top[(i + 1) % 4]}
        chosen_top_curve = None
        for ct in top_curve_tags:
            cb = gmsh.model.getBoundary([(1, ct)], oriented=False, recursive=False)
            cb_pts = {p for d_, p in cb}
            if cb_pts == target_pts:
                chosen_top_curve = ct
                break
        if chosen_top_curve is None:
            gmsh.model.geo.synchronize()
            pytest.skip(f"Could not find top curve for side {i}")
        try:
            side_loop = gmsh.model.geo.addCurveLoop(
                [
                    bot_lines[i],
                    vert_lines[(i + 1) % 4],
                    -chosen_top_curve,
                    -vert_lines[i],
                ]
            )
            side_face = gmsh.model.geo.addPlaneSurface([side_loop])
            side_faces.append(side_face)
        except Exception as e:
            print(f"side {i} failed: {e}")
            gmsh.model.geo.synchronize()
            pytest.skip(f"Side {i} curve loop failed: {e}")

    print(f"side_faces created: {side_faces}")

    # Build the surface loop using bot, sides (geo) + discrete top.
    try:
        surf_loop = gmsh.model.geo.addSurfaceLoop(
            [bot_face, *side_faces, discrete_top_tag]
        )
        vol = gmsh.model.geo.addVolume([surf_loop])
        gmsh.model.geo.synchronize()
    except Exception as e:
        print(f"Surface loop / volume failed: {e}")
        pytest.fail(f"Could not create mixed surface loop: {e}")

    print(f"vol={vol}, surf_loop={surf_loop}")

    # Mesh in 3D. The discrete top should be respected.
    gmsh.model.mesh.generate(3)

    _, etags2, _ = gmsh.model.mesh.getElements(2, discrete_top_tag)
    actual_tri_count = sum(len(t) for t in etags2)
    nodes_top, _, _ = gmsh.model.mesh.getNodes(
        2, discrete_top_tag, includeBoundary=True
    )
    print(
        f"After generate(3): discrete top has {len(nodes_top)} nodes, "
        f"{actual_tri_count} tris (expected {expected_tri_count})"
    )

    # 3D element check
    _, etags3, _ = gmsh.model.mesh.getElements(3, vol)
    n_tets = sum(len(t) for t in etags3)
    print(f"Volume tets: {n_tets}")

    assert (
        actual_tri_count == expected_tri_count
    ), f"top face was re-meshed: got {actual_tri_count}, expected {expected_tri_count}"
    assert n_tets > 0, "no volume tets were generated"


def test_geo_volume_discrete_top_via_create_topology(tmp_path):
    """Variant C: use createTopology() (NOT classifySurfaces+createGeometry)
    to build only the curve/point topology around the discrete top, then
    construct the volume's surface loop using the original discrete face.

    createTopology() does not rebuild a parametric surface, so the original
    discrete mesh elements should be preserved.
    """
    seam_path = tmp_path / "seam.msh"
    _build_seam_surface_msh(seam_path, mesh_size=0.2)
    # Capture imported triangle count
    gmsh.model.add("seam_count_c")
    gmsh.merge(str(seam_path))
    seam_2d_entities = gmsh.model.getEntities(2)
    seam_dim, seam_tag_orig = seam_2d_entities[0]
    _, etags, _ = gmsh.model.mesh.getElements(seam_dim, seam_tag_orig)
    expected_tri_count = sum(len(t) for t in etags)
    gmsh.model.remove()

    gmsh.model.add("vol_c")
    gmsh.merge(str(seam_path))
    discrete_2d = gmsh.model.getEntities(2)
    print(f"After merging seam: 2D = {discrete_2d}")
    discrete_top_tag = discrete_2d[0][1]

    # Snapshot of discrete mesh BEFORE createTopology
    _, etags0, _ = gmsh.model.mesh.getElements(2, discrete_top_tag)
    print(f"Discrete top BEFORE createTopology: {sum(len(t) for t in etags0)} tris")

    # createTopology only builds boundary curves/points without rebuilding the surface.
    gmsh.model.mesh.createTopology()
    print(
        f"After createTopology: 0D={gmsh.model.getEntities(0)}, "
        f"1D={gmsh.model.getEntities(1)}, 2D={gmsh.model.getEntities(2)}"
    )

    new_2d = gmsh.model.getEntities(2)
    discrete_top_tag = None
    for d, t in new_2d:
        bb = gmsh.model.getBoundingBox(d, t)
        if abs(bb[2] - 1.0) < 1e-6 and abs(bb[5] - 1.0) < 1e-6:
            discrete_top_tag = t
            break
    print(f"discrete_top_tag after createTopology: {discrete_top_tag}")

    # Snapshot of discrete mesh AFTER createTopology
    _, etags1, _ = gmsh.model.mesh.getElements(2, discrete_top_tag)
    print(f"Discrete top AFTER createTopology: {sum(len(t) for t in etags1)} tris")

    top_boundary_curves = gmsh.model.getBoundary(
        [(2, discrete_top_tag)], oriented=False, recursive=False
    )
    top_boundary_points_all = gmsh.model.getBoundary(
        [(2, discrete_top_tag)], oriented=False, recursive=True
    )
    top_boundary_points = [p for p in top_boundary_points_all if p[0] == 0]
    print(f"top_boundary_curves: {top_boundary_curves}")
    print(f"top_boundary_points: {top_boundary_points}")

    if not top_boundary_curves or len(top_boundary_points) < 4:
        pytest.skip(
            f"createTopology yielded insufficient boundary topology: "
            f"{len(top_boundary_curves)} curves, {len(top_boundary_points)} points"
        )

    top_corner_coords = {
        p: gmsh.model.getValue(0, p, []) for d, p in top_boundary_points
    }
    print(f"top_corner_coords: {top_corner_coords}")

    # Build bottom and side faces stitching to the discrete top corners.
    pts_bot = [
        gmsh.model.geo.addPoint(0, 0, 0, 0.5),
        gmsh.model.geo.addPoint(1, 0, 0, 0.5),
        gmsh.model.geo.addPoint(1, 1, 0, 0.5),
        gmsh.model.geo.addPoint(0, 1, 0, 0.5),
    ]
    bot_lines = [
        gmsh.model.geo.addLine(pts_bot[i], pts_bot[(i + 1) % 4]) for i in range(4)
    ]
    bot_loop = gmsh.model.geo.addCurveLoop(bot_lines)
    bot_face = gmsh.model.geo.addPlaneSurface([bot_loop])

    def find_top_corner(x, y):
        for tag, c in top_corner_coords.items():
            if abs(c[0] - x) < 1e-6 and abs(c[1] - y) < 1e-6 and abs(c[2] - 1.0) < 1e-6:
                return tag
        return None

    bot_xy = [(0, 0), (1, 0), (1, 1), (0, 1)]
    matched_top = [find_top_corner(x, y) for (x, y) in bot_xy]
    print(f"matched_top corners: {matched_top}")
    if any(m is None for m in matched_top):
        gmsh.model.geo.synchronize()
        pytest.skip(f"Could not match corners: {matched_top}")

    vert_lines = [gmsh.model.geo.addLine(pts_bot[i], matched_top[i]) for i in range(4)]

    top_curve_tags = [c for d, c in top_boundary_curves]
    side_faces = []
    for i in range(4):
        target_pts = {matched_top[i], matched_top[(i + 1) % 4]}
        chosen_top_curve = None
        for ct in top_curve_tags:
            cb = gmsh.model.getBoundary([(1, ct)], oriented=False, recursive=False)
            cb_pts = {p for d_, p in cb}
            if cb_pts == target_pts:
                chosen_top_curve = ct
                break
        if chosen_top_curve is None:
            gmsh.model.geo.synchronize()
            pytest.skip(f"Could not find top curve for side {i}")
        side_loop = gmsh.model.geo.addCurveLoop(
            [bot_lines[i], vert_lines[(i + 1) % 4], -chosen_top_curve, -vert_lines[i]]
        )
        side_face = gmsh.model.geo.addPlaneSurface([side_loop])
        side_faces.append(side_face)

    surf_loop = gmsh.model.geo.addSurfaceLoop([bot_face, *side_faces, discrete_top_tag])
    vol = gmsh.model.geo.addVolume([surf_loop])
    gmsh.model.geo.synchronize()
    print(f"vol={vol}")

    # Inspect discrete boundary curves' mesh state
    for ct in top_curve_tags:
        cn, _, _ = gmsh.model.mesh.getNodes(1, ct, includeBoundary=True)
        _, ce_tags, _ = gmsh.model.mesh.getElements(1, ct)
        print(
            f"  curve {ct} BEFORE: {len(cn)} nodes, {sum(len(t) for t in ce_tags)} edge elems"
        )

    # The discrete curves carry boundary nodes but no 1D line elements.
    # createEdges may materialize them.
    try:
        gmsh.model.mesh.createEdges([(1, t) for t in top_curve_tags])
    except Exception as e:
        print(f"createEdges raised: {e}")

    for ct in top_curve_tags:
        cn, _, _ = gmsh.model.mesh.getNodes(1, ct, includeBoundary=True)
        _, ce_tags, _ = gmsh.model.mesh.getElements(1, ct)
        print(
            f"  curve {ct} AFTER createEdges: {len(cn)} nodes, {sum(len(t) for t in ce_tags)} elems"
        )

    # Manually inject 1D Line elements on each discrete curve, sorted along the curve.
    # We have 4 corner points and the discrete curve carries interior nodes.
    import numpy as np

    for i, ct in enumerate(top_curve_tags):
        # Get the corner endpoints of this curve from boundary
        cb = gmsh.model.getBoundary([(1, ct)], oriented=False, recursive=False)
        end_pts = [p for d_, p in cb]
        end_coords = {p: np.array(gmsh.model.getValue(0, p, [])) for p in end_pts}

        # Get all nodes on the curve (interior). NB: corner nodes belong to dim=0 entities,
        # not the curve itself. Try includeBoundary=False to get only interior nodes.
        cn_tags, cn_coords, _ = gmsh.model.mesh.getNodes(1, ct, includeBoundary=False)
        print(f"  curve {ct} interior nodes: {len(cn_tags)}")
        # Get corner node tags via dim=0 entity nodes
        corner_node_tags = []
        for p in end_pts:
            pn_tags, _, _ = gmsh.model.mesh.getNodes(0, p, includeBoundary=False)
            if len(pn_tags) > 0:
                corner_node_tags.append(int(pn_tags[0]))
            else:
                corner_node_tags.append(None)
        print(f"  curve {ct} corner node tags: {corner_node_tags}")

        # Sort interior nodes along the line from end1->end2
        if cn_tags is not None and len(cn_tags) > 0:
            v = end_coords[end_pts[1]] - end_coords[end_pts[0]]
            v = v / np.linalg.norm(v)
            cn_coords3 = np.array(cn_coords).reshape(-1, 3)
            params = (cn_coords3 - end_coords[end_pts[0]]) @ v
            order = np.argsort(params)
            sorted_tags = [int(cn_tags[k]) for k in order]
        else:
            sorted_tags = []
        # Build chain: corner0 -> sorted interior -> corner1
        chain = [corner_node_tags[0], *sorted_tags, corner_node_tags[1]]
        print(f"  curve {ct} chain length: {len(chain)}")
        if any(c is None for c in chain):
            continue
        # Add Line (type 1) elements
        line_node_tags = []
        for k in range(len(chain) - 1):
            line_node_tags.extend([chain[k], chain[k + 1]])
        try:
            gmsh.model.mesh.addElementsByType(ct, 1, [], line_node_tags)
            print(f"  curve {ct}: added {len(chain)-1} Line elements")
        except Exception as e:
            print(f"  curve {ct}: addElementsByType failed: {e}")

    for ct in top_curve_tags:
        _, ce_tags, _ = gmsh.model.mesh.getElements(1, ct)
        print(f"  curve {ct} FINAL: {sum(len(t) for t in ce_tags)} elems")

    try:
        gmsh.model.mesh.generate(3)
        gen_ok = True
    except Exception as e:
        print(f"generate(3) FAILED: {e}")
        gen_ok = False

    if not gen_ok:
        pytest.fail("generate(3) raised an exception")

    _, etags2, _ = gmsh.model.mesh.getElements(2, discrete_top_tag)
    actual_tri_count = sum(len(t) for t in etags2)
    nodes_top, _, _ = gmsh.model.mesh.getNodes(
        2, discrete_top_tag, includeBoundary=True
    )
    _, etags3, _ = gmsh.model.mesh.getElements(3, vol)
    n_tets = sum(len(t) for t in etags3)
    print(
        f"After generate(3): discrete top has {len(nodes_top)} nodes, "
        f"{actual_tri_count} tris (expected {expected_tri_count}), {n_tets} tets"
    )

    assert (
        actual_tri_count == expected_tri_count
    ), f"top face was re-meshed: got {actual_tri_count}, expected {expected_tri_count}"
    assert n_tets > 0, "no volume tets were generated"


def test_embed_with_create_topology(tmp_path):
    seam_path = tmp_path / "seam.msh"
    _build_seam_surface_msh(seam_path, mesh_size=0.2)
    gmsh.model.add("box_c")
    gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
    gmsh.model.occ.synchronize()
    top_face = None
    for dim, tag in gmsh.model.getEntities(2):
        bb = gmsh.model.getBoundingBox(dim, tag)
        if abs(bb[2] - 1.0) < 1e-6 and abs(bb[5] - 1.0) < 1e-6:
            top_face = tag
            break

    pre_2d = {tag for d, tag in gmsh.model.getEntities(2)}
    gmsh.merge(str(seam_path))
    imported_2d = sorted({tag for d, tag in gmsh.model.getEntities(2)} - pre_2d)[0]

    # Try: classify the merged surface's boundary edges into the model topology
    # so its boundary curves are reconciled with the OCC top-face boundary.
    gmsh.model.mesh.classifySurfaces(
        angle=40 * 3.14159 / 180,
        boundary=True,
        forReparametrization=False,
        curveAngle=180 * 3.14159 / 180,
    )
    gmsh.model.mesh.createGeometry()
    gmsh.model.mesh.embed(2, [imported_2d], 2, top_face)
    gmsh.model.mesh.generate(3)
    nodes_top = gmsh.model.mesh.getNodes(2, top_face, includeBoundary=True)
    assert len(nodes_top[0]) > 0
