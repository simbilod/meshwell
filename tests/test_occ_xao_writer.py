"""Tests for the fully-tagged XAO writer + injection."""

from __future__ import annotations

import shapely
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCP.gp import gp_Pnt

import gmsh
from meshwell.cad_occ import cad_occ
from meshwell.model import ModelManager
from meshwell.occ_entity import OCC_entity
from meshwell.occ_xao_writer import write_xao
from meshwell.polyline import PolyLine
from meshwell.polyprism import PolyPrism
from meshwell.polysurface import PolySurface


def test_xao_writer_produces_single_self_contained_file(tmp_path):
    """XAO includes full tagging: entities, A___B interface, A___None exteriors.

    Runs the inputs through ``cad_occ`` so BOPAlgo canonicalises the shared
    TShape at x=1 -- only then does the writer's interface detector fire.
    """
    a = OCC_entity(
        occ_function=lambda: BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 1, 1, 1).Shape(),
        physical_name="A",
        mesh_order=1,
        dimension=3,
    )
    b = OCC_entity(
        occ_function=lambda: BRepPrimAPI_MakeBox(gp_Pnt(1, 0, 0), 1, 1, 1).Shape(),
        physical_name="B",
        mesh_order=2,
        dimension=3,
    )
    ents = cad_occ([a, b])
    xao = tmp_path / "m.xao"
    write_xao(ents, xao)

    assert xao.exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["m.xao"]

    content = xao.read_text()
    assert "<![CDATA[" in content
    assert "DBRep_DrawableShape" in content

    gmsh.initialize()
    try:
        gmsh.open(str(xao))
        gmsh.model.occ.synchronize()

        vol_names = {
            gmsh.model.getPhysicalName(d, t) for d, t in gmsh.model.getPhysicalGroups(3)
        }
        assert {"A", "B"} <= vol_names

        surf_names = {
            gmsh.model.getPhysicalName(d, t) for d, t in gmsh.model.getPhysicalGroups(2)
        }
        assert surf_names & {"A___B", "B___A"}
        assert {"A___None", "B___None"} <= surf_names
    finally:
        gmsh.finalize()


def test_inject_full_mixed_scene():
    """Mixed 3D/2D/1D entities end up with correct physical groups + masses."""
    square_A = shapely.Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
    square_B = shapely.Polygon([(1, 0), (2, 0), (2, 1), (1, 1), (1, 0)])
    prism_A = PolyPrism(
        polygons=square_A,
        buffers={0.0: 0.0, 2.0: 0.0},
        physical_name="A",
        mesh_order=1,
    )
    prism_B = PolyPrism(
        polygons=square_B,
        buffers={0.0: 0.0, 2.0: 0.0},
        physical_name="B",
        mesh_order=2,
    )
    cut = PolySurface(
        polygons=square_A,
        physical_name="cut",
        mesh_order=0,
        translation=(0.0, 0.0, 1.0),
    )
    wire = PolyLine(
        linestrings=shapely.LineString([(6, 0), (7, 0)]),
        physical_name="wire",
        mesh_order=3,
    )

    occ_ents = cad_occ([cut, prism_A, prism_B, wire])
    mm = ModelManager(filename="test_xao_mixed")
    try:
        mm.load_occ_entities(occ_ents)
        all_names = {
            gmsh.model.getPhysicalName(d, t)
            for dim in (0, 1, 2, 3)
            for d, t in gmsh.model.getPhysicalGroups(dim)
        }
        assert {"A", "B", "cut", "wire"} <= all_names

        # Volume totals per entity match hand-computed values.
        vol_by_name: dict[str, float] = {}
        for d, t in gmsh.model.getPhysicalGroups(3):
            name = gmsh.model.getPhysicalName(d, t)
            total = sum(
                gmsh.model.occ.getMass(d, tag)
                for tag in gmsh.model.getEntitiesForPhysicalGroup(d, t)
            )
            vol_by_name[name] = total
        assert abs(vol_by_name["A"] - 2.0) < 1e-6
        assert abs(vol_by_name["B"] - 2.0) < 1e-6

        # Shared A___B interface exists, and it is area ~ 2.
        surf_groups = gmsh.model.getPhysicalGroups(2)
        interface_name = next(
            (
                gmsh.model.getPhysicalName(d, t)
                for d, t in surf_groups
                if gmsh.model.getPhysicalName(d, t) in {"A___B", "B___A"}
            ),
            None,
        )
        assert interface_name is not None
        interface_tag = next(
            t
            for d, t in surf_groups
            if gmsh.model.getPhysicalName(d, t) == interface_name
        )
        ifc_area = sum(
            gmsh.model.occ.getMass(2, tag)
            for tag in gmsh.model.getEntitiesForPhysicalGroup(2, interface_tag)
        )
        assert abs(ifc_area - 2.0) < 1e-6
    finally:
        mm.finalize()


def test_inject_two_touching_boxes_disjoint_volumes():
    """Per-entity volume lookup produces disjoint gmsh-entity sets."""
    a = OCC_entity(
        occ_function=lambda: BRepPrimAPI_MakeBox(
            gp_Pnt(0, 0, 0), 1.0, 1.0, 1.0
        ).Shape(),
        physical_name="a",
        mesh_order=1,
        dimension=3,
    )
    b = OCC_entity(
        occ_function=lambda: BRepPrimAPI_MakeBox(
            gp_Pnt(1, 0, 0), 1.0, 1.0, 1.0
        ).Shape(),
        physical_name="b",
        mesh_order=2,
        dimension=3,
    )
    occ_ents = cad_occ([a, b])

    mm = ModelManager(filename="test_xao_volumes")
    try:
        mm.load_occ_entities(occ_ents)
        names_to_vols = {
            gmsh.model.getPhysicalName(d, t): set(
                gmsh.model.getEntitiesForPhysicalGroup(d, t)
            )
            for d, t in gmsh.model.getPhysicalGroups(3)
        }
        assert set(names_to_vols) == {"a", "b"}
        assert len(names_to_vols["a"]) == 1
        assert len(names_to_vols["b"]) == 1
        assert names_to_vols["a"].isdisjoint(names_to_vols["b"])
        surf_names = {
            gmsh.model.getPhysicalName(d, t) for d, t in gmsh.model.getPhysicalGroups(2)
        }
        assert surf_names & {"a___b", "b___a"}
    finally:
        mm.finalize()


def test_keep_false_entity_removed_but_interface_named():
    """A keep=False helper still names boundaries it shares with kept entities."""
    kept = OCC_entity(
        occ_function=lambda: BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 1, 1, 1).Shape(),
        physical_name="kept",
        mesh_order=2,
        dimension=3,
    )
    helper = OCC_entity(
        occ_function=lambda: BRepPrimAPI_MakeBox(gp_Pnt(1, 0, 0), 1, 1, 1).Shape(),
        physical_name="helper",
        mesh_order=1,
        mesh_bool=False,  # keep=False in OCCLabeledEntity.
        dimension=3,
    )
    occ_ents = cad_occ([kept, helper])
    mm = ModelManager(filename="test_xao_keep_false")
    try:
        mm.load_occ_entities(occ_ents)
        vol_names = {
            gmsh.model.getPhysicalName(d, t) for d, t in gmsh.model.getPhysicalGroups(3)
        }
        # Only the kept entity is tagged as a 3D physical group.
        assert vol_names == {"kept"}
        # gmsh has only one volume now.
        assert len(gmsh.model.getEntities(3)) == 1
        # The kept___helper interface (or helper___kept) still exists as a
        # named dim-2 group even though the helper solid was removed.
        surf_names = {
            gmsh.model.getPhysicalName(d, t) for d, t in gmsh.model.getPhysicalGroups(2)
        }
        assert surf_names & {"kept___helper", "helper___kept"}
    finally:
        mm.finalize()


def test_keep_false_3d_fully_inside_leaves_interior_void():
    """keep=False 3D helper fully inside a kept 3D volume leaves a void.

    helper = 4x4x4 at (3,3,3); kept = 10x10x10 at origin. helper has higher
    priority (mesh_order=1) so during ``cad_occ`` it wins the overlap
    region. After load the mesh has one hollow solid: outer volume
    1000 - 64 = 936, bounded by 6 outer faces tagged ``kept___None`` and
    6 inner faces tagged ``helper___kept`` (or the permutation). The
    void region itself is absent from the mesh.
    """
    helper = OCC_entity(
        occ_function=lambda: BRepPrimAPI_MakeBox(gp_Pnt(3, 3, 3), 4, 4, 4).Shape(),
        physical_name="helper",
        mesh_order=1,
        mesh_bool=False,
        dimension=3,
    )
    kept = OCC_entity(
        occ_function=lambda: BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 10, 10, 10).Shape(),
        physical_name="kept",
        mesh_order=2,
        dimension=3,
    )
    occ_ents = cad_occ([helper, kept])
    mm = ModelManager(filename="test_xao_keep_false_void")
    try:
        mm.load_occ_entities(occ_ents)
        assert len(gmsh.model.getEntities(3)) == 1
        (vol_tag,) = [t for _, t in gmsh.model.getEntities(3)]
        kept_vol = gmsh.model.occ.getMass(3, vol_tag)
        assert abs(kept_vol - (1000 - 64)) < 1e-6

        surf_names = {
            gmsh.model.getPhysicalName(d, t) for d, t in gmsh.model.getPhysicalGroups(2)
        }
        assert "kept___None" in surf_names
        assert surf_names & {"helper___kept", "kept___helper"}

        # Outer + void walls = 12 named surfaces; neither group is empty.
        name_to_tags = {
            gmsh.model.getPhysicalName(d, t): list(
                gmsh.model.getEntitiesForPhysicalGroup(d, t)
            )
            for d, t in gmsh.model.getPhysicalGroups(2)
        }
        assert len(name_to_tags["kept___None"]) == 6
        interface_key = next(
            k for k in ("helper___kept", "kept___helper") if k in name_to_tags
        )
        assert len(name_to_tags[interface_key]) == 6
    finally:
        mm.finalize()


def test_keep_false_polyprism_void_is_not_meshed(tmp_path):
    """Meshing a keep=False void leaves the interior empty of tets.

    Beyond the OCC-level 'void___outer interface tagged' check, the
    resulting MSH must actually have zero tetrahedra with centroids
    inside the void, and summed tet volume equal to outer-minus-inner.
    That is the only way to confirm gmsh treated the void as absent
    rather than as a separately-meshed sub-region.
    """
    import meshio
    import numpy as np

    from meshwell.mesh import mesh

    inner = PolyPrism(
        polygons=shapely.box(3, 3, 7, 7),
        buffers={3.0: 0.0, 7.0: 0.0},
        physical_name="void",
        mesh_order=1,
        mesh_bool=False,
    )
    outer = PolyPrism(
        polygons=shapely.box(0, 0, 10, 10),
        buffers={0.0: 0.0, 10.0: 0.0},
        physical_name="outer",
        mesh_order=2,
    )
    xao = tmp_path / "hollow.xao"
    msh = tmp_path / "hollow.msh"
    write_xao(cad_occ([inner, outer]), xao)
    mesh(
        dim=3,
        input_file=xao,
        output_file=msh,
        default_characteristic_length=1.0,
        n_threads=1,
    )
    m = meshio.read(str(msh))

    tets = np.vstack([cb.data for cb in m.cells if cb.type == "tetra"])
    centroids = m.points[tets].mean(axis=1)
    in_void = np.all((centroids > 3) & (centroids < 7), axis=1)
    assert in_void.sum() == 0, "tets found inside the void region"

    # Summed tet volume matches outer minus inner (to meshing precision).
    def tet_vol(tet):
        a, b, c, d = [m.points[i] for i in tet]
        return abs(np.dot(b - a, np.cross(c - a, d - a))) / 6.0

    total_vol = sum(tet_vol(t) for t in tets)
    assert abs(total_vol - (1000 - 64)) < 1.0


def test_shared_physical_name_across_entities():
    """Entities sharing a physical_name collapse into one physical group.

    Three shared-name patterns to pin:

    - Two DISJOINT kept boxes named 'metal': one physical group 'metal'
      with two volumes, no 'metal___metal' interface (same-name pairs
      are skipped by the interface pass).
    - Two OVERLAPPING kept boxes named 'bulk': BOPAlgo splits into three
      pieces, all three go into the 'bulk' group. Volume equals union
      2^3 + 2^3 - 1^3 = 15.
    - Two disjoint keep=False voids both named 'void': the slab's
      `slab___void` interface group contains all void walls (6 + 6 = 12
      faces), not two separate groups.
    """
    # --- Pattern A: disjoint kept, shared name ---
    a1 = OCC_entity(
        occ_function=lambda: BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 1, 1, 1).Shape(),
        physical_name="metal",
        mesh_order=1,
        dimension=3,
    )
    a2 = OCC_entity(
        occ_function=lambda: BRepPrimAPI_MakeBox(gp_Pnt(5, 0, 0), 1, 1, 1).Shape(),
        physical_name="metal",
        mesh_order=2,
        dimension=3,
    )
    mm = ModelManager(filename="test_shared_name_disjoint")
    try:
        mm.load_occ_entities(cad_occ([a1, a2]))
        pgroups = {
            gmsh.model.getPhysicalName(d, t): list(
                gmsh.model.getEntitiesForPhysicalGroup(d, t)
            )
            for d, t in gmsh.model.getPhysicalGroups(3)
        }
        assert list(pgroups) == ["metal"]
        assert len(pgroups["metal"]) == 2
        # No same-name interface.
        surf = {
            gmsh.model.getPhysicalName(d, t) for d, t in gmsh.model.getPhysicalGroups(2)
        }
        assert "metal___metal" not in surf
    finally:
        mm.finalize()

    # --- Pattern B: overlapping kept, shared name ---
    b1 = OCC_entity(
        occ_function=lambda: BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 2, 2, 2).Shape(),
        physical_name="bulk",
        mesh_order=1,
        dimension=3,
    )
    b2 = OCC_entity(
        occ_function=lambda: BRepPrimAPI_MakeBox(gp_Pnt(1, 1, 1), 2, 2, 2).Shape(),
        physical_name="bulk",
        mesh_order=2,
        dimension=3,
    )
    mm = ModelManager(filename="test_shared_name_overlap")
    try:
        mm.load_occ_entities(cad_occ([b1, b2]))
        bulk_tags = [
            t
            for d, t in gmsh.model.getPhysicalGroups(3)
            if gmsh.model.getPhysicalName(d, t) == "bulk"
            for t in gmsh.model.getEntitiesForPhysicalGroup(d, t)
        ]
        assert len(bulk_tags) == 3  # union splits into three pieces
        total_vol = sum(gmsh.model.occ.getMass(3, tt) for tt in bulk_tags)
        assert abs(total_vol - 15.0) < 1e-6  # 2^3 + 2^3 - 1^3
    finally:
        mm.finalize()

    # --- Pattern C: multiple keep=False helpers sharing a name ---
    slab = PolyPrism(
        polygons=shapely.box(0, 0, 10, 10),
        buffers={0.0: 0.0, 2.0: 0.0},
        physical_name="slab",
        mesh_order=5,
    )
    v_a = OCC_entity(
        occ_function=lambda: BRepPrimAPI_MakeBox(gp_Pnt(1, 1, 0.5), 1, 1, 1).Shape(),
        physical_name="void",
        mesh_order=1,
        mesh_bool=False,
        dimension=3,
    )
    v_b = OCC_entity(
        occ_function=lambda: BRepPrimAPI_MakeBox(gp_Pnt(5, 5, 0.5), 1, 1, 1).Shape(),
        physical_name="void",
        mesh_order=1,
        mesh_bool=False,
        dimension=3,
    )
    mm = ModelManager(filename="test_shared_name_voids")
    try:
        mm.load_occ_entities(cad_occ([slab, v_a, v_b]))
        # Both voids' walls unified under slab___void.
        surf_by_name = {
            gmsh.model.getPhysicalName(d, t): list(
                gmsh.model.getEntitiesForPhysicalGroup(d, t)
            )
            for d, t in gmsh.model.getPhysicalGroups(2)
        }
        interface_key = next(
            k for k in ("slab___void", "void___slab") if k in surf_by_name
        )
        assert len(surf_by_name[interface_key]) == 12  # 6 + 6 void walls
        # No separate void_a/void_b groups.
        assert "void" not in surf_by_name  # keep=False -> no own group
        # Slab mass = 200 - 2 * 1^3.
        slab_tags = [
            t
            for d, t in gmsh.model.getPhysicalGroups(3)
            if gmsh.model.getPhysicalName(d, t) == "slab"
            for t in gmsh.model.getEntitiesForPhysicalGroup(d, t)
        ]
        assert (
            abs(sum(gmsh.model.occ.getMass(3, tt) for tt in slab_tags) - 198.0) < 1e-6
        )
    finally:
        mm.finalize()


def test_keep_false_polyprism_fully_inside_leaves_interior_void():
    """Same hollow-solid semantic as the OCC_entity case, via PolyPrism.

    outer 10x10x10 PolyPrism, void 4x4x4 PolyPrism fully inside, keep=False.
    Mirrors :func:`test_keep_false_3d_fully_inside_leaves_interior_void`
    but uses the shapely/PolyPrism entry path users actually write in the
    wild.
    """
    inner = PolyPrism(
        polygons=shapely.box(3, 3, 7, 7),
        buffers={3.0: 0.0, 7.0: 0.0},
        physical_name="void",
        mesh_order=1,
        mesh_bool=False,
    )
    outer = PolyPrism(
        polygons=shapely.box(0, 0, 10, 10),
        buffers={0.0: 0.0, 10.0: 0.0},
        physical_name="outer",
        mesh_order=2,
    )
    occ_ents = cad_occ([inner, outer])
    mm = ModelManager(filename="test_xao_keep_false_polyprism_void")
    try:
        mm.load_occ_entities(occ_ents)
        assert len(gmsh.model.getEntities(3)) == 1
        (vol_tag,) = [t for _, t in gmsh.model.getEntities(3)]
        assert abs(gmsh.model.occ.getMass(3, vol_tag) - (1000 - 64)) < 1e-6

        name_to_tags = {
            gmsh.model.getPhysicalName(d, t): list(
                gmsh.model.getEntitiesForPhysicalGroup(d, t)
            )
            for d, t in gmsh.model.getPhysicalGroups(2)
        }
        assert len(name_to_tags["outer___None"]) == 6
        interface_key = next(
            k for k in ("void___outer", "outer___void") if k in name_to_tags
        )
        assert len(name_to_tags[interface_key]) == 6
    finally:
        mm.finalize()


def test_keep_false_lower_dim_cut_surface_is_tagged():
    """A keep=False 2D cut inside a kept 3D box must tag the resulting face.

    The helper has higher priority (mesh_order=1) than the box (mesh_order=2),
    so it wins the overlap during ``cad_occ``'s all-fragment pass. The box's
    solid gets split into two sub-solids whose shared face is the cut plane.
    That face must carry the helper's physical_name ("cut") even though the
    helper itself has keep=False -- it's the reason a user adds a
    keep=False lower-dim entity in the first place.
    """
    box = PolyPrism(
        polygons=shapely.box(0, 0, 1, 1),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="box",
        mesh_order=2,
    )
    cut = PolySurface(
        polygons=shapely.box(0, 0, 1, 1),
        physical_name="cut",
        mesh_order=1,
        mesh_bool=False,
        translation=(0.0, 0.0, 0.5),
    )
    occ_ents = cad_occ([cut, box])
    mm = ModelManager(filename="test_xao_keep_false_cut")
    try:
        mm.load_occ_entities(occ_ents)
        # Box split in two.
        assert len(gmsh.model.getEntities(3)) == 2
        # "cut" is tagged at dim=2 with exactly one face (the shared cut plane).
        surf_groups = gmsh.model.getPhysicalGroups(2)
        name_to_tags = {
            gmsh.model.getPhysicalName(d, t): list(
                gmsh.model.getEntitiesForPhysicalGroup(d, t)
            )
            for d, t in surf_groups
        }
        assert "cut" in name_to_tags
        assert len(name_to_tags["cut"]) == 1
        cut_tag = name_to_tags["cut"][0]
        # The cut face is excluded from the box's exterior (box___None).
        assert cut_tag not in name_to_tags.get("box___None", [])
    finally:
        mm.finalize()
