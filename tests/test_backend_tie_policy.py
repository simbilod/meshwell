"""gmsh backend must match OCC: no tie cuts, no A___A groups."""
import gmsh
from shapely.geometry import Polygon

from meshwell.cad_gmsh import CAD_GMSH, strip_suffix
from meshwell.polysurface import PolySurface


def _overlapping_pair(mesh_order_b):
    a = PolySurface(
        polygons=Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
        physical_name="a",
        mesh_order=1,
    )
    b = PolySurface(
        polygons=Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
        physical_name="b",
        mesh_order=mesh_order_b,
    )
    return [a, b]


def _run(entities):
    proc = CAD_GMSH()
    try:
        labeled = proc.process_entities(entities)
        groups = {
            gmsh.model.getPhysicalName(dim, tag)
            for dim, tag in gmsh.model.getPhysicalGroups()
        }
        owner_areas = {
            tuple(strip_suffix(n) for n in ent.physical_name): sum(
                gmsh.model.occ.getMass(dim, tag) for dim, tag in ent.dimtags
            )
            for ent in labeled
            if ent.dimtags
        }
        return groups, owner_areas
    finally:
        proc.model_manager.finalize()


def test_tie_overlap_earlier_entity_wins():
    _, areas = _run(_overlapping_pair(mesh_order_b=1))  # tie
    # earlier-declared entity keeps the 1x1 overlap: a=4.0, b=4.0-1.0=3.0.
    # Tolerance accommodates the backend's ~1e-5 outward perturbation
    # buffer (areas inflate to ~4.00008 / ~3.00004); still far below the
    # 1.0 gap that a reversed ownership (a=3, b=4) would produce.
    assert abs(areas[("a",)] - 4.0) < 1e-3
    assert abs(areas[("b",)] - 3.0) < 1e-3


def test_tie_and_nontie_ownership_agree():
    _, tie_areas = _run(_overlapping_pair(mesh_order_b=1))
    _, cut_areas = _run(_overlapping_pair(mesh_order_b=2))
    for k in tie_areas:
        assert abs(tie_areas[k] - cut_areas[k]) < 1e-6


def test_no_same_material_interface_group():
    a = PolySurface(
        polygons=Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        physical_name="m",
        mesh_order=1,
    )
    b = PolySurface(
        polygons=Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
        physical_name="m",
        mesh_order=1,
    )
    groups, _ = _run([a, b])
    assert "m___m" not in groups
    # the shared edge must also not appear in m's exterior boundary
    assert "m___None" in groups
