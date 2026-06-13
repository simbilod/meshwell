"""Focused test for the AABB fallback path in occ_xao_writer.

Scene: a structured cohort + an unstructured neighbour at z=0. The
shared horizontal face often has different TShape IDs on the two
sides (BOP fragmenting), so the TShape-identity path returns empty
and the AABB-proximity fallback rescues the match.

This test pins the fallback's *behaviour* (correctness of the
matched physical groups), enabling the inner-loop refactor in the
next task to be verified without regressions.
"""
import sys
from pathlib import Path

import gmsh
from shapely.geometry import Polygon

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from meshwell.cad_occ import cad_occ
from meshwell.occ_xao_writer import write_xao
from meshwell.polyprism import PolyPrism
from meshwell.structured.pipeline import structured_post_pass, structured_pre_pass


def _rect(x1, y1, x2, y2):
    return Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])


def test_aabb_fallback_finds_shared_horizontal_interface(tmp_path):
    """Cohort + unstructured base meet at z=0.

    The shared face must appear in the bg___base interface group,
    even though BOP may have produced separate TShapes for the two
    sides.
    """
    bg = PolyPrism(
        _rect(-5, -5, 5, 5),
        {0.0: 0.0, 1.0: 0.0},
        physical_name="bg",
        structured=True,
        mesh_order=2.0,
    )
    base = PolyPrism(
        _rect(-10, -10, 10, 10),
        {-2.0: 0.0, 0.0: 0.0},
        physical_name="base",
        mesh_order=3.0,
    )

    state = structured_pre_pass([bg, base], point_tolerance=1e-3)
    occ_entities = cad_occ(state.entities_out, prepared=True)
    final = structured_post_pass(occ_entities, state)

    xao = tmp_path / "scene.xao"
    write_xao(final, xao)

    # Snapshot physical groups from the XAO
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("aabb_test")
        gmsh.merge(str(xao))
        gmsh.model.occ.synchronize()
        groups = {
            gmsh.model.getPhysicalName(d, t): set(
                gmsh.model.getEntitiesForPhysicalGroup(d, t)
            )
            for d, t in gmsh.model.getPhysicalGroups()
        }
    finally:
        gmsh.finalize()

    # bg___base interface (either ordering) must exist with at least 1 face.
    iface_name = next((n for n in groups if n in {"bg___base", "base___bg"}), None)
    assert (
        iface_name is not None
    ), f"expected bg___base interface in groups: {sorted(groups)}"
    assert (
        len(groups[iface_name]) >= 1
    ), f"interface group {iface_name} should contain >= 1 face"


def test_aabb_match_face_excluded_from_neighbour_none_group(tmp_path):
    """Regression: AABB-matched faces excluded from both ___None groups.

    Bug history: when the fallback matched sid1 (cohort side) with
    sid2 (neighbour side) where sid1 != sid2 (because BOP produced
    distinct TShapes), the writer added sid1 to both sides'
    ``entity_interface_ids``. But the neighbour's boundary IDs
    contain sid2, not sid1 — so the exclusion step did not remove
    the matched face, and it appeared in BOTH ``bg___base`` and
    ``base___None``. Faces in the interface MUST NOT appear in the
    neighbour's ``___None`` group.
    """
    bg = PolyPrism(
        _rect(-5, -5, 5, 5),
        {0.0: 0.0, 1.0: 0.0},
        physical_name="bg",
        structured=True,
        mesh_order=2.0,
    )
    base = PolyPrism(
        _rect(-10, -10, 10, 10),
        {-2.0: 0.0, 0.0: 0.0},
        physical_name="base",
        mesh_order=3.0,
    )

    state = structured_pre_pass([bg, base], point_tolerance=1e-3)
    occ_entities = cad_occ(state.entities_out, prepared=True)
    final = structured_post_pass(occ_entities, state)

    xao = tmp_path / "scene.xao"
    write_xao(final, xao)

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("aabb_none_test")
        gmsh.merge(str(xao))
        gmsh.model.occ.synchronize()
        groups = {
            gmsh.model.getPhysicalName(d, t): set(
                gmsh.model.getEntitiesForPhysicalGroup(d, t)
            )
            for d, t in gmsh.model.getPhysicalGroups()
        }
    finally:
        gmsh.finalize()

    iface = groups.get("bg___base", set()) | groups.get("base___bg", set())
    base_none = groups.get("base___None", set())
    bg_none = groups.get("bg___None", set())

    overlap_base = iface & base_none
    overlap_bg = iface & bg_none
    assert (
        not overlap_base
    ), f"bg___base interface faces appear in base___None: {overlap_base}"
    assert (
        not overlap_bg
    ), f"bg___base interface faces appear in bg___None: {overlap_bg}"
