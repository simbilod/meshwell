"""Tests for the post-fragment near-coincident face audit."""
from __future__ import annotations

import numpy as np
import pytest
import shapely

from meshwell.cad_occ import OCCLabeledEntity, cad_occ
from meshwell.occ_fragment_audit import (
    CoincidentFacesError,
    audit_fragment_faces,
    format_coincident_groups,
)
from meshwell.polyprism import PolyPrism
from meshwell.polysurface import PolySurface


def _arc_polygon(n: int = 30, r: float = 1.0) -> shapely.Polygon:
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return shapely.Polygon([(r * np.cos(t), r * np.sin(t)) for t in theta])


def test_clean_scene_reports_no_coincidences():
    """Clean adjacent-prism scene must report no coincident faces.

    The face cache ensures BOPAlgo fuses the shared boundary cleanly.
    """
    left = shapely.Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
    right = shapely.affinity.translate(left, xoff=5)
    buffers = {0.0: 0.0, 1.0: 0.0}
    ents = cad_occ(
        [
            PolyPrism(polygons=left, buffers=buffers, physical_name="L"),
            PolyPrism(polygons=right, buffers=buffers, physical_name="R"),
        ]
    )
    assert audit_fragment_faces(ents, point_tolerance=1e-3) == []


def test_clean_arc_scene_reports_no_coincidences():
    """Same scene, but with an arc-heavy circular PolySurface pair."""
    poly = _arc_polygon(n=40, r=1.0)
    ents = cad_occ(
        [
            PolySurface(polygons=poly, physical_name="outer", mesh_order=2),
            PolySurface(
                polygons=shapely.affinity.scale(poly, xfact=0.4, yfact=0.4),
                physical_name="inner",
                mesh_order=1,
            ),
        ]
    )
    assert audit_fragment_faces(ents, point_tolerance=1e-3) == []


def _box_entity(origin: tuple[float, float, float], size: float, name: str):
    """Build an OCCLabeledEntity directly from a BRepPrimAPI_MakeBox.

    No ``OCCGeometryCache`` is involved, so two boxes at the same origin
    produce *distinct* TShapes for every face -- the exact pathology the
    audit must catch.
    """
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.gp import gp_Pnt

    shape = BRepPrimAPI_MakeBox(gp_Pnt(*origin), size, size, size).Shape()
    return OCCLabeledEntity(
        shapes=[shape],
        physical_name=(name,),
        index=0,
        keep=True,
        dim=3,
        mesh_order=1,
    )


def test_audit_detects_overlapping_duplicate_boxes():
    """Two overlapping cache-free boxes produce six coincident-face groups.

    Each face pair is geometrically identical but TShape-distinct.
    """
    a = _box_entity((0, 0, 0), 1.0, "A")
    b = _box_entity((0, 0, 0), 1.0, "B")
    groups = audit_fragment_faces([a, b], point_tolerance=1e-3)
    assert len(groups) == 6
    # Each group must include both owners.
    for g in groups:
        owners_flat = {n for owner in g.owners for n in owner}
        assert owners_flat == {"A", "B"}
        assert len(g.tshape_hashes) == 2


def test_audit_detects_shared_boundary_face_when_cache_bypassed():
    """Two adjacent cache-free boxes share one face — only that face duplicates.

    The other faces are geometrically distinct between the two boxes.
    """
    a = _box_entity((0, 0, 0), 1.0, "A")
    b = _box_entity((1, 0, 0), 1.0, "B")
    groups = audit_fragment_faces([a, b], point_tolerance=1e-3)
    # Exactly one bucket: the shared face at x=1.
    assert len(groups) == 1
    g = groups[0]
    assert g.centroid == (1.0, 0.5, 0.5)
    owners_flat = {n for owner in g.owners for n in owner}
    assert owners_flat == {"A", "B"}


def test_validate_fragment_raises_on_duplicates(monkeypatch):
    """``validate_fragment=True`` raises CoincidentFacesError on duplicates.

    A real BOPAlgo pass on identical boxes would fuse the overlap and
    produce no duplicates, so we stub ``_fragment_all`` to a no-op --
    the stub preserves the two distinct-TShape boxes that the audit
    then catches.
    """
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.gp import gp_Pnt

    from meshwell.cad_occ import CAD_OCC
    from meshwell.occ_entity import OCC_entity

    def _box():
        return BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 1, 1, 1).Shape()

    def _noop_fragment(_self, ents, progress_bars=False):  # noqa: ARG001
        return ents

    monkeypatch.setattr(CAD_OCC, "_fragment_all", _noop_fragment)

    entities = [
        OCC_entity(occ_function=_box, physical_name="A", mesh_order=1),
        OCC_entity(occ_function=_box, physical_name="B", mesh_order=2),
    ]
    with pytest.raises(CoincidentFacesError) as exc:
        cad_occ(entities, validate_fragment=True)
    msg = str(exc.value)
    assert "near-coincident face group(s) detected" in msg
    assert "A" in msg
    assert "B" in msg


def test_validate_fragment_passes_on_clean_scene():
    """``validate_fragment=True`` on a clean scene must be a no-op."""
    left = shapely.Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
    right = shapely.affinity.translate(left, xoff=5)
    buffers = {0.0: 0.0, 1.0: 0.0}
    ents = cad_occ(
        [
            PolyPrism(polygons=left, buffers=buffers, physical_name="L"),
            PolyPrism(polygons=right, buffers=buffers, physical_name="R"),
        ],
        validate_fragment=True,
    )
    assert len(ents) == 2


def test_format_coincident_groups_flags_orientation_mismatch():
    """Opposite-orientation duplicates get a wire-winding hint in the report."""
    a = _box_entity((0, 0, 0), 1.0, "A")
    # Build B's shape with all faces reversed so orientations differ.
    # The audit keys on TShape (not orientation), so this still bucket-
    # collides with A at the six face centroids.
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.gp import gp_Pnt

    reversed_box = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 1, 1, 1).Shape().Reversed()
    b = OCCLabeledEntity(
        shapes=[reversed_box],
        physical_name=("B",),
        index=1,
        keep=True,
        dim=3,
        mesh_order=1,
    )
    groups = audit_fragment_faces([a, b], point_tolerance=1e-3)
    assert groups
    report = format_coincident_groups(groups)
    assert "mismatched wire winding" in report
