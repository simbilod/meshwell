"""Post-fragment sanity checks for near-coincident face duplicates.

The OCC bridge assumes ``BOPAlgo_Builder`` fuses shared boundaries into one
TShape per unique face. When it doesn't — because of sub-fuzzy geometric
drift, opposite wire windings, or a cache miss on the instantiation side —
two face TShapes survive at the same geometric location and reach tetgen
as duplicate facets. Tetgen emits ``found two self-intersecting facets,
dihedral angle 0`` or ``could not recover boundary mesh`` at that point
and users are left chasing a meshing-side symptom for a CAD-side bug.

This module walks the fragmented entity compound and reports near-coincident
but TShape-distinct faces before they reach gmsh. It's the CAD-stage
equivalent of a sanity assertion — cheap, and it converts a cryptic mesher
crash into a named diagnostic pointing at the two owning physical names.

Matching bucket: ``(centroid_xyz_rounded, area_rounded)``. Tolerance
defaults to ``10 * point_tolerance`` so it flags legitimate drift and
not just exact matches; tighten if you have legitimate sub-feature-scale
geometry.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import floor, log10
from typing import TYPE_CHECKING

from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
from OCP.TopAbs import TopAbs_FACE
from OCP.TopExp import TopExp_Explorer
from OCP.TopTools import TopTools_ShapeMapHasher

if TYPE_CHECKING:
    from meshwell.cad_occ import OCCLabeledEntity

_HASHER = TopTools_ShapeMapHasher()


@dataclass
class CoincidentFaceGroup:
    """One bucket of near-coincident faces that don't share a TShape.

    Attributes:
        centroid: Quantized centroid (x, y, z) of the group.
        area: Quantized face area.
        tshape_hashes: The distinct TShape hashes in the bucket.
        orientations: Orientation flag per tshape (same indexing as hashes).
            ``0`` = FORWARD, ``1`` = REVERSED, ``2`` = INTERNAL, ``3`` =
            EXTERNAL. Two faces at the same centroid+area with opposite
            orientations usually mean opposite wire windings upstream.
        owners: Physical-name tuples, one per distinct TShape, describing
            which entities produced each face.
    """

    centroid: tuple[float, float, float]
    area: float
    tshape_hashes: list[int]
    orientations: list[int]
    owners: list[tuple[str, ...]]


class CoincidentFacesError(RuntimeError):
    """Raised when the post-fragment audit finds near-coincident duplicates.

    The message lists every offending group so a user can see *which*
    entity pair produced the duplicate without having to walk the
    BREP by hand.
    """


def audit_fragment_faces(
    entities: list[OCCLabeledEntity],
    point_tolerance: float = 1e-3,
    tolerance_multiplier: float = 10.0,
) -> list[CoincidentFaceGroup]:
    """Return groups of near-coincident faces with distinct TShapes.

    Args:
        entities: Entities returned by ``CAD_OCC._fragment_all``.
        point_tolerance: Per-axis quantization tolerance. Usually matches
            the value passed to ``cad_occ``.
        tolerance_multiplier: Audit-side multiplier on ``point_tolerance``;
            two faces whose centroid + area agree to this product land
            in the same bucket. Defaults to ``10``.

    Returns:
        List of :class:`CoincidentFaceGroup` — one per bucket with >1
        distinct TShape. Empty list means no duplicates were detected.
    """
    tol = point_tolerance * tolerance_multiplier
    ndigits = max(0, int(-floor(log10(tol))) if tol > 0 else 0)

    # Bucket faces by quantized (centroid, area). Per-bucket we record the
    # distinct TShape hashes, their orientations, and their owning entity
    # physical names.
    buckets: dict[
        tuple[float, float, float, float], dict[int, tuple[int, set[tuple[str, ...]]]]
    ] = defaultdict(dict)

    for ent in entities:
        for shape in ent.shapes:
            exp = TopExp_Explorer(shape, TopAbs_FACE)
            while exp.More():
                face = exp.Current()
                props = GProp_GProps()
                BRepGProp.SurfaceProperties_s(face, props)
                c = props.CentreOfMass()
                area = props.Mass()
                key = (
                    round(float(c.X()), ndigits),
                    round(float(c.Y()), ndigits),
                    round(float(c.Z()), ndigits),
                    round(float(area), ndigits),
                )
                tshape = _HASHER(face)
                orientation = int(face.Orientation())
                entry = buckets[key].get(tshape)
                if entry is None:
                    buckets[key][tshape] = (orientation, {tuple(ent.physical_name)})
                else:
                    entry[1].add(tuple(ent.physical_name))
                exp.Next()

    groups: list[CoincidentFaceGroup] = []
    for key, by_tshape in buckets.items():
        if len(by_tshape) < 2:
            continue
        hashes = list(by_tshape.keys())
        orientations = [by_tshape[h][0] for h in hashes]
        owners = [
            tuple(sorted(o for owner in by_tshape[h][1] for o in owner)) for h in hashes
        ]
        groups.append(
            CoincidentFaceGroup(
                centroid=(key[0], key[1], key[2]),
                area=key[3],
                tshape_hashes=hashes,
                orientations=orientations,
                owners=owners,
            )
        )
    return groups


def format_coincident_groups(groups: list[CoincidentFaceGroup]) -> str:
    """Render an audit report as a human-readable diagnostic string."""
    lines = [f"{len(groups)} near-coincident face group(s) detected:"]
    orientation_names = {0: "FORWARD", 1: "REVERSED", 2: "INTERNAL", 3: "EXTERNAL"}
    for i, g in enumerate(groups, start=1):
        lines.append(
            f"  [{i}] centroid={g.centroid} area={g.area} "
            f"({len(g.tshape_hashes)} distinct TShapes)"
        )
        for h, o, owners in zip(g.tshape_hashes, g.orientations, g.owners):
            lines.append(
                f"      TShape={h} orientation={orientation_names.get(o, o)} "
                f"owners={owners}"
            )
        # Orientation-mismatch hint: if the bucket has exactly two TShapes
        # with opposite orientations, upstream wires were likely wound the
        # wrong way around the shared face.
        if len(g.orientations) == 2 and set(g.orientations) == {0, 1}:
            lines.append(
                "      hint: opposite orientations suggest mismatched wire "
                "winding between the two entities"
            )
    return "\n".join(lines)
