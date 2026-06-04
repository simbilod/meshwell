"""Cohort-adjacent unstructured PolyPrism.

Produced by ``decompose_cohorts`` when a user-input PolyPrism touches a
cohort z-plane. Owns the cohort-adjacency state as typed fields and
(starting from Task 7) overrides ``instanciate_occ`` to build a custom
shell that shares ``TopoDS_Face`` TShapes with the cohort by reference.
"""
from __future__ import annotations

from copy import copy
from typing import ClassVar

from shapely.geometry import MultiPolygon, Polygon

from meshwell.polyprism import PolyPrism


class CohortNeighbourUnstructured(PolyPrism):
    """A PolyPrism that touches at least one cohort z-plane.

    Holds the cohort-adjacency state (`cohort_adjacency`, `tile_polygons`)
    as typed instance fields. Until Task 7, inherits `instanciate_occ`
    from PolyPrism unchanged.

    Construction: produced only by ``CohortNeighbourUnstructured.from_polyprism``.
    Users should not construct directly — the structured pipeline upgrades
    a user PolyPrism into this subclass during ``decompose_cohorts`` when
    adjacency is detected.
    """

    # Per-process registry of cohort_index -> EdgeRegistry. Populated by
    # the orchestrator before cad_occ() runs; cleared after. Moved here
    # from PolyPrism so PolyPrism stays free of cohort coupling.
    _cohort_edge_registries: ClassVar[dict] = {}

    @classmethod
    def _set_cohort_edge_registries(cls, registries):
        cls._cohort_edge_registries = dict(registries) if registries else {}

    # Same for FaceRegistry.
    _cohort_face_registries: ClassVar[dict] = {}

    @classmethod
    def _set_cohort_face_registries(cls, registries):
        cls._cohort_face_registries = dict(registries) if registries else {}

    @classmethod
    def from_polyprism(
        cls,
        original: PolyPrism,
        cohort_adjacency: list[tuple[int, float]],
        tile_polygons: tuple[Polygon, ...],
    ) -> "CohortNeighbourUnstructured":
        """Upgrade a user PolyPrism into a CohortNeighbourUnstructured.

        Shallow-copies the original, replaces ``polygons`` with the
        arrangement tile MultiPolygon (or a single Polygon when there's
        exactly one tile), and attaches the cohort-adjacency state.
        """
        # Shallow copy preserves __init__-time computed attributes
        # (e.g. buffers, zmin, zmax, extrude) without re-running the
        # buffer/snap path.
        nbr = copy(original)
        nbr.__class__ = cls
        if len(tile_polygons) == 1:
            nbr.polygons = tile_polygons[0]
        else:
            nbr.polygons = MultiPolygon(list(tile_polygons))
        nbr.cohort_adjacency = list(cohort_adjacency)
        nbr.tile_polygons = tuple(tile_polygons)
        return nbr
