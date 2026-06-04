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

    def instanciate_occ(self):
        """Build OCC volume via cohort-shell construction.

        Looks up the cohort whose touched plane matches our zmin or zmax,
        then delegates to ``build_neighbour_shell``. The latter constructs
        per-tile prisms via ``BRepPrimAPI_MakePrism`` on the cached cohort
        top face for each tile (preserving the TShape) and Fuses them when
        there are multiple tiles. The cohort-cached top face survives by
        TShape identity into the result, so the cohort sub-piece's bot/top
        face and this neighbour's prism share one ``TopoDS_Face`` and BOP
        does not re-fragment it.

        When no cohort registry is available (e.g. in isolated unit tests
        that do not install the registries), falls back to PolyPrism's
        legacy MakePrism + Fuse path.
        """
        from meshwell.structured.cohort_neighbour_shell import (
            build_neighbour_shell,
        )

        cls = type(self)
        # Pick the cohort whose touched plane matches our zmin or zmax.
        z_touched: float | None = None
        z_far: float | None = None
        shared_ci: int | None = None
        for ci, z_shared in self.cohort_adjacency:
            if cls._cohort_face_registries.get(ci) is None:
                continue
            if z_shared == self.zmin:
                shared_ci = ci
                z_touched = self.zmin
                z_far = self.zmax
                break
            if z_shared == self.zmax:
                shared_ci = ci
                z_touched = self.zmax
                z_far = self.zmin
                break

        if shared_ci is None:
            # No cohort registry available — fall back to PolyPrism's
            # legacy path. Can happen during isolated unit tests that
            # don't install the registries.
            return super().instanciate_occ()

        face_registry = cls._cohort_face_registries[shared_ci]
        edge_registry = cls._cohort_edge_registries[shared_ci]
        solid = build_neighbour_shell(
            tiles=self.tile_polygons,
            z_touched=z_touched,
            z_far=z_far,
            face_registry=face_registry,
            edge_registry=edge_registry,
            identify_arcs=self.identify_arcs,
            min_arc_points=self.min_arc_points,
            arc_tolerance=self.arc_tolerance,
        )
        rotation_point = self._get_rotation_point(self.polygons)
        return self._apply_transformation_occ(solid, rotation_point)
