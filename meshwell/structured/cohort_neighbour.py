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
        """Build OCC volume with cohort-cached touched-plane face.

        For each tile polygon, look up the cached cohort TopoDS_Face for
        the touched z-plane and use it as the prism's base. Build via
        BRepPrimAPI_MakePrism (preserves input face TShape). Combine
        multi-tile prisms via BRepAlgoAPI_Fuse — Task 8 replaces this
        with custom shell construction.
        """
        from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
        from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
        from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
        from OCP.gp import gp_Vec
        from shapely.geometry.polygon import orient

        cls = type(self)
        adjacency = self.cohort_adjacency
        polys = (
            self.polygons.geoms if hasattr(self.polygons, "geoms") else [self.polygons]
        )
        volumes = []
        for poly in polys:
            if poly.interiors:
                poly = orient(poly, sign=1.0)

            shared_ci: int | None = None
            build_z = self.zmin
            build_vec = gp_Vec(0, 0, self.zmax - self.zmin)
            for ci, z_shared in adjacency:
                if cls._cohort_edge_registries.get(ci) is None:
                    continue
                if z_shared == self.zmin:
                    shared_ci = ci
                    build_z = self.zmin
                    build_vec = gp_Vec(0, 0, self.zmax - self.zmin)
                    break
                if z_shared == self.zmax:
                    shared_ci = ci
                    build_z = self.zmax
                    build_vec = gp_Vec(0, 0, self.zmin - self.zmax)
                    break

            shared_face_registry = (
                cls._cohort_face_registries.get(shared_ci)
                if shared_ci is not None
                else None
            )
            shared_edge_registry = (
                cls._cohort_edge_registries.get(shared_ci)
                if shared_ci is not None
                else None
            )

            if shared_face_registry is not None:
                face = shared_face_registry.face_xy(
                    poly,
                    build_z,
                    self.identify_arcs,
                    self.min_arc_points,
                    self.arc_tolerance,
                )
            else:
                exterior_vertices = [(x, y, build_z) for x, y in poly.exterior.coords]
                outer_wire = self._make_occ_wire_from_vertices(
                    exterior_vertices,
                    identify_arcs=self.identify_arcs,
                    min_arc_points=self.min_arc_points,
                    arc_tolerance=self.arc_tolerance,
                    edge_registry=shared_edge_registry,
                )
                mf = BRepBuilderAPI_MakeFace(outer_wire)
                for interior in poly.interiors:
                    hole_vertices = [(x, y, build_z) for x, y in interior.coords]
                    hole_wire = self._make_occ_wire_from_vertices(
                        hole_vertices,
                        identify_arcs=self.identify_arcs,
                        min_arc_points=self.min_arc_points,
                        arc_tolerance=self.arc_tolerance,
                        edge_registry=shared_edge_registry,
                    )
                    mf.Add(hole_wire)
                face = mf.Face()

            volumes.append(BRepPrimAPI_MakePrism(face, build_vec).Shape())

        if not volumes:
            return None

        result = volumes[0]
        for v in volumes[1:]:
            fuse_api = BRepAlgoAPI_Fuse(result, v)
            fuse_api.Build()
            result = fuse_api.Shape()

        rotation_point = self._get_rotation_point(self.polygons)
        return self._apply_transformation_occ(result, rotation_point)
