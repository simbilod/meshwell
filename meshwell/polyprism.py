"""Gmsh polyprism definitions."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import gmsh
import shapely
from shapely.geometry import MultiPolygon, Polygon

from meshwell.geometry_entity import GeometryEntity
from meshwell.structured.exceptions import StructuredExtrudeRequiredError
from meshwell.validation import format_physical_name

if TYPE_CHECKING:
    from OCP.TopoDS import TopoDS_Shape


class PolyPrism(GeometryEntity):
    """Creates a bottom-up GMSH "prism" formed by a polygon associated with (optional) z-dependent grow/shrink operations.

    Attributes:
        polygons: list of shapely (Multi)Polygon
        buffers: dict of {z: buffer} used to shrink/grow base polygons at specified z-values
        physical_name: name of the physical this entity will belong to
        mesh_order: priority of the entity if it overlaps with others (lower numbers override higher numbers)
        mesh_bool: if True, entity will be meshed; if not, will not be meshed (useful to tag boundaries)

    """

    # Per-process registry of cohort-index -> EdgeRegistry, populated by
    # the cad_occ entry point in the structured pipeline. Cleared after
    # each cad_occ() invocation so cross-test contamination cannot occur.
    _cohort_edge_registries: ClassVar[dict] = {}

    @classmethod
    def _set_cohort_edge_registries(cls, registries):
        """Install a mapping from cohort_index -> EdgeRegistry.

        Called by structured_pre_pass's caller (cad_occ wrapper) before
        building polyprism OCC representations. Pass an empty dict to
        clear.
        """
        cls._cohort_edge_registries = dict(registries) if registries else {}

    # Per-process registry of cohort-index -> FaceRegistry, populated by
    # the cad_occ entry point in the structured pipeline. Cleared after
    # each cad_occ() invocation so cross-test contamination cannot occur.
    _cohort_face_registries: ClassVar[dict] = {}

    @classmethod
    def _set_cohort_face_registries(cls, registries):
        """Install a mapping from cohort_index -> FaceRegistry.

        Called by structured_pre_pass's caller (cad_occ wrapper) before
        building polyprism OCC representations. Pass an empty dict to
        clear.
        """
        cls._cohort_face_registries = dict(registries) if registries else {}

    def __init__(
        self,
        polygons: Polygon | list[Polygon] | MultiPolygon | list[MultiPolygon],
        buffers: dict[float, float],
        physical_name: str | tuple[str, ...] | None = None,
        mesh_order: float | None = None,
        mesh_bool: bool = True,
        additive: bool = False,
        subdivision: tuple[int, int, int] | None = None,
        point_tolerance: float = 1e-3,
        identify_arcs: bool | None = None,
        min_arc_points: int = 5,
        arc_tolerance: float = 1e-3,
        translation: tuple[float, float, float] | None = None,
        rotation_axis: tuple[float, float, float] | None = None,
        rotation_point: tuple[float, float, float] | None = None,
        rotation_angle: float = 0.0,
        structured: bool = False,
    ):
        # Initialize parent class with point tracking and transformation parameters
        super().__init__(
            point_tolerance=point_tolerance,
            translation=translation,
            rotation_axis=rotation_axis,
            rotation_point=rotation_point,
            rotation_angle=rotation_angle,
        )

        # Parse buffers or prepare extrusion
        if point_tolerance > 0:
            # Snap input polygons to user grid before storing / buffering.
            if isinstance(polygons, list):
                polygons = [
                    shapely.set_precision(
                        p, grid_size=point_tolerance, mode="pointwise"
                    )
                    for p in polygons
                ]
            else:
                polygons = shapely.set_precision(
                    polygons, grid_size=point_tolerance, mode="pointwise"
                )
        self.polygons = polygons
        if all(buffer == 0 for buffer in buffers.values()):
            self.extrude = True
            self.zmin, self.zmax = min(buffers.keys()), max(buffers.keys())
        else:
            self.extrude = False
            self.buffered_polygons: list[
                tuple[float, Polygon]
            ] = self._get_buffered_polygons(polygons, buffers)

        # Structured validation and identify_arcs resolution
        # entity_index=-1 is a placeholder: the real index is unknown at
        # construction time; the cad_occ pre-pass will re-raise with the
        # correct entity index once it has scanned the full entity list.
        if structured and not self.extrude:
            raise StructuredExtrudeRequiredError(entity_index=-1)
        # identify_arcs must be EXPLICITLY set to True; structured=True alone
        # does NOT imply arc detection. Automatically enabling it caused
        # polygons with all vertices co-circular (e.g. rectangles, whose corners
        # lie on the circumscribed circle) to be mis-built as disk-shaped solids.
        if identify_arcs is None:
            identify_arcs = False
        self.structured = structured
        self.identify_arcs = identify_arcs

        if self.identify_arcs and not self.extrude:
            raise NotImplementedError(
                "Arc identification is currently only supported for PolyPrism when extrude=True."
            )

        # Store other attributes
        self.buffers = buffers
        self.mesh_order = mesh_order
        self.additive = additive
        self.dimension = 3
        self.subdivision = subdivision
        self.min_arc_points = min_arc_points
        self.arc_tolerance = arc_tolerance

        # Format physical name
        self.physical_name = format_physical_name(physical_name)
        self.mesh_bool = mesh_bool
        self.additive = additive

    def _create_volumes_directly(self) -> list[int]:
        """Create GMSH volumes directly without using CAD class methods."""
        if self.extrude:
            surfaces = self._create_surfaces_with_holes_at_z(self.polygons, self.zmin)
            surface_dimtags = [(2, surface) for surface in surfaces]
            entities = gmsh.model.occ.extrude(
                surface_dimtags, 0, 0, self.zmax - self.zmin
            )
            volumes = [tag for dim, tag in entities if dim == 3]
        else:
            volumes = [
                tag
                for entry in self.buffered_polygons
                if (tag := self._create_volume_with_holes_directly(entry)) != 0
            ]

        # Fuse multiple volumes if needed
        if len(volumes) <= 1:
            return volumes

        fused_dimtags = gmsh.model.occ.fuse(
            [(3, volumes[0])],
            [(3, tag) for tag in volumes[1:]],
            removeObject=True,
            removeTool=True,
        )[0]
        # Clear caches after boolean operations that may invalidate geometry IDs
        self._clear_caches()
        return [tag for dim, tag in fused_dimtags]

    def _get_buffered_polygons(
        self, polygons: list[Polygon], buffers: dict[float, float]
    ) -> list[tuple[float, Polygon]]:
        """Break up polygons on each layer into lists of (z,polygon) tuples according to buffer entries.

        Arguments (implicit):
            polygons: list of (Multi)Polygons to bufferize
            buffers: {z: buffer} values to apply to the polygons

        Returns:
            buffered_polygons: list of (z, buffered_polygons)

        """
        all_polygons_list = []
        for polygon in polygons.geoms if hasattr(polygons, "geoms") else [polygons]:
            current_buffers = []
            for z, width_buffer in buffers.items():
                current_buffers.append((z, polygon.buffer(width_buffer, join_style=2)))
            all_polygons_list.append(current_buffers)

        return all_polygons_list

    def _create_volume_directly(
        self,
        entry: list[tuple[float, Polygon]],
        exterior: bool = True,
        interior_index: int = 0,
    ) -> int:
        """Create volume directly using GMSH calls."""
        curve_loops = []
        for z, polygon in entry:
            vertices = self.xy_surface_vertices(
                polygon=polygon,
                polygon_z=z,
                exterior=exterior,
                interior_index=interior_index,
            )
            # Create a curve loop for this polygon at this z
            points = self._create_points_from_vertices(vertices)
            lines = []
            for i in range(len(points) - 1):
                line_id = self._add_line_with_cache(points[i], points[i + 1])
                if line_id != 0:
                    lines.append(line_id)
            if lines:
                loop_id = gmsh.model.occ.addCurveLoop(lines)
                curve_loops.append(loop_id)

        if not curve_loops:
            return 0

        # If only one curve loop, we can't build a volume
        if len(curve_loops) < 2:
            return 0

        # Return volume from thru-sections
        try:
            volume_dimtags = gmsh.model.occ.addThruSections(
                curve_loops, makeSolid=True, makeRuled=True
            )
            gmsh.model.occ.synchronize()
            if volume_dimtags and volume_dimtags[0][0] == 3:
                return volume_dimtags[0][1]
        except Exception:
            return 0

        return 0

    def xy_surface_vertices(
        self,
        polygon: Polygon,
        polygon_z: float,
        exterior: bool,
        interior_index: int,
    ) -> list[tuple[float, float, float]]:
        """Draw xy surface."""
        return (
            [(x, y, polygon_z) for x, y in polygon.exterior.coords]
            if exterior
            else [
                (x, y, polygon_z) for x, y in polygon.interiors[interior_index].coords
            ]
        )

    def _create_volume_with_holes_directly(
        self, entry: list[tuple[float, Polygon]]
    ) -> int:
        """Create volume with holes directly using GMSH calls."""
        exterior = self._create_volume_directly(entry, exterior=True)
        if exterior == 0:
            return 0
        interiors = [
            tag
            for interior_index in range(len(entry[0][1].interiors))
            if (
                tag := self._create_volume_directly(
                    entry,
                    exterior=False,
                    interior_index=interior_index,
                )
            )
            != 0
        ]
        if interiors:
            cut_result = gmsh.model.occ.cut(
                [(3, exterior)],
                [(3, interior) for interior in interiors],
                removeObject=True,
                removeTool=True,
            )
            gmsh.model.occ.synchronize()
            # Handle cut result - if cut succeeded, use the result, otherwise check if original still exists
            if cut_result and cut_result[0]:
                exterior = cut_result[0][0][1]  # Parse `outDimTags', `outDimTagsMap'
            else:
                # If cut failed, original might have been removed due to removeObject=True
                existing_3d = {tag for dim, tag in gmsh.model.getEntities(3)}
                if exterior not in existing_3d:
                    exterior = 0
            # Clear caches after boolean operations that may invalidate geometry IDs
            self._clear_caches()
        return exterior

    def _create_surfaces_with_holes_at_z(self, polygons, z) -> list[int]:
        """Create surfaces with holes at given z level directly using GMSH calls."""
        surfaces = []
        for polygon in polygons.geoms if hasattr(polygons, "geoms") else [polygons]:
            # Create outer surface
            exterior_vertices = [(x, y, z) for x, y in polygon.exterior.coords]
            exterior = self._create_surface_from_vertices(
                exterior_vertices,
                identify_arcs=self.identify_arcs,
                min_arc_points=self.min_arc_points,
                arc_tolerance=self.arc_tolerance,
            )
            if exterior == 0:
                continue

            # Create interior surfaces (holes)
            interior_surfaces = []
            for interior in polygon.interiors:
                interior_vertices = [(x, y, z) for x, y in interior.coords]
                interior_surface = self._create_surface_from_vertices(
                    interior_vertices,
                    identify_arcs=self.identify_arcs,
                    min_arc_points=self.min_arc_points,
                    arc_tolerance=self.arc_tolerance,
                )
                if interior_surface != 0:
                    interior_surfaces.append(interior_surface)

            # Cut holes from exterior surface
            for interior_surface in interior_surfaces:
                cut_result = gmsh.model.occ.cut(
                    [(2, exterior)],
                    [(2, interior_surface)],
                    removeObject=True,
                    removeTool=True,
                )
                gmsh.model.occ.synchronize()
                # Handle cut result - if cut succeeded, use the result, otherwise keep original
                if cut_result and cut_result[0]:
                    exterior = cut_result[0][0][1]
                # Clear caches after boolean operations that may invalidate geometry IDs
                self._clear_caches()

            surfaces.append(exterior)
        return surfaces

    def subdivide(self, model, prisms, subdivision):
        """Split the prisms into subprisms according to subdivision."""
        subdivided_prisms = []
        import numpy as np

        global_xmin = np.inf
        global_ymin = np.inf
        global_zmin = np.inf
        global_xmax = -np.inf
        global_ymax = -np.inf
        global_zmax = -np.inf
        for prism in prisms:
            xmin, ymin, zmin, xmax, ymax, zmax = model.occ.getBoundingBox(3, prism)
            if xmin < global_xmin:
                global_xmin = xmin
            if ymin < global_ymin:
                global_ymin = ymin
            if zmin < global_zmin:
                global_zmin = zmin
            if xmax > global_xmax:
                global_xmax = xmax
            if ymax > global_ymax:
                global_ymax = ymax
            if zmax > global_zmax:
                global_zmax = zmax
        dx = (global_xmax - global_xmin) / subdivision[0]
        dy = (global_ymax - global_ymin) / subdivision[1]
        dz = (global_zmax - global_zmin) / subdivision[2]
        prisms_dimtags = {(3, prism) for prism in prisms}
        for x_index in range(subdivision[0]):
            for y_index in range(subdivision[1]):
                for z_index in range(subdivision[2]):
                    tool = model.occ.add_box(
                        global_xmin + x_index * dx,
                        global_ymin + y_index * dy,
                        global_zmin + z_index * dz,
                        dx,
                        dy,
                        dz,
                    )
                    intersection, intersection_map = model.occ.intersect(
                        list(prisms_dimtags),
                        [(3, tool)],
                        removeObject=False,
                        removeTool=True,
                    )
                    prisms_dimtags -= set(intersection_map[0])
                    subdivided_prisms.extend(intersection_map[0] + intersection)
        model.occ.remove(list(prisms_dimtags))
        return subdivided_prisms

    def instanciate(self, cad_model: Any) -> list[tuple[int, int]]:
        """Create GMSH volumes directly without using CAD class methods."""
        prisms = self._create_volumes_directly()
        if self.subdivision is not None:
            prisms = self.subdivide(
                cad_model.model_manager.model, prisms, self.subdivision
            )

        dimtags = [(3, prism) for prism in prisms]
        rotation_point = self._get_rotation_point(self.polygons)
        dimtags = self._apply_transformation_gmsh(dimtags, rotation_point)
        gmsh.model.occ.synchronize()
        return dimtags

    def _create_occ_volume(
        self,
        entry: list[tuple[float, Polygon]],
        exterior: bool = True,
        interior_index: int = 0,
    ) -> "TopoDS_Shape":
        """Loft a tapered solid through the per-z wires of ``entry``.

        Mirrors gmsh's ``addThruSections(makeSolid=True, makeRuled=True)``.
        Each layer in ``entry`` contributes one wire (built with the
        existing :meth:`_make_occ_wire_from_vertices` helper so arcs are
        preserved). ``BRepOffsetAPI_ThruSections`` builds bottom + top +
        lateral surface as a single closed solid -- no manual face
        stitching, no shell sealing, no MakeSolid wrapping.

        All wires must have the same vertex count (true for any
        consistently-buffered polygon, which is the only case meshwell
        currently supports). OCC raises during ``Build()`` if violated.
        """
        from OCP.BRepOffsetAPI import BRepOffsetAPI_ThruSections

        loft = BRepOffsetAPI_ThruSections(True, True)  # isSolid, isRuled
        for z, polygon in entry:
            vertices = self.xy_surface_vertices(
                polygon=polygon,
                polygon_z=z,
                exterior=exterior,
                interior_index=interior_index,
            )
            wire = self._make_occ_wire_from_vertices(
                vertices,
                identify_arcs=self.identify_arcs,
                min_arc_points=self.min_arc_points,
                arc_tolerance=self.arc_tolerance,
            )
            loft.AddWire(wire)
        loft.Build()
        return loft.Shape()

    def plot_decomposition(
        self,
        ax=None,
        line_color: str = "blue",
        arc_color: str = "red",
        show_centers: bool = True,
        **kwargs,
    ):
        """Visualize the decomposition of the base cross-section."""
        # For PolyPrism, we plot the base polygon (buffered_polygons[0] or polygons)
        if self.extrude:
            polygons = (
                self.polygons.geoms
                if hasattr(self.polygons, "geoms")
                else [self.polygons]
            )
            for polygon in polygons:
                vertices = [
                    self._parse_coords(coords) for coords in polygon.exterior.coords
                ]
                ax = super().plot_decomposition(
                    vertices,
                    ax=ax,
                    line_color=line_color,
                    arc_color=arc_color,
                    show_centers=show_centers,
                    identify_arcs=self.identify_arcs,
                    min_arc_points=self.min_arc_points,
                    arc_tolerance=self.arc_tolerance,
                    **kwargs,
                )
                for interior in polygon.interiors:
                    vertices = [
                        self._parse_coords(coords) for coords in interior.coords
                    ]
                    ax = super().plot_decomposition(
                        vertices,
                        ax=ax,
                        line_color=line_color,
                        arc_color=arc_color,
                        show_centers=show_centers,
                        identify_arcs=self.identify_arcs,
                        min_arc_points=self.min_arc_points,
                        arc_tolerance=self.arc_tolerance,
                        **kwargs,
                    )
        else:
            # For buffered polygons, we plot the first layer
            for entry in self.buffered_polygons:
                # entry is list of (z, polygon)
                z, polygon = entry[0]
                vertices = [(x, y, z) for x, y in polygon.exterior.coords]
                ax = super().plot_decomposition(
                    vertices,
                    ax=ax,
                    line_color=line_color,
                    arc_color=arc_color,
                    show_centers=show_centers,
                    identify_arcs=self.identify_arcs,
                    min_arc_points=self.min_arc_points,
                    arc_tolerance=self.arc_tolerance,
                    **kwargs,
                )
                for interior in polygon.interiors:
                    vertices = [(x, y, z) for x, y in interior.coords]
                    ax = super().plot_decomposition(
                        vertices,
                        ax=ax,
                        line_color=line_color,
                        arc_color=arc_color,
                        show_centers=show_centers,
                        identify_arcs=self.identify_arcs,
                        min_arc_points=self.min_arc_points,
                        arc_tolerance=self.arc_tolerance,
                        **kwargs,
                    )
        return ax

    def _create_occ_volume_with_holes(
        self,
        entry: list[tuple[float, Polygon]],
    ) -> TopoDS_Shape:
        """Create OCC volume with holes directly using OCP.

        Non-extrude path uses ``BRepAlgoAPI_Cut`` for the z-varying buffer
        case; the extrude path in :meth:`instanciate_occ` builds the
        prism as a single swept face.
        """
        from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut

        exterior = self._create_occ_volume(entry, exterior=True)
        interior_count = len(entry[0][1].interiors)

        for i in range(interior_count):
            interior = self._create_occ_volume(entry, exterior=False, interior_index=i)
            cut_api = BRepAlgoAPI_Cut(exterior, interior)
            cut_api.Build()
            exterior = cut_api.Shape()

        return exterior

    def instanciate_occ(self) -> TopoDS_Shape:
        """Create OCC volumes directly using OCP."""
        from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
        from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
        from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
        from OCP.gp import gp_Vec
        from shapely.geometry.polygon import orient

        volumes = []
        if self.extrude:
            polys = (
                self.polygons.geoms
                if hasattr(self.polygons, "geoms")
                else [self.polygons]
            )
            for poly in polys:
                # For polygons with holes, canonicalize to OGC convention
                # (CCW exterior + CW interiors) so OCC's face-with-hole
                # construction works regardless of the input's shapely
                # orientation. ``cad_common.prepare_entities`` runs a
                # ``buffer(...).intersection(bbox)`` that silently flips
                # the exterior to CW while leaving interiors CCW; without
                # this orient pass the resulting BRep face has both wires
                # CCW and the prism's volume includes the hole. Skipped
                # when there are no interiors to keep mesh output
                # bit-identical to pre-fix reference files for the common
                # no-hole case.
                if poly.interiors:
                    poly = orient(poly, sign=1.0)

                # Determine whether to use a cohort's EdgeRegistry for
                # this polygon's boundary wire, and at which z to build
                # the polygon face. BRepPrimAPI_MakePrism builds the
                # face at the user-supplied z and extrudes it along the
                # supplied vector. If the cohort touches at our zmin,
                # build at zmin and extrude up: the bot face IS the
                # shared face. If the cohort touches at our zmax, build
                # at zmax and extrude DOWN: the top face IS the shared
                # face. Either way the "user-built" face's edges go
                # through the shared registry.
                adjacency = getattr(self, "_cohort_adjacency", None) or []
                shared_registry = None
                build_z = self.zmin
                build_vec = gp_Vec(0, 0, self.zmax - self.zmin)
                for ci, z_shared in adjacency:
                    reg = PolyPrism._cohort_edge_registries.get(ci)
                    if reg is None:
                        continue
                    if z_shared == self.zmin:
                        shared_registry = reg
                        build_z = self.zmin
                        build_vec = gp_Vec(0, 0, self.zmax - self.zmin)
                        break
                    if z_shared == self.zmax:
                        shared_registry = reg
                        build_z = self.zmax
                        build_vec = gp_Vec(0, 0, self.zmin - self.zmax)
                        break

                exterior_vertices = [(x, y, build_z) for x, y in poly.exterior.coords]
                outer_wire = self._make_occ_wire_from_vertices(
                    exterior_vertices,
                    identify_arcs=self.identify_arcs,
                    min_arc_points=self.min_arc_points,
                    arc_tolerance=self.arc_tolerance,
                    edge_registry=shared_registry,
                )
                mf = BRepBuilderAPI_MakeFace(outer_wire)
                for interior in poly.interiors:
                    hole_vertices = [(x, y, build_z) for x, y in interior.coords]
                    hole_wire = self._make_occ_wire_from_vertices(
                        hole_vertices,
                        identify_arcs=self.identify_arcs,
                        min_arc_points=self.min_arc_points,
                        arc_tolerance=self.arc_tolerance,
                        edge_registry=shared_registry,
                    )
                    mf.Add(hole_wire)
                face = mf.Face()

                volumes.append(BRepPrimAPI_MakePrism(face, build_vec).Shape())
        else:
            volumes.extend(
                [
                    self._create_occ_volume_with_holes(entry)
                    for entry in self.buffered_polygons
                ]
            )

        if not volumes:
            return None

        result = volumes[0]
        for v in volumes[1:]:
            fuse_api = BRepAlgoAPI_Fuse(result, v)
            fuse_api.Build()
            result = fuse_api.Shape()

        rotation_point = self._get_rotation_point(self.polygons)
        return self._apply_transformation_occ(result, rotation_point)

    def to_dict(self) -> dict:
        """Convert entity to dictionary representation.

        Returns:
            Dictionary containing serializable entity data
        """
        import shapely.wkt
        from shapely.geometry import MultiPolygon

        if isinstance(self.polygons, MultiPolygon):
            polygons_wkt = [
                shapely.wkt.dumps(p, rounding_precision=12) for p in self.polygons.geoms
            ]
        elif isinstance(self.polygons, list):
            polygons_wkt = [
                shapely.wkt.dumps(p, rounding_precision=12) for p in self.polygons
            ]
        else:
            polygons_wkt = [shapely.wkt.dumps(self.polygons, rounding_precision=12)]

        return {
            "type": "PolyPrism",
            "polygons_wkt": polygons_wkt,
            "buffers": {str(k): v for k, v in self.buffers.items()},
            "physical_name": self.physical_name,
            "mesh_order": self.mesh_order,
            "mesh_bool": self.mesh_bool,
            "additive": self.additive,
            "point_tolerance": self.point_tolerance,
            "structured": self.structured,
            "identify_arcs": self.identify_arcs,
            "min_arc_points": self.min_arc_points,
            "arc_tolerance": self.arc_tolerance,
            "subdivision": list(self.subdivision) if self.subdivision else None,
            "translation": self.translation,
            "rotation_axis": self.rotation_axis,
            "rotation_point": self.rotation_point,
            "rotation_angle": self.rotation_angle,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PolyPrism":
        """Reconstruct entity from dictionary representation.

        Args:
            data: Dictionary containing entity data

        Returns:
            PolyPrism instance
        """
        import shapely.wkt
        from shapely.geometry import MultiPolygon

        polygons = [shapely.wkt.loads(wkt) for wkt in data["polygons_wkt"]]
        polygons = MultiPolygon(polygons) if len(polygons) > 1 else polygons[0]

        buffers = {float(k): v for k, v in data["buffers"].items()}
        subdivision = tuple(data["subdivision"]) if data["subdivision"] else None

        return cls(
            polygons=polygons,
            buffers=buffers,
            physical_name=data["physical_name"],
            mesh_order=data["mesh_order"],
            mesh_bool=data["mesh_bool"],
            additive=data["additive"],
            point_tolerance=data["point_tolerance"],
            structured=data.get("structured", False),
            identify_arcs=data["identify_arcs"],
            min_arc_points=data["min_arc_points"],
            arc_tolerance=data["arc_tolerance"],
            subdivision=subdivision,
            translation=data.get("translation"),
            rotation_axis=data.get("rotation_axis"),
            rotation_point=data.get("rotation_point"),
            rotation_angle=data.get("rotation_angle", 0.0),
        )

    def _validate_polygon_buffers(self) -> bool:
        """Check if any buffering operation changes the topology of the polygon."""
        # Get first polygon or multipolygon
        first_geom = self.buffered_polygons[0][0][1]

        # Handle both single polygons and multipolygons
        first_polygons = (
            first_geom.geoms if hasattr(first_geom, "geoms") else [first_geom]
        )

        # Get reference counts from first polygon(s)
        reference_counts = []
        for polygon in first_polygons:
            # Store exterior vertex count and interior vertex counts for this polygon
            polygon_counts = {
                "exterior": len(polygon.exterior.coords),
                "interiors": [len(interior.coords) for interior in polygon.interiors],
            }
            reference_counts.append(polygon_counts)

        # Check each buffered polygon matches reference counts
        for buffered_polygon in self.buffered_polygons[0][1:]:
            geom = buffered_polygon[1]
            polygons = geom.geoms if hasattr(geom, "geoms") else [geom]

            if len(polygons) != len(reference_counts):
                return False

            for polygon, ref_counts in zip(polygons, reference_counts):
                # Check exterior vertices match
                if len(polygon.exterior.coords) != ref_counts["exterior"]:
                    return False

                # Check interior vertices match
                polygon_interior_counts = [
                    len(interior.coords) for interior in polygon.interiors
                ]
                if len(polygon_interior_counts) != len(ref_counts["interiors"]):
                    return False

                for count, ref_count in zip(
                    polygon_interior_counts, ref_counts["interiors"]
                ):
                    if count != ref_count:
                        return False

        return True
