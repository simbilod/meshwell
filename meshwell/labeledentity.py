from pydantic import BaseModel, ConfigDict
import gmsh
from typing import List, Union, Any, Tuple
from meshwell.resolution import ResolutionSpec


class LabeledEntities(BaseModel):
    """General class to track the gmsh entities that result from the geometry definition."""

    index: int
    model: Any
    dimtags: List[Tuple[int, int]]
    physical_name: str | tuple[str, ...]
    resolutions: List[ResolutionSpec] | None = None
    keep: bool
    boundaries: List[int] = []
    interfaces: List = []

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _fuse_self(self, dimtags: List[Union[int, str]]) -> List[Union[int, str]]:
        if len(dimtags) == 0:
            return []
        elif len(dimtags) != 1:
            dimtags = gmsh.model.occ.fuse(
                [dimtags[0]],
                dimtags[1:],
                removeObject=True,
                removeTool=True,
            )[0]
            self.model.occ.synchronize()
        return dimtags

    def get_tags(self) -> List[int]:
        tags = [tag for dim, tag in self.dimtags]
        if any(isinstance(el, list) for el in tags):
            tags = [item for sublist in tags for item in sublist]
        return tags

    def get_dim(self) -> int:
        return [dim for dim, tag in self.dimtags][0]

    def update_boundaries(self) -> List[int]:
        self.boundaries = [
            tag for dim, tag in gmsh.model.getBoundary(self.dimtags, True, False, False)
        ]
        return self.boundaries

    def add_refinement_fields_to_model(
        self,
        refinement_field_indices: List,
        refinement_max_index: int,
        default_resolution: float,
        final_entity_list: List,
    ):
        """
        Adds refinement fields to the model based on base_resolution and resolution info.

        # FIXME: make a better ResolutionSpec class that handles the entity/boundary distinction
        """
        n = refinement_max_index

        if self.resolutions:
            entities = []
            boundaries = []
            boundary_lines = []
            boundary_lines_points = []

            for resolutionspec in self.resolutions:
                # Parse by dimension
                if self.get_dim() == 3:
                    entity_str = "RegionsList"
                    boundary_str = "SurfacesList"
                    curves_str = "CurvesList"
                    points_str = "PointsList"

                    # Check condition on volumes
                    if resolutionspec.resolution_volumes is None:
                        entities = []
                        entity_resolution = default_resolution
                    else:
                        entity_resolution = min(
                            resolutionspec.resolution_volumes, default_resolution
                        )
                        entities = [
                            volume_tag
                            for volume_tag in self.get_tags()
                            if resolutionspec.min_volumes
                            < self.model.occ.getMass(3, volume_tag)
                            < resolutionspec.max_volumes
                        ]

                    # Check condition on surfaces
                    if resolutionspec.resolution_surfaces is None:
                        boundaries = []
                        boundary_resolution = entity_resolution
                    else:
                        boundaries = [
                            surface_tag
                            for surface_tag in self.boundaries
                            if resolutionspec.min_area_surfaces
                            < self.model.occ.getMass(2, surface_tag)
                            < resolutionspec.max_area_surfaces
                        ]
                        boundary_resolution = min(
                            resolutionspec.resolution_surfaces, entity_resolution
                        )
                        boundary_sizemax = resolutionspec.sizemax_surfaces
                        boundary_distmax = resolutionspec.distmax_surfaces
                        boundary_sigmoid = resolutionspec.surface_sigmoid
                        boundaries_samplings = {
                            boundary: resolutionspec.calculate_sampling_surface(
                                self.model.occ.getMass(2, boundary)
                            )
                            for boundary in boundaries
                        }

                    # Check condition on surface curves
                    if resolutionspec.resolution_curves is None:
                        boundary_lines = []
                        boundary_lines_resolution = boundary_resolution
                    else:
                        boundary_lines = [
                            c
                            for b in self.boundaries
                            for cs in self.model.occ.getCurveLoops(b)[1]
                            for c in cs
                            if resolutionspec.min_length_curves
                            < self.model.occ.getMass(1, c)
                            < resolutionspec.max_length_curves
                        ]

                        boundary_lines_resolution = min(
                            resolutionspec.resolution_curves, boundary_resolution
                        )
                        boundary_lines_sizemax = resolutionspec.sizemax_curves
                        boundary_lines_distmax = resolutionspec.distmax_curves
                        boundary_line_sigmoid = resolutionspec.curve_sigmoid
                        boundary_lines_samplings = {
                            boundary_line: resolutionspec.calculate_sampling_curve(
                                self.model.occ.getMass(1, boundary_line)
                            )
                            for boundary_line in boundary_lines
                        }

                    # Check condition on surface curve points
                    if resolutionspec.resolution_points is None:
                        boundary_lines_points = []
                    else:
                        boundary_lines_points = [
                            c
                            for b in self.boundaries
                            for cs in self.model.occ.getCurveLoops(b)[1]
                            for c in cs
                            if resolutionspec.min_length_curves_for_points
                            < self.model.occ.getMass(1, c)
                            < resolutionspec.max_length_curves_for_points
                        ]
                        boundary_lines_points = self.model.getBoundary(
                            [
                                (0, boundary_line_point)
                                for boundary_line_point in boundary_lines_points
                            ]
                        )
                        boundary_lines_points_resolution = min(
                            resolutionspec.resolution_points, boundary_lines_resolution
                        )
                        boundary_lines_points_sizemax = resolutionspec.sizemax_points
                        boundary_lines_points_distmax = resolutionspec.distmax_points
                        boundary_lines_points_sigmoid = resolutionspec.point_sigmoid

                elif self.get_dim() == 2:
                    entity_str = "SurfacesList"
                    boundary_str = "CurvesList"
                    curves_str = "PointsList"

                    # Check condition on surfaces
                    if resolutionspec.resolution_surfaces is None:
                        entities = []
                        entity_resolution = default_resolution
                    else:
                        entity_resolution = min(
                            resolutionspec.resolution_surfaces, default_resolution
                        )
                        entities = [
                            surface_tag
                            for surface_tag in self.get_tags()
                            if resolutionspec.min_area_surfaces
                            < self.model.occ.getMass(2, surface_tag)
                            < resolutionspec.max_area_surfaces
                        ]

                    # Check condition on curves
                    if resolutionspec.resolution_curves is None:
                        boundaries = []
                    else:
                        boundaries = [
                            curve_tag
                            for curve_tag in self.boundaries
                            if resolutionspec.min_length_curves
                            < self.model.occ.getMass(1, curve_tag)
                            < resolutionspec.max_length_curves
                        ]

                        boundary_resolution = min(
                            resolutionspec.resolution_curves, entity_resolution
                        )
                        boundary_sizemax = resolutionspec.sizemax_curves
                        boundary_distmax = resolutionspec.distmax_curves
                        boundary_sigmoid = resolutionspec.curve_sigmoid
                        boundaries_samplings = {
                            boundary: resolutionspec.calculate_sampling_curve(
                                self.model.occ.getMass(1, boundary)
                            )
                            for boundary in boundaries
                        }

                    if resolutionspec.resolution_points is not None:
                        raise ValueError(
                            "ResolutionSpec for points not implemented in 2D yet!"
                        )

                elif self.get_dim() == 1:
                    entity_str = "CurvesList"
                    boundary_str = "PointsList"
                else:
                    entity_str = "PointsList"
                    boundary_str = None

                if entities:
                    self.model.mesh.field.add("MathEval", n)
                    self.model.mesh.field.setString(n, "F", f"{entity_resolution}")
                    self.model.mesh.field.add("Restrict", n + 1)
                    self.model.mesh.field.setNumber(n + 1, "InField", n)
                    self.model.mesh.field.setNumbers(
                        n + 1,
                        entity_str,
                        entities,
                    )
                    refinement_field_indices.extend((n + 1,))
                    n += 2

                for boundary in boundaries:
                    self.model.mesh.field.add("Distance", n)
                    self.model.mesh.field.setNumbers(n, boundary_str, [boundary])
                    self.model.mesh.field.setNumber(
                        n, "Sampling", boundaries_samplings[boundary]
                    )
                    self.model.mesh.field.add("Threshold", n + 1)
                    self.model.mesh.field.setNumber(n + 1, "InField", n)
                    self.model.mesh.field.setNumber(
                        n + 1, "SizeMin", boundary_resolution
                    )
                    self.model.mesh.field.setNumber(n + 1, "DistMin", 0)
                    if boundary_sizemax and boundary_distmax:
                        self.model.mesh.field.setNumber(
                            n + 1, "SizeMax", boundary_sizemax
                        )
                        self.model.mesh.field.setNumber(
                            n + 1, "DistMax", boundary_distmax
                        )
                    self.model.mesh.field.setNumber(n + 1, "StopAtDistMax", 1)
                    self.model.mesh.field.setNumber(
                        n + 1, "Sigmoid", int(boundary_sigmoid)
                    )
                    refinement_field_indices.extend((n + 1,))
                    n += 2

                for boundary_line in boundary_lines:
                    self.model.mesh.field.add("Distance", n)
                    self.model.mesh.field.setNumbers(n, curves_str, [boundary_line])
                    self.model.mesh.field.setNumber(
                        n, "Sampling", boundary_lines_samplings[boundary_line]
                    )
                    self.model.mesh.field.add("Threshold", n + 1)
                    self.model.mesh.field.setNumber(n + 1, "InField", n)
                    self.model.mesh.field.setNumber(
                        n + 1, "SizeMin", boundary_lines_resolution
                    )
                    self.model.mesh.field.setNumber(n + 1, "DistMin", 0)
                    if boundary_lines_sizemax and boundary_lines_distmax:
                        self.model.mesh.field.setNumber(
                            n + 1, "SizeMax", boundary_lines_sizemax
                        )
                        self.model.mesh.field.setNumber(
                            n + 1, "DistMax", boundary_lines_distmax
                        )
                    self.model.mesh.field.setNumber(n + 1, "StopAtDistMax", 1)
                    self.model.mesh.field.setNumber(
                        n + 1, "Sigmoid", int(boundary_line_sigmoid)
                    )
                    refinement_field_indices.extend((n + 1,))
                    n += 2

                if boundary_lines_points:
                    self.model.mesh.field.add("Distance", n)
                    self.model.mesh.field.setNumbers(
                        n, points_str, boundary_lines_points
                    )
                    self.model.mesh.field.add("Threshold", n + 1)
                    self.model.mesh.field.setNumber(n + 1, "InField", n)
                    self.model.mesh.field.setNumber(
                        n + 1, "SizeMin", boundary_lines_points_resolution
                    )
                    self.model.mesh.field.setNumber(n + 1, "DistMin", 0)
                    if boundary_lines_points_sizemax and boundary_lines_points_distmax:
                        self.model.mesh.field.setNumber(
                            n + 1, "SizeMax", boundary_lines_points_sizemax
                        )
                        self.model.mesh.field.setNumber(
                            n + 1, "DistMax", boundary_lines_points_distmax
                        )
                    self.model.mesh.field.setNumber(n + 1, "StopAtDistMax", 1)
                    self.model.mesh.field.setNumber(
                        n + 1, "Sigmoid", int(boundary_lines_points_sigmoid)
                    )
                    refinement_field_indices.extend((n + 1,))
                    n += 2

        return refinement_field_indices, n
