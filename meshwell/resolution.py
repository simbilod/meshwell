import numpy as np
import copy
from pydantic import BaseModel


class ResolutionSpec(BaseModel):
    """
    Object holding resolution information for an entity and its boundaries.

    # FIXME: make a better ResolutionSpec class that handles the entity/boundary distinction

    Arguments:
        # Volume
        volume_resolution (float): resolution of the volume (3D). No effect if the entity is 2D.
            defaults to inf --> default resolution, since the local resolution is always the minimum of all size fields

        # Surface
        surface_resolution (float): resolution of the surface (2D) or of all the surfaces touching the volume (3D)
            defaults to inf --> default resolution (2D) or volume_resolution (3D), since the local resolution is always the minimum of all size fields

        # Curves
        curve_resolution (float): resolution of curves constituting the volumes' surfaces (3D) or surfaces (2D)
            defaults to inf --> surface_resolution, since the local resolution is always the minimum of all size fields

        # Points
        point_resolution (float): resolution of points constituting the volumes' surfaces' curves (3D) or surfaces' curves (2D)
            defaults to inf --> curve_resolution, since the local resolution is always the minimum of all size fields
            can be filtered by the length of the associated curves
    """

    # Volume
    resolution_volumes: float | None = None
    min_volumes: float = 0
    max_volumes: float = np.inf
    # Surface
    resolution_surfaces: float | None = None
    min_area_surfaces: float = 0
    max_area_surfaces: float = np.inf
    distmax_surfaces: float | None = None
    sizemax_surfaces: float | None = None
    surface_sigmoid: bool = False
    surface_per_sampling_surfaces: float | None = None
    sampling_surface_max: int = 100
    # Curve
    resolution_curves: float | None = None
    min_length_curves: float = 0
    max_length_curves: float = np.inf
    distmax_curves: float | None = None
    sizemax_curves: float | None = None
    curve_sigmoid: bool = False
    length_per_sampling_curves: float | None = None
    sampling_curve_max: int = 100
    # Point
    resolution_points: float | None = None
    min_length_curves_for_points: float = 0
    max_length_curves_for_points: float = np.inf
    distmax_points: float | None = None
    sizemax_points: float | None = None
    point_sigmoid: bool = False

    def refine(self, resolution_factor: float):
        result = copy.copy(self)
        if result.resolution_volumes is not None:
            result.resolution_volumes *= resolution_factor
        if result.resolution_surfaces is not None:
            result.resolution_surfaces *= resolution_factor
        if result.sizemax_surfaces is not None:
            result.sizemax_surfaces *= resolution_factor
        if result.resolution_curves is not None:
            result.resolution_curves *= resolution_factor
        if result.sizemax_curves is not None:
            result.sizemax_curves *= resolution_factor
        if result.resolution_points is not None:
            result.resolution_points *= resolution_factor
        if result.sizemax_points is not None:
            result.sizemax_points *= resolution_factor
        return result

    def calculate_sampling(self, mass_per_sampling, mass, max_sampling):
        if mass_per_sampling is None:
            return 2  # avoid int(inf) error
        else:
            return min(max(2, int(mass / mass_per_sampling)), max_sampling)

    def calculate_sampling_surface(self, area):
        if self.surface_per_sampling_surfaces:
            return self.calculate_sampling(
                self.surface_per_sampling_surfaces, area, self.sampling_surface_max
            )
        else:
            return self.calculate_sampling(
                0.5 * self.resolution_surfaces, area, self.sampling_surface_max
            )

    def calculate_sampling_curve(self, length):
        if self.length_per_sampling_curves:
            return self.calculate_sampling(
                self.length_per_sampling_curves, length, self.sampling_curve_max
            )
        else:
            return self.calculate_sampling(
                0.5 * self.resolution_curves, length, self.sampling_curve_max
            )
