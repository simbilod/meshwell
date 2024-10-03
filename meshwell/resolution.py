import numpy as np
import copy
from pydantic import BaseModel


class ResolutionSpec(BaseModel):
    """
    Object holding resolution information for an entity and its boundaries.

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
    """

    # Volume
    resolution_volumes: float = np.inf
    min_volumes: float = 0
    max_volumes: float = np.inf
    # Surface
    resolution_surfaces: float = np.inf
    min_area_surfaces: float = 0
    max_area_surfaces: float = np.inf
    distmax_surfaces: float | None = None
    sizemax_surfaces: float | None = None
    # Curve
    resolution_curves: float = np.inf
    min_length_curves: float = 0
    max_length_curves: float = np.inf
    distmax_curves: float | None = None
    sizemax_curves: float | None = None
    # Point
    resolution_points: float = np.inf
    distmax_points: float | None = None
    sizemax_points: float | None = None

    def refine(self, resolution_factor: float):
        result = copy.copy(self)
        result.resolution_volumes *= resolution_factor
        result.resolution_surfaces *= resolution_factor
        if result.sizemax_surfaces is not None:
            result.sizemax_surfaces *= resolution_factor
        result.resolution_curves *= resolution_factor
        if result.sizemax_curves is not None:
            result.sizemax_curves *= resolution_factor
        result.resolution_points *= resolution_factor
        if result.sizemax_points is not None:
            result.sizemax_points *= resolution_factor
        return result
