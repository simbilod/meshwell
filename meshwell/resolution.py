import numpy as np
import copy
from typing import Literal, Any
from pydantic import BaseModel


class ResolutionSpec(BaseModel):
    """A ResolutionSpec is attached to a pre-CAD entity.

    It sets a mesh size field (see child classes) to the resulting post-CAD volumes, surfaces, curves, or points.

    The volumes, surfaces, curves can be filtered based on their mass (volume, area, length). Points can be filtered based on the length of the curve they belong to.
    """

    # Eventually we can add flags here to also consider proximity to other specific physicals (e.g. shared interfaces)
    apply_to: Literal["volumes", "surfaces", "curves", "points"]
    resolution: float  # base resolution number used across all child classes; can be different than the overall model default resolution
    min_mass: float = 0
    max_mass: float = np.inf

    @property
    def entity_str(self):
        """Convenience wrapper."""
        if self.apply_to == "volumes":
            return "RegionsList"
        elif self.apply_to == "surfaces":
            return "SurfacesList"
        elif self.apply_to == "curves":
            return "CurvesList"
        elif self.apply_to == "points":
            return "PointsList"

    @property
    def target_dimension(self):
        """Convenience wrapper."""
        if self.apply_to == "volumes":
            return 3
        elif self.apply_to == "surfaces":
            return 2
        elif self.apply_to == "curves":
            return 1
        elif self.apply_to == "points":
            return 0


class ConstantInField(ResolutionSpec):
    """Constant resolution within the entities."""

    def apply(
        self, model: Any, current_field_index: int, entities, refinement_field_indices
    ) -> int:
        model.mesh.field.add("MathEval", current_field_index)
        model.mesh.field.setString(current_field_index, "F", f"{self.resolution}")
        model.mesh.field.add("Restrict", current_field_index + 1)
        model.mesh.field.setNumber(
            current_field_index + 1, "InField", current_field_index
        )
        model.mesh.field.setNumbers(
            current_field_index + 1,
            self.entity_str,
            entities,
        )
        refinement_field_indices.extend((current_field_index + 1,))
        current_field_index += 2

        return refinement_field_indices, current_field_index

    def refine(self, resolution_factor: float):
        result = copy.copy(self)
        if result.resolution is not None:
            result.resolution *= resolution_factor

        return result


class DistanceField(ResolutionSpec):
    """Shared functionality for size fields that consider distance from the entities."""

    mass_per_sampling: float | None = None
    max_sampling: float
    samplings: int
    sizemax: float
    distmax: float
    sizemin: float | None = None
    distmin: float = 0

    def calculate_sampling(self, mass):
        if self.mass_per_sampling is None:
            return 2
        else:
            mass_per_sampling = self.mass_per_sampling or 0.5 * self.sizemin
            return min(max(2, int(mass / mass_per_sampling)), self.max_sampling)

    def refine(self, resolution_factor: float):
        result = copy.copy(self)
        if result.sizemax is not None:
            result.sizemax *= resolution_factor
        if result.sizemin is not None:
            result.sizemin *= resolution_factor


class ThresholdField(DistanceField):
    """Linear or sigmoid growth of the resolution away from the entity"""

    sigmoid: bool

    def apply(
        self, current_field_index: int, refinement_field_indices, entities
    ) -> int:
        self.model.mesh.field.add("Distance", current_field_index)
        self.model.mesh.field.setNumbers(current_field_index, self.entity_str, entities)
        self.model.mesh.field.setNumber(current_field_index, "Sampling", self.samplings)
        self.model.mesh.field.add("Threshold", current_field_index + 1)
        self.model.mesh.field.setNumber(
            current_field_index + 1, "InField", current_field_index
        )
        self.model.mesh.field.setNumber(
            current_field_index + 1, "SizeMin", self.sizemin
        )
        self.model.mesh.field.setNumber(
            current_field_index + 1, "DistMin", self.distmin
        )
        if self.sizemax and self.distmax:
            self.model.mesh.field.setNumber(
                current_field_index + 1, "SizeMax", self.sizemax
            )
            self.model.mesh.field.setNumber(
                current_field_index + 1, "DistMax", self.distmax
            )
        self.model.mesh.field.setNumber(current_field_index + 1, "StopAtDistMax", 1)
        self.model.mesh.field.setNumber(
            current_field_index + 1, "Sigmoid", int(self.sigmoid)
        )
        refinement_field_indices.extend((current_field_index + 1,))
        current_field_index += 2

        return refinement_field_indices, current_field_index


class GrowthField(DistanceField):
    """Exponential growth of the resolution away from the entity"""

    growth_factor: float

    def apply(self, current_field_index: int) -> int:
        raise NotImplementedError()
