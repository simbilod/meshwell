import numpy as np
import copy
from typing import Literal, Any, List
from pydantic import BaseModel
import warnings


class ResolutionSpec(BaseModel):
    """A ResolutionSpec is attached to a pre-CAD entity.

    It sets a mesh size field (see child classes) to the resulting post-CAD volumes, surfaces, curves, or points.

    The volumes, surfaces, curves can be filtered based on their mass (volume, area, length). Points can be filtered based on the length of the curve they belong to.
    """

    apply_to: Literal["volumes", "surfaces", "curves", "points"]
    min_mass: float = 0
    max_mass: float = np.inf
    sharing: List[str] | None = None
    not_sharing: List[str] | None = None

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

    resolution: float

    def apply(
        self,
        model: Any,
        current_field_index: int,
        entities_mass_dict,
        refinement_field_indices,
    ) -> int:
        new_field_indices = []

        model.mesh.field.add("MathEval", current_field_index)
        model.mesh.field.setString(current_field_index, "F", f"{self.resolution}")
        model.mesh.field.add("Restrict", current_field_index + 1)
        model.mesh.field.setNumber(
            current_field_index + 1, "InField", current_field_index
        )
        model.mesh.field.setNumbers(
            current_field_index + 1,
            self.entity_str,
            list(entities_mass_dict.keys()),
        )
        new_field_indices = (current_field_index + 1,)
        current_field_index += 2

        return new_field_indices, current_field_index

    def refine(self, resolution_factor: float):
        result = copy.copy(self)
        if result.resolution is not None:
            result.resolution *= resolution_factor

        return result


class SampledField(ResolutionSpec):
    """Shared functionality for size fields that require sampling the entities at points."""

    mass_per_sampling: float | None = None
    max_sampling: int = 100
    sizemin: float

    def calculate_samplings(self, entities_mass_dict):
        """Calculates a more optimal sampling from the masses"""

        if self.mass_per_sampling is None:
            # Default sampling is half the minimum resolution
            mass_per_sampling = 0.5 * self.sizemin

        return {
            tag: min(max(2, int(mass / mass_per_sampling)), self.max_sampling)
            if mass is not None
            else 1
            for tag, mass in entities_mass_dict.items()
        }


class ThresholdField(SampledField):
    """Linear growth of the resolution away from the entity"""

    sizemax: float
    sizemin: float
    distmin: float = 0
    distmax: float

    def apply(
        self,
        model: Any,
        current_field_index: int,
        entities_mass_dict,
        refinement_field_indices,
    ) -> int:
        new_field_indices = []

        # Compute samplings
        samplings_dict = self.calculate_samplings(entities_mass_dict)

        # FIXME: It is computationally cheaper to have a large sampling on all the curves rather than one field per curve; but there is probably an optimum somewhere.
        # FOr instance, the distribution should be very skewed (tiny vertical curves, tiny curves in bends, vs long horizontal ones), so there may be benefits for a small number of optimized fields.
        samplings = max(samplings_dict.values())
        entities = list(entities_mass_dict.keys())

        if self.entity_str == "RegionsList":
            warnings.warn("Cannot set a distance field on a Volume! Skipping")
        else:
            model.mesh.field.add("Distance", current_field_index)
            model.mesh.field.setNumbers(current_field_index, self.entity_str, entities)
            model.mesh.field.setNumber(current_field_index, "Sampling", samplings)
            model.mesh.field.add("Threshold", current_field_index + 1)
            model.mesh.field.setNumber(
                current_field_index + 1, "InField", current_field_index
            )
            model.mesh.field.setNumber(current_field_index + 1, "SizeMin", self.sizemin)
            model.mesh.field.setNumber(current_field_index + 1, "DistMin", self.distmin)
            if self.sizemax and self.distmax:
                model.mesh.field.setNumber(
                    current_field_index + 1, "SizeMax", self.sizemax
                )
                model.mesh.field.setNumber(
                    current_field_index + 1, "DistMax", self.distmax
                )
            model.mesh.field.setNumber(current_field_index + 1, "StopAtDistMax", 1)
            new_field_indices = (current_field_index + 1,)
        current_field_index += 2

        return new_field_indices, current_field_index

    def refine(self, resolution_factor: float):
        result = copy.copy(self)
        if result.sizemax is not None:
            result.sizemax *= resolution_factor
        if result.sizemin is not None:
            result.sizemin *= resolution_factor

        return result


class ExponentialField(SampledField):
    """Exponential growth of the characteristic length away from the entity"""

    growth_factor: float
    lengthscale: float

    def apply(
        self,
        model: Any,
        current_field_index: int,
        entities_mass_dict,
        refinement_field_indices,
    ) -> int:
        new_field_indices = []

        if self.entity_str == "RegionsList":
            warnings.warn("Cannot set a distance field on a Volume! Skipping")
        else:
            # Compute samplings
            samplings_dict = self.calculate_samplings(entities_mass_dict)

            # FIXME: It is computationally cheaper to have a large sampling on all the curves rather than one field per curve; but there is probably an optimum somewhere.
            # FOr instance, the distribution should be very skewed (tiny vertical curves, tiny curves in bends, vs long horizontal ones), so there may be benefits for a small number of optimized fields.
            samplings = max(samplings_dict.values())
            entities = list(entities_mass_dict.keys())

            # Sampled distance field
            model.mesh.field.add("Distance", current_field_index)
            model.mesh.field.setNumbers(current_field_index, self.entity_str, entities)
            model.mesh.field.setNumber(current_field_index, "Sampling", samplings)

            # Math field
            model.mesh.field.add("MathEval", current_field_index + 1)
            model.mesh.field.setString(
                current_field_index + 1,
                "F",
                f"{self.sizemin} * {self.growth_factor}^(F{current_field_index} / {self.lengthscale})",
            )

            new_field_indices = (current_field_index + 1,)
            current_field_index += 2

        return new_field_indices, current_field_index

    def refine(self, resolution_factor: float):
        result = copy.copy(self)
        if result.sizemin is not None:
            result.sizemin *= resolution_factor

        return result
