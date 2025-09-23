"""Resolution specifications."""
import copy
import warnings
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel


class ResolutionSpec(BaseModel):
    """A ResolutionSpec is attached to a pre-CAD entity.

    It sets a mesh size field (see child classes) to the resulting post-CAD volumes, surfaces, curves, or points.

    The volumes, surfaces, curves can be filtered based on their mass (volume, area, length). Points can be filtered based on the length of the curve they belong to.
    """

    apply_to: Literal["volumes", "surfaces", "curves", "points"]
    min_mass: float = 0
    max_mass: float = np.inf
    sharing: list[str] | None = None
    not_sharing: list[str] | None = None
    restrict_to: list[str] | None = None

    @property
    def entity_str(self):
        """Convenience wrapper."""
        if self.apply_to == "volumes":
            return "RegionsList"
        if self.apply_to == "surfaces":
            return "SurfacesList"
        if self.apply_to == "curves":
            return "CurvesList"
        if self.apply_to == "points":
            return "PointsList"
        return None

    @property
    def target_dimension(self):
        """Convenience wrapper."""
        if self.apply_to == "volumes":
            return 3
        if self.apply_to == "surfaces":
            return 2
        if self.apply_to == "curves":
            return 1
        if self.apply_to == "points":
            return 0
        return None


class ConstantInField(ResolutionSpec):
    """Provides constant resolution within specified entities.

    This class implements a resolution specification that applies a uniform
    mesh size throughout the specified geometric entities.

    Attributes:
        resolution: The constant mesh resolution to apply.
    """

    resolution: float

    def apply(self, model: Any, entities_mass_dict, **kwargs) -> int:  # noqa: ARG002
        """Apply constant resolution field to the model.

        Creates a MathEval field with constant resolution and restricts it
        to the specified entities.

        Args:
            model: The mesh model to apply the field to.
            entities_mass_dict: Dictionary mapping entity tags to their masses.
            **kwargs: Unused kwargs

        Returns:
            int: Index of the created restrict field.
        """
        matheval_field_index = model.mesh.field.add("MathEval")
        model.mesh.field.setString(matheval_field_index, "F", f"{self.resolution}")
        restrict_field_index = model.mesh.field.add("Restrict")
        model.mesh.field.setNumber(
            restrict_field_index, "InField", matheval_field_index
        )
        model.mesh.field.setNumbers(
            restrict_field_index,
            self.entity_str,
            list(entities_mass_dict.keys()),
        )

        return restrict_field_index

    def refine(self, resolution_factor: float):
        """Create a refined copy with adjusted resolution.

        Args:
            resolution_factor: Factor to multiply the resolution by.

        Returns:
            ConstantInField: A new instance with refined resolution.
        """
        result = copy.copy(self)
        if result.resolution is not None:
            result.resolution *= resolution_factor

        return result


class SampledField(ResolutionSpec):
    """Base class for size fields that require sampling entities at points.

    This class provides shared functionality for resolution specifications
    that need to sample geometric entities to determine appropriate mesh sizing.

    Attributes:
        mass_per_sampling: Mass threshold per sampling point.
            If None, defaults to 0.5 * sizemin.
        max_sampling: Maximum number of sampling points allowed (default: 100).
        sizemin: Minimum mesh size.
    """

    mass_per_sampling: float | None = None
    max_sampling: int = 100
    sizemin: float

    def calculate_samplings(self, entities_mass_dict):
        """Calculate optimal sampling distribution based on entity masses.

        Determines the number of sampling points for each entity based on
        its mass and the specified mass_per_sampling ratio.

        Args:
            entities_mass_dict: Dictionary mapping entity tags to their masses.

        Returns:
            dict: Mapping of entity tags to their calculated sampling counts.
        """
        if self.mass_per_sampling is None:
            # Default sampling is half the minimum resolution
            mass_per_sampling = 0.5 * self.sizemin
        else:
            mass_per_sampling = self.mass_per_sampling

        return {
            tag: min(max(2, int(mass / mass_per_sampling)), self.max_sampling)
            if mass is not None
            else 1
            for tag, mass in entities_mass_dict.items()
        }

    def apply_distance(self, model: Any, entities_mass_dict):
        """Create and configure a distance field for the specified entities.

        Args:
            model: The mesh model to apply the field to.
            entities_mass_dict: Dictionary mapping entity tags to their masses.

        Returns:
            int: Index of the created distance field.
        """
        # Compute optimal samplings for each entity
        samplings_dict = self.calculate_samplings(entities_mass_dict)

        # FIXME: It is computationally cheaper to have a large sampling on all the curves rather than one field per curve; but there is probably an optimum somewhere.
        # For instance, the distribution should be very skewed (tiny vertical curves, tiny curves in bends, vs long horizontal ones), so there may be benefits for a small number of optimized fields.
        samplings = max(samplings_dict.values())
        entities = list(entities_mass_dict.keys())

        distance_field_index = model.mesh.field.add("Distance")
        model.mesh.field.setNumbers(distance_field_index, self.entity_str, entities)
        model.mesh.field.setNumber(distance_field_index, "Sampling", samplings)
        return distance_field_index

    def apply_restrict(
        self,
        model: Any,
        target_field_index: int,
        restrict_to_str: str,
        restrict_to_tags=None,
    ):
        """Apply restriction to limit a field to specific entities.

        Creates a restriction field that limits the application of another
        field to specified geometric entities.

        Args:
            model: The mesh model to apply the restriction to.
            target_field_index (int): Index of the field to restrict.
            restrict_to_str (str): String identifier for the restriction type.
            restrict_to_tags: List of entity tags to restrict to.

        Returns:
            int: Index of the created restriction field.
        """
        restrict_field_index = model.mesh.field.add("Restrict")
        model.mesh.field.setNumber(restrict_field_index, "InField", target_field_index)
        model.mesh.field.setNumbers(
            restrict_field_index,
            restrict_to_str,
            restrict_to_tags,
        )

        return restrict_field_index


class ThresholdField(SampledField):
    """Implements linear growth of resolution away from entities.

    This class creates a threshold field that provides fine resolution near
    specified entities and gradually increases the mesh size with distance.

    Attributes:
        sizemax (float): Maximum mesh size at far distances.
        sizemin (float): Minimum mesh size near entities.
        distmin (float): Distance where minimum size applies (default: 0).
        distmax (float): Distance where maximum size applies.
    """

    sizemax: float
    sizemin: float
    distmin: float = 0
    distmax: float

    def apply(
        self,
        model: Any,
        entities_mass_dict,
        restrict_to_str,
        restrict_to_tags=None,
    ) -> int:
        """Apply threshold field with linear resolution growth.

        Creates a distance-based field that transitions from minimum to maximum
        mesh size over the specified distance range.

        Args:
            model: The mesh model to apply the field to.
            entities_mass_dict: Dictionary mapping entity tags to their masses.
            restrict_to_str (str): String identifier for restriction type.
            restrict_to_tags: List of entity tags to restrict the field to.

        Returns:
            int | None: Index of the created field, or None if skipped.

        Warnings:
            UserWarning: If attempting to set distance field on a Volume.
        """
        if self.entity_str == "RegionsList":
            warnings.warn(
                "Cannot set a distance field on a Volume! Skipping", stacklevel=2
            )
        else:
            distance_field_index = self.apply_distance(
                model=model,
                entities_mass_dict=entities_mass_dict,
            )
            threshold_field_index = model.mesh.field.add("Threshold")
            model.mesh.field.setNumber(
                threshold_field_index, "InField", distance_field_index
            )
            model.mesh.field.setNumber(threshold_field_index, "SizeMin", self.sizemin)
            model.mesh.field.setNumber(threshold_field_index, "DistMin", self.distmin)
            if self.sizemax and self.distmax:
                model.mesh.field.setNumber(
                    threshold_field_index, "SizeMax", self.sizemax
                )
                model.mesh.field.setNumber(
                    threshold_field_index, "DistMax", self.distmax
                )
            model.mesh.field.setNumber(threshold_field_index, "StopAtDistMax", 1)

            # Restriction field
            if restrict_to_tags:
                return self.apply_restrict(
                    model, threshold_field_index, restrict_to_str, restrict_to_tags
                )
            return threshold_field_index
        return None

    def refine(self, resolution_factor: float):
        """Create a refined copy with adjusted size parameters.

        Args:
            resolution_factor (float): Factor to multiply size parameters by.

        Returns:
            ThresholdField: A new instance with refined size parameters.
        """
        result = copy.copy(self)

        if result.sizemax is not None:
            result.sizemax *= resolution_factor
        if result.sizemin is not None:
            result.sizemin *= resolution_factor

        return result


class ExponentialField(SampledField):
    """Exponential growth of the characteristic length away from the entity.

    Attributes:
        growth_factor: Factor by which the mesh size grows exponentially
        lengthscale: Characteristic length scale for the exponential growth
    """

    growth_factor: float
    lengthscale: float

    def apply(
        self,
        model: Any,
        entities_mass_dict,
        restrict_to_str,
        restrict_to_tags=None,
    ) -> int | None:
        """Apply the exponential field to the mesh model.

        Args:
            model: The mesh model to apply the field to
            entities_mass_dict: Dictionary mapping entities to their mass properties
            restrict_to_str: String representation of restriction criteria
            restrict_to_tags: Optional list of tags to restrict the field to

        Returns:
            int: Index of the created field, or None if field cannot be applied

        Warnings:
            UserWarning: If attempting to set distance field on a Volume
        """
        if self.entity_str == "RegionsList":
            warnings.warn(
                "Cannot set a distance field on a Volume! Skipping", stacklevel=2
            )
        else:
            distance_field_index = self.apply_distance(
                model=model,
                entities_mass_dict=entities_mass_dict,
            )

            # Math field
            matheval_field_index = model.mesh.field.add("MathEval")
            model.mesh.field.setString(
                matheval_field_index,
                "F",
                f"{self.sizemin} * {self.growth_factor}^(F{distance_field_index} / {self.lengthscale})",
            )

            # Restriction field
            if restrict_to_tags:
                return self.apply_restrict(
                    model, matheval_field_index, restrict_to_str, restrict_to_tags
                )
            return matheval_field_index
        return None

    def refine(self, resolution_factor: float):
        """Create a refined copy of this field with adjusted resolution.

        The minimum size is scaled by the resolution factor to create a finer
        or coarser mesh as needed.

        Args:
            resolution_factor (float): Factor to scale the minimum size by.
                                     Values < 1.0 create finer meshes,
                                     values > 1.0 create coarser meshes

        Returns:
            ExponentialField: A new field instance with adjusted resolution
        """
        result = copy.copy(self)
        if result.sizemin is not None:
            result.sizemin *= resolution_factor

        return result
