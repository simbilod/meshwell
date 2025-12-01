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
    restrict_to: List[str] | None = None

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

    def apply(self, model: Any, entities_mass_dict, **kwargs) -> int:
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

    def apply_distance(self, model: Any, entities_mass_dict):
        # Compute samplings
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
        """Common application"""
        restrict_field_index = model.mesh.field.add("Restrict")
        model.mesh.field.setNumber(restrict_field_index, "InField", target_field_index)
        model.mesh.field.setNumbers(
            restrict_field_index,
            restrict_to_str,
            restrict_to_tags,
        )
        return restrict_field_index


class ThresholdField(SampledField):
    """Linear growth of the resolution away from the entity"""

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
        if self.entity_str == "RegionsList":
            warnings.warn("Cannot set a distance field on a Volume! Skipping")
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
                restrict_field_index = self.apply_restrict(
                    model, threshold_field_index, restrict_to_str, restrict_to_tags
                )
                return restrict_field_index
            else:
                return threshold_field_index

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
        entities_mass_dict,
        restrict_to_str,
        restrict_to_tags=None,
    ) -> int:
        if self.entity_str == "RegionsList":
            warnings.warn("Cannot set a distance field on a Volume! Skipping")
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
                restrict_field_index = self.apply_restrict(
                    model, matheval_field_index, restrict_to_str, restrict_to_tags
                )
                return restrict_field_index
            else:
                return matheval_field_index

    def refine(self, resolution_factor: float):
        result = copy.copy(self)
        if result.sizemin is not None:
            result.sizemin *= resolution_factor

        return result


class DirectSizeSpecification(ResolutionSpec):
    """Dataclass for directly specifying mesh sizes from data points."""

    refinement_data: np.ndarray
    min_size: float | None = None
    max_size: float | None = None

    # Apply to all dimensions by default
    apply_to: Literal["volumes", "surfaces", "curves", "points"] | None = None

    class Config:
        arbitrary_types_allowed = True

    def apply(self, model: Any, entities_mass_dict, **kwargs) -> int:
        import gmsh
        import tempfile
        import os

        # 1. Use Data Directly
        r_data = self.refinement_data

        # Ensure 2D array
        if r_data.ndim == 1:
            r_data = r_data.reshape(1, -1)

        # Parse coords and values (Assume x,y,z,size)
        if r_data.shape[1] != 4:
            raise ValueError(
                f"refinement_data must be (N, 4) [x,y,z,size], got shape {r_data.shape}"
            )

        coords = r_data[:, :3]
        sizes = r_data[:, 3]

        # 2. Clamp sizes
        if self.min_size:
            sizes = np.maximum(sizes, self.min_size)
        if self.max_size:
            sizes = np.minimum(sizes, self.max_size)

        # 3. Create PostView
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pos", delete=False) as tmp:
            tmp_path = tmp.name
            tmp.write('View "size_field" {\n')
            for i in range(len(coords)):
                x, y, z = coords[i]
                s = sizes[i]
                tmp.write(f"SP({x}, {y}, {z}){{{s}}};\n")
            tmp.write("};\n")

        gmsh.merge(tmp_path)
        view_tag = gmsh.view.getTags()[-1]
        os.unlink(tmp_path)

        # 4. Create Field
        field_index = model.mesh.field.add("PostView")
        model.mesh.field.setNumber(field_index, "ViewTag", view_tag)

        # 5. Apply Restriction
        if kwargs.get("restrict_to_tags"):
            restrict_field = model.mesh.field.add("Restrict")
            model.mesh.field.setNumber(restrict_field, "InField", field_index)
            model.mesh.field.setNumbers(
                restrict_field, self.entity_str, kwargs["restrict_to_tags"]
            )
            return restrict_field

        return field_index
