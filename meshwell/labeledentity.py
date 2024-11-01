from pydantic import BaseModel, ConfigDict
import gmsh
from typing import List, Union, Any, Tuple
from meshwell.resolution import ResolutionSpec
import warnings


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

    @property
    def tags(self) -> List[int]:
        tags = [tag for dim, tag in self.dimtags]
        if any(isinstance(el, list) for el in tags):
            tags = [item for sublist in tags for item in sublist]
        return tags

    @property
    def dim(self) -> int:
        return [dim for dim, tag in self.dimtags][0]

    def update_boundaries(self) -> List[int]:
        self.boundaries = [
            tag for dim, tag in gmsh.model.getBoundary(self.dimtags, True, False, False)
        ]
        return self.boundaries

    def filter_by_mass(self, target_dimension: int, min_mass, max_mass):
        def filter_by_target_and_tags(target_dimension, tags, min_mass, max_mass):
            return [
                tag
                for tag in tags
                if min_mass < self.model.occ.getMass(target_dimension, tag) < max_mass
            ]

        # Filter the tags based on current dimension and target
        if self.dim == 3:
            if target_dimension == 3:
                tags = self.tags
            elif target_dimension == 2:
                tags = self.boundaries
            elif target_dimension == 1 or target_dimension == 0:
                tags = [self.model.occ.getCurveLoops(b)[1] for b in self.boundaries]
        elif self.dim == 2:
            if target_dimension == 3:
                warnings.warn("Applying volume ResolutionSpec to surface, skipping")
            elif target_dimension == 2:
                tags = self.tags
            elif target_dimension == 1:
                tags = self.boundaries
            elif target_dimension == 0:
                raise NotImplementedError(
                    "ResolutionSpec for points not implemented in 2D yet!"
                )

        if target_dimension == 0:
            filtered_tags = filter_by_target_and_tags(1, tags, min_mass, max_mass)
            return self.model.getBoundary([(0, tag) for tag in filtered_tags])
        else:
            return filter_by_target_and_tags(target_dimension, tags, min_mass, max_mass)

    def add_refinement_fields_to_model(
        self,
        refinement_field_indices: List,
        refinement_max_index: int,
        default_resolution: float,
    ):
        """
        Adds refinement fields to the model based on base_resolution and resolution info.
        """
        n = refinement_max_index

        if self.resolutions:
            for resolutionspec in self.resolutions:
                resolutionspec.resolution = min(
                    default_resolution, resolutionspec.resolution
                )
                entities = self.filter_by_mass(
                    target_dimension=resolutionspec.target_dimension,
                    min_mass=resolutionspec.min_mass,
                    max_mass=resolutionspec.max_mass,
                )
                new_field_indices, n = resolutionspec.apply(
                    model=self.model,
                    current_field_index=n,
                    refinement_field_indices=refinement_field_indices,
                    entities=entities,
                )
                refinement_field_indices.extend(new_field_indices)

        return refinement_field_indices, n
