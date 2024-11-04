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
            # Returns tags and masses
            tags = [
                tag
                for tag in tags
                if min_mass < self.model.occ.getMass(target_dimension, tag) < max_mass
            ]
            return (
                {tag: self.model.occ.getMass(target_dimension, tag) for tag in tags}
                if tags is not []
                else {}
            )

        # Filter the tags based on current dimension and target
        match self.dim - target_dimension:
            case 0:
                tags = self.tags
            case 1:
                tags = self.boundaries
            case 2 | 3:
                tags = [
                    c
                    for b in self.boundaries
                    for cs in self.model.occ.getCurveLoops(b)[1]
                    for c in cs
                ]
            case -1:
                warnings.warn("Applying volume ResolutionSpec to surface, skipping")
                return {}

        # If targeting points, need post-filtering filtering
        if target_dimension == 0:
            filtered_tags = filter_by_target_and_tags(1, tags, min_mass, max_mass)
            return {
                p[1]: None
                for p in self.model.getBoundary(
                    [(1, tag) for tag in filtered_tags.keys()]
                )
            }  # no mass in 0D
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
                entities_mass_dict = self.filter_by_mass(
                    target_dimension=resolutionspec.target_dimension,
                    min_mass=resolutionspec.min_mass,
                    max_mass=resolutionspec.max_mass,
                )
                new_field_indices, n = resolutionspec.apply(
                    model=self.model,
                    current_field_index=n,
                    refinement_field_indices=refinement_field_indices,
                    entities_mass_dict=entities_mass_dict,
                )
                refinement_field_indices.extend(new_field_indices)

        return refinement_field_indices, n
