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
    mesh_edge_name_interfaces: List = []

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

    def filter_tags_by_target_dimension(self, target_dimension):
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
                warnings.warn(
                    "Target dimension requested is 3, but entity is 2D; skipping resolution assignment."
                )
                return []

        return tags

    def filter_mesh_boundary_tags_by_target_dimension(self, target_dimension):
        match self.dim - target_dimension:
            case 0:
                raise ValueError("Not a boundary!")
            case 1:
                tags = self.mesh_edge_name_interfaces
            case 2 | 3:
                tags = [
                    c
                    for b in self.mesh_edge_name_interfaces
                    for cs in self.model.occ.getCurveLoops(b)[1]
                    for c in cs
                ]
            case -1:
                raise ValueError("Not a boundary!")

        return tags

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
        tags = self.filter_tags_by_target_dimension(target_dimension)

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
        all_entities_dict,
        boundary_delimiter,
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

                # Filter by shared or not shared as well;include boundary
                entities_mass_dict_sharing = {}
                if resolutionspec.sharing is None:
                    superset = {x for xs in all_entities_dict.keys() for x in xs}
                    include_boundary = True
                else:
                    include_boundary = (
                        True if boundary_delimiter in resolutionspec.sharing else False
                    )
                    superset = set(resolutionspec.sharing)
                if resolutionspec.not_sharing is not None:
                    include_boundary = (
                        False
                        if boundary_delimiter in resolutionspec.not_sharing
                        else True
                    )
                    superset -= set(resolutionspec.not_sharing)
                for other_name, other_entity in all_entities_dict.items():
                    # If itself
                    if all(item in other_name for item in self.physical_name):
                        tags = self.filter_tags_by_target_dimension(
                            resolutionspec.target_dimension
                        )
                        if not include_boundary:
                            tags = set(tags) - set(
                                self.filter_mesh_boundary_tags_by_target_dimension(
                                    resolutionspec.target_dimension
                                )
                            )
                        for tag in self.tags:
                            if tag in entities_mass_dict:
                                entities_mass_dict_sharing[tag] = entities_mass_dict[
                                    tag
                                ]
                        continue
                    if any(item in other_name for item in superset):
                        other_tags = other_entity.filter_tags_by_target_dimension(
                            resolutionspec.target_dimension
                        )
                        # Special case if other tag contains a boundary line also shared with self
                        if (
                            not include_boundary
                            and resolutionspec.target_dimension == 1
                        ):
                            other_tags = set(other_tags) - (
                                set(
                                    other_entity.filter_mesh_boundary_tags_by_target_dimension(
                                        resolutionspec.target_dimension
                                    )
                                )
                                & set(
                                    self.filter_mesh_boundary_tags_by_target_dimension(
                                        resolutionspec.target_dimension
                                    )
                                )
                            )
                        for tag in other_tags:
                            if tag in entities_mass_dict:
                                entities_mass_dict_sharing[tag] = entities_mass_dict[
                                    tag
                                ]

                if entities_mass_dict_sharing:
                    new_field_indices, n = resolutionspec.apply(
                        model=self.model,
                        current_field_index=n,
                        refinement_field_indices=refinement_field_indices,
                        entities_mass_dict=entities_mass_dict_sharing,
                    )
                    refinement_field_indices.extend(new_field_indices)

        return refinement_field_indices, n
