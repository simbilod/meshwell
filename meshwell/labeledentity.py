import gmsh
from typing import List, Union, Any, Tuple
from meshwell.resolution import ResolutionSpec
import warnings


class LabeledEntities:
    """General class to track the gmsh entities that result from the geometry definition."""

    def __init__(
        self,
        index: int,
        model: Any,
        dimtags: List[Tuple[int, int]],
        physical_name: Union[str, tuple[str, ...]],
        resolutions: List[ResolutionSpec] | None = None,
        keep: bool = True,
        boundaries: List[int] = None,
        interfaces: List = None,
        mesh_edge_name_interfaces: List = None,
    ):
        self.index = index
        self.model = model
        self.dimtags = dimtags
        self.physical_name = physical_name
        self.resolutions = resolutions
        self.keep = keep
        self.boundaries = boundaries or []
        self.interfaces = interfaces or []
        self.mesh_edge_name_interfaces = mesh_edge_name_interfaces or []

    def to_dict(self) -> dict:
        """Convert entity to dictionary representation.

        Returns:
            Dictionary containing serializable entity data
        """
        return {
            "index": self.index,
            "dimtags": self.dimtags,
            "physical_name": self.physical_name,
            "resolutions": [r.__dict__ for r in self.resolutions]
            if self.resolutions
            else None,
            "keep": self.keep,
            "boundaries": self.boundaries,
            "interfaces": self.interfaces,
            "mesh_edge_name_interfaces": self.mesh_edge_name_interfaces,
        }

    def update_boundaries(self) -> List[int]:
        # Filter out non-existent entities before getting boundaries
        valid_dimtags = []
        all_entities = {}

        # Get all existing entities by dimension
        for dim in range(4):
            try:
                entities = gmsh.model.getEntities(dim)
                all_entities[dim] = {tag for _, tag in entities}
            except:  # noqa: E722
                all_entities[dim] = set()

        # Filter dimtags to only include existing entities
        for dim, tag in self.dimtags:
            if tag in all_entities.get(dim, set()):
                valid_dimtags.append((dim, tag))

        # Update dimtags to only valid ones
        self.dimtags = valid_dimtags

        # Get boundaries only for valid entities
        if valid_dimtags:
            self.boundaries = [
                tag
                for dim, tag in gmsh.model.getBoundary(
                    valid_dimtags, True, False, False
                )
            ]
        else:
            self.boundaries = []

        return self.boundaries

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
        if not self.dimtags:
            return -1  # Invalid dimension for empty entities
        return [dim for dim, tag in self.dimtags][0]

    def boundaries(self) -> List[int]:
        # Use the same safe logic as update_boundaries
        return self.update_boundaries()

    def filter_tags_by_target_dimension(self, target_dimension):
        match self.dim - target_dimension:
            case 0:
                tags = self.tags
            case 1:
                tags = self.boundaries
            case 2 | 3:
                if self.dim == 2:
                    tags = self.boundaries  # boundaries are curves already
                else:
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
                if self.dim == 2:
                    tags = (
                        self.mesh_edge_name_interfaces
                    )  # boundaries are curves already
                else:
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
            points_boundaries_dimtags = [
                self.model.getBoundary([(1, tag)]) for tag in filtered_tags.keys()
            ]
            points_dimtags = [x for xs in points_boundaries_dimtags for x in xs]
            points_mass_dict = {p[1]: None for p in points_dimtags}
            return points_mass_dict
        else:
            return filter_by_target_and_tags(target_dimension, tags, min_mass, max_mass)

    def add_refinement_fields_to_model(
        self,
        all_entities_dict,
        boundary_delimiter,
        dimtags_physical_mapping: dict,
    ):
        """
        Adds refinement fields to the model based on base_resolution and resolution info.
        """
        refinement_field_indices = []

        if self.resolutions:
            for resolutionspec in self.resolutions:
                # This returns a dict of tags: mass that pass the filtering
                entities_mass_dict = self.filter_by_mass(
                    target_dimension=resolutionspec.target_dimension,
                    min_mass=resolutionspec.min_mass,
                    max_mass=resolutionspec.max_mass,
                )

                # Further filter by shared or not shared with others
                entities_mass_dict_sharing = {}

                # If sharing not specified, all entities are candidates
                if resolutionspec.sharing is None:
                    superset = set(all_entities_dict.keys())
                    include_boundary = True
                # Otherwise only considier the specified ones in sharing
                else:
                    include_boundary = (
                        True if boundary_delimiter in resolutionspec.sharing else False
                    )
                    superset = set(resolutionspec.sharing)

                # Reduce the superset by removing not_sharing entities
                if resolutionspec.not_sharing is not None:
                    include_boundary = (
                        False
                        if boundary_delimiter in resolutionspec.not_sharing
                        else True
                    )
                    superset -= set(resolutionspec.not_sharing)

                # Now loop over all entities to find shared tags
                for other_name, other_entity in all_entities_dict.items():
                    # If self
                    if all(item in other_name for item in self.physical_name):
                        tags = self.filter_tags_by_target_dimension(
                            resolutionspec.target_dimension
                        )
                        # Don't consider boundary tags if needed
                        if not include_boundary:
                            tags = set(tags) - set(
                                self.filter_mesh_boundary_tags_by_target_dimension(
                                    resolutionspec.target_dimension
                                )
                            )
                        # Perform filtering
                        for tag in tags:
                            if tag in entities_mass_dict:
                                entities_mass_dict_sharing[tag] = entities_mass_dict[
                                    tag
                                ]
                        continue

                    # If other
                    if any(item in other_name for item in superset):
                        # Get tags of the other entity
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

                        # Perform filtering
                        for tag in other_tags:
                            if tag in entities_mass_dict:
                                entities_mass_dict_sharing[tag] = entities_mass_dict[
                                    tag
                                ]

                # Find tags to restrict to as well
                restrict_to_tags = []
                if resolutionspec.restrict_to is not None:
                    for other_name, other_entity in all_entities_dict.items():
                        if any(
                            item in other_name for item in resolutionspec.restrict_to
                        ):
                            restrict_to_tags.extend(other_entity.tags)
                else:
                    restrict_to_tags = None

                if self.dim == 3:
                    restrict_to_str = "VolumesList"
                elif self.dim == 2:
                    restrict_to_str = "SurfacesList"
                elif self.dim == 1:
                    restrict_to_str = "CurvesList"
                elif self.dim == 0:
                    restrict_to_str = "PointsList"

                # If there are entities left, apply the refinement fields
                if entities_mass_dict_sharing:
                    refinement_field_indices.append(
                        resolutionspec.apply(
                            model=self.model,
                            entities_mass_dict=entities_mass_dict_sharing,
                            restrict_to_str=restrict_to_str,  # depends on entity dimensionality
                            restrict_to_tags=restrict_to_tags,
                        )
                    )

        return refinement_field_indices

    def recover_interfaces(
        self, boundary_delimiter: str, dimtags_physical_mapping: dict
    ):
        """For a model that was loaded from CAD (and hence is fully tagged already)."""
        self.update_boundaries()  # get all the boundaries back

        recovered_interfaces = []
        recovered_mesh_edge_name_interfaces = []
        for boundary_tag in self.boundaries:
            boundary_physical = dimtags_physical_mapping[(self.dim - 1, boundary_tag)]
            if boundary_delimiter in boundary_physical:
                recovered_mesh_edge_name_interfaces.append(boundary_tag)
            else:
                recovered_interfaces.append(boundary_tag)

        self.interfaces = recovered_interfaces
        self.mesh_edge_name_interfaces = recovered_mesh_edge_name_interfaces
