"""Class definition for tracking gmsh entities."""
import warnings
from typing import Any

import gmsh

from meshwell.resolution import ResolutionSpec


class LabeledEntities:
    """General class to track the gmsh entities that result from the geometry definition."""

    def __init__(
        self,
        index: int,
        model: Any,
        dimtags: list[tuple[int, int]],
        physical_name: str | tuple[str, ...],
        resolutions: list[ResolutionSpec] | None = None,
        keep: bool = True,
        boundaries: list[int] | None = None,
        interfaces: list | None = None,
        mesh_edge_name_interfaces: list | None = None,
    ):
        """Initialize a geometric entity.

        Args:
            index: Unique identifier for this entity
            model: The geometric model (typically a GMSH model)
            dimtags: List of (dimension, tag) pairs identifying geometric entities
            physical_name: Name or names for physical groups
            resolutions: Optional list of mesh resolution specifications
            keep: Whether to keep this entity during operations
            boundaries: Optional list of boundary entity tags
            interfaces: Optional list of interface entities
            mesh_edge_name_interfaces: Optional list of mesh edge interface names
        """
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

    def update_boundaries(self) -> list[int]:
        """Update and return boundary tags for valid entities.

        Filters out non-existent entities before computing boundaries
        to avoid GMSH errors.

        Returns:
            List of boundary entity tags
        """
        # Filter out non-existent entities before getting boundaries
        valid_dimtags = []
        all_entities = {}

        # Get all existing entities by dimension
        for dim in range(4):
            try:
                entities = gmsh.model.getEntities(dim)
                all_entities[dim] = {tag for _, tag in entities}
            except Exception:
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

    def _fuse_self(self, dimtags: list[int | str]) -> list[int | str]:
        """Fuse multiple geometric entities into a single entity.

        Args:
            dimtags: List of entity identifiers to fuse

        Returns:
            List containing the fused entity identifier(s)
        """
        if len(dimtags) == 0:
            return []

        if len(dimtags) != 1:
            dimtags = gmsh.model.occ.fuse(
                [dimtags[0]],
                dimtags[1:],
                removeObject=True,
                removeTool=True,
            )[0]
            self.model.occ.synchronize()
        return dimtags

    @property
    def tags(self) -> list[int]:
        """Extract entity tags from dimension-tag pairs.

        Returns:
            Flattened list of entity tags
        """
        tags = [tag for dim, tag in self.dimtags]
        if any(isinstance(el, list) for el in tags):
            tags = [item for sublist in tags for item in sublist]
        return tags

    @property
    def dim(self) -> int:
        """Get the dimension of this entity.

        Returns:
            The geometric dimension (0=point, 1=curve, 2=surface, 3=volume)
            or -1 if no entities are present
        """
        if not self.dimtags:
            return -1  # Invalid dimension for empty entities
        return next(dim for dim, tag in self.dimtags)

    def boundaries(self) -> list[int]:
        """Get boundary entity tags.

        Returns:
            List of boundary entity tags using safe update logic
        """
        return self.update_boundaries()

    def filter_tags_by_target_dimension(self, target_dimension: int) -> list[int]:
        """Filter entity tags based on target dimension.

        Args:
            target_dimension: The desired geometric dimension

        Returns:
            List of entity tags matching the target dimension

        Warnings:
            Issues warning if target dimension is incompatible
        """
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
                    "Target dimension requested is 3, but entity is 2D; "
                    "skipping resolution assignment.",
                    stacklevel=2,
                )
                return []

        return tags

    def filter_mesh_boundary_tags_by_target_dimension(
        self, target_dimension: int
    ) -> list[int]:
        """Filter mesh boundary tags by target dimension.

        Args:
            target_dimension: The desired geometric dimension

        Returns:
            List of mesh boundary tags matching the target dimension

        Raises:
            ValueError: If the operation is not valid for boundaries
        """
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

    def filter_by_mass(
        self, target_dimension: int, min_mass: float, max_mass: float
    ) -> dict[int, float | None]:
        """Filter entities by mass within specified bounds.

        Args:
            target_dimension: The geometric dimension to filter
            min_mass: Minimum mass threshold (exclusive)
            max_mass: Maximum mass threshold (exclusive)

        Returns:
            Dictionary mapping entity tags to their masses (or None for points)
        """

        def filter_by_target_and_tags(
            target_dimension: int, tags: list[int], min_mass: float, max_mass: float
        ) -> dict[int, float]:
            """Filter tags by mass and return tag-mass mapping.

            Returns:
                Dictionary mapping tags to their masses
            """
            filtered_tags = [
                tag
                for tag in tags
                if min_mass < self.model.occ.getMass(target_dimension, tag) < max_mass
            ]
            return (
                {
                    tag: self.model.occ.getMass(target_dimension, tag)
                    for tag in filtered_tags
                }
                if filtered_tags
                else {}
            )

        # Filter the tags based on current dimension and target
        tags = self.filter_tags_by_target_dimension(target_dimension)

        # If targeting points, need post-filtering filtering
        if target_dimension == 0:
            filtered_tags = filter_by_target_and_tags(1, tags, min_mass, max_mass)
            points_boundaries_dimtags = [
                self.model.getBoundary([(1, tag)]) for tag in filtered_tags
            ]
            points_dimtags = [x for xs in points_boundaries_dimtags for x in xs]
            return {p[1]: None for p in points_dimtags}

        return filter_by_target_and_tags(target_dimension, tags, min_mass, max_mass)

    def add_refinement_fields_to_model(
        self,
        all_entities_dict,
        boundary_delimiter,
    ):
        """Adds refinement fields to the model based on base_resolution and resolution info."""
        refinement_field_indices = []

        if self.resolutions:
            for resolutionspec in self.resolutions:
                entities_mass_dict = self.filter_by_mass(
                    target_dimension=resolutionspec.target_dimension,
                    min_mass=resolutionspec.min_mass,
                    max_mass=resolutionspec.max_mass,
                )

                # Filter by shared or not shared as well; include boundary
                entities_mass_dict_sharing = {}
                if resolutionspec.sharing is None:
                    superset = set(all_entities_dict.keys())
                    include_boundary = True
                else:
                    include_boundary = boundary_delimiter in resolutionspec.sharing
                    superset = set(resolutionspec.sharing)
                if resolutionspec.not_sharing is not None:
                    include_boundary = (
                        boundary_delimiter not in resolutionspec.not_sharing
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
                        for tag in tags:
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

                # Also retrieve tags of entities restricted_to
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

                if entities_mass_dict_sharing:
                    refinement_field_indices.append(
                        resolutionspec.apply(
                            model=self.model,
                            entities_mass_dict=entities_mass_dict_sharing,
                            restrict_to_str=restrict_to_str,  # RegionsList or SurfaceLists, depends on model dimensionality
                            restrict_to_tags=restrict_to_tags,
                        )
                    )

        return refinement_field_indices
