"""Internal per-entity state for mesh.py's resolution / refinement driver.

This module is private (underscore prefix). External callers should not
import ``_MeshEntity`` -- it's an implementation detail of the
resolution-spec engine inside :mod:`meshwell.mesh`. The public OCC -> gmsh
bridge lives in :mod:`meshwell.occ_xao_writer`.
"""
import warnings
from typing import Any

import gmsh

from meshwell.resolution import CURVE_SUBTYPE_MAP, DIM_TO_LIST_FIELD, ResolutionSpec


def entity_name_set(physical_name: str | tuple[str, ...]) -> set[str]:
    """Normalize a physical_name (str or tuple of str) to a set of exact names.

    Matching between entities must always be by exact name — substring or
    per-character containment makes "metal" match "metal2".
    """
    if isinstance(physical_name, str):
        return {physical_name}
    return set(physical_name)


class _MeshEntity:
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
        self._explicit_dim = None

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
        if self._explicit_dim is not None:
            return self._explicit_dim

        if not self.dimtags:
            return -1  # Invalid dimension for empty entities

        return next(dim for dim, tag in self.dimtags)

    def filter_tags_by_target_dimension(self, target_dimension: int) -> list[int]:
        """Filter entity tags based on target dimension.

        Args:
            target_dimension: The desired geometric dimension

        Returns:
            List of entity tags at the target dimension: the entity's own tags
            when they coincide (diff 0); its boundaries one dimension below
            (diff 1); recursive-boundary points when targeting dimension 0; and
            the curves of each boundary surface's curve loops for a volume's
            curves (dim 3, target 1). Returns an empty list (after a warning)
            when the target dimension exceeds the entity's.

        Warnings:
            Issues warning if target dimension is incompatible
        """
        diff = self.dim - target_dimension

        if diff < 0:
            warnings.warn(
                f"Target dimension {target_dimension} exceeds entity dimension "
                f"{self.dim}; skipping resolution assignment.",
                stacklevel=2,
            )
            return []
        if diff == 0:
            return list(self.tags)
        if diff == 1:
            return list(self.boundaries)
        if target_dimension == 0:
            # Points from a surface (diff 2) or volume (diff 3): recursive
            # boundary walk straight to dimension 0.
            dimtags = self.model.getBoundary(
                [(self.dim, tag) for tag in self.tags],
                combined=False,
                oriented=False,
                recursive=True,
            )
            return [tag for dim, tag in dimtags if dim == 0]

        # Remaining case: dim 3, target 1 -- curves via each boundary
        # surface's curve loops.
        tags: list[int] = []
        for b in self.boundaries:
            try:
                for cs in self.model.occ.getCurveLoops(b)[1]:
                    tags.extend(cs)
            except Exception as e:
                # Surface may be unknown to OCC (e.g. discrete); skip it.
                import logging

                logging.getLogger(__name__).debug(
                    f"Failed to get curve loops for {b}: {e}"
                )
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
                tags = []
                for b in self.mesh_edge_name_interfaces:
                    try:
                        for cs in self.model.occ.getCurveLoops(b)[1]:
                            tags.extend(cs)
                    except Exception as e:
                        import logging

                        logging.getLogger(__name__).debug(
                            f"Failed to get curve loops for {b}: {e}"
                        )
            case -1:
                raise ValueError("Not a boundary!")

        return tags

    def filter_by_mass(
        self,
        target_dimension: int,
        min_mass: float,
        max_mass: float,
        entity_type_filter: str | None = None,
    ) -> dict[int, float | None]:
        """Filter entities by mass within specified bounds.

        Args:
            target_dimension: The geometric dimension to filter
            min_mass: Minimum mass threshold (exclusive)
            max_mass: Maximum mass threshold (exclusive)
            entity_type_filter: Optional gmsh entity type string (e.g. "Line",
                "Circle") to further filter dim-1 entities by their OCC type.

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

        # If targeting points, need post-filtering filtering
        if target_dimension == 0:
            # Points are selected as the endpoints of curves whose LENGTH
            # is within the mass bounds; fetch the curves explicitly.
            curve_tags = self.filter_tags_by_target_dimension(1)
            filtered_tags = filter_by_target_and_tags(1, curve_tags, min_mass, max_mass)
            points_boundaries_dimtags = [
                self.model.getBoundary([(1, tag)]) for tag in filtered_tags
            ]
            points_dimtags = [x for xs in points_boundaries_dimtags for x in xs]
            return {p[1]: None for p in points_dimtags}

        # Filter the tags based on current dimension and target
        tags = self.filter_tags_by_target_dimension(target_dimension)

        result = filter_by_target_and_tags(target_dimension, tags, min_mass, max_mass)

        # Apply entity type filtering for curve subtypes (e.g. "Line", "Circle")
        if entity_type_filter is not None and target_dimension == 1:
            result = {
                tag: mass
                for tag, mass in result.items()
                if gmsh.model.getEntityType(1, tag) == entity_type_filter
            }

        return result

    def add_refinement_fields_to_model(
        self,
        all_entities_dict,
        boundary_delimiter,
        constant_collector=None,
        tag_to_entity_names=None,
    ):
        """Adds refinement fields to the model based on base_resolution and resolution info."""
        from meshwell.resolution import ConstantInField

        refinement_field_indices = []

        if self.resolutions:
            for resolutionspec in self.resolutions:
                target_dim = resolutionspec.target_dimension
                if target_dim is None:
                    target_dim = self.dim

                entities_mass_dict = self.filter_by_mass(
                    target_dimension=target_dim,
                    min_mass=resolutionspec.min_mass,
                    max_mass=resolutionspec.max_mass,
                    entity_type_filter=CURVE_SUBTYPE_MAP.get(resolutionspec.apply_to),
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

                # Use tag_to_entity_names for O(N) lookup instead of O(N^2)
                if tag_to_entity_names is not None:
                    # Filter entities_mass_dict using the reverse index
                    self_names = entity_name_set(self.physical_name)

                    boundary_tags = set()

                    if not include_boundary:
                        boundary_tags = set(
                            self.filter_mesh_boundary_tags_by_target_dimension(
                                target_dim
                            )
                        )

                    for tag, mass in entities_mass_dict.items():
                        tag_owners = tag_to_entity_names.get((target_dim, tag), set())

                        # If tag is owned by self or by any entity in superset
                        if (tag_owners & self_names) or (tag_owners & superset):
                            # Handle boundary filtering
                            if (
                                not include_boundary
                                and target_dim == 1
                                and tag in boundary_tags
                            ):
                                # Handle tag being boundary of self
                                is_shared_boundary = False
                                other_owners = tag_owners - self_names
                                for owner_name in other_owners:
                                    if owner_name in superset:
                                        other_entity = all_entities_dict[owner_name]
                                        other_boundary_tags = other_entity.filter_mesh_boundary_tags_by_target_dimension(
                                            target_dim
                                        )
                                        if tag not in other_boundary_tags:
                                            is_shared_boundary = True
                                            break

                                if not is_shared_boundary:
                                    continue

                            entities_mass_dict_sharing[tag] = mass

                else:
                    # Legacy O(N^2) fallback
                    for other_name, other_entity in all_entities_dict.items():
                        # If itself
                        if other_name in entity_name_set(self.physical_name):
                            tags = self.filter_tags_by_target_dimension(target_dim)
                            if not include_boundary:
                                tags = set(tags) - set(
                                    self.filter_mesh_boundary_tags_by_target_dimension(
                                        target_dim
                                    )
                                )
                            for tag in tags:
                                if tag in entities_mass_dict:
                                    entities_mass_dict_sharing[
                                        tag
                                    ] = entities_mass_dict[tag]
                            continue
                        if other_name in superset:
                            other_tags = other_entity.filter_tags_by_target_dimension(
                                target_dim
                            )
                            # Special case if other tag contains a boundary line also shared with self
                            if not include_boundary and target_dim == 1:
                                other_tags = set(other_tags) - (
                                    set(
                                        other_entity.filter_mesh_boundary_tags_by_target_dimension(
                                            target_dim
                                        )
                                    )
                                    & set(
                                        self.filter_mesh_boundary_tags_by_target_dimension(
                                            target_dim
                                        )
                                    )
                                )
                            for tag in other_tags:
                                if tag in entities_mass_dict:
                                    entities_mass_dict_sharing[
                                        tag
                                    ] = entities_mass_dict[tag]

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

                restrict_to_str = DIM_TO_LIST_FIELD.get(self.dim)

                if entities_mass_dict_sharing:
                    if constant_collector is not None and isinstance(
                        resolutionspec, ConstantInField
                    ):
                        constant_collector[resolutionspec.resolution][
                            resolutionspec.entity_str
                        ].extend(entities_mass_dict_sharing.keys())
                    else:
                        field_idx = resolutionspec.apply(
                            model=self.model,
                            entities_mass_dict=entities_mass_dict_sharing,
                            restrict_to_str=restrict_to_str,  # RegionsList or SurfaceLists, depends on model dimensionality
                            restrict_to_tags=restrict_to_tags,
                        )
                        if field_idx is not None:
                            refinement_field_indices.append(field_idx)

        return refinement_field_indices
