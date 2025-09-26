"""Class definition for generating geometry and saving to a CAD file."""

from __future__ import annotations

from os import cpu_count
from pathlib import Path

from meshwell.labeledentity import LabeledEntities
from meshwell.model import ModelManager
from meshwell.tag import tag_boundaries, tag_entities, tag_interfaces
from meshwell.validation import (
    unpack_dimtags,
)


class CAD:
    """CAD class for generating geometry and saving to .xao format."""

    def __init__(
        self,
        point_tolerance: float = 1e-3,
        n_threads: int = cpu_count(),
        filename: str = "temp",
        model: ModelManager | None = None,
    ):
        """Initialize CAD processor.

        Args:
            point_tolerance: Tolerance for point merging
            n_threads: Number of threads for processing
            filename: Base filename for the model
            model: Optional Model instance to use (creates new if None)

        """
        # Use provided model or create new one
        if model is None:
            self.model_manager = ModelManager(
                n_threads=n_threads,
                filename=filename,
                point_tolerance=point_tolerance,
            )
            self._owns_model = True
        else:
            self.model_manager = model
            self._owns_model = False

        # Store parameters for backward compatibility
        self.point_tolerance = point_tolerance

        # Shared point cache for geometry entities that support point deduplication
        self._shared_point_cache: dict[tuple[float, float, float], int] = {}

    def _instanciate_entity(
        self, index: int, entity_obj, progress_bars: bool
    ) -> LabeledEntities:
        """Common logic for instantiating entities."""
        physical_name = entity_obj.physical_name
        if progress_bars and physical_name:
            print(f"Processing {physical_name} - instantiation")

        # Set up shared point cache for geometry entities that support it
        if hasattr(entity_obj, "_set_point_cache"):
            entity_obj._set_point_cache(self._shared_point_cache)

        # instantiate entity
        dimtags_out = entity_obj.instanciate(self)
        dimtags = unpack_dimtags(dimtags_out)

        return LabeledEntities(
            index=index,
            dimtags=dimtags,
            physical_name=physical_name,
            keep=entity_obj.mesh_bool,
            model=self.model_manager.model,
        )

    def _process_entities(
        self,
        entities_list: list,
        progress_bars: bool,
    ) -> tuple[list, int]:
        """Process structura entities."""
        # Separate and order entities
        structural_entities = [e for e in entities_list if not e.additive]

        # Process structural entities
        structural_entity_list, max_dim = self._process_multidimensional_entities(
            structural_entities, progress_bars
        )

        return structural_entity_list, max_dim

    def _process_multidimensional_entities(
        self, structural_entities: list, progress_bars: bool
    ) -> tuple[list, int]:
        """Process entities with mixed dimensions using hierarchical approach."""
        entity_dimensions = []
        max_dim = 0
        for index, entity_obj in enumerate(structural_entities):
            dim = entity_obj.dimension
            max_dim = max(dim, max_dim)
            entity_dimensions.append((dim, index, entity_obj))

        # Group entities by dimension and sort by mesh_order within each dimension
        dimension_groups = {0: [], 1: [], 2: [], 3: []}
        for dim, index, entity_obj in entity_dimensions:
            dimension_groups[dim].append((index, entity_obj))

        # Sort each dimension group by mesh_order
        for dim in dimension_groups:
            dimension_groups[dim].sort(
                key=lambda x: x[1].mesh_order
                if x[1].mesh_order is not None
                else float("inf")
            )

        # Process entities by dimension (highest first)
        all_processed_entities = []
        dim = max_dim

        while dim >= 0:
            if dimension_groups[dim]:
                # Process entities of same dimension in mesh_order sequence
                current_dimension_entities = self._process_dimension_group_cuts(
                    dimension_groups[dim], progress_bars
                )

                # Second pass: Fragment against higher dimensional entities
                if all_processed_entities:
                    current_dimension_entities = (
                        self._process_dimension_group_fragments(
                            current_dimension_entities,
                            progress_bars,
                            all_processed_entities,
                        )
                    )

                all_processed_entities.extend(current_dimension_entities)
            dim -= 1

        return all_processed_entities, max_dim

    def _process_dimension_group_fragments(
        self,
        entity_group: list[LabeledEntities],
        progress_bars: bool,
        higher_dim_entities: list[LabeledEntities],
    ) -> list[LabeledEntities]:
        """Fragment processing for entities against higher dimensional entities."""
        processed_entities = []

        for current_entity in entity_group:
            if progress_bars and current_entity.physical_name:
                print(
                    f"Processing {current_entity.physical_name} - fragment integration"
                )

            # Fragment against higher-dimensional entities if they exist
            if higher_dim_entities and current_entity.dimtags:
                for higher_dim_entity in higher_dim_entities:
                    if not higher_dim_entity.dimtags or not current_entity.dimtags:
                        continue

                    fragment_result = self.model_manager.model.occ.fragment(
                        current_entity.dimtags,
                        higher_dim_entity.dimtags,
                        removeObject=True,
                        removeTool=True,
                    )
                    self.model_manager.sync_model()

                    # Update current_entity dimtags based on fragment result
                    if (
                        fragment_result
                        and len(fragment_result) > 1
                        and fragment_result[1]
                    ):
                        num_object_dimtags = len(current_entity.dimtags)
                        new_object_dimtags = []

                        for idx in range(num_object_dimtags):
                            if idx < len(fragment_result[1]):
                                mapping = fragment_result[1][idx]
                                if mapping:
                                    if isinstance(mapping, list):
                                        new_object_dimtags.extend(mapping)
                                    else:
                                        new_object_dimtags.append(mapping)

                        current_entity.dimtags = new_object_dimtags

            if current_entity.dimtags:
                processed_entities.append(current_entity)

        return processed_entities

    def _process_dimension_group_cuts(
        self, entity_group: list, progress_bars: bool
    ) -> list[LabeledEntities]:
        """Process entities of same dimension using cuts."""
        processed_entities = []

        for i, (index, entity_obj) in enumerate(entity_group):
            # instantiate entity using helper method
            current_entity = self._instanciate_entity(index, entity_obj, progress_bars)

            if i == 0:
                processed_entities.append(current_entity)
            else:
                # Cut against previously processed entities
                tool_dimtags = [
                    dimtag
                    for prev_entity in processed_entities
                    for dimtag in prev_entity.dimtags
                    if dimtag
                ]

                if tool_dimtags and current_entity.dimtags:
                    cut = self.model_manager.model.occ.cut(
                        current_entity.dimtags,
                        tool_dimtags,
                        removeObject=True,
                        removeTool=False,
                    )
                    if cut and cut[0]:
                        current_entity.dimtags = list(set(cut[0]))

                if current_entity.dimtags:
                    processed_entities.append(current_entity)
                self.model_manager.sync_model()
                # Clear shared point cache after boolean operations
                self._shared_point_cache.clear()

        return processed_entities

    def _tag_mesh_components(
        self,
        final_entity_list: list,
        max_dim: int,
        interface_delimiter: str,
        boundary_delimiter: str,
    ) -> None:
        """Tag entities, interfaces, and boundaries."""
        # Update entity boundaries
        for entity in final_entity_list:
            entity.update_boundaries()

        # Filter out entities that became invalid after boundary update
        valid_final_entities = [
            entity for entity in final_entity_list if entity.dimtags and entity.dim >= 0
        ]

        # Tag entities
        if valid_final_entities:
            tag_entities(valid_final_entities, self.model_manager.model)
            # Tag highest-dimension model interfaces, model boundaries
            valid_final_entities = tag_interfaces(
                valid_final_entities,
                max_dim,
                interface_delimiter,
                self.model_manager.model,
            )
            # Tag model boundaries
            tag_boundaries(
                valid_final_entities,
                max_dim,
                interface_delimiter,
                boundary_delimiter,
                self.model_manager.model,
            )

    def process_entities(
        self,
        entities_list: list,
        interface_delimiter: str = "___",
        boundary_delimiter: str = "None",
        progress_bars: bool = False,
    ) -> list:
        """Process entities and return final entity list (no file I/O).

        Args:
            entities_list: List of entities to process
            interface_delimiter: Delimiter for interface names
            boundary_delimiter: Delimiter for boundary names
            progress_bars: Show progress bars during processing

        Returns:
            List of processed LabeledEntities

        """
        self.model_manager.ensure_initialized(str(self.model_manager.filename))

        # Process entities and get max dimension
        final_entity_list, max_dim = self._process_entities(
            entities_list=entities_list,
            progress_bars=progress_bars,
        )

        # Tag entities and boundaries (filtering happens inside _tag_mesh_components)
        self._tag_mesh_components(
            final_entity_list=final_entity_list,
            max_dim=max_dim,
            interface_delimiter=interface_delimiter,
            boundary_delimiter=boundary_delimiter,
        )

        # Delete entities that are not marked to keep
        for entity in final_entity_list:
            if not entity.keep and entity.dimtags:
                self.model_manager.model.occ.remove(entity.dimtags, recursive=True)
                self.model_manager.model.occ.synchronize()

        return final_entity_list

    def to_xao(self, output_file: Path) -> None:
        """Save current model state to .xao file.

        Args:
            output_file: Output file path (will be suffixed with .xao)

        """
        self.model_manager.save_to_xao(output_file)


def cad(
    entities_list: list,
    output_file: Path,
    point_tolerance: float = 1e-3,
    n_threads: int = cpu_count(),
    filename: str = "temp",
    interface_delimiter: str = "___",
    boundary_delimiter: str = "None",
    progress_bars: bool = False,
    model: ModelManager | None = None,
) -> None:
    """Utility function that wraps the CAD class for easier usage.

    Args:
        entities_list: List of entities to process
        output_file: Output file path
        point_tolerance: Tolerance for point merging
        n_threads: Number of threads to use for processing
        filename: Temporary filename for GMSH model
        interface_delimiter: Delimiter for interface names
        boundary_delimiter: Delimiter for boundary names
        progress_bars: Show progress bars during processing
        model: Optional Model instance to use (creates new if None)

    """
    cad_processor = CAD(
        point_tolerance=point_tolerance,
        n_threads=n_threads,
        filename=filename,
        model=model,
    )

    # Process entities
    cad_processor.process_entities(
        entities_list=entities_list,
        interface_delimiter=interface_delimiter,
        boundary_delimiter=boundary_delimiter,
        progress_bars=progress_bars,
    )

    # Save to file
    cad_processor.to_xao(output_file)

    # Finalize if we created the model
    if model is None:
        cad_processor.model_manager.finalize()
