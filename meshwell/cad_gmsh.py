"""CAD processor."""
from __future__ import annotations

from os import cpu_count
from pathlib import Path

import gmsh

from meshwell.labeledentity import LabeledEntities
from meshwell.model import ModelManager
from meshwell.tag import tag_boundaries, tag_entities, tag_interfaces
from meshwell.validation import (
    unpack_dimtags,
)


def _validate_dimtags_exist(dimtags):
    """Validate that dimtags exist in the model. Returns only valid ones."""
    if not dimtags:
        return []

    valid_dimtags = []
    for dim, tag in dimtags:
        all_entities = gmsh.model.getEntities(dim)
        existing_tags = {entity_tag for _, entity_tag in all_entities}
        if tag in existing_tags:
            valid_dimtags.append((dim, tag))

    return valid_dimtags


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

    def _instantiate_entity(
        self, index: int, entity_obj, progress_bars: bool
    ) -> LabeledEntities:
        """Common logic for instantiating entities."""
        physical_name = entity_obj.physical_name
        if progress_bars and physical_name:
            print(f"Processing {physical_name} - instantiation")

        # Instantiate entity
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
        """Process entities."""
        from meshwell.validation import validate_sweep_topology

        # Validate structured sweep topology before processing
        structural_entities = [
            e for e in entities_list if not getattr(e, "additive", False)
        ]
        validate_sweep_topology(structural_entities)

        # Process all entities
        structural_entity_list, max_dim = self._process_multidimensional_entities(
            entities_list, progress_bars
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
        if not higher_dim_entities or not entity_group:
            return entity_group

        # Collect all dimtags for group fragment operation
        object_dimtags = []
        tool_dimtags = []

        for entity in entity_group:
            if entity.dimtags:
                object_dimtags.extend(entity.dimtags)

        for entity in higher_dim_entities:
            if entity.dimtags:
                tool_dimtags.extend(entity.dimtags)

        if not object_dimtags or not tool_dimtags:
            return entity_group

        if progress_bars:
            print("Processing fragment integration for dimension group")

        # Validate all dimtags exist before fragment
        object_dimtags = _validate_dimtags_exist(object_dimtags)
        tool_dimtags = _validate_dimtags_exist(tool_dimtags)

        # Perform single fragment operation for all entities
        fragment_result = self.model_manager.model.occ.fragment(
            object_dimtags,
            tool_dimtags,
            removeObject=True,
            removeTool=True,
        )
        self.model_manager.model.occ.synchronize()

        if not fragment_result or len(fragment_result) < 2:
            raise ValueError("Fragment operation failed or returned invalid result")

        mapping = fragment_result[1]

        if not mapping or len(mapping) != (len(object_dimtags) + len(tool_dimtags)):
            raise ValueError(
                f"Fragment mapping incomplete: expected {len(object_dimtags) + len(tool_dimtags)} entries, got {len(mapping) if mapping else 0}"
            )

        # Update entity dimtags based on mapping
        object_idx = 0
        for entity in entity_group:
            if entity.dimtags:
                new_dimtags = []
                for _ in entity.dimtags:
                    if object_idx < len(mapping) and mapping[object_idx]:
                        if isinstance(mapping[object_idx], list):
                            new_dimtags.extend(mapping[object_idx])
                        else:
                            new_dimtags.append(mapping[object_idx])
                    object_idx += 1

                entity.dimtags = list(set(new_dimtags))

        # Update higher dimension entities
        tool_idx = len(object_dimtags)
        for entity in higher_dim_entities:
            if entity.dimtags:
                new_dimtags = []
                for _ in entity.dimtags:
                    if tool_idx < len(mapping) and mapping[tool_idx]:
                        if isinstance(mapping[tool_idx], list):
                            new_dimtags.extend(mapping[tool_idx])
                        else:
                            new_dimtags.append(mapping[tool_idx])
                    tool_idx += 1

                entity.dimtags = list(set(new_dimtags))

        self.model_manager.model.occ.synchronize()

        # Return entities that still have valid dimtags
        return [entity for entity in entity_group if entity.dimtags]

    def _process_dimension_group_cuts(
        self, entity_group: list, progress_bars: bool
    ) -> list[LabeledEntities]:
        """Process entities of same dimension using unified fragment and mesh_order selection."""
        if not entity_group:
            return []

        # 1. Instantiate all entities independently
        labeled_entities_with_objs = []
        for index, entity_obj in entity_group:
            ent = self._instantiate_entity(index, entity_obj, progress_bars)
            labeled_entities_with_objs.append((ent, entity_obj))

        all_dimtags = []
        for ent, _ in labeled_entities_with_objs:
            all_dimtags.extend(ent.dimtags)

        if not all_dimtags:
            return []

        # If only one entity, no need to fragment
        if len(all_dimtags) == 1:
            self.model_manager.sync_model()
            return [ent for ent, _ in labeled_entities_with_objs if ent.dimtags]

        # 2. Single fragment operation to resolve all overlaps
        # We use an empty tool list to fragment everything against everything
        fragment_result = self.model_manager.model.occ.fragment(
            all_dimtags, [], removeObject=True, removeTool=True
        )
        self.model_manager.model.occ.synchronize()

        if not fragment_result or len(fragment_result) < 2:
            return [ent for ent, _ in labeled_entities_with_objs if ent.dimtags]

        mapping = fragment_result[1]

        # 3. Assign each fragment to the entity with the lowest mesh_order
        piece_to_owners = {}  # (dim, tag) -> list of (ent, mesh_order)
        dimtag_idx = 0
        for ent, obj in labeled_entities_with_objs:
            mo = obj.mesh_order if obj.mesh_order is not None else float("inf")
            for _ in ent.dimtags:
                for piece in mapping[dimtag_idx]:
                    if piece not in piece_to_owners:
                        piece_to_owners[piece] = []
                    piece_to_owners[piece].append((ent, mo))
                dimtag_idx += 1

        # Reset entity tags and reassign
        for ent, _ in labeled_entities_with_objs:
            ent.dimtags = []

        for piece, owners in piece_to_owners.items():
            # Find owner with minimum mesh_order
            # In case of tie, first in list (original order) wins
            best_ent = min(owners, key=lambda x: x[1])[0]
            best_ent.dimtags.append(piece)

        # 4. Self-fusion: Merge fragments belonging to the same entity
        # This recovers the "single surface" behavior and removes fake internal interfaces
        self.model_manager.model.occ.synchronize()

        # DISABLED self-fusion.
        # OpenCASCADE fuse drops disjoint elements when fusing a list of non-touching elements.
        # Keeping separate conformal fragments is safe and preserves all elements.
        # for ent, _obj in labeled_entities_with_objs:
        #     if len(ent.dimtags) > 1:
        #         try:
        #             fuse_result = self.model_manager.model.occ.fuse(
        #                 [ent.dimtags[0]],
        #                 ent.dimtags[1:],
        #                 removeObject=True,
        #                 removeTool=True,
        #             )
        #             self.model_manager.model.occ.synchronize()
        #             if fuse_result and len(fuse_result) >= 1:
        #                 ent.dimtags = fuse_result[0]
        #         except Exception:
        #             print(
        #                 f"Cannot fuse {ent.dimtags[0]} and {ent.dimtags[1:]}; keeping separate."
        #             )
        #             pass

        # Final cleanup and sync
        self.model_manager.sync_model()

        return [ent for ent, _ in labeled_entities_with_objs if ent.dimtags]

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
) -> dict:
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

    # Create blueprint
    import json

    blueprint = {}
    for entity in entities_list:
        p_name = getattr(entity, "physical_name", None)
        if p_name:
            p_name_keys = p_name if isinstance(p_name, tuple) else [p_name]

            for k in p_name_keys:
                blueprint[k] = {
                    "dim": getattr(entity, "dimension", None),
                    "mesh_structured": getattr(entity, "mesh_structured", False),
                    "extrusion_layers": getattr(entity, "extrusion_layers", None),
                    "recombine": getattr(entity, "recombine", False),
                }

                # Check for z bounds or buffers if it's a PolyPrism
                if hasattr(entity, "buffers"):
                    # the first z and last z
                    zs = list(entity.buffers.keys())
                    if len(zs) >= 2:
                        z_min, z_max = min(zs), max(zs)
                        blueprint[k]["extrusion_vector"] = [
                            0.0,
                            0.0,
                            float(z_max - z_min),
                        ]

    if output_file:
        output_file = Path(output_file)
        blueprint_file = output_file.with_suffix(".json")
        with blueprint_file.open("w") as f:
            json.dump(blueprint, f, indent=2)

    # Save to file
    cad_processor.to_xao(output_file)

    # Finalize if we created the model
    if model is None:
        cad_processor.model_manager.finalize()

    return blueprint
