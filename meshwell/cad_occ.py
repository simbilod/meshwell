"""OCC CAD processor using OCP (OpenCASCADE Python) bindings."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from os import cpu_count
from typing import TYPE_CHECKING, Any

from OCP.BOPAlgo import BOPAlgo_Builder
from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_SOLID, TopAbs_VERTEX
from OCP.TopExp import TopExp_Explorer

if TYPE_CHECKING:
    from OCP.TopoDS import TopoDS_Shape


@dataclass
class OCCLabeledEntity:
    """Dataclass to store OCC shape(s) and associated metadata.

    shapes holds the fragment pieces this entity owns after the all-fragment pass.
    """

    shapes: list[TopoDS_Shape]
    physical_name: tuple[str, ...]
    index: int
    keep: bool
    dim: int


def _resolve_piece_ownership(
    piece_candidates: dict[Any, list[tuple[int, float]]],
) -> dict[Any, int]:
    """Pick the owning entity index for each fragment piece.

    Rule: lowest mesh_order wins. On tie, first candidate in insertion order wins.

    Args:
        piece_candidates: maps piece key -> list of (entity_index, mesh_order).

    Returns:
        dict mapping piece key -> winning entity_index.
    """
    owners: dict[Any, int] = {}
    for piece, candidates in piece_candidates.items():
        best_idx = candidates[0][0]
        best_mo = candidates[0][1]
        for idx, mo in candidates[1:]:
            if mo < best_mo:
                best_idx = idx
                best_mo = mo
        owners[piece] = best_idx
    return owners


class CAD_OCC:
    """CAD class for generating geometry using OpenCASCADE (via OCP)."""

    def __init__(
        self,
        point_tolerance: float = 1e-3,
        n_threads: int = cpu_count(),
    ):
        """Initialize OCC CAD processor.

        Args:
            point_tolerance: Tolerance for boolean operations (Fuzzy value)
            n_threads: Number of threads for parallel processing
        """
        self.point_tolerance = point_tolerance
        self.n_threads = n_threads

    def _get_shape_dimension(self, shape: TopoDS_Shape) -> int:
        """Infer dimension from TopoDS_Shape type."""
        # Check for Solids (3D)
        explorer = TopExp_Explorer(shape, TopAbs_SOLID)
        if explorer.More():
            return 3

        # Check for Faces (2D)
        explorer = TopExp_Explorer(shape, TopAbs_FACE)
        if explorer.More():
            return 2

        # Check for Edges (1D)
        explorer = TopExp_Explorer(shape, TopAbs_EDGE)
        if explorer.More():
            return 1

        # Check for Vertices (0D)
        explorer = TopExp_Explorer(shape, TopAbs_VERTEX)
        if explorer.More():
            return 0

        return -1

    def _instantiate_entity_occ(self, index: int, entity_obj: Any) -> OCCLabeledEntity:
        """Instantiate a single entity into an OCC shape."""
        shape = entity_obj.instanciate_occ()
        dim = self._get_shape_dimension(shape)

        return OCCLabeledEntity(
            shapes=[shape],
            physical_name=entity_obj.physical_name,
            index=index,
            keep=entity_obj.mesh_bool,
            dim=dim,
        )

    def _process_dimension_group_cuts_occ(
        self, entities: list[OCCLabeledEntity], ent_objs: list[Any]
    ) -> list[OCCLabeledEntity]:
        """Process entities of same dimension using cuts based on mesh_order.

        This implementation groups by mesh_order to leverage batch parallelization
        offered by BOPAlgo_Builder.
        """
        if not entities:
            return []

        # Group entities by mesh_order
        from collections import defaultdict

        groups = defaultdict(list)
        for ent, obj in zip(entities, ent_objs):
            mo = obj.mesh_order if obj.mesh_order is not None else float("inf")
            groups[mo].append(ent)

        sorted_orders = sorted(groups.keys())

        all_processed_entities = []
        accumulated_shapes = []  # Shapes that will cut future groups

        for mo in sorted_orders:
            current_group = groups[mo]

            # 1. Resolve overlaps within the group using BOPAlgo_Builder (Parallel)
            if len(current_group) > 1:
                builder = BOPAlgo_Builder()
                builder.SetRunParallel(self.n_threads > 1)
                builder.SetFuzzyValue(self.point_tolerance)
                builder.SetNonDestructive(False)

                for ent in current_group:
                    builder.AddArgument(ent.shape)

                builder.Perform()

                # Update each entity shape from builder results
                for ent in current_group:
                    modified = builder.Modified(ent.shape)
                    if not modified.IsEmpty():
                        from OCP.BRep import BRep_Builder
                        from OCP.TopoDS import TopoDS_Compound

                        comp_builder = BRep_Builder()
                        new_comp = TopoDS_Compound()
                        comp_builder.MakeCompound(new_comp)
                        for s in modified:
                            comp_builder.Add(new_comp, s)
                        ent.shape = new_comp
                    elif builder.IsDeleted(ent.shape):
                        # Shape was completely swallowed by another in the same group
                        # (Possible if shapes are identical or one contains another)
                        pass

            # 2. Cut current group against ALL previous shapes (Parallel)
            if accumulated_shapes:
                from OCP.BRep import BRep_Builder
                from OCP.TopoDS import TopoDS_Compound

                comp_builder = BRep_Builder()
                compound_tool = TopoDS_Compound()
                comp_builder.MakeCompound(compound_tool)
                for s in accumulated_shapes:
                    comp_builder.Add(compound_tool, s)

                for ent in current_group:
                    cut_api = BRepAlgoAPI_Cut(ent.shape, compound_tool)
                    cut_api.SetRunParallel(self.n_threads > 1)
                    cut_api.SetFuzzyValue(self.point_tolerance)
                    cut_api.SetNonDestructive(False)
                    cut_api.Build()
                    ent.shape = cut_api.Shape()

            # 3. Add to processed and accumulated
            all_processed_entities.extend(current_group)
            accumulated_shapes.extend([ent.shape for ent in current_group])

        return all_processed_entities

    def _process_dimension_group_fragments_occ(
        self,
        entity_group: list[OCCLabeledEntity],
        higher_dim_entities: list[OCCLabeledEntity],
    ) -> list[OCCLabeledEntity]:
        """Fragment processing for entities against higher dimensional entities."""
        if not higher_dim_entities or not entity_group:
            return entity_group

        builder = BOPAlgo_Builder()
        builder.SetRunParallel(self.n_threads > 1)
        builder.SetFuzzyValue(self.point_tolerance)
        builder.SetNonDestructive(False)

        # Add all shapes to the builder
        all_entities = entity_group + higher_dim_entities
        shape_to_entity_map = {}

        for ent in all_entities:
            builder.AddArgument(ent.shape)
            # Store the original shape to map back later
            # Note: We use the hash of the shape if possible, or just the identity
            shape_to_entity_map[ent.shape] = ent

        builder.Perform()

        # Update each entity with its modified counterparts
        for ent in all_entities:
            modified_shapes = builder.Modified(ent.shape)
            if not modified_shapes.IsEmpty():
                # If modified, the new shape is a compound of the modified parts
                # or we just need the new resulting shapes.
                # In OCC, builder.Modified() returns a list of shapes.
                # We need to collect them.
                from OCP.BRep import BRep_Builder
                from OCP.TopoDS import TopoDS_Compound

                comp_builder = BRep_Builder()
                new_compound = TopoDS_Compound()
                comp_builder.MakeCompound(new_compound)

                for modified_shape in modified_shapes:
                    comp_builder.Add(new_compound, modified_shape)
                ent.shape = new_compound
            elif builder.IsDeleted(ent.shape):
                # If deleted, we should probably mark it or handle it
                # For now just keep existing shape if NOT deleted,
                # but if deleted it might mean it's been swallowed or failed.
                pass

        return entity_group

    def process_entities(
        self,
        entities_list: list[Any],
        _progress_bars: bool = False,  # Included for interface compatibility
    ) -> list[OCCLabeledEntity]:
        """Process entities and return list of OCCLabeledEntity objects.

        Args:
            entities_list: List of entity objects (PolyPrism, PolySurface, etc.)
            progress_bars: Ignored, for interface compatibility

        Returns:
            list[OCCLabeledEntity]: Processed entities ready for meshing or export
        """
        # Group by dimension and sort by mesh_order
        dimension_groups: dict[int, list[Any]] = {0: [], 1: [], 2: [], 3: []}
        max_dim = 0

        # Instantiate and infer dimension early in parallel
        with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
            labeled_entities = list(
                executor.map(
                    lambda x: self._instantiate_entity_occ(x[0], x[1]),
                    enumerate(entities_list),
                )
            )

        for ent_obj, labeled_ent in zip(entities_list, labeled_entities):
            # Use dimension from entity if available, otherwise from shape
            dim = getattr(ent_obj, "dimension", None)
            if dim is None:
                dim = labeled_ent.dim

            # Ensure dim is an integer and not None for max()
            if dim is None:
                dim = -1

            max_dim = max(max_dim, dim)

            # Ensure dimension group exists
            if dim not in dimension_groups:
                dimension_groups[dim] = []

            dimension_groups[dim].append((ent_obj, labeled_ent))

        # Sort each group by mesh_order
        for d in range(4):
            dimension_groups[d].sort(
                key=lambda x: x[0].mesh_order
                if x[0].mesh_order is not None
                else float("inf")
            )

        all_processed_entities: list[OCCLabeledEntity] = []

        # Process from highest dimension down
        for d in range(max_dim, -1, -1):
            if not dimension_groups[d]:
                continue

            # Entities of current dimension
            inc_labeled_entities = [x[1] for x in dimension_groups[d]]
            current_dim_entities = self._process_dimension_group_cuts_occ(
                inc_labeled_entities, [x[0] for x in dimension_groups[d]]
            )

            # 2. Fragment against higher dimensional entities
            if all_processed_entities:
                current_dim_entities = self._process_dimension_group_fragments_occ(
                    current_dim_entities, all_processed_entities
                )

            all_processed_entities.extend(current_dim_entities)

        return all_processed_entities


def cad_occ(
    entities_list: list[Any],
    point_tolerance: float = 1e-3,
    n_threads: int = cpu_count(),
) -> list[OCCLabeledEntity]:
    """Utility function for OCC-based CAD processing."""
    processor = CAD_OCC(point_tolerance=point_tolerance, n_threads=n_threads)
    return processor.process_entities(entities_list)
