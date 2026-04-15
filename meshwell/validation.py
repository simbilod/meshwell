"""Mesh validation routines."""
import math

from meshwell.labeledentity import LabeledEntities


def validate_dimtags(dimtags):
    """Validate that all dimension-tag pairs have the same dimension.

    Args:
        dimtags: List of (dimension, tag) tuples representing geometric entities

    Returns:
        int: The common dimension of all entities

    Raises:
        ValueError: If entities have different dimensions
    """
    dims = [dim for dim, tag in dimtags]
    if len(set(dims)) != 1:
        raise ValueError(
            "All the entities corresponding to a mesh physical_name must be of the same dimension."
        )
    return dims[0]


def format_physical_name(physical_name: str | tuple[str, ...]) -> tuple[str, ...]:
    """Format a physical name to ensure consistent tuple representation.

    Args:
        physical_name: The physical name to format

    Returns:
        tuple: A tuple containing the physical name
    """
    if isinstance(physical_name, str):
        return (physical_name,)
    return physical_name


def unpack_dimtags(dimtags):
    """Unpack and flatten dimension-tag pairs into a consistent format.

    Takes a list of (dimension, tag) pairs and ensures all tags are at the same
    level, flattening any nested lists of tags while preserving the dimension.

    Args:
        dimtags: List of (dimension, tag) tuples, where tags may be nested lists

    Returns:
        list: List of (dimension, tag) tuples with flattened tags
    """
    if not dimtags:
        return []
    dim = next(dim for dim, tag in dimtags)
    tags = [tag for dim, tag in dimtags]
    if any(isinstance(el, list) for el in tags):
        tags = [item for sublist in tags for item in sublist]
    return [(dim, tag) for tag in tags]


def assign_mesh_order_from_ordering(entities, start_index: int = 0):
    """Assigns a mesh_order according to the ordering of entities in the list."""
    for index, entity in enumerate(entities, start=start_index):
        entity.mesh_order = index
    return entities


def sort_entities_by_mesh_order(entities):
    """Returns a list of entities, sorted by mesh_order."""
    return sorted(entities, key=lambda entity: entity.mesh_order)


def order_entities(entities):
    """Returns a list of entities, sorted by mesh_order, assigning a mesh order corresponding to the ordering if not defined."""
    defined_order_entities = [
        entity for entity in entities if entity.mesh_order is not None
    ]
    ordered_defined_entities = sort_entities_by_mesh_order(defined_order_entities)
    undefined_order_entities = [
        entity for entity in entities if entity.mesh_order is None
    ]
    if ordered_defined_entities:
        start_index = math.ceil(ordered_defined_entities[-1].mesh_order) + 1
    else:
        start_index = 1
    ordered_undefined_entities = assign_mesh_order_from_ordering(
        undefined_order_entities,
        start_index=start_index,
    )
    return ordered_defined_entities + ordered_undefined_entities


def consolidate_entities_by_physical_name(entities):
    """Returns a new list of LabeledEntities, with a single entity per physical_name."""
    physical_name_dict = {}

    for entity in entities:
        if entity.physical_name not in physical_name_dict:
            physical_name_dict[entity.physical_name] = entity
        else:
            existing_entity = physical_name_dict[entity.physical_name]
            combined_dimtags = existing_entity.dimtags + entity.dimtags
            combined_entity = LabeledEntities(
                index=existing_entity.index,
                dimtags=combined_dimtags,
                physical_name=existing_entity.physical_name,
                resolutions=existing_entity.resolutions,
                keep=existing_entity.keep,
                model=existing_entity.model,
            )
            physical_name_dict[entity.physical_name] = combined_entity

    return list(physical_name_dict.values())


def validate_sweep_topology(entities: list):
    """Validate that structured extrusions have higher priority than unstructured entities.

    If an entity is marked as `mesh_structured=True`, it must never be cut or mutated
    by an unstructured entity during 3D Boolean operations, otherwise its mathematically
    sweepable topology will be destroyed. This means structured entities must have a
    strictly lower `mesh_order` value (higher resolving priority) than unstructured ones.

    Args:
        entities: List of CAD entities.

    Raises:
        ValueError: If an unstructured entity has a higher or equal priority than a structured entity.
    """
    structured_entities = [e for e in entities if getattr(e, "mesh_structured", False)]
    unstructured_entities = [
        e for e in entities if not getattr(e, "mesh_structured", False)
    ]

    if not structured_entities or not unstructured_entities:
        return

    # Treat None mesh_order as float("inf")
    def get_order(e):
        mo = getattr(e, "mesh_order", None)
        return float("inf") if mo is None else mo

    # Find the weakest structured priority (maximum mesh_order value)
    structured_orders = [get_order(e) for e in structured_entities]
    max_structured_order = max(structured_orders)

    # Find the strongest unstructured priority (minimum mesh_order value)
    unstructured_orders = [get_order(e) for e in unstructured_entities]
    min_unstructured_order = min(unstructured_orders)

    if max_structured_order >= min_unstructured_order:
        # Find exactly which ones are offending to give a good error message
        offending_structured = [
            e for e in structured_entities if get_order(e) == max_structured_order
        ]
        offending_unstructured = [
            e for e in unstructured_entities if get_order(e) == min_unstructured_order
        ]

        struct_names = [
            getattr(e, "physical_name", "Unknown") for e in offending_structured
        ]
        unstruct_names = [
            getattr(e, "physical_name", "Unknown") for e in offending_unstructured
        ]

        raise ValueError(
            f"Topological violation: Structured extruded prisms must have higher priority "
            f"(lower mesh_order value) than unstructured entities to preserve their "
            f"sweepability during Boolean intersections.\n"
            f"Structured entities {struct_names} have mesh_order {max_structured_order}, but "
            f"unstructured entities {unstruct_names} have mesh_order {min_unstructured_order}."
        )
