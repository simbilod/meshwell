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
