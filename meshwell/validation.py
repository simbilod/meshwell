"""Mesh validation routines."""
import math
import warnings


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


def validate_tolerance_ladder(
    perturbation: float,
    cut_fuzzy_value: float,
    fragment_fuzzy_value: float,
) -> None:
    """Enforce the cad_occ tolerance-ladder invariants.

    The OCC pipeline relies on::

        cut_fuzzy_value < perturbation < 2*perturbation < fragment_fuzzy_value

    * A cut fuzzy at or above the perturbation can merge the buffered
      overlap into the lower-priority entity and erase the carved face.
    * A fragment fuzzy below ``2*perturbation`` leaves the buffered
      face gap unmerged: coincident faces keep distinct TShapes and
      ``A___B`` interface groups are silently dropped by the XAO writer.

    Raises:
        ValueError: when the ladder is inverted
            (``fragment_fuzzy_value <= cut_fuzzy_value``).

    Warns:
        UserWarning: when ``fragment_fuzzy_value < 2*perturbation`` or
            ``cut_fuzzy_value >= perturbation`` (with ``perturbation > 0``).
    """
    if fragment_fuzzy_value <= cut_fuzzy_value:
        raise ValueError(
            f"fragment_fuzzy_value ({fragment_fuzzy_value:g}) must exceed "
            f"cut_fuzzy_value ({cut_fuzzy_value:g}): the final fragment must "
            f"be at least as permissive as the per-entity cuts."
        )
    if perturbation > 0 and fragment_fuzzy_value < 2 * perturbation:
        warnings.warn(
            f"fragment_fuzzy_value ({fragment_fuzzy_value:g}) is below "
            f"2*perturbation ({2 * perturbation:g}): faces separated by the "
            f"perturbation buffer may keep distinct TShapes and interface "
            f"groups (A___B) can be silently dropped. Raise "
            f"fragment_fuzzy_value / point_tolerance, or lower perturbation.",
            stacklevel=3,
        )
    if perturbation > 0 and cut_fuzzy_value >= perturbation:
        warnings.warn(
            f"cut_fuzzy_value ({cut_fuzzy_value:g}) is >= perturbation "
            f"({perturbation:g}): a loose cut fuzzy can merge the buffered "
            f"overlap into the lower-priority entity and erase the carved "
            f"face.",
            stacklevel=3,
        )
