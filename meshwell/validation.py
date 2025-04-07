from meshwell.labeledentity import LabeledEntities
import math


def validate_dimtags(dimtags):
    dims = [dim for dim, tag in dimtags]
    if len(set(dims)) != 1:
        raise ValueError(
            "All the entities corresponding to a mesh physical_name must be of the same dimension."
        )
    else:
        return dims[0]


def unpack_dimtags(dimtags):
    dim = [dim for dim, tag in dimtags][0]
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
    """Returns a list of entities, sorted first by dimension (higher dimensions first) and then by mesh_order,
    assigning a mesh order corresponding to the ordering if not defined."""
    # Group entities by dimension
    entities_by_dim = {}
    for entity in entities:
        dim = entity.dimension
        if dim not in entities_by_dim:
            entities_by_dim[dim] = []
        entities_by_dim[dim].append(entity)

    ordered_entities = []
    # Process dimensions in descending order
    for dim in sorted(entities_by_dim.keys(), reverse=True):
        dim_entities = entities_by_dim[dim]

        # Split into defined and undefined mesh_order entities
        defined_order = [e for e in dim_entities if e.mesh_order is not None]
        undefined_order = [e for e in dim_entities if e.mesh_order is None]

        # Sort defined order entities
        ordered_defined = sort_entities_by_mesh_order(defined_order)

        # Calculate start index for undefined entities
        if ordered_defined:
            start_index = math.ceil(ordered_defined[-1].mesh_order) + 1
        else:
            start_index = 1

        # Assign mesh_order to undefined entities
        ordered_undefined = assign_mesh_order_from_ordering(
            undefined_order,
            start_index=start_index,
        )

        ordered_entities.extend(ordered_defined + ordered_undefined)

    return ordered_entities


def consolidate_entities_by_physical_name(entities):
    """Returns a new list of LabeledEntities, with a single entity per physical_name."""
    consolidated_entities = []
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

    for entity in physical_name_dict.values():
        consolidated_entities.append(entity)

    return consolidated_entities
