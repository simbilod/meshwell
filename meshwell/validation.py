from meshwell.labeledentity import LabeledEntities


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


def sort_entities_by_mesh_order(entities):
    """Returns a list of entities, sorted by mesh_order."""
    return sorted(entities, key=lambda entity: entity.mesh_order)


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
                resolution=existing_entity.resolution,
                keep=existing_entity.keep,
                model=existing_entity.model,
            )
            physical_name_dict[entity.physical_name] = combined_entity

    for entity in physical_name_dict.values():
        consolidated_entities.append(entity)

    return consolidated_entities
