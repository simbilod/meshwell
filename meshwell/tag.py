from typing import List
import gmsh
from itertools import combinations, product


def tag_entities(entity_list: List):
    """Adds physical labels to the entities in the model."""
    # One pass to get the global name --> dimtags mapping
    names_to_tags = {
        0: {},
        1: {},
        2: {},
        3: {},
    }
    for entities in entity_list:
        dim = entities.dim
        for physical_name in entities.physical_name:
            if physical_name not in names_to_tags[dim]:
                names_to_tags[dim][physical_name] = []
            names_to_tags[dim][physical_name].extend(entities.tags)

    for dim in names_to_tags.keys():
        for physical_name, tags in names_to_tags[dim].items():
            gmsh.model.addPhysicalGroup(dim, tags, name=physical_name)


def tag_interfaces(entity_list: List, max_dim: int, boundary_delimiter: str):
    """Adds physical labels to the interfaces between entities in entity_list."""
    names_to_tags = {
        0: {},
        1: {},
        2: {},
    }
    for entity1, entity2 in combinations(entity_list, 2):
        if entity1.physical_name == entity2.physical_name:
            continue
        elif entity1.dim != entity2.dim:
            continue
        elif entity1.dim != max_dim:
            continue
        else:
            dim = entity1.dim - 1
            common_interfaces = list(
                set(entity1.boundaries).intersection(entity2.boundaries)
            )
            if common_interfaces:
                # Update entity interface logs
                entity1.interfaces.extend(common_interfaces)
                entity2.interfaces.extend(common_interfaces)
                # Prepare physical tags
                for entity1_physical_name, entity2_physical_name in product(
                    entity1.physical_name, entity2.physical_name
                ):
                    interface_name = f"{entity1_physical_name}{boundary_delimiter}{entity2_physical_name}"
                    if interface_name not in names_to_tags[dim]:
                        names_to_tags[dim][interface_name] = []
                    names_to_tags[dim][interface_name].extend(common_interfaces)

    for dim in names_to_tags.keys():
        for physical_name, tags in names_to_tags[dim].items():
            gmsh.model.addPhysicalGroup(dim, tags, name=physical_name)

    return entity_list


def tag_boundaries(
    entity_list: List, max_dim: int, boundary_delimiter: str, mesh_edge_name: str
):
    """Adds physical labels to the boundaries of the entities in entity_list."""
    names_to_tags = {
        0: {},
        1: {},
        2: {},
    }
    for entity in entity_list:
        if entity.dim != max_dim:
            continue
        dim = entity.dim - 1
        boundaries = list(set(entity.boundaries) - set(entity.interfaces))
        entity.mesh_edge_name_interfaces.extend(boundaries)
        for entity_physical_name in entity.physical_name:
            boundary_name = (
                f"{entity_physical_name}{boundary_delimiter}{mesh_edge_name}"
            )
            if boundary_name not in names_to_tags[dim]:
                names_to_tags[dim][boundary_name] = []
            names_to_tags[dim][boundary_name].extend(boundaries)

    for dim in names_to_tags.keys():
        for physical_name, tags in names_to_tags[dim].items():
            gmsh.model.addPhysicalGroup(dim, tags, name=physical_name)
