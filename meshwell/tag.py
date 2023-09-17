from typing import List
import gmsh
from itertools import combinations


def tag_entities(entity_list: List):
    """Adds physical labels to the entities in the model."""
    for entities in entity_list:
        if entities.physical_name:
            gmsh.model.addPhysicalGroup(
                entities.get_dim(), entities.get_tags(), name=entities.physical_name
            )


def tag_interfaces(entity_list: List, max_dim: int, boundary_delimiter: str):
    """Adds physical labels to the interfaces between entities in entity_list."""
    for entity1, entity2 in combinations(entity_list, 2):
        if entity1.physical_name == entity2.physical_name:
            continue
        elif entity1.get_dim() != entity2.get_dim():
            continue
        elif entity1.get_dim() != max_dim:
            continue
        else:
            common_interfaces = list(
                set(entity1.boundaries).intersection(entity2.boundaries)
            )
            if common_interfaces:
                # Remember which boundaries were interfaces with another entity
                entity1.interfaces.extend(common_interfaces)
                entity2.interfaces.extend(common_interfaces)
                gmsh.model.addPhysicalGroup(
                    max_dim - 1,
                    common_interfaces,
                    name=f"{entity1.physical_name}{boundary_delimiter}{entity2.physical_name}",
                )

    return entity_list


def tag_boundaries(
    entity_list: List, max_dim: int, boundary_delimiter: str, mesh_edge_name: str
):
    """Adds physical labels to the boundaries of the entities in entity_list."""
    for entity in entity_list:
        if entity.get_dim() != max_dim:
            continue
        boundaries = list(set(entity.boundaries) - set(entity.interfaces))
        gmsh.model.addPhysicalGroup(
            max_dim - 1,
            boundaries,
            name=f"{entity.physical_name}{boundary_delimiter}{mesh_edge_name}",
        )
