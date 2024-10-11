from typing import List
import gmsh
from itertools import combinations, product


def tag_entities(entity_list: List):
    """Adds physical labels to the entities in the model."""
    for entities in entity_list:
        if entities.physical_name:
            if isinstance(entities.physical_name, str):
                physical_names = [entities.physical_name]
            else:
                physical_names = entities.physical_name
            for physical_name in physical_names:
                gmsh.model.addPhysicalGroup(
                    entities.get_dim(), entities.get_tags(), name=physical_name
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
                if isinstance(entity1.physical_name, str):
                    entity1_physical_names = [entity1.physical_name]
                else:
                    entity1_physical_names = entity1.physical_name
                if isinstance(entity2.physical_name, str):
                    entity2_physical_names = [entity2.physical_name]
                else:
                    entity2_physical_names = entity2.physical_name
                for entity1_physical_name, entity2_physical_name in product(
                    entity1_physical_names, entity2_physical_names
                ):
                    gmsh.model.addPhysicalGroup(
                        max_dim - 1,
                        common_interfaces,
                        name=f"{entity1_physical_name}{boundary_delimiter}{entity2_physical_name}",
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
        if isinstance(entity.physical_name, str):
            entity_physical_names = [entity.physical_name]
        else:
            entity_physical_names = entity.physical_name
        for entity_physical_name in entity_physical_names:
            gmsh.model.addPhysicalGroup(
                max_dim - 1,
                boundaries,
                name=f"{entity_physical_name}{boundary_delimiter}{mesh_edge_name}",
            )
