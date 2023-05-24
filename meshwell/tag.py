import gmsh
from itertools import combinations


def tag_entities(entity_list):
    """Adds physical labels to the entities in the model."""
    for entities in entity_list:
        gmsh.model.addPhysicalGroup(
            entities.get_dim(), entities.get_tags(), name=entities.label
        )


def tag_interfaces(entity_list, max_dim, boundary_delimiter):
    """Adds physical labels to the interfaces between entities in entity_list."""
    for entity1, entity2 in combinations(entity_list, 2):
        if entity1.label == entity2.label:
            continue
        elif entity1.get_dim() != entity2.get_dim():
            continue
        elif entity1.get_dim() != max_dim:
            continue
        else:
            common_interfaces = list(
                set(entity1.boundaries).intersection(entity2.boundaries)
            )
            # Remember which boundaries were interfaces with another entity
            entity1.interfaces.extend(common_interfaces)
            entity2.interfaces.extend(common_interfaces)
            if entity1.keep is False:
                gmsh.model.addPhysicalGroup(
                    max_dim - 1,
                    common_interfaces,
                    name=f"{entity1.label}",
                )
            elif entity2.keep is False:
                gmsh.model.addPhysicalGroup(
                    max_dim - 1,
                    common_interfaces,
                    name=f"{entity2.label}",
                )
            else:
                gmsh.model.addPhysicalGroup(
                    max_dim - 1,
                    common_interfaces,
                    name=f"{entity1.label}{boundary_delimiter}{entity2.label}",
                )

    return entity_list


def tag_boundaries(entity_list, max_dim, boundary_delimiter, mesh_edge_name):
    """Adds physical labels to the boundaries of the entities in entity_list."""
    for entity in entity_list:
        if entity.get_dim() != max_dim:
            continue
        boundaries = list(set(entity.boundaries) - set(entity.interfaces))
        gmsh.model.addPhysicalGroup(
            max_dim - 1,
            boundaries,
            name=f"{entity.label}{boundary_delimiter}{mesh_edge_name}",
        )
