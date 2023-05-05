from __future__ import annotations

from typing import Dict, Optional

import gmsh
import shapely
from meshwell.validation import validate_dimtags, unpack_dimtags
from meshwell.entity import LabeledEntities
from meshwell.tag import tag_entities, tag_interfaces, tag_boundaries
from meshwell.refinement import constant_refinement
from collections import OrderedDict


def mesh(
    dimtags_dict: OrderedDict,
    model,
    resolutions: Optional[Dict] = None,
    default_characteristic_length: float = 0.5,
    filename: Optional[str] = "mesh.msh",
    verbosity: Optional[bool] = False,
    interface_delimiter: str = "___",
    boundary_delimiter: str = "None",
):
    """Creates a GMSH mesh with proper physical tagging from a dict of {labels: list(GMSH entities)}.

    Args:
        dimtags_dict: OrderedDict of key: physical name, and value: (dim, tags) of  GMSH entities
        model: gmsh model to synchronize
        resolutions (Dict): Pairs {"physical name": {"resolution": float, "distance": "float}}
        default_characteristic_length (float): if resolutions is not specified for this physical, will use this value instead
        filename (str, path): where to save the .msh file
        verbosity: GMSH stdout while meshing (True or False)
        interface_delimiter: string characters to use when naming interfaces between entities
        boundary_delimiter: string characters to use when defining an interface between an entity and nothing (simulation boundary)
    """
    resolutions = resolutions if resolutions else {}

    # Validate and unpack dim tags, and detect mesh dimension
    max_dim = 0
    for label, dimtags in dimtags_dict.items():
        dim = validate_dimtags(dimtags)
        max_dim = max(dim, max_dim)
        dimtags_dict[label] = unpack_dimtags(dimtags)

    # Preserve ID numbering
    gmsh.option.setNumber("Geometry.OCCBooleanPreserveNumbering", 1)

    # Iterate through OrderedDict of entities, logging the volumes/surfaces in order
    final_entity_list = []
    for index, (label, dimtags) in enumerate(dimtags_dict.items()):
        base_resolution = (
            resolutions[label]["resolution"]
            if label in resolutions
            else default_characteristic_length
        )
        current_entities = LabeledEntities(
            index=index,
            dimtags=dimtags,
            label=label,
            model=model,
            base_resolution=base_resolution,
        )
        if index != 0:
            current_dimtags_cut = []
            for current_dimtags in current_entities.dimtags:
                for previous_entities in final_entity_list:
                    if cut := model.cut(
                        [current_dimtags],
                        previous_entities.dimtags,
                        removeObject=True,  # Only keep the difference
                        removeTool=False,  # Tool (previous entities) should remain untouched
                    ):
                        current_dimtags_cut.append(cut[0])
                    model.synchronize()
                    model.removeAllDuplicates()  # Heal interfaces now
                    model.synchronize()
                    previous_entities.update_boundaries()
            current_entities.dimtags = [
                item for sublist in current_dimtags_cut for item in sublist
            ]
        current_entities.update_boundaries()
        final_entity_list.append(current_entities)

    # Tag entities, interfaces, and boundaries
    tag_entities(final_entity_list)
    final_entity_list = tag_interfaces(final_entity_list, max_dim, interface_delimiter)
    tag_boundaries(final_entity_list, max_dim, interface_delimiter, boundary_delimiter)

    # Perform refinement
    refinement_fields = []
    refinement_index = 0
    refinement_fields_constant, refinement_index = constant_refinement(
        final_entity_list, refinement_field_index=0
    )
    refinement_fields.extend(refinement_fields_constant)

    # Use the smallest element size overall
    gmsh.model.mesh.field.add("Min", refinement_index)
    gmsh.model.mesh.field.setNumbers(refinement_index, "FieldsList", refinement_fields)
    gmsh.model.mesh.field.setAsBackgroundMesh(refinement_index)

    # Turn off default meshing options
    gmsh.model.mesh.MeshSizeFromPoints = 0
    gmsh.model.mesh.MeshSizeFromCurvature = 0
    gmsh.model.mesh.MeshSizeExtendFromBoundary = 0

    model.synchronize()

    gmsh.option.setNumber(
        "General.Terminal", 1 if verbosity else 0
    )  # 1 verbose, 0 otherwise
    gmsh.model.mesh.generate(3)
    gmsh.write(filename)

    # Mesh
    gmsh.finalize()
    return True


if __name__ == "__main__":
    from meshwell.prism import Prism

    polygon1 = shapely.Polygon(
        [[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]],
        holes=([[0.5, 0.5], [1.5, 0.5], [1.5, 1.5], [0.5, 1.5], [0.5, 0.5]],),
    )
    polygon2 = shapely.Polygon([[-1, -1], [-2, -1], [-2, -2], [-1, -2], [-1, -1]])
    polygon = shapely.MultiPolygon([polygon1, polygon2])

    buffers = {0.0: 0.0, 1.0: -0.1}

    gmsh.initialize()
    occ = gmsh.model.occ
    poly3D = Prism(polygons=polygon, buffers=buffers, model=occ)

    dimtags_dict = OrderedDict(
        {
            "first_physical": [(3, poly3D)],
            "second_entity": [(3, occ.addSphere(0, 0, 0, 1))],
        }
    )

    resolutions = {
        "first_physical": {"resolution": 0.1, "distance": 1},
    }

    geometry = mesh(
        dimtags_dict=dimtags_dict,
        model=occ,
        resolutions=resolutions,
        default_characteristic_length=0.5,
        verbosity=False,
    )
