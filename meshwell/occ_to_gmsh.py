"""Bridge between OCC and GMSH for meshwell."""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from OCP.BRepTools import BRepTools

from meshwell.labeledentity import LabeledEntities
from meshwell.tag import tag_boundaries, tag_entities, tag_interfaces

if TYPE_CHECKING:
    from meshwell.cad_occ import OCCLabeledEntity
    from meshwell.model import ModelManager


def inject_occ_entities_into_gmsh(
    occ_entities: list[OCCLabeledEntity],
    model_manager: ModelManager | None = None,
    interface_delimiter: str = "___",
    boundary_delimiter: str = "None",
) -> list[LabeledEntities]:
    """Inject OCC shapes into gmsh and tag them.

    Args:
        occ_entities: List of OCCLabeledEntity objects
        model_manager: ModelManager instance
        interface_delimiter: Delimiter for interface names
        boundary_delimiter: Delimiter for boundary names

    Returns:
        list[LabeledEntities]: List of GMSH-tagged entities
    """
    from meshwell.model import ModelManager

    owns_model = False
    if model_manager is None:
        model_manager = ModelManager()
        owns_model = True

    model_manager.ensure_initialized(str(model_manager.filename))
    gmsh_model = model_manager.model

    final_entity_list = []

    # Store max dimension seen
    max_dim = 0
    for ent in occ_entities:
        max_dim = max(max_dim, ent.dim)

    # Inject each entity into gmsh
    with tempfile.TemporaryDirectory() as tmpdir:
        for i, occ_ent in enumerate(occ_entities):
            # 1. Serialize to BREP
            brep_file = Path(tmpdir) / f"entity_{i}.brep"
            BRepTools.Write_s(occ_ent.shape, str(brep_file))

            # 2. Import into gmsh
            # importShapes returns a list of (dim, tag)
            new_dimtags = gmsh_model.occ.importShapes(str(brep_file))
            gmsh_model.occ.synchronize()

            # 3. Create LabeledEntities object
            labeled_ent = LabeledEntities(
                index=occ_ent.index,
                dimtags=new_dimtags,
                physical_name=occ_ent.physical_name,
                keep=occ_ent.keep,
                model=gmsh_model,
            )
            final_entity_list.append(labeled_ent)

    # Filter out non-kept entities early if needed
    # (Actually we want to tag them all first for interface calculation)

    # Update boundaries for all injected entities
    for entity in final_entity_list:
        entity.update_boundaries()

    # Tag everything using existing tag.py logic
    if final_entity_list:
        tag_entities(final_entity_list, gmsh_model)
        tag_interfaces(
            final_entity_list,
            max_dim,
            interface_delimiter,
            gmsh_model,
        )
        tag_boundaries(
            final_entity_list,
            max_dim,
            interface_delimiter,
            boundary_delimiter,
            gmsh_model,
        )

    # Finally, remove entities that were NOT marked as keep=True
    for entity in final_entity_list:
        if not entity.keep and entity.dimtags:
            gmsh_model.occ.remove(entity.dimtags, recursive=False)
            gmsh_model.occ.synchronize()

    # Clean up any leftover boundary entities that do not bound any higher-dimensional entities
    if max_dim == 3:
        dangling_surfaces = []
        for dim, tag in gmsh_model.getEntities(2):
            upward_adj, _ = gmsh_model.getAdjacencies(dim, tag)
            if len(upward_adj) == 0:
                dangling_surfaces.append((dim, tag))

        if dangling_surfaces:
            gmsh_model.occ.remove(dangling_surfaces, recursive=True)
            gmsh_model.occ.synchronize()
    elif max_dim == 2:
        dangling_curves = []
        for dim, tag in gmsh_model.getEntities(1):
            upward_adj, _ = gmsh_model.getAdjacencies(dim, tag)
            if len(upward_adj) == 0:
                dangling_curves.append((dim, tag))

        if dangling_curves:
            gmsh_model.occ.remove(dangling_curves, recursive=True)
            gmsh_model.occ.synchronize()

    if owns_model:
        model_manager.finalize()

    return final_entity_list


def occ_to_xao(
    occ_entities: list[OCCLabeledEntity],
    output_file: Path,
    model_manager: ModelManager | None = None,
    interface_delimiter: str = "___",
    boundary_delimiter: str = "None",
) -> None:
    """Convenience function to inject and save to .xao.

    Args:
        occ_entities: List of OCCLabeledEntity objects
        output_file: Output path for .xao
        model_manager: ModelManager instance
        interface_delimiter: Delimiter for interface names
        boundary_delimiter: Delimiter for boundary names
    """
    from meshwell.model import ModelManager

    owns_model = False
    if model_manager is None:
        model_manager = ModelManager()
        owns_model = True

    inject_occ_entities_into_gmsh(
        occ_entities=occ_entities,
        model_manager=model_manager,
        interface_delimiter=interface_delimiter,
        boundary_delimiter=boundary_delimiter,
    )
    model_manager.save_to_xao(output_file)

    if owns_model:
        model_manager.finalize()
