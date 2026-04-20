"""Bridge between OCC and GMSH for meshwell."""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from OCP.BRepTools import BRepTools
from tqdm.auto import tqdm

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
    progress_bars: bool = False,
    remove_all_duplicates: bool = False,
) -> list[LabeledEntities]:
    """Inject OCC shapes into gmsh and tag them.

    All entity shapes are packed into one TopoDS_Compound (entity-ordered),
    written to a single BREP file, and imported in one importShapes call.
    BREP serialization preserves sub-shape sharing, so coincident faces
    stay coincident in GMSH — this is what lets tag_interfaces find the
    shared boundaries between entities.

    Args:
        occ_entities: list of OCCLabeledEntity objects. Each may carry
            multiple fragment pieces in ``shapes``.
        model_manager: ModelManager instance. A fresh one is created if None.
        interface_delimiter: delimiter for interface physical names.
        boundary_delimiter: delimiter for exterior boundary physical names.
        progress_bars: if True, show tqdm progress bars for per-entity steps.
        remove_all_duplicates: if True, run a gmsh-level fragment safety net
            across all imported dimtags after importShapes. Equivalent to
            ``gmsh.model.occ.removeAllDuplicates()`` but preserves per-entity
            physical tagging by remapping dimtags through the fragment map
            (``removeAllDuplicates`` issues fresh tags and drops the mapping).
    """
    from OCP.BRep import BRep_Builder
    from OCP.TopoDS import TopoDS_Compound

    from meshwell.model import ModelManager

    owns_model = False
    if model_manager is None:
        model_manager = ModelManager()
        owns_model = True

    model_manager.ensure_initialized(str(model_manager.filename))
    gmsh_model = model_manager.model

    # More accurate bounding boxes from STL tessellation. Tag-safe (bbox-only).
    # Helps bbox-based queries (e.g. getEntitiesInBoundingBox) and some internal
    # coincidence checks during boundary recovery.
    import gmsh as _gmsh

    _gmsh.option.setNumber("Geometry.OCCBoundsUseStl", 1)

    max_dim = 0
    for ent in occ_entities:
        if ent.shapes:
            max_dim = max(max_dim, ent.dim)

    # Build one compound. Record how many top-level children each entity
    # contributes so we can slice the returned dimtag list afterwards.
    comp_builder = BRep_Builder()
    compound = TopoDS_Compound()
    comp_builder.MakeCompound(compound)

    piece_counts: list[int] = []
    for ent in tqdm(
        occ_entities,
        desc="Packing OCC compound",
        disable=not progress_bars,
        leave=False,
    ):
        piece_counts.append(len(ent.shapes))
        for s in ent.shapes:
            comp_builder.Add(compound, s)

    final_entity_list: list[LabeledEntities] = []

    if progress_bars:
        print(
            f"Writing BREP compound and importing into gmsh ({sum(piece_counts)} pieces)…",
            flush=True,
        )
    with tempfile.TemporaryDirectory() as tmpdir:
        brep_file = Path(tmpdir) / "all_entities.brep"
        BRepTools.Write_s(compound, str(brep_file))

        imported_dimtags = gmsh_model.occ.importShapes(
            str(brep_file), highestDimOnly=False
        )
        gmsh_model.occ.synchronize()

    # Optional gmsh-level duplicate safety net. Mirrors occ.removeAllDuplicates
    # (which calls booleanFragments(-1, allDimTags, ...) internally) but keeps
    # the ``outDimTagsMap`` so we can forward each entity's dimtags to the new
    # post-fragment tags. Plain removeAllDuplicates would drop that mapping and
    # leave our per-entity dimtag slice pointing at deleted shapes.
    if remove_all_duplicates and imported_dimtags:
        if progress_bars:
            print(
                f"Fragment safety net across {len(imported_dimtags)} imported pieces…",
                flush=True,
            )
        _, out_map = gmsh_model.occ.fragment(
            list(imported_dimtags),
            [],
            removeObject=True,
            removeTool=True,
        )
        gmsh_model.occ.synchronize()
        # out_map[i] holds the new dimtags for imported_dimtags[i]. Rewrite
        # imported_dimtags in place so the slice-by-piece_counts loop below
        # still works, but may yield a different count per entity (absorbed
        # duplicates collapse to the same dimtag in multiple slots — downstream
        # tag_interfaces treats shared surfaces correctly).
        remapped: list[tuple[int, int]] = []
        new_counts: list[int] = []
        cursor = 0
        for count in piece_counts:
            entity_new: list[tuple[int, int]] = []
            for i in range(cursor, cursor + count):
                entity_new.extend(out_map[i])
            cursor += count
            new_counts.append(len(entity_new))
            remapped.extend(entity_new)
        imported_dimtags = remapped
        piece_counts = new_counts

    # importShapes preserves top-level iteration order of the compound, so
    # slicing by piece_counts yields each entity's dimtags.
    cursor = 0
    for ent, count in zip(occ_entities, piece_counts):
        dimtags = imported_dimtags[cursor : cursor + count]
        cursor += count
        # Entity was fully absorbed by a higher-priority overlap. It has no
        # GMSH topology to tag; drop it here rather than letting an empty
        # LabeledEntities reach tag_entities (which keys on .dim and crashes
        # on dim=-1).
        if count == 0:
            continue
        final_entity_list.append(
            LabeledEntities(
                index=ent.index,
                dimtags=list(dimtags),
                physical_name=ent.physical_name,
                keep=ent.keep,
                model=gmsh_model,
            )
        )

    for entity in tqdm(
        final_entity_list,
        desc="Updating boundaries",
        disable=not progress_bars,
        leave=False,
    ):
        entity.update_boundaries()

    if final_entity_list:
        if progress_bars:
            print(
                f"Tagging {len(final_entity_list)} entities, interfaces, boundaries…",
                flush=True,
            )
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

    # Remove entities marked keep=False (e.g. helpers used only to cut).
    for entity in final_entity_list:
        if not entity.keep and entity.dimtags:
            gmsh_model.occ.remove(entity.dimtags, recursive=False)
            gmsh_model.occ.synchronize()

    # Strip boundary/curve entities left over without a higher-dim parent.
    if max_dim == 3:
        dangling = [
            (dim, tag)
            for dim, tag in gmsh_model.getEntities(2)
            if len(gmsh_model.getAdjacencies(dim, tag)[0]) == 0
        ]
        if dangling:
            gmsh_model.occ.remove(dangling, recursive=True)
            gmsh_model.occ.synchronize()
    elif max_dim == 2:
        dangling = [
            (dim, tag)
            for dim, tag in gmsh_model.getEntities(1)
            if len(gmsh_model.getAdjacencies(dim, tag)[0]) == 0
        ]
        if dangling:
            gmsh_model.occ.remove(dangling, recursive=True)
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
    progress_bars: bool = False,
    remove_all_duplicates: bool = False,
) -> None:
    """Convenience function to inject and save to .xao.

    Args:
        occ_entities: List of OCCLabeledEntity objects
        output_file: Output path for .xao
        model_manager: ModelManager instance
        interface_delimiter: Delimiter for interface names
        boundary_delimiter: Delimiter for boundary names
        progress_bars: if True, show tqdm progress bars for per-entity steps.
        remove_all_duplicates: if True, run a gmsh-level fragment safety net
            across all imported pieces after importShapes. See
            :func:`inject_occ_entities_into_gmsh`.
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
        progress_bars=progress_bars,
        remove_all_duplicates=remove_all_duplicates,
    )
    model_manager.save_to_xao(output_file)

    if owns_model:
        model_manager.finalize()
