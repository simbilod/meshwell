"""Bridge between OCC and GMSH for meshwell."""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from OCP.BRepGProp import BRepGProp
from OCP.BRepTools import BRepTools
from OCP.GProp import GProp_GProps
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_SOLID, TopAbs_VERTEX
from OCP.TopExp import TopExp_Explorer
from tqdm.auto import tqdm

from meshwell.labeledentity import LabeledEntities
from meshwell.tag import tag_boundaries, tag_entities, tag_interfaces

if TYPE_CHECKING:
    from OCP.TopoDS import TopoDS_Shape

    from meshwell.cad_occ import OCCLabeledEntity
    from meshwell.model import ModelManager


_TOPABS_FOR_DIM = {
    3: TopAbs_SOLID,
    2: TopAbs_FACE,
    1: TopAbs_EDGE,
    0: TopAbs_VERTEX,
}


def _leaf_shapes(shape: TopoDS_Shape, dim: int) -> list[TopoDS_Shape]:
    """Return sub-shapes of ``shape`` at the TopAbs class matching ``dim``.

    Importing into gmsh binds at the leaf TopAbs class (SOLID/FACE/EDGE/VERTEX);
    a WIRE decomposes to edges, a SHELL to faces, etc. We enumerate leaves here
    so the downstream gmsh-entity lookup can match by (mass, centroid) at the
    same granularity gmsh uses.
    """
    topabs = _TOPABS_FOR_DIM[dim]
    if shape.ShapeType() == topabs:
        return [shape]
    out: list[TopoDS_Shape] = []
    exp = TopExp_Explorer(shape, topabs)
    while exp.More():
        out.append(exp.Current())
        exp.Next()
    return out


def _shape_mass_centroid(
    shape: TopoDS_Shape, dim: int
) -> tuple[float, tuple[float, float, float]]:
    """Compute mass (length/area/volume) and centroid for a leaf shape."""
    props = GProp_GProps()
    if dim == 3:
        BRepGProp.VolumeProperties_s(shape, props)
    elif dim == 2:
        BRepGProp.SurfaceProperties_s(shape, props)
    elif dim == 1:
        BRepGProp.LinearProperties_s(shape, props)
    else:
        # dim == 0: a vertex has no mass; fall back to its point coordinates.
        from OCP.BRep import BRep_Tool
        from OCP.TopoDS import TopoDS

        p = BRep_Tool.Pnt_s(TopoDS.Vertex_s(shape))
        return 0.0, (p.X(), p.Y(), p.Z())
    com = props.CentreOfMass()
    return props.Mass(), (com.X(), com.Y(), com.Z())


def _import_via_brep_and_mass_lookup(
    occ_entities: list[OCCLabeledEntity],
    gmsh_model,
    progress_bars: bool = False,
) -> list[list[tuple[int, int]]]:
    """Import BREP compound, match per-entity shapes by mass + centroid.

    Legacy path: returns ``[[dimtag, ...], ...]`` aligned with
    ``occ_entities``.
    """
    from OCP.BRep import BRep_Builder
    from OCP.TopoDS import TopoDS_Compound

    comp_builder = BRep_Builder()
    compound = TopoDS_Compound()
    comp_builder.MakeCompound(compound)
    total_pieces = 0
    for ent in tqdm(
        occ_entities,
        desc="Packing OCC compound",
        disable=not progress_bars,
        leave=False,
    ):
        for s in ent.shapes:
            comp_builder.Add(compound, s)
            total_pieces += 1

    if progress_bars:
        print(
            f"Writing BREP compound and importing into gmsh ({total_pieces} pieces)…",
            flush=True,
        )
    with tempfile.TemporaryDirectory() as tmpdir:
        brep_file = Path(tmpdir) / "all_entities.brep"
        BRepTools.Write_s(compound, str(brep_file))

        gmsh_model.occ.importShapes(str(brep_file), highestDimOnly=False)
        gmsh_model.occ.synchronize()

    # gmsh's ``_multiBind`` iterates ALL sub-shapes per TopAbs class with
    # ``returnNewOnly=false``, so the returned dimtag list mixes our top-level
    # pieces with every sub-face/sub-edge of every solid. Compound-position
    # slicing is therefore unreliable across dims. Instead, identify each
    # OCCLabeledEntity's gmsh entities by matching mass + centroid. Mass and
    # centroid come straight from OCC BRepGProp, so they equal gmsh's own
    # ``getMass``/``getCenterOfMass`` values within float rounding.
    def _match_shape(shape, dim, used):
        exp_mass, exp_c = _shape_mass_centroid(shape, dim)
        best_key = None
        best_err = float("inf")
        for _, tag in gmsh_model.getEntities(dim):
            key = (dim, tag)
            if key in used:
                continue
            if dim == 0:
                m = 0.0
                c = gmsh_model.getValue(0, tag, [])
                c = (c[0], c[1], c[2])
            else:
                m = gmsh_model.occ.getMass(dim, tag)
                c = tuple(gmsh_model.occ.getCenterOfMass(dim, tag))
            dx = c[0] - exp_c[0]
            dy = c[1] - exp_c[1]
            dz = c[2] - exp_c[2]
            err = abs(m - exp_mass) + (dx * dx + dy * dy + dz * dz) ** 0.5
            if err < best_err:
                best_err = err
                best_key = key
        return best_key

    used: set[tuple[int, int]] = set()
    entity_dimtags: list[list[tuple[int, int]]] = []
    for ent in occ_entities:
        dimtags: list[tuple[int, int]] = []
        for s in ent.shapes:
            for leaf in _leaf_shapes(s, ent.dim):
                match = _match_shape(leaf, ent.dim, used)
                if match is not None:
                    used.add(match)
                    dimtags.append(match)
        entity_dimtags.append(dimtags)
    return entity_dimtags


def _import_via_xao(
    occ_entities: list[OCCLabeledEntity],
    gmsh_model,
    progress_bars: bool = False,
) -> list[list[tuple[int, int]]]:
    """Import via XAO + per-entity marker physical groups.

    Writes an inline XAO with unique marker physical groups per entity,
    loads it via ``gmsh.open``, recovers each entity's dimtags from its
    marker group, and strips the markers so the real physical-tagging
    runs normally.
    """
    from meshwell.occ_xao_writer import write_xao

    if progress_bars:
        print(
            f"Writing XAO for {len(occ_entities)} entities and importing into gmsh…",
            flush=True,
        )
    with tempfile.TemporaryDirectory() as tmpdir:
        xao_path = Path(tmpdir) / "all_entities.xao"
        markers = write_xao(occ_entities, xao_path)

        import gmsh as _gmsh

        _gmsh.open(str(xao_path))
        gmsh_model.occ.synchronize()

    # Reverse-lookup: marker physical-group name -> (dim, physical tag).
    marker_to_group: dict[str, tuple[int, int]] = {}
    for dim in (0, 1, 2, 3):
        for d, tag in gmsh_model.getPhysicalGroups(dim):
            name = gmsh_model.getPhysicalName(d, tag)
            if name in markers.values():
                marker_to_group[name] = (d, tag)

    entity_dimtags: list[list[tuple[int, int]]] = []
    for ent in occ_entities:
        marker = markers.get(ent.index)
        if marker is None or marker not in marker_to_group:
            entity_dimtags.append([])
            continue
        d, ptag = marker_to_group[marker]
        dimtags = [(d, int(t)) for t in gmsh_model.getEntitiesForPhysicalGroup(d, ptag)]
        entity_dimtags.append(dimtags)

    # Strip the markers so tag_entities can set real physical names without
    # colliding with our temporary ones.
    for d, ptag in marker_to_group.values():
        gmsh_model.removePhysicalGroups([(d, ptag)])

    return entity_dimtags


def inject_occ_entities_into_gmsh(
    occ_entities: list[OCCLabeledEntity],
    model_manager: ModelManager | None = None,
    interface_delimiter: str = "___",
    boundary_delimiter: str = "None",
    progress_bars: bool = False,
    remove_all_duplicates: bool = False,
    use_xao: bool = False,
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
        use_xao: if True, hand the shapes to gmsh via an inline XAO file
            with per-entity marker physical groups (see
            :mod:`meshwell.occ_xao_writer`). Per-entity dimtags come from
            ``getEntitiesForPhysicalGroup(marker)`` — exact TShape identity
            rather than mass+centroid heuristic. Default False preserves the
            legacy BREP + mass-lookup path.
    """
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

    final_entity_list: list[LabeledEntities] = []

    if use_xao:
        entity_dimtags = _import_via_xao(
            occ_entities, gmsh_model, progress_bars=progress_bars
        )
    else:
        entity_dimtags = _import_via_brep_and_mass_lookup(
            occ_entities, gmsh_model, progress_bars=progress_bars
        )

    # Optional gmsh-level duplicate safety net. Mirrors occ.removeAllDuplicates
    # (which calls booleanFragments(-1, allDimTags, ...) internally) but keeps
    # the ``outDimTagsMap`` so we can forward each entity's dimtags to the new
    # post-fragment tags. Plain removeAllDuplicates would drop that mapping and
    # leave each entity's saved dimtags pointing at deleted shapes.
    if remove_all_duplicates and any(entity_dimtags):
        # Build a de-duplicated list of dimtags to feed to fragment, plus a
        # per-entity index into that list so we can rewire results.
        unique_dimtags: list[tuple[int, int]] = []
        dimtag_to_input_idx: dict[tuple[int, int], int] = {}
        input_idx_per_entity: list[list[int]] = []
        for dimtags in entity_dimtags:
            idxs: list[int] = []
            for dt in dimtags:
                if dt not in dimtag_to_input_idx:
                    dimtag_to_input_idx[dt] = len(unique_dimtags)
                    unique_dimtags.append(dt)
                idxs.append(dimtag_to_input_idx[dt])
            input_idx_per_entity.append(idxs)

        if progress_bars:
            print(
                f"Fragment safety net across {len(unique_dimtags)} dimtags…",
                flush=True,
            )
        _, out_map = gmsh_model.occ.fragment(
            unique_dimtags,
            [],
            removeObject=True,
            removeTool=True,
        )
        gmsh_model.occ.synchronize()

        # Rewrite each entity's dimtags through out_map. A single input dimtag
        # may expand to multiple outputs (shape was split further); duplicates
        # across entities collapse to the same output dimtag, which downstream
        # ``tag_interfaces`` handles correctly for shared surfaces.
        new_entity_dimtags: list[list[tuple[int, int]]] = []
        for idxs in input_idx_per_entity:
            collected: list[tuple[int, int]] = []
            seen_local: set[tuple[int, int]] = set()
            for i in idxs:
                for dt in out_map[i]:
                    if dt not in seen_local:
                        seen_local.add(dt)
                        collected.append(dt)
            new_entity_dimtags.append(collected)
        entity_dimtags = new_entity_dimtags

    for ent, dimtags in zip(occ_entities, entity_dimtags):
        # Entity was fully absorbed by a higher-priority overlap. It has no
        # GMSH topology to tag; drop it here rather than letting an empty
        # LabeledEntities reach tag_entities (which keys on .dim and crashes
        # on dim=-1).
        if not dimtags:
            continue
        final_entity_list.append(
            LabeledEntities(
                index=ent.index,
                dimtags=dimtags,
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
    use_xao: bool = False,
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
        use_xao: if True, use the XAO import path (TShape-exact identity via
            marker physical groups). See :func:`inject_occ_entities_into_gmsh`.
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
        use_xao=use_xao,
    )
    model_manager.save_to_xao(output_file)

    if owns_model:
        model_manager.finalize()
