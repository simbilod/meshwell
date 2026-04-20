"""OCC -> gmsh bridge via XAO.

This module is the sole path from ``OCCLabeledEntity`` to a tagged gmsh
model. The tagging (entities, inter-entity interfaces, exterior
boundaries) is computed entirely at OCP level using ``TopoDS_Shape``
TShape identity and emitted directly into the XAO file. gmsh loads the
file and every physical group is already in place -- no post-import
tagging pipeline.

Pipeline
--------

    OCCLabeledEntity(s)  (from meshwell.cad_occ)
            |
            v  write_xao: OCP tagging + BREP (kept only) + groups
    .xao (self-contained, CDATA-inline BREP)
            |
            v  gmsh.open  (done inside inject_occ_entities_into_gmsh)
    gmsh model with every physical group applied

The resolution / refinement engine in :mod:`meshwell.mesh` then recovers
per-entity state from those physical groups when meshing runs -- the
bridge does not need to hand back any Python-side per-entity objects.

Design notes
------------

- **BREP is required**: XAO is a metadata wrapper, not its own geometry
  format. Its ``<shape format="BREP">`` element and the
  ``TopExp::MapShapes`` reference indices need a bit-identical shape
  on both sides. Embedded inline in CDATA (single self-contained file).

- **keep=False entities** are *not* serialized into the BREP. Their
  shape memory is still used for interface computation -- a face F
  shared between kept A and helper B (via BOPAlgo TShape identity) is
  already a sub-face of A's solid, so F travels into the BREP through
  A, and the ``A___B`` interface group references it by topology index.
  Nothing is imported for B, so nothing needs to be removed after load;
  "keep=False" means literally "not in the final model".
"""

from __future__ import annotations

import tempfile
import xml.etree.ElementTree as ET
from itertools import combinations, product
from pathlib import Path
from typing import TYPE_CHECKING

from OCP.BRep import BRep_Builder
from OCP.BRepTools import BRepTools
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_SOLID, TopAbs_VERTEX
from OCP.TopExp import TopExp, TopExp_Explorer
from OCP.TopoDS import TopoDS_Compound
from OCP.TopTools import TopTools_IndexedMapOfShape, TopTools_ShapeMapHasher

if TYPE_CHECKING:
    from meshwell.cad_occ import OCCLabeledEntity
    from meshwell.model import ModelManager


_HASHER = TopTools_ShapeMapHasher()

_DIM_TO_TOPABS = {
    3: TopAbs_SOLID,
    2: TopAbs_FACE,
    1: TopAbs_EDGE,
    0: TopAbs_VERTEX,
}
_DIM_TO_XAO_ELEM = {3: "solid", 2: "face", 1: "edge", 0: "vertex"}
_DIM_TO_XAO_GROUP = {3: "solids", 2: "faces", 1: "edges", 0: "vertices"}

# ---------------------------------------------------------------------------
# OCP-level helpers
# ---------------------------------------------------------------------------


def _leaf_subshapes(shape, dim):
    """Yield ``(sub_shape, tshape_id)`` pairs at the TopAbs class for ``dim``.

    Mirrors the leaf-enumeration gmsh's ``_multiBind`` does when binding
    imported OCC shapes (one gmsh entity per leaf TopAbs). Dedupe by
    TShape identity.
    """
    topabs = _DIM_TO_TOPABS[dim]
    seen: set[int] = set()
    exp = TopExp_Explorer(shape, topabs)
    while exp.More():
        sub = exp.Current()
        key = _HASHER(sub)
        if key not in seen:
            seen.add(key)
            yield sub, key
        exp.Next()


def _compute_physical_groups(
    entities: list[OCCLabeledEntity],
    interface_delimiter: str,
    boundary_delimiter: str,
) -> dict[tuple[int, str], list]:
    """OCP reconstruction of ``tag_entities``/``tag_interfaces``/``tag_boundaries``.

    Returns ``{(dim, physical_name): [TopoDS_Shape, ...]}``. Interfaces are
    named by TShape identity of shared sub-shapes. keep=False entities
    participate in interface tagging (so kept neighbours can name shared
    boundaries) but do not get their own entity or exterior groups --
    they will be removed from gmsh after load.
    """
    if not entities:
        return {}

    max_dim = max((e.dim for e in entities if e.shapes), default=0)

    # entity_leaves[i]  : TopoDS_Shape list at ent.dim (entity's own tags).
    # entity_boundary[i]: {tshape_id: TopoDS_Shape} at ent.dim - 1, populated
    #                     only for top-dim entities (lower-dim entities can't
    #                     have gmsh "boundaries" in meshwell's sense).
    entity_leaves: list[list] = []
    entity_boundary: list[dict[int, object]] = []
    for ent in entities:
        leaves = [leaf for s in ent.shapes for leaf, _ in _leaf_subshapes(s, ent.dim)]
        boundaries: dict[int, object] = {}
        if ent.dim == max_dim and ent.dim > 0:
            for s in ent.shapes:
                for sub, sid in _leaf_subshapes(s, ent.dim - 1):
                    boundaries.setdefault(sid, sub)
        entity_leaves.append(leaves)
        entity_boundary.append(boundaries)

    groups: dict[tuple[int, str], list] = {}

    # 1. tag_entities: skip keep=False.
    for ent, leaves in zip(entities, entity_leaves):
        if not ent.keep:
            continue
        for name in ent.physical_name:
            groups.setdefault((ent.dim, name), []).extend(leaves)

    # 2. tag_interfaces: include pairs where at least one side is keep=True,
    #    so kept neighbours can name boundaries shared with helper
    #    (keep=False) entities. Pairs where both sides are keep=False have
    #    no shape in the emitted BREP and would reference phantom topology.
    entity_interface_ids: list[set[int]] = [set() for _ in entities]
    for (i1, ent1), (i2, ent2) in combinations(enumerate(entities), 2):
        if ent1.dim != ent2.dim or ent1.dim != max_dim:
            continue
        bid1 = set(entity_boundary[i1].keys())
        bid2 = set(entity_boundary[i2].keys())
        common = bid1 & bid2
        if not common:
            continue
        entity_interface_ids[i1].update(common)
        entity_interface_ids[i2].update(common)
        if not (ent1.keep or ent2.keep):
            continue
        if ent1.physical_name == ent2.physical_name:
            continue
        interface_dim = ent1.dim - 1
        common_shapes = [entity_boundary[i1][bid] for bid in common]
        for n1, n2 in product(ent1.physical_name, ent2.physical_name):
            name = f"{n1}{interface_delimiter}{n2}"
            groups.setdefault((interface_dim, name), []).extend(common_shapes)

    # 3. tag_boundaries: exterior only for keep=True entities. Helpers
    #    (keep=False) get removed post-load, so their "exterior" would
    #    dangle.
    lower_dim_ids: set[int] = set()
    for ent, leaves in zip(entities, entity_leaves):
        if ent.dim < max_dim:
            for leaf in leaves:
                lower_dim_ids.add(_HASHER(leaf))

    for i, ent in enumerate(entities):
        if ent.dim != max_dim or not ent.keep:
            continue
        boundary_dim = ent.dim - 1
        bids = set(entity_boundary[i].keys())
        exterior = bids - entity_interface_ids[i] - lower_dim_ids
        exterior_shapes = [entity_boundary[i][bid] for bid in exterior]
        if not exterior_shapes:
            continue
        for name in ent.physical_name:
            full_name = f"{name}{interface_delimiter}{boundary_delimiter}"
            groups.setdefault((boundary_dim, full_name), []).extend(exterior_shapes)

    return groups


# ---------------------------------------------------------------------------
# XAO writer
# ---------------------------------------------------------------------------


def write_xao(
    entities: list[OCCLabeledEntity],
    xao_path: Path,
    model_name: str = "meshwell",
    interface_delimiter: str = "___",
    boundary_delimiter: str = "None",
) -> None:
    """Serialize ``entities`` into a self-contained, fully-tagged XAO file.

    The XAO contains every physical group the meshwell tagging pipeline
    would produce (entities, interfaces, exterior boundaries), computed at
    OCP level from TShape identity.

    keep=False entities are **not** serialized into the BREP -- only
    their OCP sub-boundaries already shared (via BOPAlgo TShape identity)
    with a kept entity survive, as boundary sub-shapes of the kept
    entity's solid. That lets ``tag_interfaces`` still name ``A___helper``
    without putting helper's solid in the mesh. Interface/boundary
    computation still walks every entity's ``shapes`` list in Python
    memory; only the BREP serialization excludes keep=False.
    """
    # Build compound of keep=True shapes only. Sub-boundaries shared with
    # keep=False helpers come along for free because BOPAlgo made them
    # share TShape with the kept entity's solid.
    cb = BRep_Builder()
    compound = TopoDS_Compound()
    cb.MakeCompound(compound)
    for ent in entities:
        if not ent.keep:
            continue
        for s in ent.shapes:
            cb.Add(compound, s)

    # OCP's BRepTools only exposes Write to a file path (no stream API), so
    # round-trip through a temp file to get the BREP text for inline CDATA.
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".brep", delete=False) as tf:
        tf_path = Path(tf.name)
    try:
        BRepTools.Write_s(compound, str(tf_path))
        brep_text = tf_path.read_text()
    finally:
        tf_path.unlink(missing_ok=True)

    if "]]>" in brep_text:
        raise ValueError("BREP text contains ']]>' sequence, cannot embed in XAO CDATA")

    main_map = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(compound, main_map)

    physical_groups = _compute_physical_groups(
        entities, interface_delimiter, boundary_delimiter
    )

    # Build per-dim topology index covering every shape any group references.
    topology_index: dict[int, dict[int, int]] = {0: {}, 1: {}, 2: {}, 3: {}}
    topology_entries: dict[int, list[tuple[int, int]]] = {
        0: [],
        1: [],
        2: [],
        3: [],
    }
    for (dim, _), shapes in physical_groups.items():
        for shape in shapes:
            sid = _HASHER(shape)
            if sid not in topology_index[dim]:
                local = len(topology_entries[dim])
                topology_index[dim][sid] = local
                topology_entries[dim].append((local, main_map.FindIndex(shape)))

    root = ET.Element("XAO", version="1.0", author="meshwell")
    geom = ET.SubElement(root, "geometry", name=model_name)
    shape_el = ET.SubElement(geom, "shape", format="BREP")
    shape_el.text = f"__CDATA_PLACEHOLDER__{id(shape_el)}__"
    topology = ET.SubElement(geom, "topology")
    for dim in (0, 1, 2, 3):
        parent = ET.SubElement(
            topology,
            _DIM_TO_XAO_GROUP[dim],
            count=str(len(topology_entries[dim])),
        )
        for local, reference in topology_entries[dim]:
            ET.SubElement(
                parent,
                _DIM_TO_XAO_ELEM[dim],
                index=str(local),
                reference=str(reference),
            )

    # Skip empty groups: some entities (e.g. a fully-absorbed coincident
    # solid) emit a (dim, name) with zero shapes. Writing them as
    # ``<group count="0"/>`` self-closing tags interferes with gmsh's
    # XAO reader -- elements of the next sibling group get misattributed.
    non_empty: list[tuple[tuple[int, str], list[int]]] = []
    for (dim, name), shapes in physical_groups.items():
        seen_ids: set[int] = set()
        local_indices: list[int] = []
        for shape in shapes:
            sid = _HASHER(shape)
            if sid in seen_ids:
                continue
            seen_ids.add(sid)
            local_indices.append(topology_index[dim][sid])
        if local_indices:
            non_empty.append(((dim, name), local_indices))

    groups_el = ET.SubElement(root, "groups", count=str(len(non_empty)))
    for (dim, name), local_indices in non_empty:
        g = ET.SubElement(
            groups_el,
            "group",
            name=name,
            dimension=_DIM_TO_XAO_ELEM[dim],
            count=str(len(local_indices)),
        )
        for local in local_indices:
            ET.SubElement(g, "element", index=str(local))

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")

    xml_bytes = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    xml_text = xml_bytes.decode("utf-8")
    placeholder = f"__CDATA_PLACEHOLDER__{id(shape_el)}__"
    xml_text = xml_text.replace(placeholder, f"<![CDATA[{brep_text}]]>")
    xao_path.write_text(xml_text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Injection (XAO -> gmsh, with keep=False removal + safety net)
# ---------------------------------------------------------------------------


def _resolve_entity_dimtags(
    occ_entities: list[OCCLabeledEntity],
    gmsh_model,
) -> list[list[tuple[int, int]]]:
    """Look up each entity's gmsh dimtags by its first physical_name.

    Two entities with the same physical_name collapse to the same
    dimtag list -- meshwell's convention: a shared name means a shared
    logical group.
    """
    # Build (dim, name) -> list of tags, one query per physical group.
    by_name: dict[tuple[int, str], list[int]] = {}
    for dim in (0, 1, 2, 3):
        for d, ptag in gmsh_model.getPhysicalGroups(dim):
            name = gmsh_model.getPhysicalName(d, ptag)
            by_name[(d, name)] = [
                int(t) for t in gmsh_model.getEntitiesForPhysicalGroup(d, ptag)
            ]

    entity_dimtags: list[list[tuple[int, int]]] = []
    for ent in occ_entities:
        if not ent.physical_name:
            entity_dimtags.append([])
            continue
        name = ent.physical_name[0]
        tags = by_name.get((ent.dim, name), [])
        entity_dimtags.append([(ent.dim, t) for t in tags])
    return entity_dimtags


def _run_remove_all_duplicates_safety_net(
    gmsh_model,
    entity_dimtags: list[list[tuple[int, int]]],
) -> list[list[tuple[int, int]]]:
    """Post-import gmsh-level fragment safety net preserving per-entity dimtags.

    Mirrors ``occ.removeAllDuplicates()`` (which calls
    ``booleanFragments(-1, allDimTags, ...)`` internally) but keeps the
    ``outDimTagsMap`` so we can forward each entity's dimtags to the new
    post-fragment tags.
    """
    if not any(entity_dimtags):
        return entity_dimtags

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

    _, out_map = gmsh_model.occ.fragment(
        unique_dimtags,
        [],
        removeObject=True,
        removeTool=True,
    )
    gmsh_model.occ.synchronize()

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
    return new_entity_dimtags


def _strip_dangling_sub_entities(gmsh_model, max_dim: int) -> None:
    """Remove dim-1 gmsh entities with no parent after keep=False removal."""
    if max_dim not in (2, 3):
        return
    below = max_dim - 1
    dangling = [
        (dim, tag)
        for dim, tag in gmsh_model.getEntities(below)
        if len(gmsh_model.getAdjacencies(dim, tag)[0]) == 0
    ]
    if dangling:
        gmsh_model.occ.remove(dangling, recursive=True)
        gmsh_model.occ.synchronize()


def inject_occ_entities_into_gmsh(
    occ_entities: list[OCCLabeledEntity],
    model_manager: ModelManager | None = None,
    interface_delimiter: str = "___",
    boundary_delimiter: str = "None",
    progress_bars: bool = False,
    remove_all_duplicates: bool = False,
) -> None:
    """Inject OCC entities into a gmsh model (full tagging + optional safety net).

    Pipeline: write fully-tagged XAO -> ``gmsh.open`` -> optional fragment
    safety net -> dangling cleanup. All physical groups (entities,
    interfaces, exteriors) come from the XAO; the resolution / refinement
    engine in :mod:`meshwell.mesh` recovers per-entity state from those
    groups when meshing runs.

    Args:
        occ_entities: list of ``OCCLabeledEntity`` objects from
            :mod:`meshwell.cad_occ`.
        model_manager: ``ModelManager`` to load into. A fresh one is created
            if ``None``.
        interface_delimiter: separator between physical names in interface
            and exterior-boundary group names.
        boundary_delimiter: marker appended after ``interface_delimiter`` for
            exterior boundaries (e.g. ``A___None``).
        progress_bars: emit status lines during the bridge.
        remove_all_duplicates: run a gmsh-level fragment safety net across
            all imported dimtags after loading. Complements the OCP-side
            canonicalization for rare cases where BREP round-trip fragments
            residuals.
    """
    from meshwell.model import ModelManager as _ModelManager

    owns_model = False
    if model_manager is None:
        model_manager = _ModelManager()
        owns_model = True

    model_manager.ensure_initialized(str(model_manager.filename))
    gmsh_model = model_manager.model

    # More accurate bbox queries from STL tessellation (tag-safe).
    import gmsh as _gmsh

    _gmsh.option.setNumber("Geometry.OCCBoundsUseStl", 1)

    max_dim = max((e.dim for e in occ_entities if e.shapes), default=0)

    if progress_bars:
        print(
            f"Writing XAO for {len(occ_entities)} entities and importing into gmsh…",
            flush=True,
        )
    with tempfile.TemporaryDirectory() as tmpdir:
        xao_path = Path(tmpdir) / "all_entities.xao"
        write_xao(
            occ_entities,
            xao_path,
            interface_delimiter=interface_delimiter,
            boundary_delimiter=boundary_delimiter,
        )
        _gmsh.open(str(xao_path))
        gmsh_model.occ.synchronize()

    if remove_all_duplicates:
        if progress_bars:
            print("Fragment safety net across imported dimtags…", flush=True)
        entity_dimtags = _resolve_entity_dimtags(occ_entities, gmsh_model)
        _run_remove_all_duplicates_safety_net(gmsh_model, entity_dimtags)

    _strip_dangling_sub_entities(gmsh_model, max_dim)

    if owns_model:
        model_manager.finalize()


def occ_to_xao(
    occ_entities: list[OCCLabeledEntity],
    output_file: Path,
    model_manager: ModelManager | None = None,
    interface_delimiter: str = "___",
    boundary_delimiter: str = "None",
    progress_bars: bool = False,
    remove_all_duplicates: bool = False,
) -> None:
    """Inject the entities into a gmsh model and save it to ``output_file``.

    Convenience wrapper: calls :func:`inject_occ_entities_into_gmsh` then
    ``model_manager.save_to_xao`` so the saved file carries the full
    tagged state gmsh built.
    """
    from meshwell.model import ModelManager as _ModelManager

    owns_model = False
    if model_manager is None:
        model_manager = _ModelManager()
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
