"""XAO writer for OCC → gmsh handoff.

Writing XAO lets us piggyback on gmsh's built-in XAO reader (see
``gmsh/src/geo/GModelIO_OCC.cpp:6623``) which maps named physical groups
to gmsh entities via ``TopExp::MapShapes`` references — exact TShape
identity, no mass/centroid heuristic. We assign each entity a unique
*marker* physical-group name so the caller can recover per-entity
dimtags via ``getEntitiesForPhysicalGroup`` after load, then drops the
markers so the real physical-tagging (``tag_entities``) runs as usual.

The BREP blob is embedded inline in CDATA — single self-contained file,
matching gmsh's own ``_writeXAO`` convention. BREP is required by XAO:
it is the only shape format gmsh's XAO reader accepts, and the
topology ``reference`` integers are indices into
``TopExp::MapShapes(mainShape)`` — an OCC enumeration that depends on
a bit-identical shape on both sides of the round-trip.
"""
from __future__ import annotations

import tempfile
import xml.etree.ElementTree as ET
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


_HASHER = TopTools_ShapeMapHasher()

_DIM_TO_TOPABS = {
    3: TopAbs_SOLID,
    2: TopAbs_FACE,
    1: TopAbs_EDGE,
    0: TopAbs_VERTEX,
}
_DIM_TO_XAO_ELEM = {3: "solid", 2: "face", 1: "edge", 0: "vertex"}
_DIM_TO_XAO_GROUP = {3: "solids", 2: "faces", 1: "edges", 0: "vertices"}


def _marker_name(entity_index: int) -> str:
    """Unique-per-entity XAO group name used to reverse-lookup dimtags."""
    return f"_meshwell_xao_marker_{entity_index}"


def _leaf_subshapes(shape, dim):
    """Yield all sub-shapes of `shape` at the TopAbs class matching `dim`.

    Mirrors the leaf-enumeration gmsh's ``_multiBind`` does when binding
    imported OCC shapes (one gmsh entity per leaf TopAbs). Dedupe by
    TShape identity via :class:`TopTools_ShapeMapHasher`.
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


def write_xao(
    entities: list[OCCLabeledEntity],
    xao_path: Path,
    model_name: str = "meshwell",
) -> dict[int, str]:
    """Serialize ``entities`` into a single self-contained XAO for gmsh.

    The XAO writes:
    - ``<shape format="BREP">`` with the BREP blob inline in CDATA.
    - A ``<topology>`` section with a ``<solid/face/edge/vertex>`` entry for
      every leaf sub-shape referenced by any entity, keyed by a dim-local
      ``index`` (entry order within its class) and a global ``reference``
      (index in ``TopExp::MapShapes`` over the compound — what the reader
      round-trips back into a ``TopoDS_Shape``).
    - A ``<groups>`` section with one group per entity, using a unique
      marker name so the caller can recover per-entity dimtags via
      ``gmsh.model.getEntitiesForPhysicalGroup``.

    Args:
        entities: OCCLabeledEntity list (same input order preserved).
        xao_path: where to write the XAO file.
        model_name: written as ``<geometry name="...">``; cosmetic.

    Returns:
        Map ``{entity.index: marker_name}``. After gmsh loads the XAO,
        ``marker_name`` is a physical-group name whose members are this
        entity's dimtags. Caller typically strips these markers after
        using them to populate LabeledEntities.
    """
    # Build compound of all shapes (original entity order).
    cb = BRep_Builder()
    compound = TopoDS_Compound()
    cb.MakeCompound(compound)
    for ent in entities:
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

    # Defensive: BREP is plain ASCII but if some locale/encoding ever emits
    # a literal ``]]>`` we'd break the CDATA. Raise instead of corrupting.
    if "]]>" in brep_text:
        raise ValueError("BREP text contains ']]>' sequence, cannot embed in XAO CDATA")

    # Global index: TopoDS_Shape -> integer. Reader uses mainMap.FindKey(K).
    main_map = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(compound, main_map)

    # Per-dim tables:
    #   topology_index[dim][_shape_id] = local index within dim class (0..N-1)
    #   topology_entries[dim] = list of (local_index, reference) in write order.
    topology_index: dict[int, dict[int, int]] = {0: {}, 1: {}, 2: {}, 3: {}}
    topology_entries: dict[int, list[tuple[int, int]]] = {0: [], 1: [], 2: [], 3: []}
    # Cache leaf-enumeration results so we don't redo TopExp_Explorer per-entity.
    leaves_per_entity: list[list[tuple[int, int]]] = []  # [(dim, local_index), ...]

    for ent in entities:
        entity_leaves: list[tuple[int, int]] = []
        for s in ent.shapes:
            for leaf, leaf_id in _leaf_subshapes(s, ent.dim):
                t_idx = topology_index[ent.dim]
                if leaf_id not in t_idx:
                    local = len(topology_entries[ent.dim])
                    t_idx[leaf_id] = local
                    reference = main_map.FindIndex(leaf)
                    topology_entries[ent.dim].append((local, reference))
                entity_leaves.append((ent.dim, t_idx[leaf_id]))
        leaves_per_entity.append(entity_leaves)

    # Build the XML tree.
    root = ET.Element("XAO", version="1.0", author="meshwell")
    geom = ET.SubElement(root, "geometry", name=model_name)
    shape_el = ET.SubElement(geom, "shape", format="BREP")
    # ElementTree has no CDATA primitive; insert the raw marker and post-process.
    shape_el.text = f"__CDATA_PLACEHOLDER__{id(shape_el)}__"
    topology = ET.SubElement(geom, "topology")
    for dim in (0, 1, 2, 3):
        group_tag = _DIM_TO_XAO_GROUP[dim]
        elem_tag = _DIM_TO_XAO_ELEM[dim]
        parent = ET.SubElement(
            topology, group_tag, count=str(len(topology_entries[dim]))
        )
        for local, reference in topology_entries[dim]:
            ET.SubElement(
                parent,
                elem_tag,
                index=str(local),
                reference=str(reference),
            )

    markers: dict[int, str] = {}
    groups = ET.SubElement(root, "groups", count=str(len(entities)))
    for ent, entity_leaves in zip(entities, leaves_per_entity):
        if not entity_leaves:
            continue
        name = _marker_name(ent.index)
        markers[ent.index] = name
        # All leaves of one entity share the same dim by construction.
        dim = entity_leaves[0][0]
        g = ET.SubElement(
            groups,
            "group",
            name=name,
            dimension=_DIM_TO_XAO_ELEM[dim],
            count=str(len(entity_leaves)),
        )
        for _, local in entity_leaves:
            ET.SubElement(g, "element", index=str(local))

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")

    # Serialize to string so we can swap the placeholder for a real CDATA
    # block. ElementTree's built-in writer escapes '<' and '>' in text.
    xml_bytes = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    xml_text = xml_bytes.decode("utf-8")
    placeholder = f"__CDATA_PLACEHOLDER__{id(shape_el)}__"
    xml_text = xml_text.replace(placeholder, f"<![CDATA[{brep_text}]]>")
    xao_path.write_text(xml_text, encoding="utf-8")

    return markers
