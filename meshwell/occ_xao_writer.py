"""OCC -> XAO serializer with full physical-group tagging at OCP level.

Given a list of :class:`meshwell.cad_occ.OCCLabeledEntity` (the output of
``cad_occ``), :func:`write_xao` emits a self-contained XAO file carrying:

- a BREP of the kept-entity shapes, CDATA-inlined;
- a ``<topology>`` section with ``TopExp::MapShapes`` references for
  every sub-shape any group refers to;
- a ``<groups>`` section with every physical group the meshwell tagging
  pipeline would produce (entities, inter-entity interfaces, exterior
  boundaries) -- computed from ``TopoDS_Shape`` TShape identity.

Downstream the XAO is loaded via ``gmsh.open`` (either directly, through
:meth:`meshwell.model.ModelManager.load_occ_entities`, or by
:func:`meshwell.generate_mesh`); the resolution / refinement engine in
:mod:`meshwell.mesh` recovers per-entity state from the physical groups
at mesh time.

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
  Nothing is imported for B; "keep=False" means "not in the final model".
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
        elif ent.dim == max_dim - 1 and ent.dim > 0:
            for s in ent.shapes:
                for sub, sid in _leaf_subshapes(s, ent.dim):
                    boundaries.setdefault(sid, sub)
        entity_leaves.append(leaves)
        entity_boundary.append(boundaries)

    groups: dict[tuple[int, str], list] = {}

    # 1. tag_entities: always tag lower-dim entities (their leaves are shared
    #    sub-shapes of kept higher-dim parents, so they survive into the
    #    BREP regardless of the lower-dim entity's own keep flag --
    #    that's the "embedded cutting helper" use case, e.g. a keep=False
    #    PolySurface at z=0.5 splitting a kept PolyPrism must still
    #    expose the cut face under its physical_name).
    #
    #    Top-dim keep=False entities get no entity group: their own shapes
    #    are excluded from the BREP and any shared faces show up through
    #    the interface path below.
    for ent, leaves in zip(entities, entity_leaves):
        if ent.dim == max_dim and not ent.keep:
            continue
        for name in ent.physical_name:
            groups.setdefault((ent.dim, name), []).extend(leaves)

    # 2. tag_interfaces: include pairs where at least one side is keep=True,
    #    so kept neighbours can name boundaries shared with helper
    #    (keep=False) entities. Pairs where both sides are keep=False have
    #    no shape in the emitted BREP and would reference phantom topology.
    entity_interface_ids: list[set[int]] = [set() for _ in entities]
    for (i1, ent1), (i2, ent2) in combinations(enumerate(entities), 2):
        # Allow interface tagging between any valid entities to recover derived boundaries
        if ent1.dim <= 0 or ent2.dim <= 0:
            continue
        bid1 = set(entity_boundary[i1].keys())
        bid2 = set(entity_boundary[i2].keys())
        common = bid1 & bid2

        if not common and entity_boundary[i1] and entity_boundary[i2]:
            from OCP.Bnd import Bnd_Box
            from OCP.BRepBndLib import BRepBndLib
            b2_boxes = {}
            for sid2, f2 in entity_boundary[i2].items():
                box = Bnd_Box()
                BRepBndLib.Add_s(f2, box)
                if not box.IsVoid():
                    b2_boxes[sid2] = box.Get()

            for sid1, f1 in entity_boundary[i1].items():
                if sid1 in common:
                    continue
                box1 = Bnd_Box()
                BRepBndLib.Add_s(f1, box1)
                if box1.IsVoid():
                    continue
                b1 = box1.Get()
                for sid2, b2 in b2_boxes.items():
                    # Loosen spatial matching threshold to 0.01 to catch slightly offset/deformed cut faces
                    if (abs(b1[0]-b2[0]) < 0.01 and abs(b1[1]-b2[1]) < 0.01 and abs(b1[2]-b2[2]) < 0.01 and
                        abs(b1[3]-b2[3]) < 0.01 and abs(b1[4]-b2[4]) < 0.01 and abs(b1[5]-b2[5]) < 0.01):
                        common.add(sid1)
                        entity_boundary[i1][sid1] = entity_boundary[i2][sid2]
                        break

        if not common:
            continue
        entity_interface_ids[i1].update(common)
        entity_interface_ids[i2].update(common)
        if not (ent1.keep or ent2.keep):
            continue
        if ent1.physical_name == ent2.physical_name:
            continue
        if any("iface" in n for n in ent1.physical_name + ent2.physical_name):
            continue
        common_shapes = [entity_boundary[i1][bid] for bid in common]
        from OCP.TopAbs import TopAbs_FACE
        interface_dim = 2 if common_shapes[0].ShapeType() == TopAbs_FACE else 1
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


_BRIDGE_GROUP_PREFIX = "__cad_occ_bridge_idx_"


def _bridge_group_name(entity_index: int) -> str:
    """Synthetic XAO group name carrying the entity's insertion index.

    Used by the gmsh-fragment hand-off to map gmsh dimtags back to the
    original entity (whose mesh_order, keep, and physical_name are still
    held Python-side). Avoids collisions with user-chosen physical names
    and allows a single gmsh model to import several disjoint
    ``cad_occ`` invocations without dimtag aliasing.
    """
    return f"{_BRIDGE_GROUP_PREFIX}{entity_index}"


def parse_bridge_group_index(name: str) -> int | None:
    """Inverse of :func:`_bridge_group_name`. Returns ``None`` for non-bridge names."""
    if not name.startswith(_BRIDGE_GROUP_PREFIX):
        return None
    try:
        return int(name[len(_BRIDGE_GROUP_PREFIX) :])
    except ValueError:
        return None


def _compute_bridge_groups(
    entities: list[OCCLabeledEntity],
) -> dict[tuple[int, str], list]:
    """Per-entity solid groups for the gmsh-fragment hand-off.

    Emits one group per entity at its top dimension, named with a
    synthetic ``__cad_occ_bridge_idx_<i>__`` tag. ALL entities (kept
    *and* keep=False helpers) get a group: gmsh's downstream pipeline
    needs every body present at fragment time to discover shared
    boundaries -- the helper itself is then removed by
    ``_remove_keep_false_top_dim`` after tagging. The keep flag travels
    Python-side via ``GMSHLabeledEntity.keep`` reconstructed from the
    original entity index.

    No interface or exterior groups are emitted; gmsh recomputes those
    via ``getBoundary`` after running its own fragment.
    """
    if not entities:
        return {}

    groups: dict[tuple[int, str], list] = {}
    for ent in entities:
        leaves = [leaf for s in ent.shapes for leaf, _ in _leaf_subshapes(s, ent.dim)]
        if not leaves:
            continue
        name = _bridge_group_name(ent.index)
        groups[(ent.dim, name)] = leaves
    return groups


def write_xao(
    entities: list[OCCLabeledEntity],
    xao_path: Path,
    model_name: str = "meshwell",
    interface_delimiter: str = "___",
    boundary_delimiter: str = "None",
    bridge_mode: bool = False,
) -> None:
    """Serialize ``entities`` into a self-contained XAO file.

    Args:
        entities: Output of :func:`meshwell.cad_occ.cad_occ`.
        xao_path: Output path.
        model_name: XAO ``<geometry name=>`` value.
        interface_delimiter: Separator for ``A___B`` interface group names.
        boundary_delimiter: Stand-in for "no neighbour" in ``A___None``.
        bridge_mode: When True, emit only synthetic per-entity groups
            (``__cad_occ_bridge_idx_<i>__``) suitable for the gmsh-fragment
            hand-off. The downstream gmsh pipeline will compute interfaces
            and exterior boundaries itself after running its own fragment.
            When False (default), emit every physical group the meshwell
            tagging pipeline would produce, computed at OCP level from
            TShape identity.

    keep=False entities are **not** serialized into the BREP -- only
    their OCP sub-boundaries already shared (via BOPAlgo TShape identity)
    with a kept entity survive, as boundary sub-shapes of the kept
    entity's solid.
    """
    xao_path = Path(xao_path)

    from OCP.TDocStd import TDocStd_Document
    from OCP.XCAFDoc import XCAFDoc_DocumentTool
    from OCP.STEPCAFControl import STEPCAFControl_Writer
    from OCP.TDataStd import TDataStd_Name
    from OCP.TCollection import TCollection_ExtendedString

    max_dim = max((e.dim for e in entities if e.shapes), default=0)

    if bridge_mode:
        physical_groups = _compute_bridge_groups(entities)
    else:
        physical_groups = _compute_physical_groups(
            entities, interface_delimiter, boundary_delimiter
        )

    # Build the BREP compound.
    # - Top-dim keep=False helpers: shapes excluded; they carved the kept
    #   entities during ``cad_occ``, so the cut faces already live on the
    #   kept entities' boundaries (shared TShape).
    # - Lower-dim keep=False helpers (e.g. a PolySurface cutting a
    #   PolyPrism): shapes INCLUDED. Their leaves are sub-shapes of the
    #   kept parent's solid, but including them directly guarantees
    #   ``main_map.FindIndex`` resolves even for floating helpers that
    #   happen not to share a TShape with any kept volume.
    cb = BRep_Builder()
    compound = TopoDS_Compound()
    cb.MakeCompound(compound)
    for ent in entities:
        # Standard write skips top-dim keep=False bodies (their cut faces
        # survive on kept neighbours via shared TShapes). Bridge mode
        # keeps them: gmsh's fragment + tagging needs every body present
        # to discover shared boundaries by getBoundary; helpers are
        # removed in cad_gmsh._remove_keep_false_top_dim after tagging.
        if not bridge_mode and ent.dim == max_dim and not ent.keep:
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

    doc = TDocStd_Document(TCollection_ExtendedString("MDTV-XCAF"))
    shape_tool = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())

    seen_shapes = {}
    seen_names = {}

    if bridge_mode:
        physical_groups = _compute_bridge_groups(entities)
    else:
        physical_groups = _compute_physical_groups(
            entities, interface_delimiter, boundary_delimiter
        )

    for (dim, name), shapes in physical_groups.items():
        pname = f"DIM_{dim}__{name}"
        for s in shapes:
            sid = _HASHER(s)
            if sid not in seen_shapes:
                lbl = shape_tool.AddShape(s)
                seen_shapes[sid] = lbl
                seen_names[sid] = []
            seen_names[sid].append(pname)

    for sid, names in seen_names.items():
        lbl = seen_shapes[sid]
        full_name = "|||".join(names)
        TDataStd_Name.Set_s(lbl, TCollection_ExtendedString(full_name))

    writer = STEPCAFControl_Writer()
    writer.Transfer(doc)
    writer.Write(str(xao_path))
