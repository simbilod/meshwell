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

- **Why not STEP+XCAF**: an earlier iteration attached XCAF labels per
  sub-shape and exported as STEP, but ``XCAFDoc_ShapeTool::AddShape`` on
  a face that is already a sub-face of an added solid produces a
  sub-component label whose ``TDataStd_Name`` is silently dropped at
  STEP export time -- only the parent solid keeps its name. That cost us
  the ``A___B`` interface and ``B___None`` boundary groups, plus a
  large perf regression (~2-3x on the write+load round trip for
  hundred-prism scenes) from the XCAF tree walking gmsh performs on
  import.

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

import numpy as np
from OCP.Bnd import Bnd_Box
from OCP.BRep import BRep_Builder
from OCP.BRepBndLib import BRepBndLib
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

# Used by the spatial fallback in ``_compute_physical_groups`` when
# TShape-identity matching fails to pair a cut face with its
# coincident neighbour. Picks up cut faces drifted by ~mm at user
# scale; only kicks in when identity matching produced no commons.
_AABB_INTERFACE_TOL = 1e-2

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


def _shape_aabb(shape) -> tuple[float, ...] | None:
    """Return the (xmin, ymin, zmin, xmax, ymax, zmax) AABB or None if void."""
    box = Bnd_Box()
    BRepBndLib.Add_s(shape, box)
    if box.IsVoid():
        return None
    return box.Get()


def _aabbs_close(b1: tuple[float, ...], b2: tuple[float, ...], tol: float) -> bool:
    """Per-corner L_inf proximity between two AABBs."""
    return all(abs(b1[i] - b2[i]) < tol for i in range(6))


def _is_purely_synthetic(ent: OCCLabeledEntity) -> bool:
    """True if ``ent`` is a structured-pipeline bookkeeping companion only.

    Structured-pipeline post-pass emits two flavours of entities carrying
    a synthetic ``__cohort_<ci>__slab_<si>[...]`` name:

    1. Real cohort sub-solids: their ``physical_name`` is a tuple where
       the synthetic name is **appended** to the user's real name
       (e.g. ``("lower", "__cohort_0__slab_0")``). These are real
       geometry — they own faces, can interface with neighbours, and
       must produce ``A___B`` / ``A___None`` groups normally.

    2. Synthetic 2D face annotators: their ``physical_name`` consists
       **only** of a synthetic name (e.g. ``("__cohort_0__slab_0__top",)``).
       They duplicate the parent solid's face TShape so the orchestrator
       can resolve face → gmsh tag by physical-group name lookup after
       gmsh.merge. They must not affect interface or boundary detection
       — purely an annotator on an existing face.

    We use "all names start with ``__cohort_``" rather than "any" so the
    real sub-solid (flavour 1) is treated as real geometry while the
    annotator (flavour 2) is correctly identified as bookkeeping-only.
    """
    return bool(ent.physical_name) and all(
        n.startswith("__cohort_") for n in ent.physical_name
    )


def _filter_real_names(names: tuple[str, ...]) -> tuple[str, ...]:
    """Drop synthetic ``__cohort_*`` names; keep the user-visible names only.

    Used when forming ``A___B`` interface group names from a pair of real
    cohort sub-solids: their ``physical_name`` tuple has a synthetic name
    appended, but we only want the user-visible ``A___B``, not also the
    spurious ``A_____cohort_X``, ``__cohort_X___B``, ``__cohort_X_____cohort_Y``.
    """
    return tuple(n for n in names if not n.startswith("__cohort_"))


def _compute_physical_groups(
    entities: list[OCCLabeledEntity],
    interface_delimiter: str,
    boundary_delimiter: str,
) -> dict[tuple[int, str], list]:
    """OCP reconstruction of ``tag_entities``/``tag_interfaces``/``tag_boundaries``.

    Returns ``{(dim, physical_name): [TopoDS_Shape, ...]}``. Interfaces are
    named by TShape identity of shared sub-shapes, with an AABB-proximity
    fallback for cut faces whose TShapes drifted apart during fragmentation.
    keep=False entities participate in interface tagging (so kept neighbours
    can name shared boundaries) but do not get their own entity or exterior
    groups -- they will be removed from gmsh after load.
    """
    if not entities:
        return {}

    max_dim = max((e.dim for e in entities if e.shapes), default=0)

    # entity_leaves[i]  : TopoDS_Shape list at ent.dim (entity's own tags).
    # entity_boundary[i]: {tshape_id: TopoDS_Shape} at the appropriate
    #                     boundary dim, populated for top-dim entities (their
    #                     dim-1 faces) and for dim-1 entities (their own dim
    #                     leaves -- used to wire up internal embedded
    #                     surfaces back to a top-dim parent's faces).
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
            # Lower-dim entity: its own leaves act as the boundary index
            # so the interface pass can match them against a parent's
            # faces (embedded internal surface case).
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

    # Pre-compute AABBs once per entity (was recomputed per pair before).
    # Keyed by tshape_id; None when the shape's bbox is void.
    entity_aabbs: list[dict[int, tuple[float, ...]]] = []
    for i in range(len(entities)):
        boxes: dict[int, tuple[float, ...]] = {}
        for sid, face in entity_boundary[i].items():
            box = _shape_aabb(face)
            if box is not None:
                boxes[sid] = box
        entity_aabbs.append(boxes)

    # Pre-stack each entity's AABBs into a numpy array for vectorized
    # proximity matching in the fallback below. ``entity_sids[i]`` is
    # the sid-in-array-order list; entity_aabb_arr[i] is the (F, 6) array.
    entity_sids: list[list[int]] = []
    entity_aabb_arr: list[np.ndarray] = []
    for boxes in entity_aabbs:
        sids = list(boxes.keys())
        entity_sids.append(sids)
        entity_aabb_arr.append(
            np.array([boxes[s] for s in sids], dtype=float)
            if sids
            else np.zeros((0, 6), dtype=float)
        )

    entity_interface_ids: list[set[int]] = [set() for _ in entities]
    for (i1, ent1), (i2, ent2) in combinations(enumerate(entities), 2):
        if ent1.dim <= 0 or ent2.dim <= 0:
            continue
        bid1 = set(entity_boundary[i1].keys())
        bid2 = set(entity_boundary[i2].keys())
        common = bid1 & bid2

        # Spatial AABB fallback: cut faces that drifted apart in
        # TShape identity but still occupy the same region. Only runs
        # when identity-based matching produced nothing. Vectorized
        # with numpy: for each face on entity 1, compute the L_inf
        # per-corner distance to every face on entity 2 in one shot,
        # then pick the first index whose distance is within tolerance.
        if not common and entity_aabbs[i1] and entity_aabbs[i2]:
            arr2 = entity_aabb_arr[i2]
            sids2 = entity_sids[i2]
            for sid1, b1 in entity_aabbs[i1].items():
                b1_arr = np.asarray(b1, dtype=float)
                # Per-row L_inf corner distance to b1 across all of entity 2.
                dists = np.abs(arr2 - b1_arr).max(axis=1)
                matches = np.where(dists < _AABB_INTERFACE_TOL)[0]
                if matches.size == 0:
                    continue
                # Preserve "pick first match by sid-array order" semantics
                # for determinism.
                sid2 = sids2[int(matches[0])]
                common.add(sid1)
                entity_boundary[i1][sid1] = entity_boundary[i2][sid2]

        if not common:
            continue
        # Purely-synthetic structured-pipeline 2D annotators duplicate the
        # parent solid's face TShape so the orchestrator can name-resolve
        # ``__cohort_X__slab_Y__role`` -> gmsh face tag after merge. They
        # are bookkeeping companions, not real geometry: they must NOT
        # consume their parent's exterior boundary faces (otherwise
        # ``A___None`` is empty), and they must NOT contribute to "shared
        # between A and B" interface detection (otherwise the natural
        # ``A___B`` interface gets attributed to ``A___annotator`` and
        # the real ``A___B`` is never emitted).
        #
        # Real cohort sub-solids ALSO carry a synthetic name appended to
        # their tuple (e.g. ``("lower", "__cohort_0__slab_0")``) but they
        # are real geometry — their interfaces still need to be detected
        # normally. ``_is_purely_synthetic`` distinguishes the two flavours.
        if _is_purely_synthetic(ent1) or _is_purely_synthetic(ent2):
            continue
        entity_interface_ids[i1].update(common)
        entity_interface_ids[i2].update(common)
        if not (ent1.keep or ent2.keep):
            continue
        # Skip synthetic names from the cross product: a cohort sub-solid
        # named ``("lower", "__cohort_0__slab_0")`` should yield the user
        # ``lower___upper`` interface only, not ``__cohort_0__slab_0___upper``.
        # Fall back to the full tuple if filtering left nothing — only
        # happens for purely-synthetic pairs, which we skipped above.
        names1 = _filter_real_names(ent1.physical_name) or ent1.physical_name
        names2 = _filter_real_names(ent2.physical_name) or ent2.physical_name
        # Two cohort sub-solids from the same user entity (e.g. ``b``
        # split into two slabs) share an internal slab boundary face;
        # that's an interior face of ``b``, not a ``b___b`` group.
        # Compare on the real-name view so the synthetic appendix
        # doesn't fool the same-entity check.
        if names1 == names2:
            continue
        # Internal "iface" helpers (named e.g. ``oxide_iface``) carry their
        # own physical_name and should not also be glued to a neighbour pair.
        if any("iface" in n for n in ent1.physical_name + ent2.physical_name):
            continue
        common_shapes = [entity_boundary[i1][bid] for bid in common]
        # interface_dim is determined by the actual shape type of the shared
        # sub-shape rather than ``ent.dim - 1`` -- when a lower-dim embedded
        # entity matches a top-dim entity's face, both are FACEs (dim=2)
        # even though ent.dim differs.
        interface_dim = 2 if common_shapes[0].ShapeType() == TopAbs_FACE else 1
        for n1, n2 in product(names1, names2):
            name = f"{n1}{interface_delimiter}{n2}"
            groups.setdefault((interface_dim, name), []).extend(common_shapes)

    # 3. tag_boundaries: exterior only for keep=True entities. Helpers
    #    (keep=False) get removed post-load, so their "exterior" would
    #    dangle.
    lower_dim_ids: set[int] = set()
    for ent, leaves in zip(entities, entity_leaves):
        if ent.dim < max_dim:
            # Purely-synthetic structured-pipeline 2D annotators tag the
            # same TShape as their user-visible parent; they must not
            # steal those faces out of the parent's ``___None`` exterior
            # boundary group.
            if _is_purely_synthetic(ent):
                continue
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
    """Serialize ``entities`` into a self-contained XAO file.

    Args:
        entities: Output of :func:`meshwell.cad_occ.cad_occ`.
        xao_path: Output path.
        model_name: XAO ``<geometry name=>`` value.
        interface_delimiter: Separator for ``A___B`` interface group names.
        boundary_delimiter: Stand-in for "no neighbour" in ``A___None``.

    keep=False entities are **not** serialized into the BREP -- only
    their OCP sub-boundaries already shared (via BOPAlgo TShape identity)
    with a kept entity survive, as boundary sub-shapes of the kept
    entity's solid.
    """
    xao_path = Path(xao_path)

    max_dim = max((e.dim for e in entities if e.shapes), default=0)

    # Build the BREP compound.
    # - Top-dim keep=False helpers: shapes excluded; they carved the kept
    #   entities during ``cad_occ``, so the cut faces already live on the
    #   kept entities' boundaries (shared TShape).
    # - Lower-dim keep=False helpers: shapes INCLUDED. Their leaves are
    #   sub-shapes of the kept parent's solid, but including them
    #   directly guarantees ``shape_reference_map.FindIndex`` resolves
    #   for floating helpers that happen not to share a TShape with any
    #   kept volume.
    compound_builder = BRep_Builder()
    brep_compound = TopoDS_Compound()
    compound_builder.MakeCompound(brep_compound)
    for ent in entities:
        if ent.dim == max_dim and not ent.keep:
            continue
        for s in ent.shapes:
            compound_builder.Add(brep_compound, s)

    # OCP's BRepTools only exposes Write to a file path (no stream API), so
    # round-trip through a temp file to get the BREP text for inline CDATA.
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".brep", delete=False) as tf:
        tf_path = Path(tf.name)
    try:
        BRepTools.Write_s(brep_compound, str(tf_path))
        brep_text = tf_path.read_text()
    finally:
        tf_path.unlink(missing_ok=True)

    if "]]>" in brep_text:
        raise ValueError("BREP text contains ']]>' sequence, cannot embed in XAO CDATA")

    shape_reference_map = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(brep_compound, shape_reference_map)

    physical_groups = _compute_physical_groups(
        entities, interface_delimiter, boundary_delimiter
    )

    # Build per-dim topology index covering every shape any group references.
    # ``local_index`` is the position within the per-dim ``<topology>`` list;
    # ``reference`` is the index into the BREP's TopExp shape map.
    topology_local_index: dict[int, dict[int, int]] = {0: {}, 1: {}, 2: {}, 3: {}}
    topology_entries: dict[int, list[tuple[int, int]]] = {0: [], 1: [], 2: [], 3: []}
    for (dim, _), shapes in physical_groups.items():
        for shape in shapes:
            sid = _HASHER(shape)
            if sid not in topology_local_index[dim]:
                local = len(topology_entries[dim])
                topology_local_index[dim][sid] = local
                topology_entries[dim].append(
                    (local, shape_reference_map.FindIndex(shape))
                )

    root = ET.Element("XAO", version="1.0", author="meshwell")
    geometry_el = ET.SubElement(root, "geometry", name=model_name)
    shape_el = ET.SubElement(geometry_el, "shape", format="BREP")
    cdata_placeholder = f"__CDATA_PLACEHOLDER__{id(shape_el)}__"
    shape_el.text = cdata_placeholder
    topology_el = ET.SubElement(geometry_el, "topology")
    for dim in (0, 1, 2, 3):
        per_dim_parent = ET.SubElement(
            topology_el,
            _DIM_TO_XAO_GROUP[dim],
            count=str(len(topology_entries[dim])),
        )
        for local, reference in topology_entries[dim]:
            ET.SubElement(
                per_dim_parent,
                _DIM_TO_XAO_ELEM[dim],
                index=str(local),
                reference=str(reference),
            )

    # Skip empty groups: some entities (e.g. a fully-absorbed coincident
    # solid) emit a (dim, name) with zero shapes. Writing them as
    # ``<group count="0"/>`` self-closing tags interferes with gmsh's
    # XAO reader -- elements of the next sibling group get misattributed.
    non_empty_groups: list[tuple[tuple[int, str], list[int]]] = []
    for (dim, name), shapes in physical_groups.items():
        seen_ids: set[int] = set()
        local_indices: list[int] = []
        for shape in shapes:
            sid = _HASHER(shape)
            if sid in seen_ids:
                continue
            seen_ids.add(sid)
            local_indices.append(topology_local_index[dim][sid])
        if local_indices:
            non_empty_groups.append(((dim, name), local_indices))

    groups_el = ET.SubElement(root, "groups", count=str(len(non_empty_groups)))
    for (dim, name), local_indices in non_empty_groups:
        group_el = ET.SubElement(
            groups_el,
            "group",
            name=name,
            dimension=_DIM_TO_XAO_ELEM[dim],
            count=str(len(local_indices)),
        )
        for local in local_indices:
            ET.SubElement(group_el, "element", index=str(local))

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")

    xml_bytes = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    xml_text = xml_bytes.decode("utf-8")
    xml_text = xml_text.replace(cdata_placeholder, f"<![CDATA[{brep_text}]]>")
    xao_path.write_text(xml_text, encoding="utf-8")
