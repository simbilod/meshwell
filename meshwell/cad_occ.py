"""OCC CAD processor: fragment + mesh_order ownership via OCP.

Mirrors :mod:`meshwell.cad_gmsh` but drives OCCT directly through OCP
instead of the gmsh API. The two backends are intentionally kept
structurally identical so users can compare outputs head-to-head:

1. Initialize per-session geometry cache (vertex / edge sharing is the
   OCP-side equivalent of ``gmsh.model.occ``'s built-in TShape reuse).
2. Call each entity's ``instanciate_occ`` to produce ``TopoDS_Shape``
   instances.
3. ``BOPAlgo_Builder`` all-fragment over every input shape; per-input
   ``Modified()`` tells us which fragment piece descends from which
   input, which we invert to build the per-piece candidate list.
4. Resolve multi-claim pieces by lowest ``mesh_order`` (first-inserted
   wins on tie).

``keep=False`` handling differs from the gmsh backend: here the helper's
shapes are preserved through fragmentation and the XAO writer
(:mod:`meshwell.occ_xao_writer`) skips their bodies at serialization
time while still using the shared TShapes to name
``neighbour___helper`` interfaces. The gmsh backend calls
``occ.remove(..., recursive=True)`` at model level for the same
user-visible result.

Ownership semantics match :func:`meshwell.cad_gmsh._resolve_piece_ownership`
exactly; tests that pin one pin the other.
"""
from __future__ import annotations

import contextlib
from collections import defaultdict
from dataclasses import dataclass
from os import cpu_count
from pathlib import Path
from typing import TYPE_CHECKING, Any

from OCP.Bnd import Bnd_Box
from OCP.BOPAlgo import BOPAlgo_Builder
from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepExtrema import BRepExtrema_DistShapeShape
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_SOLID, TopAbs_VERTEX
from OCP.TopExp import TopExp_Explorer
from OCP.TopTools import TopTools_ShapeMapHasher
from tqdm.auto import tqdm

import gmsh

if TYPE_CHECKING:
    from OCP.TopoDS import TopoDS_Shape


@dataclass
class OCCLabeledEntity:
    """Per-entity record produced by :func:`cad_occ`.

    ``shapes`` holds the fragment pieces this entity owns after the
    all-fragment pass.
    """

    shapes: list[TopoDS_Shape]
    physical_name: tuple[str, ...]
    index: int
    keep: bool
    dim: int
    mesh_order: float | None = None


_SHAPE_HASHER = TopTools_ShapeMapHasher()


def _shape_key(shape: TopoDS_Shape) -> tuple[int, int]:
    """Return a hashable identity key for a TopoDS_Shape.

    Uses the TShape pointer plus orientation so reversed shapes compare
    distinct when BOPAlgo differentiates them and equal when it does not.
    OCP returns a fresh Python wrapper each time ``TShape()`` is called,
    so ``id()``/default ``__hash__`` isn't stable -- ``TopTools_ShapeMapHasher``
    hashes on the underlying ``TShape*`` pointer instead.
    """
    return (_SHAPE_HASHER(shape), int(shape.Orientation()))


def _resolve_piece_ownership(
    piece_candidates: dict[Any, list[tuple[int, float]]],
) -> dict[Any, int]:
    """Pick the owning entity index for each fragment piece.

    Rule: lowest ``mesh_order`` wins; first candidate in insertion order
    wins on tie. Matches :func:`meshwell.cad_gmsh._resolve_piece_ownership`.
    """
    owners: dict[Any, int] = {}
    for piece, candidates in piece_candidates.items():
        best_idx = candidates[0][0]
        best_mo = candidates[0][1]
        for idx, mo in candidates[1:]:
            if mo < best_mo:
                best_idx = idx
                best_mo = mo
        owners[piece] = best_idx
    return owners


class CAD_OCC:
    """OCP-driven CAD processor: fragment + mesh_order ownership."""

    def __init__(
        self,
        point_tolerance: float = 1e-3,
        n_threads: int = cpu_count(),
        fuzzy_value: float | None = None,
        perturbation: float | None = None,
    ):
        """Initialize OCC CAD processor.

        Args:
            point_tolerance: Coordinate quantization applied inside entity
                ``instanciate_occ`` hooks (e.g. PolySurface's
                ``shapely.set_precision`` pass). Vertices closer than this
                get snapped before the TopoDS graph is built.
            n_threads: Thread count for ``BOPAlgo_Builder.SetRunParallel``.
            fuzzy_value: BOPAlgo / BRepAlgoAPI_Cut fuzzy used by the
                sequential cut cascade and the final all-fragment pass.
                Defaults to ``point_tolerance`` -- intentionally LOOSER than
                cad_gmsh's ``tolerance_boolean = perturbation / 2``. cad_occ
                tags interfaces via raw ``TShape`` identity on per-entity
                leaves; gmsh tags via ``getBoundary`` set intersection which
                relies on gmsh's fragment merging near-coincident TShapes
                aggressively. Tightening cad_occ's fuzzy below the
                perturbation gap (~2e-5) leaves coincident faces with
                distinct TShapes, dropping ``A___B`` interfaces. Until
                cad_occ's tagging moves to a geometric-coincidence test
                (or the fragment is delegated to gmsh) the loose fuzzy
                must stay.
            perturbation: Outward shapely buffer applied to polygon entities
                before the sequential cut cascade. Mirrors cad_gmsh default
                (1e-5). Used for the shared shapely pre-pass (polygon buffer
                + InterfaceTag snap distance).
        """
        self.point_tolerance = point_tolerance
        self.n_threads = n_threads
        self.fuzzy_value = point_tolerance if fuzzy_value is None else fuzzy_value
        # Mirrors cad_gmsh default (1e-5). Used for the shared shapely
        # pre-pass (polygon buffer + InterfaceTag snap distance).
        self.perturbation = perturbation if perturbation is not None else 1e-5

    def _get_shape_dimension(self, shape: TopoDS_Shape) -> int:
        """Infer dimension from the first non-empty TopAbs class."""
        for kind, dim in (
            (TopAbs_SOLID, 3),
            (TopAbs_FACE, 2),
            (TopAbs_EDGE, 1),
            (TopAbs_VERTEX, 0),
        ):
            if TopExp_Explorer(shape, kind).More():
                return dim
        return -1

    def _unwrap_shape(self, shape: TopoDS_Shape, dim: int) -> list[TopoDS_Shape]:
        """Unwrap a compound into its constituent dim-level shapes.

        ``BRepAlgoAPI_Cut`` always returns a ``TopoDS_Compound`` wrapper,
        even when the result is a single solid. BOPAlgo_Builder.Modified()
        tracks modifications at the primitive-shape level (Solid, Face, etc.)
        but not at the Compound level. Flattening the cut result into its
        constituent sub-shapes before adding them to the fragment builder
        ensures that Modified() can locate the fragment pieces originating
        from each cut entity.
        """
        from OCP.TopAbs import TopAbs_ShapeEnum

        if shape.ShapeType() != TopAbs_ShapeEnum.TopAbs_COMPOUND:
            return [shape]

        kind_map = {3: TopAbs_SOLID, 2: TopAbs_FACE, 1: TopAbs_EDGE, 0: TopAbs_VERTEX}
        kind = kind_map.get(dim)
        if kind is None:
            return [shape]

        out: list[TopoDS_Shape] = []
        exp = TopExp_Explorer(shape, kind)
        while exp.More():
            out.append(exp.Current())
            exp.Next()
        return out if out else [shape]

    def _shape_bbox(
        self, shape: TopoDS_Shape
    ) -> tuple[float, float, float, float, float, float] | None:
        """Return (xmin, ymin, zmin, xmax, ymax, zmax) bounding box of shape.

        Returns ``None`` for void / empty shapes.
        """
        box = Bnd_Box()
        BRepBndLib.Add_s(shape, box)
        if box.IsVoid():
            return None
        return box.Get()

    def _shapes_actually_overlap(self, s1: TopoDS_Shape, s2: TopoDS_Shape) -> bool:
        """Return True iff two shapes are within ``fuzzy_value`` of touching.

        AABB overlap is necessary but not sufficient for non-convex
        geometry such as annular sectors, L-shapes, or any body whose
        bounding box is much larger than its actual material region. For
        a 90-degree annular sector at r in [14, 17], the AABB covers the
        full upper-right quadrant of [0, 17]^2, which AABB-overlaps with
        any other body in that quadrant -- even ones that share no
        material with the sector.

        Calling ``BRepAlgoAPI_Cut`` on AABB-overlapping but
        volumetrically-disjoint shapes is unsafe: OCC has been observed
        to silently SPLIT the object (returning a Compound of multiple
        SOLIDs where the input was one) and to leave duplicate face
        TShapes in the result. Subsequent fragment + meshing then
        produces malformed solids whose volumes silently fail to
        tetrahedralize.

        ``BRepExtrema_DistShapeShape`` measures the true minimum
        distance between two shapes; if it exceeds ``fuzzy_value`` the
        shapes are definitively disjoint and the cut would be a no-op
        modulo numerical noise.
        """
        ext = BRepExtrema_DistShapeShape(s1, s2)
        ext.Perform()
        if not ext.IsDone():
            return True
        return ext.Value() <= self.fuzzy_value

    def _bboxes_overlap(
        self,
        b1: tuple[float, ...],
        b2: tuple[float, ...],
    ) -> bool:
        """Return True iff the two AABBs overlap or touch.

        Plain AABB intersection. Disjoint shapes (with separated AABBs)
        skip the cut; overlapping AND touching shapes both proceed to
        ``BRepAlgoAPI_Cut`` -- mirroring cad_gmsh's unconditional cut.

        Earlier versions used ``gap = self.fuzzy_value`` to skip pairs
        that "only touch" (shared face, zero-volume overlap). That
        decision was wrong on two counts: (1) ``fuzzy_value`` defaults
        to ``point_tolerance = 1e-3``, which is 50x larger than the
        ~2e-5 perturbation-induced volumetric overlap, so REAL
        overlapping pairs were classified as "touching" and the cut
        cascade silently became a no-op for those pairs (BOPAlgo_Builder
        had to resolve all overlaps in the final fragment, producing
        unclean interfaces); (2) cutting genuinely-touching pairs is
        actually desirable -- it splits the boundary face along the
        contact, which the subsequent fragment then merges into a
        shared TShape, giving cleanly uniquified interfaces.
        """
        return (
            b1[0] <= b2[3]
            and b1[3] >= b2[0]
            and b1[1] <= b2[4]
            and b1[4] >= b2[1]
            and b1[2] <= b2[5]
            and b1[5] >= b2[2]
        )

    def _instantiate_entity_occ(
        self,
        index: int,
        entity_obj: Any,
    ) -> OCCLabeledEntity:
        """Instantiate a single entity into an OCC shape."""
        shape = entity_obj.instanciate_occ()
        dim = getattr(entity_obj, "dimension", None)
        if dim is None:
            dim = self._get_shape_dimension(shape)
        physical_name = entity_obj.physical_name
        if isinstance(physical_name, str):
            physical_name = (physical_name,)
        return OCCLabeledEntity(
            shapes=[shape],
            physical_name=physical_name,
            index=index,
            keep=getattr(entity_obj, "mesh_bool", True),
            dim=dim,
            mesh_order=getattr(entity_obj, "mesh_order", None),
        )

    def _fragment_all(
        self,
        entities: list[OCCLabeledEntity],
        progress_bars: bool = False,
    ) -> list[OCCLabeledEntity]:
        """Fragment all entity shapes; assign pieces by mesh_order priority.

        Each entity's ``shapes`` list is replaced with the fragment
        pieces it owns. Matches ``gmsh.model.occ.fragment`` + mesh_order
        post-processing semantically.
        """
        if not entities:
            return []
        if len(entities) == 1:
            return entities

        builder = BOPAlgo_Builder()
        builder.SetRunParallel(self.n_threads > 1)
        builder.SetFuzzyValue(self.fuzzy_value)
        builder.SetNonDestructive(False)

        originals_per_entity: list[list[TopoDS_Shape]] = []
        for ent in tqdm(
            entities,
            desc="BOPAlgo add arguments",
            disable=not progress_bars,
            leave=False,
        ):
            originals_per_entity.append(list(ent.shapes))
            for s in ent.shapes:
                builder.AddArgument(s)

        if progress_bars:
            print(
                f"BOPAlgo_Builder.Perform() on {len(entities)} entities…",
                flush=True,
            )
        builder.Perform()

        piece_candidates: dict[tuple[int, int], list[tuple[int, float]]] = defaultdict(
            list
        )
        piece_shapes: dict[tuple[int, int], TopoDS_Shape] = {}

        for ent_idx, ent in enumerate(
            tqdm(
                entities,
                desc="Collecting fragment pieces",
                disable=not progress_bars,
                leave=False,
            )
        ):
            mo = ent.mesh_order
            if mo is None:
                mo = float("inf")
            for original in originals_per_entity[ent_idx]:
                modified = builder.Modified(original)
                if modified.IsEmpty() and not builder.IsDeleted(original):
                    pieces = [original]
                else:
                    pieces = list(modified)
                for piece in pieces:
                    k = _shape_key(piece)
                    piece_shapes.setdefault(k, piece)
                    piece_candidates[k].append((ent_idx, mo))

        owners = _resolve_piece_ownership(piece_candidates)

        for ent in entities:
            ent.shapes = []
        for key, ent_idx in owners.items():
            entities[ent_idx].shapes.append(piece_shapes[key])

        return entities

    def process_entities_instantiate_only(
        self,
        entities_list: list[Any],
    ) -> list[OCCLabeledEntity]:
        """Run prepare + sort + instantiate. No cut, no fragment.

        This is the cleanest hand-off point for the gmsh-cut+fragment
        pipeline: OCP just builds the per-entity TopoDS_Shape (which is
        the OCP perf advantage) and gmsh handles all subsequent BOP
        (cut cascade + fragment). Skips OCP's BRepAlgoAPI_Cut entirely,
        which has been observed to produce topology that gmsh's PLC
        mesher rejects (sliver faces around arc/rect cuts) -- gmsh's own
        ``gmsh.model.occ.cut`` doesn't have this problem.
        """
        if not entities_list:
            return []

        from meshwell.cad_common import prepare_entities

        prepare_entities(
            entities_list,
            perturbation=self.perturbation,
            resolve_snap=max(self.perturbation, self.point_tolerance),
        )

        indexed = list(enumerate(entities_list))
        indexed.sort(
            key=lambda pair: (
                pair[1].mesh_order if pair[1].mesh_order is not None else float("inf"),
                pair[0],
            )
        )

        instantiated: list[OCCLabeledEntity | None] = [None] * len(entities_list)
        for orig_idx, ent in indexed:
            instantiated[orig_idx] = self._instantiate_entity_occ(orig_idx, ent)
        return [le for le in instantiated if le is not None]

    def process_entities_cut_only(
        self,
        entities_list: list[Any],
        progress_bars: bool = False,
    ) -> list[OCCLabeledEntity]:
        """Run the OCP-side prepare + sort + instantiate + sequential-cut phase.

        Stops short of ``_fragment_all``: each returned entity holds its
        post-cut shapes but no piece-ownership reassignment has happened.
        This is the bridge point used by the gmsh-fragment hand-off, where
        gmsh re-fragments the cut shapes and runs its own tagging pipeline.
        """
        if not entities_list:
            return []

        from meshwell.cad_common import prepare_entities

        # ``resolve_snap`` controls the InterfaceTag snap distance. Mirror
        # cad_gmsh: pass ``max(perturbation, point_tolerance)`` so the
        # resolved strip is wide enough for non-degenerate panels (at
        # least 2*point_tolerance per side). Without this override OCC
        # defaults snap to ``perturbation`` (1e-5) and InterfaceTags can
        # produce zero-area panels at user scale.
        prepare_entities(
            entities_list,
            perturbation=self.perturbation,
            resolve_snap=max(self.perturbation, self.point_tolerance),
        )

        # Sort by mesh_order (lowest first); preserve insertion order on ties.
        indexed = list(enumerate(entities_list))
        indexed.sort(
            key=lambda pair: (
                pair[1].mesh_order if pair[1].mesh_order is not None else float("inf"),
                pair[0],
            )
        )

        instantiated: list[OCCLabeledEntity | None] = [None] * len(entities_list)
        for orig_idx, ent in tqdm(
            indexed,
            desc="Instantiating + cutting OCC entities",
            disable=not progress_bars,
            leave=False,
        ):
            labeled = self._instantiate_entity_occ(orig_idx, ent)

            # Build a compound of all previously-instantiated same-dim
            # tool shapes whose bounding boxes VOLUMETRICALLY overlap
            # with the current entity and cut once -- mirrors
            # gmsh.model.occ.cut(object, [all_tools]).
            # Bounding-box pre-filtering skips non-overlapping tools;
            # cutting against them would cause BRepAlgoAPI_Cut to split
            # the object's faces along the tool boundary even when no
            # material is removed (OCC always performs a full BOP).
            obj_bboxes = [
                b for s in labeled.shapes if (b := self._shape_bbox(s)) is not None
            ]
            tool_shapes: list[TopoDS_Shape] = []
            for prev in instantiated:
                if prev is None or prev.dim != labeled.dim:
                    continue
                for ts in prev.shapes:
                    tb = self._shape_bbox(ts)
                    if tb is None:
                        continue
                    if not any(self._bboxes_overlap(ob, tb) for ob in obj_bboxes):
                        continue
                    # AABB overlap is too coarse for non-convex shapes
                    # (annular sectors, L-shapes). Confirm a real
                    # geometric overlap before cutting -- BRepAlgoAPI_Cut
                    # on AABB-overlapping but volume-disjoint shapes can
                    # silently split the object into multiple SOLIDs and
                    # produce duplicate face TShapes that prevent
                    # downstream meshing.
                    if not any(
                        self._shapes_actually_overlap(s, ts) for s in labeled.shapes
                    ):
                        continue
                    tool_shapes.append(ts)

            if tool_shapes and labeled.shapes:
                # Sequential per-tool cuts. Bundling all tools into a single
                # ``TopoDS_Compound`` and calling ``BRepAlgoAPI_Cut`` once was
                # observed to produce empty results (zero SOLIDs) for large
                # background bodies like a substrate cut against a compound
                # of ~10 metal+helper bodies, even though the same body
                # against each tool individually retains 1 SOLID per cut.
                # Sequential cutting matches gmsh.model.occ.cut(obj, [tools])
                # which iterates internally.
                from OCP.TopTools import TopTools_ListOfShape

                # Single Cut() with all tools as a TopTools_ListOfShape --
                # matches gmsh.model.occ.cut(obj, [tools]) semantics.
                # Bundling all tools into a TopoDS_Compound and calling
                # ``BRepAlgoAPI_Cut(s, compound)`` (the original code) was
                # observed to produce empty results (zero SOLIDs) for
                # large bodies cut against ~10 small ones, even though
                # the same body against each tool individually works.
                new_shapes: list[TopoDS_Shape] = []
                for s in labeled.shapes:
                    try:
                        cut_op = BRepAlgoAPI_Cut()
                        args = TopTools_ListOfShape()
                        args.Append(s)
                        tools = TopTools_ListOfShape()
                        for ts in tool_shapes:
                            tools.Append(ts)
                        cut_op.SetArguments(args)
                        cut_op.SetTools(tools)
                        cut_op.SetFuzzyValue(self.fuzzy_value)
                        cut_op.Build()
                        result = cut_op.Shape()
                    except Exception as e:  # pragma: no cover -- defensive
                        print(
                            f"Warning: BRepAlgoAPI_Cut failed for entity "
                            f"{orig_idx}: {e}"
                        )
                        result = s
                    if result is not None:
                        # Flatten compound wrapper so BOPAlgo_Builder.Modified()
                        # in the final fragment pass tracks sub-shape provenance.
                        new_shapes.extend(self._unwrap_shape(result, labeled.dim))
                labeled.shapes = new_shapes

            instantiated[orig_idx] = labeled

        # ``instantiated`` now has one entry per original entity, in
        # insertion order. Filter out any None (defensive; should not
        # happen) before fragment.
        return [le for le in instantiated if le is not None]

    def process_entities(
        self,
        entities_list: list[Any],
        progress_bars: bool = False,
    ) -> list[OCCLabeledEntity]:
        """Instantiate, sequentially cut, then fragment all entities.

        Pipeline mirrors ``cad_gmsh.process_entities``:

        1. Shared shapely pre-pass (buffer + InterfaceTag resolve) via
           :func:`meshwell.cad_common.prepare_entities`.
        2. Sort by mesh_order (lowest first).
        3. For each entity in sorted order: instantiate via
           ``instanciate_occ``, then ``BRepAlgoAPI_Cut`` against a
           ``TopoDS_Compound`` of all previously-instantiated same-dim
           shapes. The lowest-mesh_order entity keeps its full buffered
           geometry; higher-mesh_order entities have the overlap carved.
        4. Final all-fragment pass via ``BOPAlgo_Builder`` (existing
           ownership resolution by mesh_order).

        ``keep=False`` helpers keep their shapes for the XAO writer's
        interface-naming pass; the writer itself excludes their bodies
        from the emitted BREP.
        """
        labeled_entities = self.process_entities_cut_only(
            entities_list, progress_bars=progress_bars
        )
        if not labeled_entities:
            return []
        return self._fragment_all(labeled_entities, progress_bars=progress_bars)


def cad_occ(
    entities_list: list[Any],
    point_tolerance: float = 1e-3,
    n_threads: int = cpu_count(),
    progress_bars: bool = False,
    fuzzy_value: float | None = None,
    perturbation: float | None = None,
) -> list[OCCLabeledEntity]:
    """Utility function for OCC-based CAD processing.

    Mirrors :func:`meshwell.cad_gmsh.cad_gmsh`'s signature (minus the
    gmsh-specific ``model`` / ``filename`` / tagging kwargs); the result
    feeds :func:`meshwell.occ_xao_writer.write_xao` to produce a tagged
    XAO gmsh can load.
    """
    processor = CAD_OCC(
        point_tolerance=point_tolerance,
        n_threads=n_threads,
        fuzzy_value=fuzzy_value,
        perturbation=perturbation,
    )
    return processor.process_entities(entities_list, progress_bars=progress_bars)


def cad_occ_with_gmsh_fragment(
    entities_list: list[Any],
    point_tolerance: float = 1e-3,
    n_threads: int = cpu_count(),
    progress_bars: bool = False,
    fuzzy_value: float | None = None,
    perturbation: float | None = None,
    filename: str = "temp",
    model: Any | None = None,
    interface_delimiter: str = "___",
    boundary_delimiter: str = "None",
) -> tuple[list[Any], Any]:
    """OCP instantiate + gmsh cut+fragment+tag, returning a meshable gmsh model.

    Splits the work between the two backends to combine OCP's faster
    instantiation with gmsh's robust BOP + boundary-tagging:

    1. **OCP phase** -- prepare + sort by mesh_order + instantiate every
       entity (no cut, no fragment). OCP's perf advantage is in
       instantiation; doing the cut in OCP produces topology that gmsh's
       PLC mesher rejects (sliver faces around arc/rect cuts produce
       silent 0-tetra meshed volumes), so cuts move to gmsh.
    2. **Bridge** -- serialize the per-entity TopoDS_Shape to a
       temporary XAO with synthetic ``__cad_occ_bridge_idx_<i>__`` group
       names and ``gmsh.merge`` into a fresh ``ModelManager``.
    3. **gmsh phase** -- map physical groups back to per-entity dimtags,
       then run the existing ``cad_gmsh`` cut cascade + fragment +
       ``getBoundary`` tagging + keep=False removal. Cuts use
       ``gmsh.model.occ.cut`` which produces clean, mesher-friendly
       topology where ``BRepAlgoAPI_Cut`` does not.

    Returns ``(labeled, model_manager)`` exactly like
    :func:`meshwell.cad_gmsh.cad_gmsh`, so downstream
    :func:`meshwell.mesh.mesh` calls are identical.
    """
    import tempfile

    from meshwell.cad_gmsh import CAD_GMSH, GMSHLabeledEntity
    from meshwell.occ_xao_writer import parse_bridge_group_index, write_xao

    # ----- OCP phase: instantiate only (no cut, no fragment) --------------
    ocp_processor = CAD_OCC(
        point_tolerance=point_tolerance,
        n_threads=n_threads,
        fuzzy_value=fuzzy_value,
        perturbation=perturbation,
    )
    ocp_labeled = ocp_processor.process_entities_instantiate_only(entities_list)
    if not ocp_labeled:
        gmsh_proc = CAD_GMSH(
            point_tolerance=point_tolerance,
            n_threads=n_threads,
            filename=filename,
            model=model,
            perturbation=perturbation,
        )
        gmsh_proc.model_manager.ensure_initialized(filename)
        return [], gmsh_proc.model_manager

    # ----- Bridge: write XAO of raw shapes, gmsh.merge --------------------
    bridge_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".xao", delete=False, mode="w") as tf:
            bridge_path = tf.name
        write_xao(ocp_labeled, bridge_path, bridge_mode=True)

        gmsh_proc = CAD_GMSH(
            point_tolerance=point_tolerance,
            n_threads=n_threads,
            filename=filename,
            model=model,
            perturbation=perturbation,
        )
        gmsh_proc.model_manager.ensure_initialized(filename)
        gmsh.merge(bridge_path)
        gmsh_proc.model_manager.sync_model()
        # OCP-built shapes imported via XAO arrive as topologically-
        # independent solids whose touching faces have distinct TShapes.
        # gmsh.model.occ.cut + fragment need a sufficiently wide BOP
        # tolerance to detect geometric coincidence and merge them; the
        # default ``perturbation/2`` (5e-6) is too tight when OCP-side
        # numerics drift the imported faces by ~1e-7. ``point_tolerance``
        # (1e-3) is what cad_gmsh uses successfully on the same scenes.
        gmsh.option.setNumber("Geometry.ToleranceBoolean", point_tolerance)
    finally:
        with contextlib.suppress(OSError):
            if bridge_path:
                Path(bridge_path).unlink()

    # ----- Map gmsh dimtags back to original entities ---------------------
    index_to_dimtags: dict[int, list[tuple[int, int]]] = {}
    for dim, tag in gmsh.model.getPhysicalGroups():
        name = gmsh.model.getPhysicalName(dim, tag)
        idx = parse_bridge_group_index(name)
        if idx is None:
            continue
        ent_dimtags = [
            (dim, t) for t in gmsh.model.getEntitiesForPhysicalGroup(dim, tag)
        ]
        index_to_dimtags.setdefault(idx, []).extend(ent_dimtags)
        gmsh.model.removePhysicalGroups([(dim, tag)])
    gmsh_proc.model_manager.sync_model()

    # Build GMSHLabeledEntity list with __#index suffix, in original order.
    gmsh_entities: list[GMSHLabeledEntity] = []
    for orig_idx, ent in enumerate(entities_list):
        dimtags = index_to_dimtags.get(orig_idx, [])
        physical_name = ent.physical_name
        if isinstance(physical_name, str):
            physical_name = (physical_name,)
        suffixed = tuple(f"{n}__#{orig_idx}" for n in physical_name)
        dim = getattr(ent, "dimension", None)
        if dim is None:
            dim = dimtags[0][0] if dimtags else -1
        gmsh_entities.append(
            GMSHLabeledEntity(
                dimtags=dimtags,
                physical_name=suffixed,
                index=orig_idx,
                keep=getattr(ent, "mesh_bool", True),
                dim=dim,
                mesh_order=getattr(ent, "mesh_order", None),
            )
        )

    # ----- gmsh phase: cut cascade + fragment + tag + cleanup -------------
    # Replicate cad_gmsh.process_entities's sequential cut: sort by
    # mesh_order, then cut each entity's dimtags against all previously-
    # processed same-dim tools via gmsh.model.occ.cut. This is the part
    # that produces clean topology gmsh's mesher accepts (BRepAlgoAPI_Cut
    # in OCP does not).
    indexed_gmsh = sorted(
        enumerate(gmsh_entities),
        key=lambda p: (
            p[1].mesh_order if p[1].mesh_order is not None else float("inf"),
            p[0],
        ),
    )
    processed: list[GMSHLabeledEntity] = []
    for _, ent in indexed_gmsh:
        if ent.dimtags:
            tool_dimtags: list[tuple[int, int]] = []
            for prev in processed:
                if prev.dim == ent.dim and prev.dimtags:
                    tool_dimtags.extend(prev.dimtags)
            if tool_dimtags:
                try:
                    out_dimtags, _ = gmsh.model.occ.cut(
                        ent.dimtags,
                        tool_dimtags,
                        removeObject=True,
                        removeTool=False,
                    )
                    gmsh_proc.model_manager.sync_model()
                    ent.dimtags = out_dimtags
                except Exception as e:
                    if progress_bars:
                        print(f"Warning: gmsh cut failed for entity {ent.index}: {e}")
        processed.append(ent)

    labeled = gmsh_proc._fragment_all(gmsh_entities, progress_bars=progress_bars)
    gmsh_proc._tag_entities(labeled, interface_delimiter, boundary_delimiter)
    gmsh_proc._remove_keep_false_top_dim(labeled)
    gmsh_proc.model_manager.sync_model()

    return labeled, gmsh_proc.model_manager
