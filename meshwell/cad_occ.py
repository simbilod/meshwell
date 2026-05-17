"""OCC CAD processor: fragment + mesh_order ownership via OCP.

Mirrors :mod:`meshwell.cad_gmsh` but drives OCCT directly through OCP
instead of the gmsh API. The two backends are intentionally kept
structurally identical so users can compare outputs head-to-head:

1. Shared shapely pre-pass (perturbation buffer + InterfaceTag resolve)
   via :func:`meshwell.cad_common.prepare_entities`.
2. Call each entity's ``instanciate_occ`` to produce ``TopoDS_Shape``
   instances.
3. Sequential per-entity ``BRepAlgoAPI_Cut`` cascade against
   previously-instantiated same-dim tools (lowest ``mesh_order`` wins).
4. ``BOPAlgo_Builder`` all-fragment over every cut shape; per-input
   ``Modified()`` tells us which fragment piece descends from which
   input, which we invert to build the per-piece candidate list.
5. Resolve multi-claim pieces by lowest ``mesh_order`` (first-inserted
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

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from os import cpu_count
from typing import TYPE_CHECKING, Any

from OCP.Bnd import Bnd_Box
from OCP.BOPAlgo import BOPAlgo_Builder
from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepExtrema import BRepExtrema_DistShapeShape
from OCP.TopAbs import (
    TopAbs_EDGE,
    TopAbs_FACE,
    TopAbs_ShapeEnum,
    TopAbs_SOLID,
    TopAbs_VERTEX,
)
from OCP.TopExp import TopExp_Explorer
from OCP.TopTools import TopTools_ShapeMapHasher
from tqdm.auto import tqdm

from meshwell.cad_common import prepare_entities

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
        cut_fuzzy_value: float | None = None,
        fragment_fuzzy_value: float | None = None,
        perturbation: float | None = None,
    ):
        """Initialize OCC CAD processor.

        Args:
            point_tolerance: Coordinate quantization applied inside entity
                ``instanciate_occ`` hooks (e.g. PolySurface's
                ``shapely.set_precision`` pass). Vertices closer than this
                get snapped before the TopoDS graph is built.
            n_threads: Thread count for ``BOPAlgo_Builder.SetRunParallel``.
            cut_fuzzy_value: Fuzzy passed to ``BRepAlgoAPI_Cut`` in the
                sequential per-entity cut cascade. Defaults to
                ``perturbation / 2`` (mirrors cad_gmsh's
                ``tolerance_boolean = perturbation / 2``). Tight by design --
                a loose cut fuzzy merges the buffered overlap into the lower
                entity and erases the carved face.
            fragment_fuzzy_value: Fuzzy passed to the final ``BOPAlgo_Builder``
                all-fragment pass. Defaults to ``point_tolerance``,
                intentionally LOOSER than the cut fuzzy: cad_occ tags
                interfaces via raw ``TShape`` identity on per-entity leaves;
                tightening this below the perturbation gap (~2e-5) leaves
                coincident faces with distinct TShapes and drops ``A___B``
                interfaces.
            perturbation: Outward shapely buffer applied to polygon entities
                before the sequential cut cascade. Mirrors cad_gmsh default
                (1e-5). Used for the shared shapely pre-pass (polygon buffer
                + InterfaceTag snap distance).
        """
        self.point_tolerance = point_tolerance
        self.n_threads = n_threads
        self.perturbation = perturbation if perturbation is not None else 1e-5
        self.cut_fuzzy_value = (
            self.perturbation / 2 if cut_fuzzy_value is None else cut_fuzzy_value
        )
        self.fragment_fuzzy_value = (
            point_tolerance if fragment_fuzzy_value is None else fragment_fuzzy_value
        )

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
        """Return True iff two shapes are within ``cut_fuzzy_value`` of touching.

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
        distance between two shapes; if it exceeds ``cut_fuzzy_value``
        the shapes are definitively disjoint and the cut would be a
        no-op modulo numerical noise. Matches the fuzzy the cut itself
        will use, so the gate and the BOP agree on "near enough".
        """
        ext = BRepExtrema_DistShapeShape(s1, s2)
        ext.Perform()
        if not ext.IsDone():
            return True
        return ext.Value() <= self.cut_fuzzy_value

    def _bboxes_overlap(
        self,
        b1: tuple[float, ...],
        b2: tuple[float, ...],
    ) -> bool:
        """Return True iff the two AABBs overlap or touch.

        Plain AABB intersection. Disjoint shapes (with separated AABBs)
        skip the cut; overlapping AND touching shapes both proceed to
        ``BRepAlgoAPI_Cut`` -- mirroring cad_gmsh's unconditional cut.

        Earlier versions used a fuzzy ``gap`` to skip pairs that "only
        touch" (shared face, zero-volume overlap). That decision was
        wrong on two counts: (1) the gap defaulted to
        ``point_tolerance = 1e-3``, which is 50x larger than the
        ~2e-5 perturbation-induced volumetric overlap, so REAL
        overlapping pairs were classified as "touching" and the cut
        cascade silently became a no-op for those pairs (the final
        fragment had to resolve all overlaps, producing unclean
        interfaces); (2) cutting genuinely-touching pairs is actually
        desirable -- it splits the boundary face along the contact,
        which the subsequent fragment then merges into a shared TShape,
        giving cleanly uniquified interfaces.
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
        extra_occ_shapes: list[Any] | None = None,
        cad_occ_callback: Callable[[Any], None] | None = None,
    ) -> list[OCCLabeledEntity]:
        """Fragment all entity shapes; assign pieces by mesh_order priority.

        Each entity's ``shapes`` list is replaced with the fragment
        pieces it owns. Matches ``gmsh.model.occ.fragment`` + mesh_order
        post-processing semantically.
        """
        if not entities:
            return []
        if len(entities) == 1 and not extra_occ_shapes and cad_occ_callback is None:
            return entities

        builder = BOPAlgo_Builder()
        builder.SetRunParallel(self.n_threads > 1)
        builder.SetFuzzyValue(self.fragment_fuzzy_value)
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

        for s in extra_occ_shapes or []:
            builder.AddArgument(s)

        if progress_bars:
            print(
                f"BOPAlgo_Builder.Perform() on {len(entities)} entities…",
                flush=True,
            )
        builder.Perform()

        if cad_occ_callback is not None:
            cad_occ_callback(builder)

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
            l_ord = (
                labeled.mesh_order if labeled.mesh_order is not None else float("inf")
            )
            for prev in instantiated:
                if prev is None or prev.dim != labeled.dim:
                    continue
                p_ord = prev.mesh_order if prev.mesh_order is not None else float("inf")
                if p_ord >= l_ord:
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
                # Sequential per-tool cuts -- matches
                # ``gmsh.model.occ.cut(obj, [tools])`` which iterates
                # internally. Bundling all tools into a single
                # ``TopoDS_Compound`` and calling ``BRepAlgoAPI_Cut(s,
                # compound)`` (or feeding a ``TopTools_ListOfShape``) was
                # observed to produce empty results (zero SOLIDs) for
                # large bodies like a substrate cut against ~10
                # metal+helper bodies, even though the same body against
                # each tool individually retains 1 SOLID per cut.
                new_shapes: list[TopoDS_Shape] = []
                for s in labeled.shapes:
                    try:
                        result = s
                        for ts in tool_shapes:
                            cut_op = BRepAlgoAPI_Cut(result, ts)
                            cut_op.SetFuzzyValue(self.cut_fuzzy_value)
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
        extra_occ_shapes: list[Any] | None = None,
        cad_occ_callback: Callable[[Any], None] | None = None,
    ) -> list[OCCLabeledEntity]:
        """Instantiate, sequentially cut, then fragment all entities.

        Pipeline mirrors ``cad_gmsh.process_entities``:

        1. Shared shapely pre-pass (buffer + InterfaceTag resolve) via
           :func:`meshwell.cad_common.prepare_entities`.
        2. Sort by mesh_order (lowest first).
        3. For each entity in sorted order: instantiate via
           ``instanciate_occ``, then sequentially ``BRepAlgoAPI_Cut``
           against each previously-instantiated same-dim tool that
           passes both the AABB and the geometric-overlap pre-filter.
           The lowest-mesh_order entity keeps its full buffered geometry;
           higher-mesh_order entities have the overlap carved.
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
        return self._fragment_all(
            labeled_entities,
            progress_bars=progress_bars,
            extra_occ_shapes=extra_occ_shapes,
            cad_occ_callback=cad_occ_callback,
        )


def cad_occ(
    entities_list: list[Any],
    point_tolerance: float = 1e-3,
    n_threads: int = cpu_count(),
    progress_bars: bool = False,
    cut_fuzzy_value: float | None = None,
    fragment_fuzzy_value: float | None = None,
    perturbation: float | None = None,
    extra_occ_shapes: list[Any] | None = None,
    cad_occ_callback: Callable[[Any], None] | None = None,
) -> list[OCCLabeledEntity]:
    """Utility function for OCC-based CAD processing.

    Mirrors :func:`meshwell.cad_gmsh.cad_gmsh`'s signature (minus the
    gmsh-specific ``model`` / ``filename`` / tagging kwargs); the result
    feeds :func:`meshwell.occ_xao_writer.write_xao` to produce a tagged
    XAO gmsh can load.

    Args:
        entities_list: Ordered list of mesh entities to process.
        point_tolerance: Coordinate quantization for entity instantiation.
        n_threads: Thread count for ``BOPAlgo_Builder.SetRunParallel``.
        progress_bars: Whether to show tqdm progress bars.
        cut_fuzzy_value: Fuzzy tolerance for ``BRepAlgoAPI_Cut``.
        fragment_fuzzy_value: Fuzzy tolerance for the final ``BOPAlgo_Builder``.
        perturbation: Outward shapely buffer applied before the cut cascade.
        extra_occ_shapes: Optional list of additional OCP shapes added as
            ``BOPAlgo_Builder`` arguments alongside entity shapes.  When
            *None* (default) the behaviour is identical to before.
        cad_occ_callback: Optional callable invoked with the
            ``BOPAlgo_Builder`` instance immediately after ``Perform()``
            and before history extraction.  Lets callers walk
            ``Modified()`` / ``Generated()`` for phantom-shape tracking.
            When *None* (default) no callback is made.
    """
    processor = CAD_OCC(
        point_tolerance=point_tolerance,
        n_threads=n_threads,
        cut_fuzzy_value=cut_fuzzy_value,
        fragment_fuzzy_value=fragment_fuzzy_value,
        perturbation=perturbation,
    )
    return processor.process_entities(
        entities_list,
        progress_bars=progress_bars,
        extra_occ_shapes=extra_occ_shapes,
        cad_occ_callback=cad_occ_callback,
    )
