"""OCC CAD processor: parallel fan-out + same-name fuse + global fragment.

Mirrors :mod:`meshwell.cad_gmsh` but drives OCCT directly through OCP.
Default pipeline is parallel (``executor="auto"``); the legacy sequential
cascade is reachable via ``executor="legacy"``.

Default pipeline (``executor`` in ``auto``/``serial``/``thread``/``process``):

1. Shapely pre-pass via :func:`meshwell.cad_common.prepare_entities`.
2. Stage 0: :func:`compute_cutters` returns per-entity predecessor lists,
   deterministic in ``(mesh_order, insertion_idx)``.
3. Stage 1: per-entity prefused cut via :func:`cut_one_entity` dispatched
   through the chosen executor. ``thread`` is preferred when the GIL-
   release probe passes; ``process`` is the fallback. Each entity is cut
   against the **originals** of its predecessors -- set algebra confirms
   this is equivalent to the legacy cascade.
4. Stage 2: :func:`_same_name_fuse` collapses entities sharing
   ``(physical_name, keep, dim)`` so the mesh does not carry an internal
   face between two bodies the user already declared to be the same
   material.
5. Stage 3: :func:`CAD_OCC._fragment_all` welds coincident TShapes and
   resolves piece ownership by ``(mesh_order, ent_idx)``.

Legacy pipeline (``executor="legacy"``):

The pre-refactor cascade -- each entity sequentially cut against every
previously-instantiated tool. Kept as an escape hatch; default callers
should use the parallel path.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from os import cpu_count
from typing import TYPE_CHECKING, Any

from OCP.Bnd import Bnd_Box
from OCP.BOPAlgo import BOPAlgo_BOP, BOPAlgo_Builder, BOPAlgo_Operation
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

    from meshwell.tolerances import Tolerances


@dataclass
class OCCLabeledEntity:
    """Per-entity record produced by :func:`cad_occ`.

    ``shapes`` holds the fragment pieces this entity owns after the
    all-fragment pass.

    ``overlap_footprint`` / ``overlap_zrange`` / ``overlap_exact`` are
    populated when the source entity exposes
    :meth:`meshwell.geometry_entity.GeometryEntity.overlap_metadata`.
    Consumed by :meth:`CAD_OCC._polyprism_fast_overlap` to bypass the
    OCC distance check for polyprism-vs-polyprism pairs. See spec
    ``docs/superpowers/specs/2026-05-19-cad-occ-polyprism-overlap-fastpath-design.md``.
    """

    shapes: list[TopoDS_Shape]
    physical_name: tuple[str, ...]
    index: int
    keep: bool
    dim: int
    mesh_order: float | None = None
    overlap_footprint: Any | None = None
    overlap_zrange: tuple[float, float] | None = None
    overlap_exact: bool = False
    structured: bool = False


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

    Rule: lowest ``(mesh_order, ent_idx)`` wins. Sorting by both makes the
    tie-break independent of ``BOPAlgo_Builder.Modified()`` iteration
    order, which is non-deterministic under ``SetRunParallel(True)``.
    """
    owners: dict[Any, int] = {}
    for piece, candidates in piece_candidates.items():
        best_idx, _ = min(candidates, key=lambda c: (c[1], c[0]))
        owners[piece] = best_idx
    return owners


def cut_one_entity(
    entity_shape: "TopoDS_Shape",
    tool_shapes: list["TopoDS_Shape"],
    cut_fuzzy_value: float,
    n_threads: int = 1,
) -> "TopoDS_Shape":
    """Cut ``entity_shape`` by the prefused union of ``tool_shapes``.

    Pure function: no instance state. Used by both the serial cascade and
    the parallel per-entity dispatch. Mirrors the prefused-cut block
    inside ``process_entities_cut_only`` but takes a plain shape list,
    not the surrounding ``OCCLabeledEntity``.

    When ``tool_shapes`` is empty, returns ``entity_shape`` unchanged.
    """
    if not tool_shapes:
        return entity_shape
    if len(tool_shapes) == 1:
        fused_tools = tool_shapes[0]
    else:
        fuse_op = BOPAlgo_BOP()
        fuse_op.SetOperation(BOPAlgo_Operation.BOPAlgo_FUSE)
        fuse_op.AddArgument(tool_shapes[0])
        for ts in tool_shapes[1:]:
            fuse_op.AddTool(ts)
        fuse_op.SetFuzzyValue(cut_fuzzy_value)
        fuse_op.SetRunParallel(n_threads > 1)
        fuse_op.Perform()
        fused_tools = fuse_op.Shape()
    cut_op = BRepAlgoAPI_Cut(entity_shape, fused_tools)
    cut_op.SetFuzzyValue(cut_fuzzy_value)
    cut_op.SetRunParallel(n_threads > 1)
    cut_op.Build()
    return cut_op.Shape()


def compute_cutters(
    entities_list: list[Any],
    processor: "CAD_OCC",
) -> dict[int, list[int]]:
    """For each entity, return the indices of lower-precedence overlapping entities.

    Precedence is ``(mesh_order, insertion_index)`` ascending -- same as the
    sequential cascade. Overlap is determined by the same gates the cascade
    uses (``_polyprism_fast_overlap`` fast-path when both sides have exact
    metadata, then AABB intersection, then ``_shapes_actually_overlap``
    distance check).

    Inputs are entity objects, NOT instantiated OCC shapes. To populate
    overlap metadata and check AABBs we instantiate each entity once here
    (single-threaded); the parallel pipeline re-instantiates inside each
    worker because TopoDS_Shape hand-offs across the process boundary
    require BREP round-tripping.

    Returns ``{ent_idx: [cutter_idx, ...]}`` for every ``ent_idx`` in
    ``range(len(entities_list))``. Cutter lists are sorted by
    ``(mesh_order, idx)``.
    """
    n = len(entities_list)
    if n == 0:
        return {}

    labeled: list[OCCLabeledEntity] = [
        processor._instantiate_entity_occ(i, ent) for i, ent in enumerate(entities_list)
    ]
    bboxes = [
        [b for s in le.shapes if (b := processor._shape_bbox(s)) is not None]
        for le in labeled
    ]
    cutters: dict[int, list[int]] = {i: [] for i in range(n)}

    def precedes(j: int, i: int) -> bool:
        a = labeled[j].mesh_order if labeled[j].mesh_order is not None else float("inf")
        b = labeled[i].mesh_order if labeled[i].mesh_order is not None else float("inf")
        return (a, j) < (b, i)

    for i in range(n):
        for j in range(n):
            if i == j or not precedes(j, i):
                continue
            if labeled[j].dim != labeled[i].dim:
                continue
            fast = processor._polyprism_fast_overlap(labeled[i], labeled[j])
            if fast is False:
                continue
            if fast is None:
                if not any(
                    processor._bboxes_overlap(bi, bj)
                    for bi in bboxes[i]
                    for bj in bboxes[j]
                ):
                    continue
                if not any(
                    processor._shapes_actually_overlap(s, t)
                    for s in labeled[i].shapes
                    for t in labeled[j].shapes
                ):
                    continue
            cutters[i].append(j)

    for i in cutters:
        cutters[i].sort(
            key=lambda j: (
                labeled[j].mesh_order
                if labeled[j].mesh_order is not None
                else float("inf"),
                j,
            )
        )
    return cutters


def _collapse_members(
    members: list[OCCLabeledEntity],
    shapes: list["TopoDS_Shape"] | None = None,
) -> OCCLabeledEntity:
    """Build one OCCLabeledEntity from a fused group of same-named members."""
    if shapes is None:
        shapes = [s for m in members for s in m.shapes]
    mesh_orders = [m.mesh_order for m in members if m.mesh_order is not None]
    return OCCLabeledEntity(
        shapes=shapes,
        physical_name=members[0].physical_name,
        index=min(m.index for m in members),
        keep=members[0].keep,
        dim=members[0].dim,
        mesh_order=min(mesh_orders) if mesh_orders else None,
    )


def _same_name_fuse(
    entities: list[OCCLabeledEntity],
    tolerances: "Tolerances",
    n_threads: int = 1,
) -> list[OCCLabeledEntity]:
    """Fuse same-(physical_name, keep, dim) entities into one logical entity.

    Removes the spurious internal seam between two bodies the user already
    declared to be the same material -- would otherwise survive into the
    mesh as a zero-thickness internal face.

    Groups with one member pass through unchanged. Groups with multiple
    members whose total shape count is >1 run through ``BOPAlgo_BOP(Fuse)``
    with ``fragment_fuzzy_value`` -- the same value the downstream global
    fragment uses, so coincident faces actually fuse.
    """
    groups: dict[tuple[tuple[str, ...], bool, int], list[int]] = defaultdict(list)
    for idx, ent in enumerate(entities):
        groups[(ent.physical_name, ent.keep, ent.dim)].append(idx)

    fused: list[OCCLabeledEntity] = []
    for idxs in groups.values():
        members = [entities[i] for i in idxs]
        if len(members) == 1:
            fused.append(members[0])
            continue
        # Skip the fuse if any member is structured: structured entities
        # carry per-entity mesh-grid metadata that the structured pipeline
        # consumes downstream; merging them into one logical entity would
        # erase that mapping.
        if any(m.structured for m in members):
            fused.extend(members)
            continue
        all_shapes = [s for m in members for s in m.shapes]
        if len(all_shapes) <= 1:
            fused.append(_collapse_members(members))
            continue
        fuse_op = BOPAlgo_BOP()
        fuse_op.SetOperation(BOPAlgo_Operation.BOPAlgo_FUSE)
        fuse_op.AddArgument(all_shapes[0])
        for s in all_shapes[1:]:
            fuse_op.AddTool(s)
        fuse_op.SetFuzzyValue(tolerances.fragment_fuzzy_value)
        fuse_op.SetRunParallel(n_threads > 1)
        fuse_op.Perform()
        result = fuse_op.Shape()
        # Unwrap if result is a compound (multiple disjoint solids).
        out_shapes: list = []
        if result.ShapeType() == TopAbs_ShapeEnum.TopAbs_COMPOUND:
            kind_map = {
                3: TopAbs_SOLID,
                2: TopAbs_FACE,
                1: TopAbs_EDGE,
                0: TopAbs_VERTEX,
            }
            kind = kind_map.get(members[0].dim)
            if kind is not None:
                exp = TopExp_Explorer(result, kind)
                while exp.More():
                    out_shapes.append(exp.Current())
                    exp.Next()
        if not out_shapes:
            out_shapes = [result]
        fused.append(_collapse_members(members, shapes=out_shapes))
    return fused


_GIL_PROBE_CACHE: bool | None = None


def _probe_gil_release(speedup_threshold: float = 1.5) -> bool:
    """Return True iff OCP's BRepAlgoAPI_Cut releases the GIL under threads.

    Runs a small two-thread cut benchmark and measures wall-clock vs
    serial. GIL-released BOPs give >threshold speedup; GIL-held BOPs
    give <1.1x. Cached after first call (process-lifetime).
    """
    global _GIL_PROBE_CACHE
    if _GIL_PROBE_CACHE is not None:
        return _GIL_PROBE_CACHE

    from concurrent.futures import ThreadPoolExecutor
    from time import perf_counter

    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeSphere

    def _work():
        # Cut a box by an overlapping sphere; non-trivial enough to amortize
        # the BOP setup cost.
        box = BRepPrimAPI_MakeBox(10.0, 10.0, 10.0).Shape()
        sphere = BRepPrimAPI_MakeSphere(7.0).Shape()
        op = BRepAlgoAPI_Cut(box, sphere)
        op.SetRunParallel(False)
        op.Build()
        return op.Shape()

    # Warmup so first-call costs (allocator init etc.) don't skew the measure.
    _work()

    t0 = perf_counter()
    _work()
    _work()
    t_serial = perf_counter() - t0

    with ThreadPoolExecutor(max_workers=2) as ex:
        t0 = perf_counter()
        list(ex.map(lambda _: _work(), range(2)))
        t_parallel = perf_counter() - t0

    speedup = t_serial / max(t_parallel, 1e-6)
    _GIL_PROBE_CACHE = speedup >= speedup_threshold
    return _GIL_PROBE_CACHE


def _process_worker(entity, cutters_entities, cut_fuzzy_value) -> bytes:
    """Worker for the process-pool executor.

    Instantiates the entity and its cutters inside the worker process,
    runs ``cut_one_entity``, and returns BREP bytes. On cut failure,
    falls back to the uncut entity shape (same semantics as the legacy
    sequential cascade).
    """
    from meshwell._brep_io import brep_to_bytes

    shape = entity.instanciate_occ()
    tools = [c.instanciate_occ() for c in cutters_entities]
    try:
        result = cut_one_entity(shape, tools, cut_fuzzy_value, n_threads=1)
    except Exception as e:  # pragma: no cover -- defensive
        print(
            f"Warning: BRepAlgoAPI_Cut failed in worker: {e}",
            flush=True,
        )
        result = shape
    return brep_to_bytes(result)


def _assert_picklable(entities_list: list[Any]) -> None:
    """Raise ValueError if any entity isn't picklable.

    Used as a pre-flight before ProcessPoolExecutor dispatch so the
    error names the first offending entity, instead of surfacing as a
    generic worker exception deep in the pool.
    """
    import pickle

    try:
        pickle.dumps(entities_list)
        return
    except Exception:  # noqa: S110 -- fall through to per-entity identification
        pass
    for i, ent in enumerate(entities_list):
        try:
            pickle.dumps(ent)
        except Exception as e:
            raise ValueError(
                f"entity index {i} ({type(ent).__name__}, "
                f"physical_name={getattr(ent, 'physical_name', '<unknown>')!r}) "
                f"is not picklable: {e}"
            ) from e
    raise ValueError("entities_list is not picklable for process executor")


class CAD_OCC:
    """OCP-driven CAD processor: fragment + mesh_order ownership."""

    def __init__(
        self,
        point_tolerance: float = 1e-3,
        n_threads: int = cpu_count(),
        cut_fuzzy_value: float | None = None,
        fragment_fuzzy_value: float | None = None,
        perturbation: float | None = None,
        tolerances: "Tolerances | None" = None,
    ):
        """Initialize OCC CAD processor.

        Prefer ``tolerances=Tolerances.from_characteristic_length(L)`` over
        the legacy scalar args. The legacy args are accepted for back-compat
        but synthesize a clamped ``Tolerances`` internally. See
        :mod:`meshwell.tolerances` and
        ``docs/superpowers/plans/2026-05-19-tolerance-chain-redesign.md``.

        Args:
            point_tolerance: Coordinate quantization applied inside entity
                ``instanciate_occ`` hooks (e.g. PolySurface's
                ``shapely.set_precision`` pass). Vertices closer than this
                get snapped before the TopoDS graph is built.
            n_threads: Thread count for ``BOPAlgo_Builder.SetRunParallel``.
            cut_fuzzy_value: Fuzzy passed to ``BRepAlgoAPI_Cut`` in the
                batched compound cut per entity. Defaults to
                ``perturbation / 2`` (mirrors cad_gmsh's
                ``tolerance_boolean = perturbation / 2``). Tight by design --
                a loose cut fuzzy merges the buffered overlap into the lower
                entity and erases the carved face.
                Ignored when ``tolerances`` is provided.
            fragment_fuzzy_value: Fuzzy passed to the final ``BOPAlgo_Builder``
                all-fragment pass. Defaults to ``point_tolerance`` when
                synthesized from legacy scalars (preserves the historical
                loose-welding behaviour that merges coincident TShape faces
                produced by separate cuts). New callers should use
                ``tolerances=Tolerances.from_characteristic_length(L)`` to
                get the audited 2x-perturbation default instead. Ignored
                when ``tolerances`` is provided.
            perturbation: Outward shapely buffer applied to polygon entities
                before the sequential cut cascade. Mirrors cad_gmsh default
                (1e-5). Used for the shared shapely pre-pass (polygon buffer
                + InterfaceTag snap distance). Ignored when ``tolerances``
                is provided.
            tolerances: Pre-built :class:`meshwell.tolerances.Tolerances`
                instance. When provided, ``point_tolerance``,
                ``cut_fuzzy_value``, ``fragment_fuzzy_value``, and
                ``perturbation`` are silently ignored and all values are
                taken from this object. When absent, a ``Tolerances`` is
                synthesized from the legacy scalar kwargs with clamping to
                satisfy the hierarchy invariants.
        """
        from meshwell.tolerances import Tolerances

        if tolerances is None:
            pert = perturbation if perturbation is not None else 1e-5
            cut_f = cut_fuzzy_value if cut_fuzzy_value is not None else pert / 2
            # Legacy default: fragment_fuzzy = point_tolerance. This is the
            # historical behaviour; preserves welding of coincident faces
            # whose TShape positions drift by ~perturbation during the
            # cut cascade. New callers should use ``tolerances=`` with
            # ``Tolerances.from_characteristic_length(L)`` for the audited
            # 2x-perturbation default.
            frag_f = (
                fragment_fuzzy_value
                if fragment_fuzzy_value is not None
                else point_tolerance
            )
            tolerances = Tolerances(
                point_tolerance=point_tolerance,
                perturbation=pert,
                cut_fuzzy_value=cut_f,
                fragment_fuzzy_value=frag_f,
                geometry_tolerance=point_tolerance,
                tolerance_boolean=frag_f,
                arc_chord_height_fraction=0.01,
            )

        self.tolerances = tolerances
        self.point_tolerance = tolerances.point_tolerance
        self.n_threads = n_threads
        self.perturbation = tolerances.perturbation
        self.cut_fuzzy_value = tolerances.cut_fuzzy_value
        self.fragment_fuzzy_value = tolerances.fragment_fuzzy_value

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

    @staticmethod
    def _clamp_shape_tolerance(shape: TopoDS_Shape, max_tol: float) -> None:
        """Clamp every sub-shape's tolerance to ``max_tol`` in-place.

        OCC BOPs grow vertex/edge tolerances during intersection. Across
        a cut cascade this drift makes ``cut_fuzzy_value`` a lower bound
        only; the *effective* fuzzy can become much larger silently.
        Calling ``ShapeFix_ShapeTolerance.LimitTolerance`` after each cut
        keeps the configured fuzzy honest.

        ``LimitTolerance(shape, tmin, tmax, style)`` with tmin=0 and a
        bounded tmax clamps any tolerance above ``tmax`` down to ``tmax``.
        """
        from OCP.ShapeFix import ShapeFix_ShapeTolerance
        from OCP.TopAbs import TopAbs_SHAPE

        fixer = ShapeFix_ShapeTolerance()
        fixer.LimitTolerance(shape, 0.0, max_tol, TopAbs_SHAPE)

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
        if shape.IsNull():
            return []
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

    def _polyprism_fast_overlap(
        self,
        a: OCCLabeledEntity,
        b: OCCLabeledEntity,
    ) -> bool | None:
        """Cheap shapely + z-interval overlap test for polyprism-vs-polyprism.

        Returns:
            ``False``: the entities are definitively disjoint (skip cut).
            ``True``: definitively overlapping. Only returned when BOTH
                ``overlap_exact`` flags are True -- the shapely + interval
                test is then mathematically equivalent to the OCC distance
                check, so the OCC call can be skipped.
            ``None``: cannot decide. Either side lacks metadata, or both
                sides are present but at least one is a conservative
                tapered envelope (``overlap_exact=False``) and the cheap
                test passed; the caller must fall through to
                :meth:`_shapes_actually_overlap` for confirmation.

        Spec: ``docs/superpowers/specs/2026-05-19-cad-occ-polyprism-overlap-fastpath-design.md``.
        """
        if a.overlap_footprint is None or b.overlap_footprint is None:
            return None
        az, bz = a.overlap_zrange, b.overlap_zrange
        # Signed gap between z-intervals (negative = overlap, positive = gap).
        # Clamp to 0; we only care whether the gap exceeds the fuzzy.
        z_gap = max(0.0, max(az[0], bz[0]) - min(az[1], bz[1]))
        if z_gap > self.cut_fuzzy_value:
            return False
        if not a.overlap_footprint.dwithin(b.overlap_footprint, self.cut_fuzzy_value):
            return False
        if a.overlap_exact and b.overlap_exact:
            return True
        return None

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
        shape_override: list[Any] | None = None,
    ) -> OCCLabeledEntity:
        """Instantiate a single entity into an OCC shape.

        When ``shape_override`` is provided, it replaces the result of
        ``entity_obj.instanciate_occ()``. This is the single-source-of-truth
        path for structured entities: the orchestrator passes the phantom
        solids that the planner already constructed so that BOP fragments
        only one OCC representation of each volume, avoiding sliver
        artifacts at imperfect re-construction boundaries. All identity
        metadata (physical_name, mesh_order, overlap_metadata) still flows
        from ``entity_obj``.
        """
        if shape_override is not None:
            shapes = list(shape_override)
            if shapes:
                dim = self._get_shape_dimension(shapes[0])
            else:
                dim = getattr(entity_obj, "dimension", 0)
        else:
            shape = entity_obj.instanciate_occ()
            shapes = [shape]
            dim = getattr(entity_obj, "dimension", None)
            if dim is None:
                dim = self._get_shape_dimension(shape)
        physical_name = entity_obj.physical_name
        if isinstance(physical_name, str):
            physical_name = (physical_name,)
        # Opt-in metadata for the polyprism fast-overlap path. ``getattr``
        # so entities predating the API still work.
        md_getter = getattr(entity_obj, "overlap_metadata", None)
        md = md_getter() if callable(md_getter) else None
        if md is None:
            footprint, zrange, exact = None, None, False
        else:
            footprint, zrange, exact = md
        return OCCLabeledEntity(
            shapes=shapes,
            physical_name=physical_name,
            index=index,
            keep=getattr(entity_obj, "mesh_bool", True),
            dim=dim,
            mesh_order=getattr(entity_obj, "mesh_order", None),
            overlap_footprint=footprint,
            overlap_zrange=zrange,
            overlap_exact=exact,
            structured=bool(getattr(entity_obj, "structured", False)),
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
        entity_shape_overrides: dict[int, list[Any]] | None = None,
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
            override = (
                entity_shape_overrides.get(orig_idx)
                if entity_shape_overrides is not None
                else None
            )
            labeled = self._instantiate_entity_occ(
                orig_idx, ent, shape_override=override
            )

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
                # Per-entity-pair fast-path: when both sides are polyprisms
                # with exact (footprint, z) metadata, decide overlap purely
                # from shapely + z-interval and skip OCC distance entirely.
                # Returns None when no metadata or tapered envelope -- then
                # fall through to per-(s, ts) _shapes_actually_overlap.
                # See _polyprism_fast_overlap for the full three-way contract.
                fast = self._polyprism_fast_overlap(labeled, prev)
                if fast is False:
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
                    # downstream meshing. The fast-path above already
                    # handled the polyprism-vs-polyprism case; only fall
                    # into _shapes_actually_overlap when the fast-path
                    # couldn't decide (fast is None).
                    if fast is None and not any(
                        self._shapes_actually_overlap(s, ts) for s in labeled.shapes
                    ):
                        continue
                    # fast is True: pair overlap is exact, AABB per sub-shape
                    # above is sufficient. fast is None: OCC check above
                    # confirmed real overlap.
                    tool_shapes.append(ts)

            if tool_shapes and labeled.shapes:
                # Prefused cut. Fuses all tools into one clean shape, then
                # cuts the substrate against the fused result. Replaces the
                # earlier batched-compound form, which corrupted substrates
                # when two tools shared a face/edge (tangent). See
                # docs/superpowers/specs/2026-05-19-cad-occ-prefused-cut-safety-hotfix-design.md
                # and scripts/bench_cut_strategy_sweep.py.
                new_shapes: list[TopoDS_Shape] = []
                for s in labeled.shapes:
                    try:
                        result = cut_one_entity(
                            s, tool_shapes, self.cut_fuzzy_value, self.n_threads
                        )
                        if result is not None and not result.IsNull():
                            self._clamp_shape_tolerance(result, self.cut_fuzzy_value)
                    except Exception as e:  # pragma: no cover -- defensive
                        print(
                            f"Warning: BRepAlgoAPI_Cut failed for entity "
                            f"{orig_idx}: {e}"
                        )
                        result = s
                    if result is not None:
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
        entity_shape_overrides: dict[int, list[Any]] | None = None,
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
            entities_list,
            progress_bars=progress_bars,
            entity_shape_overrides=entity_shape_overrides,
        )
        if not labeled_entities:
            return []
        labeled_entities = _same_name_fuse(
            labeled_entities, self.tolerances, n_threads=self.n_threads
        )
        return self._fragment_all(
            labeled_entities,
            progress_bars=progress_bars,
            extra_occ_shapes=extra_occ_shapes,
            cad_occ_callback=cad_occ_callback,
        )

    def process_entities_parallel(
        self,
        entities_list: list[Any],
        progress_bars: bool = False,
        extra_occ_shapes: list[Any] | None = None,
        cad_occ_callback: Callable[[Any], None] | None = None,
        executor: str = "auto",
        entity_shape_overrides: dict[int, list[Any]] | None = None,
    ) -> list[OCCLabeledEntity]:
        """Fan-out per-entity cut + same-name fuse + global fragment.

        ``executor`` in {``auto``, ``serial``, ``thread``, ``process``}.
        ``auto`` selects ``serial`` for N<4, else ``thread`` if the GIL-
        release probe passes, else ``process``.
        """
        if not entities_list:
            return []

        n = len(entities_list)

        # Choose executor.
        chosen = executor
        if chosen == "auto":
            if n < 4:
                chosen = "serial"
            else:
                chosen = "thread" if _probe_gil_release() else "process"

        if chosen not in ("serial", "thread", "process"):
            raise ValueError(f"unknown executor: {executor!r}")

        if chosen == "process":
            _assert_picklable(entities_list)

        prepare_entities(
            entities_list,
            perturbation=self.perturbation,
            resolve_snap=max(self.perturbation, self.point_tolerance),
        )

        # Stage 0: per-entity cutter discovery.
        cutters = compute_cutters(entities_list, self)

        # Stage 1: per-entity cut.
        labeled = [
            self._instantiate_entity_occ(
                i,
                ent,
                shape_override=(
                    entity_shape_overrides.get(i)
                    if entity_shape_overrides is not None
                    else None
                ),
            )
            for i, ent in enumerate(entities_list)
        ]
        results: dict[int, Any] = {}

        def _serial_work(i: int):
            tools = [labeled[j].shapes[0] for j in cutters[i]]
            try:
                return cut_one_entity(
                    labeled[i].shapes[0], tools, self.cut_fuzzy_value, n_threads=1
                )
            except Exception as e:  # pragma: no cover -- defensive
                print(
                    f"Warning: BRepAlgoAPI_Cut failed for entity {i}: {e}",
                    flush=True,
                )
                return labeled[i].shapes[0]

        if chosen == "serial":
            for i in range(n):
                results[i] = _serial_work(i)
        elif chosen == "thread":
            from concurrent.futures import ThreadPoolExecutor, as_completed

            with ThreadPoolExecutor(max_workers=self.n_threads) as pool:
                futures = {pool.submit(_serial_work, i): i for i in range(n)}
                for fut in as_completed(futures):
                    i = futures[fut]
                    results[i] = fut.result()
        elif chosen == "process":
            from concurrent.futures import ProcessPoolExecutor, as_completed

            from meshwell._brep_io import brep_from_bytes

            with ProcessPoolExecutor(max_workers=self.n_threads) as pool:
                futures = {
                    pool.submit(
                        _process_worker,
                        entities_list[i],
                        [entities_list[j] for j in cutters[i]],
                        self.cut_fuzzy_value,
                    ): i
                    for i in range(n)
                }
                for fut in as_completed(futures):
                    i = futures[fut]
                    results[i] = brep_from_bytes(fut.result())

        # Apply tolerance clamp + unwrap.
        for i in range(n):
            shape = results[i]
            if shape is not None and not shape.IsNull():
                self._clamp_shape_tolerance(shape, self.cut_fuzzy_value)
            labeled[i].shapes = self._unwrap_shape(shape, labeled[i].dim)

        # Stage 2: same-name fuse.
        labeled = _same_name_fuse(labeled, self.tolerances, n_threads=self.n_threads)

        # Stage 3: global fragment.
        return self._fragment_all(
            labeled,
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
    executor: str = "auto",
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
        executor: One of "auto" (default), "serial", "thread", "process",
            or "legacy". The first four select the parallel pipeline's
            executor mode. "legacy" runs the pre-parallel sequential
            cascade.
    """
    processor = CAD_OCC(
        point_tolerance=point_tolerance,
        n_threads=n_threads,
        cut_fuzzy_value=cut_fuzzy_value,
        fragment_fuzzy_value=fragment_fuzzy_value,
        perturbation=perturbation,
    )
    if executor == "legacy":
        return processor.process_entities(
            entities_list,
            progress_bars=progress_bars,
            extra_occ_shapes=extra_occ_shapes,
            cad_occ_callback=cad_occ_callback,
        )
    return processor.process_entities_parallel(
        entities_list,
        progress_bars=progress_bars,
        extra_occ_shapes=extra_occ_shapes,
        cad_occ_callback=cad_occ_callback,
        executor=executor,
    )
