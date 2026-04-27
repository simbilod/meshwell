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

from collections import defaultdict
from dataclasses import dataclass
from os import cpu_count
from typing import TYPE_CHECKING, Any

from OCP.Bnd import Bnd_Box
from OCP.BOPAlgo import BOPAlgo_Builder
from OCP.BRep import BRep_Builder
from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCP.BRepBndLib import BRepBndLib
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_SOLID, TopAbs_VERTEX
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS_Compound
from OCP.TopTools import TopTools_ShapeMapHasher
from tqdm.auto import tqdm

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
            fuzzy_value: BOPAlgo fuzzy value used during the all-fragment
                pass (gmsh's ``Geometry.ToleranceBoolean`` equivalent).
                Decoupled from ``point_tolerance`` so near-coincident
                interfaces can be fused without widening the vertex snap.
                Defaults to ``point_tolerance`` when ``None``.
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

    def _bboxes_overlap(
        self,
        b1: tuple[float, ...],
        b2: tuple[float, ...],
    ) -> bool:
        """Return True iff the two AABB tuples have overlapping volume.

        Uses a small gap tolerance (``self.fuzzy_value``) so entities
        that only TOUCH (shared face, zero-volume overlap) are NOT cut
        against each other. Touching entities are handled correctly by the
        subsequent ``BOPAlgo_Builder`` fragment pass; cutting them first
        with ``BRepAlgoAPI_Cut`` would split non-overlapping faces.
        """
        gap = self.fuzzy_value
        return (
            b1[0] < b2[3] - gap
            and b1[3] > b2[0] + gap
            and b1[1] < b2[4] - gap
            and b1[4] > b2[1] + gap
            and b1[2] < b2[5] - gap
            and b1[5] > b2[2] + gap
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
        if not entities_list:
            return []

        from meshwell.cad_common import prepare_entities

        prepare_entities(entities_list, perturbation=self.perturbation)

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
                    if tb is not None and any(
                        self._bboxes_overlap(ob, tb) for ob in obj_bboxes
                    ):
                        tool_shapes.append(ts)

            if tool_shapes and labeled.shapes:
                cb = BRep_Builder()
                tool_compound = TopoDS_Compound()
                cb.MakeCompound(tool_compound)
                for s in tool_shapes:
                    cb.Add(tool_compound, s)

                cut_shapes = []
                for s in labeled.shapes:
                    try:
                        cut_op = BRepAlgoAPI_Cut(s, tool_compound)
                        result = cut_op.Shape()
                    except Exception as e:  # pragma: no cover -- defensive
                        print(
                            f"Warning: BRepAlgoAPI_Cut failed for entity "
                            f"{orig_idx}: {e}"
                        )
                        result = s
                    if result is not None:
                        # Flatten compound wrapper so BOPAlgo_Builder.Modified()
                        # can track sub-shape provenance correctly.
                        cut_shapes.extend(self._unwrap_shape(result, labeled.dim))
                labeled.shapes = cut_shapes

            instantiated[orig_idx] = labeled

        # ``instantiated`` now has one entry per original entity, in
        # insertion order. Filter out any None (defensive; should not
        # happen) before fragment.
        labeled_entities = [le for le in instantiated if le is not None]

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
