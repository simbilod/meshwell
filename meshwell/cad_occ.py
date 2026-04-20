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

from OCP.BOPAlgo import BOPAlgo_Builder
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_SOLID, TopAbs_VERTEX
from OCP.TopExp import TopExp_Explorer
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
        """
        self.point_tolerance = point_tolerance
        self.n_threads = n_threads
        self.fuzzy_value = point_tolerance if fuzzy_value is None else fuzzy_value

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
        """Instantiate entities and fragment them.

        Every entity builds its geometry with fresh TShapes; BOPAlgo's
        fragment pass with ``fuzzy_value`` is the sole mechanism for
        reconciling coincident sub-geometry -- mirroring how gmsh's own
        OCC kernel handles shared boundaries between entities.
        ``keep=False`` helpers keep their shapes for the XAO writer's
        interface-naming pass; the writer itself excludes their bodies
        from the emitted BREP.
        """
        if not entities_list:
            return []

        labeled_entities = [
            self._instantiate_entity_occ(i, ent)
            for i, ent in enumerate(
                tqdm(
                    entities_list,
                    desc="Instantiating OCC entities",
                    disable=not progress_bars,
                )
            )
        ]

        return self._fragment_all(labeled_entities, progress_bars=progress_bars)


def cad_occ(
    entities_list: list[Any],
    point_tolerance: float = 1e-3,
    n_threads: int = cpu_count(),
    progress_bars: bool = False,
    fuzzy_value: float | None = None,
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
    )
    return processor.process_entities(entities_list, progress_bars=progress_bars)
