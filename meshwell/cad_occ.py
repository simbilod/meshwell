"""OCC CAD processor using OCP (OpenCASCADE Python) bindings."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from os import cpu_count
from typing import TYPE_CHECKING, Any

from OCP.BOPAlgo import BOPAlgo_Builder
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_SOLID, TopAbs_VERTEX
from OCP.TopExp import TopExp_Explorer
from OCP.TopTools import TopTools_ShapeMapHasher

if TYPE_CHECKING:
    from OCP.TopoDS import TopoDS_Shape


@dataclass
class OCCLabeledEntity:
    """Dataclass to store OCC shape(s) and associated metadata.

    shapes holds the fragment pieces this entity owns after the all-fragment pass.
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

    Uses the TShape pointer plus orientation so that reversed shapes
    (e.g. a face and its reversed twin used to glue solids) compare distinct
    when BOPAlgo differentiates them, and equal when it does not.

    Note: OCP returns a fresh Python wrapper each time ``TopoDS_Shape.TShape()``
    is called, so ``id()``/default ``__hash__`` is not stable. We instead use
    OCC's own ``TopTools_ShapeMapHasher``, which hashes on the underlying
    ``TShape*`` pointer and is the idiomatic key used throughout OCC.
    """
    return (_SHAPE_HASHER(shape), int(shape.Orientation()))


def _resolve_piece_ownership(
    piece_candidates: dict[Any, list[tuple[int, float]]],
) -> dict[Any, int]:
    """Pick the owning entity index for each fragment piece.

    Rule: lowest mesh_order wins. On tie, first candidate in insertion order wins.

    Args:
        piece_candidates: maps piece key -> list of (entity_index, mesh_order).

    Returns:
        dict mapping piece key -> winning entity_index.
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
    """CAD class for generating geometry using OpenCASCADE (via OCP)."""

    def __init__(
        self,
        point_tolerance: float = 1e-3,
        n_threads: int = cpu_count(),
    ):
        """Initialize OCC CAD processor.

        Args:
            point_tolerance: Tolerance for boolean operations (Fuzzy value)
            n_threads: Number of threads for parallel processing
        """
        self.point_tolerance = point_tolerance
        self.n_threads = n_threads

    def _get_shape_dimension(self, shape: TopoDS_Shape) -> int:
        """Infer dimension from TopoDS_Shape type."""
        # Check for Solids (3D)
        explorer = TopExp_Explorer(shape, TopAbs_SOLID)
        if explorer.More():
            return 3

        # Check for Faces (2D)
        explorer = TopExp_Explorer(shape, TopAbs_FACE)
        if explorer.More():
            return 2

        # Check for Edges (1D)
        explorer = TopExp_Explorer(shape, TopAbs_EDGE)
        if explorer.More():
            return 1

        # Check for Vertices (0D)
        explorer = TopExp_Explorer(shape, TopAbs_VERTEX)
        if explorer.More():
            return 0

        return -1

    def _instantiate_entity_occ(self, index: int, entity_obj: Any) -> OCCLabeledEntity:
        """Instantiate a single entity into an OCC shape."""
        shape = entity_obj.instanciate_occ()
        dim = getattr(entity_obj, "dimension", None)
        if dim is None:
            dim = self._get_shape_dimension(shape)

        return OCCLabeledEntity(
            shapes=[shape],
            physical_name=entity_obj.physical_name,
            index=index,
            keep=entity_obj.mesh_bool,
            dim=dim,
            mesh_order=entity_obj.mesh_order,
        )

    def _fragment_all(self, entities: list[OCCLabeledEntity]) -> list[OCCLabeledEntity]:
        """Fragment all entities together; assign pieces by mesh_order priority.

        Each input entity carries a ``mesh_order`` attribute (float or None).
        After this call, each entity's ``shapes`` list contains only the
        fragment pieces it owns. Ownership rule: lowest mesh_order wins.
        Pieces that come from only one entity are unambiguously owned by it.
        """
        if not entities:
            return []

        # Single-entity shortcut — nothing to fragment against.
        if len(entities) == 1:
            return entities

        builder = BOPAlgo_Builder()
        builder.SetRunParallel(self.n_threads > 1)
        builder.SetFuzzyValue(self.point_tolerance)
        builder.SetNonDestructive(False)

        originals_per_entity: list[list[TopoDS_Shape]] = []
        for ent in entities:
            originals_per_entity.append(list(ent.shapes))
            for s in ent.shapes:
                builder.AddArgument(s)

        builder.Perform()

        # piece_candidates: shape_key -> list of (entity_index, mesh_order).
        # piece_shapes: shape_key -> the TopoDS_Shape handle.
        piece_candidates: dict[tuple[int, int], list[tuple[int, float]]] = {}
        piece_shapes: dict[tuple[int, int], TopoDS_Shape] = {}

        for ent_idx, ent in enumerate(entities):
            mo = ent.mesh_order
            if mo is None:
                mo = float("inf")
            for original in originals_per_entity[ent_idx]:
                modified = builder.Modified(original)
                if modified.IsEmpty() and not builder.IsDeleted(original):
                    # Shape survived untouched.
                    pieces = [original]
                else:
                    pieces = list(modified)
                for piece in pieces:
                    k = _shape_key(piece)
                    piece_shapes.setdefault(k, piece)
                    piece_candidates.setdefault(k, []).append((ent_idx, mo))

        owners = _resolve_piece_ownership(piece_candidates)

        # Reset each entity's shapes and reassign by owner.
        for ent in entities:
            ent.shapes = []
        for key, ent_idx in owners.items():
            entities[ent_idx].shapes.append(piece_shapes[key])

        return entities

    def process_entities(
        self,
        entities_list: list[Any],
        _progress_bars: bool = False,
    ) -> list[OCCLabeledEntity]:
        """Instantiate entities then do one BOPAlgo_Builder pass across all of them.

        Fragment pieces are assigned to the entity with the lowest mesh_order.
        Lower-dim entities embedded in higher-dim ones end up sharing topology
        (coincident sub-faces) because BOPAlgo preserves sub-shape sharing.
        """
        if not entities_list:
            return []

        with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
            labeled_entities = list(
                executor.map(
                    lambda x: self._instantiate_entity_occ(x[0], x[1]),
                    enumerate(entities_list),
                )
            )

        return self._fragment_all(labeled_entities)


def cad_occ(
    entities_list: list[Any],
    point_tolerance: float = 1e-3,
    n_threads: int = cpu_count(),
) -> list[OCCLabeledEntity]:
    """Utility function for OCC-based CAD processing."""
    processor = CAD_OCC(point_tolerance=point_tolerance, n_threads=n_threads)
    return processor.process_entities(entities_list)
