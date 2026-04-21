"""OCC CAD processor using OCP (OpenCASCADE Python) bindings."""
from __future__ import annotations

from dataclasses import dataclass
from os import cpu_count
from typing import TYPE_CHECKING, Any

from OCP.BOPAlgo import BOPAlgo_Builder
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_SOLID, TopAbs_VERTEX
from OCP.TopExp import TopExp_Explorer
from OCP.TopTools import TopTools_ShapeMapHasher
from tqdm.auto import tqdm

from meshwell.occ_geometry_cache import OCCGeometryCache

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
        fuzzy_value: float | None = None,
        canonicalize_topology: bool = False,
        heal_shapes: bool = True,
        heal_angular_tolerance: float = 1e-6,
        validate_fragment: bool = False,
        validate_tolerance_multiplier: float = 10.0,
    ):
        """Initialize OCC CAD processor.

        Args:
            point_tolerance: Quantization tolerance for the OCCGeometryCache.
                Must be smaller than the smallest real feature in the input
                geometry — vertices within this distance get snapped to a
                single TopoDS_Vertex, and two snapped-identical vertices fed
                to ``BRepBuilderAPI_MakeEdge`` raise
                ``StdFail_NotDone: BRep_API: command not done``.
            n_threads: Number of threads for parallel processing.
            fuzzy_value: BOPAlgo_Builder fuzzy value used during the
                all-fragment pass. Independent of ``point_tolerance`` so you
                can fuse coincident interfaces that drifted more than a
                cache-safe tolerance would allow. Defaults to
                ``point_tolerance`` when ``None``.
            canonicalize_topology: if True, run
                :func:`meshwell.occ_canonicalize.canonicalize_topology` after
                the all-fragment pass. Forces TShape sharing across entities
                for coincident sub-vertices/edges/faces whose drift is below
                ``point_tolerance`` but above ``fuzzy_value``. Removes
                residual duplicates that would otherwise reach tetgen as
                "overlapping facets" — an OCC-side substitute for the
                gmsh-level ``remove_all_duplicates`` safety net.
            heal_shapes: if True (default), run
                :func:`meshwell.occ_shape_heal.heal_shapes` as the last
                CAD-stage pass. Uses ``ShapeUpgrade_UnifySameDomain`` --
                the same class OCCT's ``BRepAlgoAPI_BuilderAlgo::
                SimplifyResult`` relies on -- to merge split-face
                artifacts BOPAlgo leaves behind. Physical-name
                attribution is preserved through the unifier's
                ``BRepTools_History``.
            heal_angular_tolerance: Maximum angle (radians) between
                adjacent face normals for them to be considered
                same-domain. Default ``1e-6``.
            validate_fragment: if True, run a post-fragment audit that
                detects near-coincident faces with distinct TShapes and
                raises :class:`CoincidentFacesError` if any are found.
                Converts downstream tetgen ``dihedral 0`` / PLC errors
                into a CAD-stage diagnostic naming the offending entities
                and their face orientations. Opt-in because the audit
                costs O(F) where F is the total face count.
            validate_tolerance_multiplier: Multiplier applied to
                ``point_tolerance`` before quantizing centroids / areas
                in the audit. ``10`` is a good default; lower it if the
                scene has legitimate sub-tolerance features, raise it to
                catch looser drift.
        """
        self.point_tolerance = point_tolerance
        self.n_threads = n_threads
        self.fuzzy_value = point_tolerance if fuzzy_value is None else fuzzy_value
        self.canonicalize_topology = canonicalize_topology
        self.heal_shapes = heal_shapes
        self.heal_angular_tolerance = heal_angular_tolerance
        self.validate_fragment = validate_fragment
        self.validate_tolerance_multiplier = validate_tolerance_multiplier

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

    def _instantiate_entity_occ(
        self,
        index: int,
        entity_obj: Any,
        occ_cache: OCCGeometryCache,
    ) -> OCCLabeledEntity:
        """Instantiate a single entity into an OCC shape."""
        shape = entity_obj.instanciate_occ(occ_cache=occ_cache)
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

    def _fragment_all(
        self,
        entities: list[OCCLabeledEntity],
        progress_bars: bool = False,
    ) -> list[OCCLabeledEntity]:
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
            print(f"BOPAlgo_Builder.Perform() on {len(entities)} entities…", flush=True)
        builder.Perform()

        # piece_candidates: shape_key -> list of (entity_index, mesh_order).
        # piece_shapes: shape_key -> the TopoDS_Shape handle.
        piece_candidates: dict[tuple[int, int], list[tuple[int, float]]] = {}
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
        progress_bars: bool = False,
    ) -> list[OCCLabeledEntity]:
        """Instantiate entities then do one BOPAlgo_Builder pass across all of them.

        Fragment pieces are assigned to the entity with the lowest mesh_order.
        Lower-dim entities embedded in higher-dim ones end up sharing topology
        (coincident sub-faces) because BOPAlgo preserves sub-shape sharing.

        Instantiation is serialized so a single ``OCCGeometryCache`` can back
        every entity; coincident sub-geometry across entities then carries
        the same TopoDS TShape identity, which is what lets ``BOPAlgo_Builder``
        merge shared boundaries rather than treating them as overlapping slivers.
        """
        if not entities_list:
            return []

        occ_cache = OCCGeometryCache(point_tolerance=self.point_tolerance)
        labeled_entities = [
            self._instantiate_entity_occ(i, ent, occ_cache)
            for i, ent in enumerate(
                tqdm(
                    entities_list,
                    desc="Instantiating OCC entities",
                    disable=not progress_bars,
                )
            )
        ]

        labeled_entities = self._fragment_all(
            labeled_entities, progress_bars=progress_bars
        )

        if self.canonicalize_topology:
            from meshwell.occ_canonicalize import canonicalize_topology

            stats = canonicalize_topology(
                labeled_entities, point_tolerance=self.point_tolerance
            )
            if progress_bars and any(stats.values()):
                print(
                    f"Canonicalized TShapes: "
                    f"{stats['vertices']} vertices, "
                    f"{stats['edges']} edges, "
                    f"{stats['faces']} faces.",
                    flush=True,
                )

        if self.heal_shapes:
            from meshwell.occ_shape_heal import heal_shapes as _heal_shapes

            _heal_shapes(
                labeled_entities,
                point_tolerance=self.point_tolerance,
                angular_tolerance=self.heal_angular_tolerance,
            )

        if self.validate_fragment:
            from meshwell.occ_fragment_audit import (
                CoincidentFacesError,
                audit_fragment_faces,
                format_coincident_groups,
            )

            groups = audit_fragment_faces(
                labeled_entities,
                point_tolerance=self.point_tolerance,
                tolerance_multiplier=self.validate_tolerance_multiplier,
            )
            if groups:
                raise CoincidentFacesError(format_coincident_groups(groups))

        return labeled_entities


def cad_occ(
    entities_list: list[Any],
    point_tolerance: float = 1e-3,
    n_threads: int = cpu_count(),
    progress_bars: bool = False,
    fuzzy_value: float | None = None,
    canonicalize_topology: bool = False,
    heal_shapes: bool = True,
    heal_angular_tolerance: float = 1e-6,
    validate_fragment: bool = False,
    validate_tolerance_multiplier: float = 10.0,
) -> list[OCCLabeledEntity]:
    """Utility function for OCC-based CAD processing."""
    processor = CAD_OCC(
        point_tolerance=point_tolerance,
        n_threads=n_threads,
        fuzzy_value=fuzzy_value,
        canonicalize_topology=canonicalize_topology,
        heal_shapes=heal_shapes,
        heal_angular_tolerance=heal_angular_tolerance,
        validate_fragment=validate_fragment,
        validate_tolerance_multiplier=validate_tolerance_multiplier,
    )
    return processor.process_entities(entities_list, progress_bars=progress_bars)
