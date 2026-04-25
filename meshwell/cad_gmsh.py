"""GMSH CAD processor: fragment + mesh_order ownership via the gmsh API.

Mirrors :mod:`meshwell.cad_occ`, but builds and fragments geometry
directly inside a gmsh model rather than via the standalone OCP OpenCASCADE
bindings. After ``cad_gmsh(...)`` the gmsh model has every owning entity
tagged as a physical group, plus derived ``A___B`` interfaces and
``A___None`` domain boundaries. Downstream, :func:`meshwell.mesh.mesh`
with ``model=`` can mesh in place without an XAO round-trip.

Pipeline:

1. Initialize the gmsh model (via :class:`ModelManager`).
2. Call each entity's ``instanciate`` to produce its gmsh dimtags.
3. ``gmsh.model.occ.fragment`` over every input dimtag; use the
   returned correspondence map to know which fragment piece originated
   from which entity.
4. Resolve multi-claim pieces by lowest ``mesh_order`` (first wins on tie).
5. Assign physical groups for entities, pair-wise interfaces, and
   exterior domain boundaries.

Ownership semantics match :mod:`meshwell.cad_occ` exactly -- the same
``mesh_order`` ladder and tie-break rules apply, so tests that pin the
OCC ownership model also pin this one.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from os import cpu_count
from typing import Any

import gmsh
from tqdm.auto import tqdm

from meshwell.model import ModelManager
from meshwell.validation import unpack_dimtags


def strip_suffix(name: str) -> str:
    """Strip the unique suffix __#index from the name."""
    return name.split("__#")[0] if "__#" in name else name


@dataclass
class GMSHLabeledEntity:
    """Per-entity gmsh-dimtag record produced by :func:`cad_gmsh`.

    ``dimtags`` holds the fragment pieces this entity owns after the
    all-fragment pass. Every dimtag here lives in the gmsh model the
    processor was driven against; the physical groups are already
    written by the time :func:`cad_gmsh` returns.
    """

    dimtags: list[tuple[int, int]]
    physical_name: tuple[str, ...]
    index: int
    keep: bool
    dim: int
    mesh_order: float | None = None


def _resolve_piece_ownership(
    piece_candidates: dict[Any, list[tuple[int, float]]],
) -> dict[Any, int]:
    """Pick the owning entity index for each fragment piece.

    Rule: lowest ``mesh_order`` wins. On tie, first candidate in
    insertion order wins. Matches :func:`cad_occ._resolve_piece_ownership`
    exactly so the two backends give identical ownership outcomes.
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


class CAD_GMSH:
    """CAD processor driving geometry construction + fragmentation via gmsh."""

    def __init__(
        self,
        point_tolerance: float = 1e-3,
        n_threads: int = cpu_count(),
        filename: str = "temp",
        model: ModelManager | None = None,
    ):
        """Initialize gmsh CAD processor.

        Args:
            point_tolerance: Tolerance used for gmsh's ``Geometry.Tolerance``
                and ``Geometry.ToleranceBoolean`` when the processor owns
                the model.
            n_threads: Thread count for gmsh meshing / boolean parallelism.
            filename: Base filename for the model (used when ``model`` is
                not provided).
            model: Optional :class:`ModelManager` to reuse. When provided
                the processor does not finalize gmsh on exit -- the caller
                owns the lifecycle.
        """
        if model is None:
            self.model_manager = ModelManager(
                n_threads=n_threads,
                filename=filename,
                point_tolerance=point_tolerance,
            )
            self._owns_model = True
        else:
            self.model_manager = model
            self._owns_model = False
        self.point_tolerance = point_tolerance
        self.n_threads = n_threads

    # ``self.model_manager.model`` is the ``gmsh.model`` object; we keep a
    # compatibility alias so entity ``instanciate(cad_model)`` calls that
    # expect a ``cad_model.model_manager.model`` attribute still work.

    def _instantiate_entity(self, index: int, entity_obj: Any) -> GMSHLabeledEntity:
        """Call the entity's ``instanciate`` hook and wrap the result."""
        physical_name = entity_obj.physical_name
        if isinstance(physical_name, str):
            physical_name = (physical_name,)
        # Make physical name unique for internal tracking
        physical_name = tuple(f"{n}__#{index}" for n in physical_name)

        dimtags_out = entity_obj.instanciate(self)
        dimtags = unpack_dimtags(dimtags_out)

        # Some entities report their dimension on the instance; fall back
        # to the first dimtag when they don't.
        dim = getattr(entity_obj, "dimension", None)
        if dim is None:
            dim = dimtags[0][0] if dimtags else -1

        return GMSHLabeledEntity(
            dimtags=dimtags,
            physical_name=physical_name,
            index=index,
            keep=getattr(entity_obj, "mesh_bool", True),
            dim=dim,
            mesh_order=getattr(entity_obj, "mesh_order", None),
        )

    def _fragment_all(
        self,
        entities: list[GMSHLabeledEntity],
        progress_bars: bool = False,
    ) -> list[GMSHLabeledEntity]:
        """Fragment all entity dimtags; assign pieces by mesh_order priority.

        Uses ``gmsh.model.occ.fragment`` with the first input dimtag as the
        object and the remainder as tools. ``outDimTagsMap`` tells us
        which fragment piece descends from each input dimtag, which we
        invert to build the per-piece candidate ownership list.
        """
        if not entities:
            return []
        if len(entities) == 1:
            return entities

        # Flatten input dimtags with entity-index provenance.
        input_dimtags: list[tuple[int, int]] = []
        provenance: list[int] = []
        for ent_idx, ent in enumerate(entities):
            for dt in ent.dimtags:
                input_dimtags.append(dt)
                provenance.append(ent_idx)

        if not input_dimtags:
            return entities
        if len(input_dimtags) == 1:
            return entities

        if progress_bars:
            print(
                f"gmsh.occ.fragment on {len(input_dimtags)} dimtags "
                f"(across {len(entities)} entities)…",
                flush=True,
            )

        object_dimtags = [input_dimtags[0]]
        tool_dimtags = input_dimtags[1:]
        _, out_map = gmsh.model.occ.fragment(
            object_dimtags,
            tool_dimtags,
            removeObject=True,
            removeTool=True,
        )
        self.model_manager.sync_model()

        # ``out_map[i]`` is the list of fragment pieces that came from
        # ``input_dimtags[i]``. Build piece -> [(ent_idx, mesh_order)].
        piece_candidates: dict[tuple[int, int], list[tuple[int, float]]] = defaultdict(
            list
        )
        for i, pieces in enumerate(
            tqdm(
                out_map,
                desc="Collecting fragment pieces",
                disable=not progress_bars,
                leave=False,
            )
        ):
            ent_idx = provenance[i]
            mo = entities[ent_idx].mesh_order
            if mo is None:
                mo = float("inf")
            for piece in pieces:
                # gmsh returns pieces as (dim, tag) tuples. Force to a
                # hashable tuple so dict keys are deterministic.
                key = (int(piece[0]), int(piece[1]))
                piece_candidates[key].append((ent_idx, mo))

        owners = _resolve_piece_ownership(piece_candidates)

        # Reset and reassign.
        for ent in entities:
            ent.dimtags = []
        for piece, ent_idx in owners.items():
            entities[ent_idx].dimtags.append(piece)

        return entities

    def _tag_entities(
        self,
        entities: list[GMSHLabeledEntity],
        interface_delimiter: str,
        boundary_delimiter: str,
    ) -> None:
        """Write physical groups for entities, interfaces, and boundaries.

        ``keep=False`` semantics mirror the OCC / XAO writer:

        * Top-dim ``keep=False`` helpers: do NOT get a physical group of
           their own (their body will be removed from the model after
           tagging), but DO still appear in ``A___B`` interface names --
           so a neighbour kept entity can still name a shared boundary
           with a helper.
        * Lower-dim ``keep=False`` helpers: always tagged (the embedded
           cut surface / curve is the useful artefact).
        * Both-keep=False interface pairs are skipped: neither side
           survives to be meshed so the named boundary would dangle.
        * Exterior ``___None`` boundaries are only emitted for kept
           top-dim entities.
        """
        max_dim = max((e.dim for e in entities if e.dimtags), default=-1)

        name_dimtags: dict[str, set[tuple[int, int]]] = defaultdict(set)
        top_dim_entities: list[GMSHLabeledEntity] = []

        for ent in entities:
            if not ent.dimtags:
                continue
            if ent.dim == max_dim:
                top_dim_entities.append(ent)
                if not ent.keep:
                    # Skip tagging the helper body; its interface with
                    # kept neighbours is still recovered below.
                    continue
            for name in ent.physical_name:
                for dt in ent.dimtags:
                    name_dimtags[name].add(dt)

        stripped_name_dimtags = defaultdict(set)
        for name, dts in name_dimtags.items():
            stripped_name_dimtags[strip_suffix(name)].update(dts)

        for name, dts in stripped_name_dimtags.items():
            _add_physical_group(name, dts)

        lower_dim_dimtags: set[tuple[int, int]] = set()
        for ent in entities:
            if not ent.dimtags:
                continue
            if ent.dim < max_dim:
                lower_dim_dimtags.update(ent.dimtags)

        # Pair-wise interfaces: one dim below the top-dim entities.
        boundary_of: dict[int, set[tuple[int, int]]] = {}
        for ent in top_dim_entities:
            boundary_of[ent.index] = set(
                gmsh.model.getBoundary(
                    ent.dimtags,
                    combined=False,
                    oriented=False,
                    recursive=False,
                )
            )

        pair_dimtags: dict[tuple[str, str], set[tuple[int, int]]] = defaultdict(set)
        same_material_interfaces: set[tuple[int, int]] = set()

        for i, ei in enumerate(top_dim_entities):
            for j, ej in enumerate(top_dim_entities):
                if j <= i:
                    continue
                if not (ei.keep or ej.keep):
                    # Both helpers: their shared face is unnamed and
                    # will dangle after both removals.
                    continue
                shared = boundary_of[ei.index] & boundary_of[ej.index]
                if not shared:
                    continue
                for ni in ei.physical_name:
                    for nj in ej.physical_name:
                        if ni == nj:
                            same_material_interfaces.update(shared)
                            pair_dimtags[(ni, ni)].update(shared)
                            continue
                        key = tuple(sorted((ni, nj)))
                        pair_dimtags[key].update(shared)

        stripped_pair_dimtags = defaultdict(set)
        for (ni, nj), dts in pair_dimtags.items():
            key = tuple(sorted((strip_suffix(ni), strip_suffix(nj))))
            stripped_pair_dimtags[key].update(dts)

        for (ni, nj), dts in stripped_pair_dimtags.items():
            _add_physical_group(f"{ni}{interface_delimiter}{nj}", dts)

        # Exterior domain boundaries.
        exterior_boundaries = defaultdict(set)
        for ent in top_dim_entities:
            if not ent.keep:
                continue
            my_bnd = boundary_of[ent.index]
            others: set[tuple[int, int]] = set()
            for other in top_dim_entities:
                if other.index == ent.index:
                    continue
                others |= boundary_of[other.index]
            exterior = my_bnd - others - same_material_interfaces - lower_dim_dimtags

            if not exterior:
                continue
            for name in ent.physical_name:
                exterior_boundaries[strip_suffix(name)].update(exterior)

        for name, dts in exterior_boundaries.items():
            _add_physical_group(
                f"{name}{interface_delimiter}{boundary_delimiter}",
                dts,
            )

    def _remove_keep_false_top_dim(self, entities: list[GMSHLabeledEntity]) -> None:
        """Remove geometry of top-dim ``keep=False`` entities from the model.

        Uses ``occ.remove(..., recursive=True)`` so sub-shapes that are
        *not* shared with any kept entity are cleaned up too. Shared
        interface faces survive the removal because gmsh's recursive
        removal leaves sub-shapes with remaining parents in place.
        """
        max_dim = max((e.dim for e in entities if e.dimtags), default=-1)
        dead: list[tuple[int, int]] = []
        for ent in entities:
            if ent.dim != max_dim or ent.keep or not ent.dimtags:
                continue
            dead.extend(ent.dimtags)
            ent.dimtags = []
        if not dead:
            return
        gmsh.model.occ.remove(dead, recursive=True)
        self.model_manager.sync_model()

    def process_entities(
        self,
        entities_list: list[Any],
        progress_bars: bool = False,
        interface_delimiter: str = "___",
        boundary_delimiter: str = "None",
    ) -> list[GMSHLabeledEntity]:
        """Instantiate, fragment, and tag ``entities_list`` in gmsh.

        The gmsh model is populated with physicals by the time this
        returns; the returned :class:`GMSHLabeledEntity` list is provided
        for introspection (ownership checks, tests, plotting).
        """
        if not entities_list:
            return []

        self.model_manager.ensure_initialized(str(self.model_manager.filename))

        # Calculate global bounding box of all polygons for clipping
        from shapely.geometry import box

        xmin, ymin, xmax, ymax = (
            float("inf"),
            float("inf"),
            float("-inf"),
            float("-inf"),
        )
        for ent in entities_list:
            if hasattr(ent, "polygons"):
                polys = (
                    ent.polygons if isinstance(ent.polygons, list) else [ent.polygons]
                )
                for p in polys:
                    b = p.bounds
                    xmin = min(xmin, b[0])
                    ymin = min(ymin, b[1])
                    xmax = max(xmax, b[2])
                    ymax = max(ymax, b[3])
        global_bbox = box(xmin, ymin, xmax, ymax)
        print(f"Global bounding box for clipping: {global_bbox.bounds}")

        # Sort all input entities by mesh_order (lowest first). Break ties by original index.
        indexed_entities = [(ent, i) for i, ent in enumerate(entities_list)]
        indexed_entities.sort(
            key=lambda x: (
                x[0].mesh_order if x[0].mesh_order is not None else float("inf"),
                x[1],
            )
        )

        instantiated_entities = []

        # Step 1: Sequential Cuts across ALL entities (restricted to same dimension for safety)
        for ent, orig_idx in indexed_entities:
            if hasattr(ent, "polygons"):
                print(f"Buffering and clipping polygons for entity {orig_idx}")
                if isinstance(ent.polygons, list):
                    ent.polygons = [
                        p.buffer(2 * self.point_tolerance, join_style=2).intersection(
                            global_bbox
                        )
                        for p in ent.polygons
                    ]
                else:
                    ent.polygons = ent.polygons.buffer(
                        2 * self.point_tolerance, join_style=2
                    ).intersection(global_bbox)

            # Instantiate
            labeled_ent = self._instantiate_entity(orig_idx, ent)

            # Cut with all previously instantiated entities (higher priority)
            if labeled_ent.dimtags:
                all_tool_dimtags = []
                for prev_ent in instantiated_entities:
                    if prev_ent.dim == labeled_ent.dim and prev_ent.dimtags:
                        all_tool_dimtags.extend(prev_ent.dimtags)

                if all_tool_dimtags:
                    try:
                        out_dimtags, _ = gmsh.model.occ.cut(
                            labeled_ent.dimtags,
                            all_tool_dimtags,
                            removeObject=True,
                            removeTool=False,
                        )
                        self.model_manager.sync_model()
                        labeled_ent.dimtags = out_dimtags
                    except Exception as e:
                        print(f"Warning: Cut failed for entity {orig_idx}: {e}")

            instantiated_entities.append(labeled_ent)

        # Step 3: Final Global Fragment
        labeled = self._fragment_all(instantiated_entities, progress_bars=progress_bars)
        self._tag_entities(labeled, interface_delimiter, boundary_delimiter)
        self._remove_keep_false_top_dim(labeled)
        self.model_manager.sync_model()

        return labeled


def _add_physical_group(name: str, dimtags) -> None:
    """Add a physical group with the given dimtags; no-op if empty.

    Groups dimtags by dimension and creates a physical group for each dimension
    present, sharing the same name.
    """
    dimtags = list(dimtags)
    if not dimtags:
        return

    dim_to_tags = defaultdict(list)
    for d, t in dimtags:
        dim_to_tags[d].append(t)

    for dim, tags in dim_to_tags.items():
        pg = gmsh.model.addPhysicalGroup(dim, tags)
        gmsh.model.setPhysicalName(dim, pg, name)


def cad_gmsh(
    entities_list: list[Any],
    point_tolerance: float = 1e-3,
    n_threads: int = cpu_count(),
    progress_bars: bool = False,
    filename: str = "temp",
    model: ModelManager | None = None,
    interface_delimiter: str = "___",
    boundary_delimiter: str = "None",
) -> tuple[list[GMSHLabeledEntity], ModelManager]:
    """Build + fragment + tag ``entities_list`` in a gmsh model.

    Returns ``(labeled_entities, model_manager)``. Pass
    ``model_manager`` on to :func:`meshwell.mesh.mesh` (with
    ``model=model_manager``) to mesh without round-tripping through XAO.
    """
    processor = CAD_GMSH(
        point_tolerance=point_tolerance,
        n_threads=n_threads,
        filename=filename,
        model=model,
    )
    labeled = processor.process_entities(
        entities_list,
        progress_bars=progress_bars,
        interface_delimiter=interface_delimiter,
        boundary_delimiter=boundary_delimiter,
    )
    return labeled, processor.model_manager
