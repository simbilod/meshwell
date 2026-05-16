"""Data model for the clean structured-polyprism pipeline.

Phase 1 ships only ``StructuredExtrusionResolutionSpec`` and the CAD-stage
``Slab`` / ``OverlapPair`` / ``StructuredPlan`` dataclasses. The PhantomMap
(Layer B) and StructuredMeshPlan (mesh-stage) land in Phase 2 / Phase 3.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, field_validator

if TYPE_CHECKING:
    from shapely.geometry import MultiPolygon, Polygon


class StructuredExtrusionResolutionSpec(BaseModel):
    """Per-z-interval layer counts for a structured ``PolyPrism``.

    Attached via ``PolyPrism(..., structured=True, resolutions=[spec])``.

    Attributes:
        n_layers: One positive integer per z-interval of the owning
            ``PolyPrism``. Length must equal ``len(buffers) - 1`` where
            ``buffers`` is the prism's z-keys dict. Enforced at planner
            time (not here, because the spec has no reference to the
            owning entity).
        recombine: When True, the slab volume is meshed with hex
            elements (gmsh element type 5) instead of wedges
            (element type 6).
    """

    model_config = ConfigDict(frozen=True)

    n_layers: list[int] = Field(
        ..., min_length=1, description="positive layer counts, one per z-interval"
    )
    recombine: bool = False

    @field_validator("n_layers")
    @classmethod
    def _all_positive(cls, v: list[int]) -> list[int]:
        for i, n in enumerate(v):
            if n <= 0:
                raise ValueError(
                    f"n_layers[{i}] = {n}: layer counts must be positive integers"
                )
        return v


class StructuredOverlapError(ValueError):
    """Raised when two structured slabs volumetrically overlap with mismatched z-extents.

    Policy B rejects all overlap unless z-extents match exactly within tolerance.
    """


class StructuredBufferTaperError(ValueError):
    """Raised when ``PolyPrism(structured=True)`` is used with non-zero buffers.

    Structured mode requires uniform extrusion; tapered geometry is not supported.
    """


@dataclass
class Slab:
    """One structured-polyprism z-interval, CAD-stage data only.

    Mesh parameters (``n_layers``, ``recombine``) are NOT stored here -
    they live on the resolution spec and are resolved in a second pass
    at mesh time. This keeps Slab a pure geometry+identity record.
    """

    footprint: "Polygon | MultiPolygon"
    zlo: float
    zhi: float
    physical_name: tuple[str, ...]
    source_index: int
    z_interval_index: int
    mesh_order: float
    identify_arcs: bool = False
    min_arc_points: int = 4
    arc_tolerance: float = 1e-3
    fragment_fuzzy_value: float | None = None
    # Populated by compute_face_partition (default: one piece = the whole footprint).
    face_partition: list["Polygon"] = field(default_factory=list)


@dataclass(frozen=True)
class OverlapPair:
    """Record of a Policy-B-resolved volumetric overlap.

    The winner slab is in ``StructuredPlan.slabs``; the loser was
    dropped during planning. Mesh stage uses this to verify the loser
    spec's n_layers agreed with the winner's.
    """

    winner_slab_index: int
    loser_source_index: int
    loser_z_interval_index: int
    z_extent: tuple[float, float]


@dataclass(frozen=True)
class StructuredPlan:
    """Frozen output of the planner; consumed by phantom + builder stages."""

    slabs: list[Slab]
    z_planes: list[float]
    overlaps: list[OverlapPair]
