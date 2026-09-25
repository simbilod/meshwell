"""Wrapper exposing a cohort compound as a cad_occ-compatible entity.

The cohort compound enters cad_occ as ONE entity (one BOP argument).
The cad_occ post-pass later expands it back into per-sub-solid
OCCLabeledEntity records via slab_meta.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from OCP.TopoDS import TopoDS_Compound

from meshwell.structured.types import Cohort, ShapeKey, SlabMeta


@dataclass
class _CohortEntity:
    """Behaves like a cad_occ entity but unwraps to N sub-solids.

    Has instanciate_occ, mesh_order, physical_name, mesh_bool, dimension.

    `slab_meta` is the dict the post-pass uses to recover per-slab
    physical_name from each surviving post-BOP sub-solid.
    """

    compound: TopoDS_Compound
    slab_meta: dict[ShapeKey, SlabMeta]
    cohort: Cohort
    cohort_index: int = 0
    # OCC-valid solids covering the cohort (``build_cohort_hull`` plus
    # ``build_void_carve_tools``; cut-cascade tools only, never meshed)
    # and ``id()`` of every lower-priority unstructured entity object that
    # overlaps it (``classify_cohort_overlaps``). cad_occ cuts exactly
    # those entities with these tools; everything else keeps skipping
    # cohort cuts.
    carve_tools: list = field(default_factory=list)
    carve_targets: frozenset[int] = frozenset()

    dimension: int = field(init=False, default=3)
    mesh_bool: bool = field(init=False, default=True)
    is_cohort: bool = field(init=False, default=True)

    @property
    def mesh_order(self) -> float:
        return min(
            (s.mesh_order if s.mesh_order is not None else float("inf"))
            for s in self.cohort.slabs
        )

    @property
    def physical_name(self) -> tuple[str, ...]:
        return (f"__cohort_{self.cohort_index}",)

    # Signal to cad_occ that this compound must be passed to
    # BOPAlgo_Builder as ONE argument, not unwrapped into individual
    # solids.  Passing the sub-solids separately causes BOP to fuse
    # adjacent solids that share TShape faces, destroying the per-slab
    # structure needed by stamp_wedges.
    keep_compound_for_bop: bool = field(init=False, default=True)

    def instanciate_occ(self) -> TopoDS_Compound:
        return self.compound
