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
