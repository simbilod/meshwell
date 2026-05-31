"""Stage 1: gather structured slabs from the input entity list.

A single PolyPrism with N+1 z-boundary keys becomes N StructuredSlab
records (one per consecutive z-pair). Unstructured entities are
returned untouched for cad_occ.
"""
from __future__ import annotations

import itertools
from typing import Any

from meshwell.polyprism import PolyPrism
from meshwell.structured.exceptions import (
    StructuredEntityTypeError,
    StructuredExtrudeRequiredError,
)
from meshwell.structured.types import StructuredSlab


def collect_structured_slabs(
    entities: list[Any],
) -> tuple[list[StructuredSlab], list[Any]]:
    """Partition the input list.

    Returns:
        (structured_slabs, unstructured_entities). `structured_slabs`
        has one StructuredSlab per (PolyPrism, z-interval) pair.
        Original PolyPrism source_index is preserved on every slab
        derived from it.
    """
    structured: list[StructuredSlab] = []
    unstructured: list[Any] = []
    for idx, ent in enumerate(entities):
        if not getattr(ent, "structured", False):
            unstructured.append(ent)
            continue
        if not isinstance(ent, PolyPrism):
            raise StructuredEntityTypeError(
                entity_index=idx, type_name=type(ent).__name__
            )
        if not ent.extrude:
            raise StructuredExtrudeRequiredError(entity_index=idx)
        z_keys = sorted(ent.buffers.keys())
        for zlo, zhi in itertools.pairwise(z_keys):
            structured.append(
                StructuredSlab(
                    source_index=idx,
                    footprint=ent.polygons,
                    zlo=zlo,
                    zhi=zhi,
                    mesh_order=ent.mesh_order,
                    mesh_bool=ent.mesh_bool,
                    physical_name=ent.physical_name,
                    identify_arcs=ent.identify_arcs,
                    arc_tolerance=ent.arc_tolerance,
                    min_arc_points=ent.min_arc_points,
                )
            )
    return structured, unstructured
