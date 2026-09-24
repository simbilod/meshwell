"""Stage 1: gather structured slabs from the input entity list.

A single PolyPrism with N+1 z-boundary keys becomes N StructuredSlab
records (one per consecutive z-pair). Unstructured entities are
returned untouched for cad_occ.
"""
from __future__ import annotations

import itertools
from typing import Any

from shapely.geometry import MultiLineString
from shapely.ops import unary_union

from meshwell.interface_tag import InterfaceTag
from meshwell.polyprism import PolyPrism
from meshwell.polysurface import StructuredPolySurface
from meshwell.structured.exceptions import (
    MixedIdentifyArcsError,
    StructuredEntityTypeError,
    StructuredExtrudeRequiredError,
    StructuredVoidMeshOrderRequiredError,
)
from meshwell.structured.types import StructuredPlane, StructuredSlab


def collect_structured_entities(
    entities: list[Any],
) -> tuple[list[StructuredSlab], list[StructuredPlane], list[Any]]:
    """Partition the input list into structured slabs, structured planes, and unstructured entities.

    Returns:
        (structured_slabs, structured_planes, unstructured_entities).
    """
    if any(getattr(e, "structured", False) for e in entities):
        arcs_true: list[tuple[int, tuple[str, ...] | str]] = []
        arcs_false: list[tuple[int, tuple[str, ...] | str]] = []
        for i, e in enumerate(entities):
            if not isinstance(e, (PolyPrism, StructuredPolySurface)):
                continue
            target = arcs_true if e.identify_arcs else arcs_false
            target.append((i, e.physical_name))
        if arcs_true and arcs_false:
            raise MixedIdentifyArcsError(arcs_true=arcs_true, arcs_false=arcs_false)

    structured: list[StructuredSlab] = []
    planes: list[StructuredPlane] = []
    unstructured: list[Any] = []
    deferred_itags: list[tuple[int, InterfaceTag]] = []

    for idx, ent in enumerate(entities):
        if isinstance(ent, StructuredPolySurface):
            fp = unary_union(ent.polygons)
            planes.append(
                StructuredPlane(
                    source_index=idx,
                    orientation="horizontal",
                    footprint=fp,
                    zmin=ent.z,
                    zmax=ent.z,
                    mesh_order=ent.mesh_order,
                    mesh_bool=ent.mesh_bool,
                    physical_name=ent.physical_name,
                    identify_arcs=ent.identify_arcs,
                    arc_tolerance=ent.arc_tolerance,
                    min_arc_points=ent.min_arc_points,
                )
            )
            continue

        if isinstance(ent, InterfaceTag):
            if ent.structured:
                lss = ent.resolved_linestrings or ent.linestrings
                planes.append(
                    StructuredPlane(
                        source_index=idx,
                        orientation="vertical",
                        footprint=MultiLineString(lss),
                        zmin=ent.zmin,
                        zmax=ent.zmax,
                        mesh_order=ent.mesh_order,
                        mesh_bool=ent.mesh_bool,
                        physical_name=ent.physical_name,
                        identify_arcs=getattr(ent, "identify_arcs", False),
                        arc_tolerance=getattr(ent, "arc_tolerance", 1e-3),
                        min_arc_points=getattr(ent, "min_arc_points", 5),
                    )
                )
            else:
                deferred_itags.append((idx, ent))
            continue

        if not getattr(ent, "structured", False):
            unstructured.append(ent)
            continue
        if not isinstance(ent, PolyPrism):
            raise StructuredEntityTypeError(
                entity_index=idx, type_name=type(ent).__name__
            )
        if not ent.extrude:
            raise StructuredExtrudeRequiredError(entity_index=idx)
        if not ent.mesh_bool and ent.mesh_order is None:
            raise StructuredVoidMeshOrderRequiredError(
                entity_index=idx,
                physical_name=ent.physical_name,
            )
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

    # Auto-promote InterfaceTag(structured=False) entities that touch a structured
    # slab whose z-planes match zmin and zmax so they bind natively to cohort lateral faces.
    if structured and deferred_itags:
        slab_z_set = {s.zlo for s in structured} | {s.zhi for s in structured}
        for idx, ent in deferred_itags:
            lss = ent.resolved_linestrings or ent.linestrings
            tag_geom = MultiLineString(lss) if lss else None
            z_aligned = any(abs(ent.zmin - z) <= 1e-9 for z in slab_z_set) and any(
                abs(ent.zmax - z) <= 1e-9 for z in slab_z_set
            )
            touches_structured = (
                tag_geom is not None
                and not tag_geom.is_empty
                and any(
                    s.zlo < ent.zmax
                    and s.zhi > ent.zmin
                    and s.footprint.intersects(tag_geom)
                    for s in structured
                )
            )
            if z_aligned and touches_structured:
                planes.append(
                    StructuredPlane(
                        source_index=idx,
                        orientation="vertical",
                        footprint=tag_geom,
                        zmin=ent.zmin,
                        zmax=ent.zmax,
                        mesh_order=ent.mesh_order,
                        mesh_bool=ent.mesh_bool,
                        physical_name=ent.physical_name,
                        identify_arcs=getattr(ent, "identify_arcs", False),
                        arc_tolerance=getattr(ent, "arc_tolerance", 1e-3),
                        min_arc_points=getattr(ent, "min_arc_points", 5),
                    )
                )
            else:
                unstructured.append(ent)
    else:
        for _, ent in deferred_itags:
            unstructured.append(ent)

    return structured, planes, unstructured


def collect_structured_slabs(
    entities: list[Any],
) -> tuple[list[StructuredSlab], list[Any]]:
    """Partition the input list into (structured_slabs, unstructured_entities)."""
    slabs, _planes, unstructured = collect_structured_entities(entities)
    return slabs, unstructured
