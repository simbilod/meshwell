"""Mesh-stage structured-sweep kernel: discovery + pairing validation.

Discovers the synthetic ``__sweep|<name>|<side>|<i>`` (dim 2) and
``__sweepsrc|<name>`` (dim 1) physical groups the CAD stage
(``meshwell.structured.sweep_cad``) writes into the XAO, validates that
the groups found in the loaded model pair 1:1 with the
``StructuredSweepResolutionSpec`` entries the caller supplies, and
builds the hook that runs immediately before ``generate(2)``.

The stamping kernel that fills band faces with an exact tensor-product
mesh lands in a later task; ``_stamp_all`` is a no-op stub here so the
discovery/pairing wiring is testable on its own. ``discover_sweeps()``
returns an empty dict when the model has no sweep groups, so the whole
path is a no-op for non-sweep meshes.
"""
from __future__ import annotations

import contextlib
import logging
from collections import defaultdict
from collections.abc import Callable

import gmsh

from meshwell.resolution import (
    StructuredExtrusionResolutionSpec,
    StructuredSweepResolutionSpec,
)
from meshwell.structured.exceptions import SweepPairingError

logger = logging.getLogger(__name__)

FACE_PREFIX = "__sweep|"
SRC_PREFIX = "__sweepsrc|"


def discover_sweeps() -> dict[str, dict]:
    """Scan the loaded gmsh model's physical groups for sweep synthetics.

    Returns:
        ``{name: {"faces": {(side, i): face_tag}, "src_curves": set[int]}}``
        keyed by ``StructuredSweep.name``. Empty dict when no ``__sweep``
        groups are present in the model.

    """
    faces: dict[str, dict[tuple[str, int], int]] = defaultdict(dict)
    srcs: dict[str, set[int]] = defaultdict(set)
    for dim, gtag in gmsh.model.getPhysicalGroups():
        gname = gmsh.model.getPhysicalName(dim, gtag)
        if dim == 2 and gname.startswith(FACE_PREFIX):
            _, name, side, idx = gname.split("|")
            for tag in gmsh.model.getEntitiesForPhysicalGroup(dim, gtag):
                faces[name][(side, int(idx))] = int(tag)
        elif dim == 1 and gname.startswith(SRC_PREFIX):
            _, name = gname.split("|")
            srcs[name].update(
                int(t) for t in gmsh.model.getEntitiesForPhysicalGroup(dim, gtag)
            )
    return {
        name: {"faces": f, "src_curves": srcs.get(name, set())}
        for name, f in faces.items()
    }


def _is_sweep_spec(spec: object) -> bool:
    """True for specs that pair with a sweep (not a wedge/cohort extrusion)."""
    return isinstance(spec, StructuredSweepResolutionSpec) and not isinstance(
        spec, StructuredExtrusionResolutionSpec
    )


def validate_sweep_pairing(
    discovered: dict[str, dict], resolution_specs: dict
) -> dict[str, StructuredSweepResolutionSpec]:
    """Hard-error on unpaired sweep groups or unpaired sweep specs.

    Args:
        discovered: Output of :func:`discover_sweeps`.
        resolution_specs: The ``resolution_specs`` mapping passed to
            ``mesh()``/``generate_mesh()``. Entries that are
            ``StructuredExtrusionResolutionSpec`` instances are ignored --
            those pair with cohorts, not sweeps.

    Returns:
        ``{sweep_name: StructuredSweepResolutionSpec}`` for every
        discovered sweep.

    Raises:
        SweepPairingError: A discovered ``__sweep`` group has no matching
            spec, or a sweep spec has no matching ``__sweep`` group.

    """
    specs_by_name: dict[str, StructuredSweepResolutionSpec] = {}
    for key, specs in (resolution_specs or {}).items():
        for spec in specs:
            if _is_sweep_spec(spec):
                specs_by_name[key] = spec

    missing = set(discovered) - set(specs_by_name)
    if missing:
        raise SweepPairingError(
            f"Sweep group(s) {sorted(missing)} present in CAD but no "
            "StructuredSweepResolutionSpec provided under that name."
        )
    orphans = set(specs_by_name) - set(discovered)
    if orphans:
        raise SweepPairingError(
            f"StructuredSweepResolutionSpec(s) {sorted(orphans)} have no "
            "matching __sweep group in the loaded CAD."
        )
    return specs_by_name


def strip_sweep_groups() -> None:
    """Remove ``__sweep*`` bookkeeping groups before the .msh is written."""
    to_remove: list[tuple[int, int]] = []
    names: list[str] = []
    for dim, gtag in gmsh.model.getPhysicalGroups():
        gname = gmsh.model.getPhysicalName(dim, gtag)
        if gname.startswith("__sweep"):
            to_remove.append((dim, gtag))
            names.append(gname)
    if not to_remove:
        return
    gmsh.model.removePhysicalGroups(to_remove)
    for gname in names:
        with contextlib.suppress(Exception):
            gmsh.model.removePhysicalName(gname)


def make_sweep_pre_2d_hook(
    discovered: dict[str, dict],
    specs_by_name: dict[str, StructuredSweepResolutionSpec],
    point_tolerance: float,
) -> Callable[[], None]:
    """Build the pre-``generate(2)`` hook: stamp every sweep, then strip bookkeeping.

    Args:
        discovered: Output of :func:`discover_sweeps`.
        specs_by_name: Output of :func:`validate_sweep_pairing`.
        point_tolerance: Geometric snap tolerance used by the stamping kernel.

    Returns:
        A zero-argument callable suitable for ``pre_2d_hook``.

    """

    def _hook() -> None:
        _stamp_all(discovered, specs_by_name, point_tolerance)
        strip_sweep_groups()

    return _hook


def _stamp_all(
    discovered: dict[str, dict],
    specs_by_name: dict[str, StructuredSweepResolutionSpec],
    point_tolerance: float,
) -> None:
    """Stamp every discovered sweep face with an exact tensor-product mesh.

    Stub: the stamping kernel is implemented in a later task. For now
    this only logs, so the discovery/pairing wiring above is testable
    independently of the stamping algorithm.
    """
    del specs_by_name, point_tolerance  # unused until stamping lands
    logger.info("sweep stamping: %d sweep(s) discovered", len(discovered))
