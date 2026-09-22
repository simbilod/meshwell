"""Streamlined Entity Subgroup Copying (`CopyGroup` / `CopyInstance`) for meshwell.

Architecture:
  1. A donor unit cell or sub-assembly is meshed once into a standard Gmsh `.msh`
     file (`generate_mesh(..., output_mesh="donor.msh")`). Compound hierarchies
     (`via.msh` -> `tile.msh` -> `chip.msh`) are built by chaining `generate_mesh`
     calls where each level stamps the `.msh` produced by the previous level.
  2. In the target scene, `CopyGroup(name=..., msh_path="donor.msh", instances=[...])`
     loads the `.msh` file, matches its `(0D, 1D, 2D, 3D)` closure against each
     `CopyInstance` via `(center_of_mass, mass)` role signatures, stamps `0D/1D/2D`
     in `pre_2d_hook`, and stamps `3D` tetrahedra in `pre_3d_hook` while Gmsh
     meshes the remaining filler entities conformally (`Mesh.MeshOnlyEmpty = 1`).
"""

from __future__ import annotations

import contextlib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import gmsh
import numpy as np
from scipy.spatial import KDTree

_NODE_SNAP_TOL = 1e-8


class CopyGroupCongruenceError(ValueError):
    """Raised when an instance closure fails geometric role matching against the donor `.msh`."""


class CopyGroupPreconditionError(ValueError):
    """Raised when scene or mesher preconditions for `CopyGroup` are violated."""


@dataclass
class CopyInstance:
    """One destination instance of a :class:`CopyGroup`.

    Attributes:
        members: Mapping ``{donor_physical_name_in_msh: instance_physical_name}``
            for every 3D member volume of the donor `.msh`.
        translation: 3-vector ``(dx, dy, dz)`` applied after rotation.
        rotation_axis: Unit axis ``(ux, uy, uz)`` for rigid rotation.
        rotation_angle_deg: Right-hand rotation angle in degrees about
            ``rotation_origin`` (must satisfy ``det(R) == +1``).
        rotation_origin: Point ``(ox, oy, oz)`` about which rotation is applied
            before ``translation``.
    """

    members: dict[str, str]
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    rotation_angle_deg: float = 0.0
    rotation_origin: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def affine_matrix(self) -> np.ndarray:
        """Return the 4x4 homogeneous rigid transform matrix mapping donor -> instance."""
        theta = np.deg2rad(self.rotation_angle_deg)
        u = np.asarray(self.rotation_axis, dtype=float)
        norm = np.linalg.norm(u)
        if norm == 0.0:
            raise ValueError("rotation_axis must be non-zero")
        u = u / norm
        ux, uy, uz = u
        c, s = np.cos(theta), np.sin(theta)
        K = np.array([[0.0, -uz, uy], [uz, 0.0, -ux], [-uy, ux, 0.0]])
        R = np.eye(3) * c + (1.0 - c) * np.outer(u, u) + s * K
        o = np.asarray(self.rotation_origin, dtype=float)
        t = np.asarray(self.translation, dtype=float) + o - R @ o
        A = np.eye(4)
        A[:3, :3] = R
        A[:3, 3] = t
        return A

    def transform_points(self, pts: np.ndarray) -> np.ndarray:
        """Apply ``(R, t)`` to an ``(N, 3)`` array of donor coordinates."""
        if len(pts) == 0:
            return np.zeros((0, 3), dtype=float)
        A = self.affine_matrix()
        return np.asarray(pts, dtype=float) @ A[:3, :3].T + A[:3, 3]

    def inverse_transform_point(self, pt: np.ndarray) -> np.ndarray:
        """Map a 3D point from the instance frame back to the donor `.msh` frame."""
        A = self.affine_matrix()
        return A[:3, :3].T @ (np.asarray(pt, dtype=float) - A[:3, 3])

    def to_dict(self) -> dict[str, Any]:
        """Serialize this CopyInstance to a dictionary."""
        return {
            "members": dict(self.members),
            "translation": list(self.translation),
            "rotation_axis": list(self.rotation_axis),
            "rotation_angle_deg": float(self.rotation_angle_deg),
            "rotation_origin": list(self.rotation_origin),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CopyInstance:
        """Reconstruct a CopyInstance from a dictionary."""
        return cls(
            members=dict(data["members"]),
            translation=tuple(data.get("translation", (0.0, 0.0, 0.0))),
            rotation_axis=tuple(data.get("rotation_axis", (0.0, 0.0, 1.0))),
            rotation_angle_deg=float(data.get("rotation_angle_deg", 0.0)),
            rotation_origin=tuple(data.get("rotation_origin", (0.0, 0.0, 0.0))),
        )


@dataclass
class CopyGroup:
    """Specification for stamping a standalone donor `.msh` + `.xao` pair onto congruent CAD instances.

    Attributes:
        name: Unique identifier for diagnostics and synthetic group tagging.
        msh_path: Path to the standalone Gmsh `.msh` file containing the donor mesh.
        instances: List of :class:`CopyInstance` targets in the current model.
        xao_path: Optional path to the companion `.xao` CAD file (defaults to
            ``msh_path.with_suffix(".xao")``).
        role_tolerance: Tolerance on ``(center_of_mass, mass)`` role matching.
        surface_only: If True, stamp only the outer `2D` shell surface mesh of each
            instance and leave its `3D` interior hollow/unmeshed while Gmsh tet-meshes
            the surrounding filler conformally around the stamped shell.
    """

    name: str
    msh_path: Path
    instances: list[CopyInstance] = field(default_factory=list)
    xao_path: Path | None = None
    role_tolerance: float = 1e-6
    surface_only: bool = False

    def __post_init__(self) -> None:
        """Normalize paths and validate instances."""
        if not isinstance(self.msh_path, Path):
            self.msh_path = Path(self.msh_path)
        if self.xao_path is None:
            self.xao_path = self.msh_path.with_suffix(".xao")
        elif not isinstance(self.xao_path, Path):
            self.xao_path = Path(self.xao_path)
        if not self.instances:
            raise ValueError(f"CopyGroup {self.name!r}: 'instances' must be non-empty.")

    def canonical_donor_names(self) -> list[str]:
        """Return the ordered list of donor physical names in `msh_path`."""
        return list(self.instances[0].members.keys())

    def to_dict(self) -> dict[str, Any]:
        """Serialize this CopyGroup to a dictionary."""
        return {
            "name": self.name,
            "msh_path": str(self.msh_path),
            "xao_path": str(self.xao_path) if self.xao_path else None,
            "instances": [inst.to_dict() for inst in self.instances],
            "role_tolerance": float(self.role_tolerance),
            "surface_only": bool(self.surface_only),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CopyGroup:
        """Reconstruct a CopyGroup from a dictionary."""
        return cls(
            name=data["name"],
            msh_path=Path(data["msh_path"]),
            xao_path=Path(data["xao_path"]) if data.get("xao_path") else None,
            instances=[CopyInstance.from_dict(d) for d in data.get("instances", [])],
            role_tolerance=float(data.get("role_tolerance", 1e-6)),
            surface_only=bool(data.get("surface_only", False)),
        )


# ---------------------------------------------------------------------------
# Precondition checks
# ---------------------------------------------------------------------------


def validate_copy_group_preconditions(
    entities: list[Any],
    copy_groups: list[CopyGroup],
    point_tolerance: float,
    optimization_flags: Any = None,
    resolution_specs: dict[str, list[Any]] | None = None,
) -> None:
    """Validate scene and mesher preconditions before CAD/meshing."""
    if optimization_flags:
        raise CopyGroupPreconditionError(
            "optimization_flags cannot be used with copy_groups because Gmsh's "
            "3D mesh optimizers do not respect Mesh.MeshOnlyEmpty and would mutate "
            "stamped tetrahedra."
        )

    all_group_member_names: set[str] = set()
    for grp in copy_groups:
        if not grp.msh_path.exists():
            raise CopyGroupPreconditionError(
                f"CopyGroup {grp.name!r}: msh_path {grp.msh_path!r} does not exist."
            )
        if grp.xao_path is None or not grp.xao_path.exists():
            raise CopyGroupPreconditionError(
                f"CopyGroup {grp.name!r}: companion xao_path {grp.xao_path!r} does not exist."
            )
        for inst in grp.instances:
            all_group_member_names.update(inst.members.values())

    if resolution_specs:
        from meshwell.resolution import BoundaryLayerResolutionSpec

        for phys_name, specs in resolution_specs.items():
            if set(phys_name.split("___")) & all_group_member_names:
                for spec in specs or []:
                    if isinstance(spec, BoundaryLayerResolutionSpec):
                        raise CopyGroupPreconditionError(
                            f"BoundaryLayerResolutionSpec on {phys_name!r} is not supported "
                            "on CopyGroup members."
                        )

    order_by_name: dict[str, float] = {}
    for idx, ent in enumerate(entities):
        pnames = getattr(ent, "physical_name", None)
        if isinstance(pnames, str):
            pnames = (pnames,)
        order = getattr(ent, "mesh_order", None)
        eff_order = float(order) if order is not None else float(idx)
        for pn in pnames or ():
            order_by_name[pn] = eff_order

    for grp in copy_groups:
        eff_members = {n for inst in grp.instances for n in inst.members.values()}
        member_orders = [order_by_name[n] for n in eff_members if n in order_by_name]
        if not member_orders:
            continue
        min_order, max_order = min(member_orders), max(member_orders)
        for other_name, other_order in order_by_name.items():
            if other_name in eff_members:
                continue
            if min_order <= other_order <= max_order:
                raise CopyGroupPreconditionError(
                    f"CopyGroup {grp.name!r}: non-member entity {other_name!r} "
                    f"(mesh_order={other_order}) is interleaved within the group's "
                    f"mesh_order band [{min_order}, {max_order}]."
                )

        if point_tolerance and point_tolerance > 0:
            for idx, inst in enumerate(grp.instances):
                if abs(inst.rotation_angle_deg) < 1e-12:
                    for comp in inst.translation:
                        rem = abs(
                            comp / point_tolerance - round(comp / point_tolerance)
                        )
                        if rem > 1e-4:
                            raise CopyGroupPreconditionError(
                                f"CopyGroup {grp.name!r} instance {idx}: translation "
                                f"{inst.translation} is not a multiple of "
                                f"point_tolerance={point_tolerance}."
                            )


# ---------------------------------------------------------------------------
# Synthetic physical group tagging across XAO
# ---------------------------------------------------------------------------


def _synth_group_name(group_name: str, inst_idx: int, slot_idx: int) -> str:
    return f"__copy|{group_name}|{inst_idx}|{slot_idx}"


def tag_copy_group_occ_entities(
    occ_entities: list[Any],
    copy_groups: list[CopyGroup],
) -> None:
    """Append ``__copy|{group.name}|{inst_idx}|{slot_idx}`` to member OCCLabeledEntities."""
    by_phys: dict[str, list[Any]] = {}
    for ent in occ_entities:
        for pn in ent.physical_name:
            by_phys.setdefault(pn, []).append(ent)

    for grp in copy_groups:
        donor_keys = grp.canonical_donor_names()
        for inst_idx, inst in enumerate(grp.instances):
            for slot_idx, dname in enumerate(donor_keys):
                iname = inst.members[dname]
                sname = _synth_group_name(grp.name, inst_idx, slot_idx)
                for ent in by_phys.get(iname, []):
                    if sname not in ent.physical_name:
                        ent.physical_name = (*ent.physical_name, sname)


def strip_copy_group_physical_groups() -> None:
    """Remove all ``__copy|...`` synthetic physical groups from the active gmsh model."""
    to_remove: list[tuple[int, int]] = []
    names_to_drop: list[str] = []
    for dim, gtag in gmsh.model.getPhysicalGroups():
        gname = gmsh.model.getPhysicalName(dim, gtag)
        if gname.startswith("__copy|"):
            to_remove.append((dim, gtag))
            names_to_drop.append(gname)
    if to_remove:
        gmsh.model.removePhysicalGroups(to_remove)
        for gname in names_to_drop:
            with contextlib.suppress(Exception):
                gmsh.model.removePhysicalName(gname)


# ---------------------------------------------------------------------------
# Closure, role matching, and `.xao` + `.msh` loading
# ---------------------------------------------------------------------------


def closure_of_volumes(vols: list[int]) -> dict[int, list[int]]:
    """Compute ``{0: pts, 1: curves, 2: faces, 3: vols}`` closure of ``vols``."""
    d3 = sorted({int(abs(v)) for v in vols})
    d2 = sorted(
        {
            int(abs(t))
            for v in d3
            for d, t in gmsh.model.getBoundary([(3, v)], oriented=False)
            if d == 2
        }
    )
    d1 = sorted(
        {
            int(abs(t))
            for f in d2
            for d, t in gmsh.model.getBoundary([(2, f)], oriented=False)
            if d == 1
        }
    )
    d0 = sorted(
        {
            int(abs(t))
            for c in d1
            for d, t in gmsh.model.getBoundary([(1, c)], oriented=False)
            if d == 0
        }
    )
    return {0: d0, 1: d1, 2: d2, 3: d3}


def shell_closure_of_volumes(
    vols: list[int],
) -> tuple[dict[int, list[int]], list[tuple[int, int]]]:
    """Return `(shell_closure, interior_dimtags)` for `vols`.

    `shell_closure` contains `{0: shell_pts, 1: shell_curves, 2: shell_faces, 3: d3}`
    where `shell_faces` are the outer boundary faces (`incidence == 1` across `vols`).
    `interior_dimtags` lists all `(dim, tag)` entities (`dim in (0, 1, 2)`) that lie
    strictly inside the union of `vols` (`incidence >= 2` internal faces and curves/points
    not shared with `shell_faces`).
    """
    full_cl = closure_of_volumes(vols)
    face_counts: Counter = Counter()
    for v in full_cl[3]:
        for d, t in gmsh.model.getBoundary([(3, v)], oriented=False):
            if d == 2:
                face_counts[int(abs(t))] += 1
    shell_faces = sorted(f for f, c in face_counts.items() if c == 1)
    shell_curves = sorted(
        {
            int(abs(t))
            for f in shell_faces
            for d, t in gmsh.model.getBoundary([(2, f)], oriented=False)
            if d == 1
        }
    )
    shell_pts = sorted(
        {
            int(abs(t))
            for c in shell_curves
            for d, t in gmsh.model.getBoundary([(1, c)], oriented=False)
            if d == 0
        }
    )
    shell_cl = {0: shell_pts, 1: shell_curves, 2: shell_faces, 3: full_cl[3]}
    interior_dimtags: list[tuple[int, int]] = (
        [(2, f) for f in full_cl[2] if f not in set(shell_faces)]
        + [(1, c) for c in full_cl[1] if c not in set(shell_curves)]
        + [(0, p) for p in full_cl[0] if p not in set(shell_pts)]
    )
    return shell_cl, interior_dimtags


def _entity_com_and_mass(dim: int, tag: int) -> tuple[np.ndarray, float]:
    """Return ``(center_of_mass_3d, mass)`` for a CAD entity ``(dim, tag)``."""
    if dim == 0:
        vals = gmsh.model.getValue(0, tag, [])
        if len(vals) == 3:
            return np.asarray(vals, dtype=float), 0.0
        b = gmsh.model.getBoundingBox(0, tag)
        return np.array([b[0], b[1], b[2]], dtype=float), 0.0
    com = np.asarray(gmsh.model.occ.getCenterOfMass(dim, tag), dtype=float)
    mass = float(gmsh.model.occ.getMass(dim, tag))
    return com, mass


def _resolve_group_instance_tags(grp: CopyGroup) -> dict[int, list[int]]:
    """Return ``{inst_idx: [vol_tag_slot_0, ..., vol_tag_slot_M-1]}`` from ``__copy|...`` groups."""
    synth_map: dict[tuple[int, int], int] = {}
    prefix = f"__copy|{grp.name}|"
    for dim, gtag in gmsh.model.getPhysicalGroups():
        gname = gmsh.model.getPhysicalName(dim, gtag)
        if not gname.startswith(prefix):
            continue
        parts = gname.split("|")
        if len(parts) != 4:
            continue
        inst_idx, slot_idx = int(parts[2]), int(parts[3])
        tags = gmsh.model.getEntitiesForPhysicalGroup(dim, gtag)
        if len(tags) != 1:
            raise CopyGroupCongruenceError(
                f"CopyGroup {grp.name!r} instance {inst_idx} slot {slot_idx} "
                f"resolved to {len(tags)} entities ({list(tags)}); expected 1."
            )
        synth_map[(inst_idx, slot_idx)] = int(tags[0])

    n_slots = len(grp.canonical_donor_names())
    out: dict[int, list[int]] = {}
    for inst_idx in range(len(grp.instances)):
        vols: list[int] = []
        for s in range(n_slots):
            if (inst_idx, s) not in synth_map:
                raise CopyGroupCongruenceError(
                    f"CopyGroup {grp.name!r}: missing synthetic group for "
                    f"instance {inst_idx}, member slot {s}."
                )
            vols.append(synth_map[(inst_idx, s)])
        out[inst_idx] = vols
    return out


def _load_donor_records_from_msh_file(
    msh_path: Path,
    xao_path: Path,
    member_names: list[str],
    surface_only: bool = False,
) -> list[dict[str, Any]]:
    """Load a donor `(0D/1D/2D/3D)` mesh template from `xao_path` + `msh_path`."""
    prev_model = gmsh.model.getCurrent()
    gmsh.model.add("__msh_donor_loader")
    try:
        gmsh.merge(str(xao_path))
        gmsh.merge(str(msh_path))
        phys_to_vol: dict[str, int] = {}
        for dim, gtag in gmsh.model.getPhysicalGroups(3):
            gname = gmsh.model.getPhysicalName(dim, gtag)
            tags = sorted(
                {int(t) for t in gmsh.model.getEntitiesForPhysicalGroup(dim, gtag)}
            )
            if len(tags) == 1:
                phys_to_vol[gname] = tags[0]

        missing = [n for n in member_names if n not in phys_to_vol]
        if missing:
            raise CopyGroupPreconditionError(
                f"Donor files ({xao_path}, {msh_path}) missing 3D physical groups: {missing}"
            )
        vols = [phys_to_vol[n] for n in member_names]
        cl = closure_of_volumes(vols)
        slot_by_vol = {int(v): s for s, v in enumerate(vols)}

        records: list[dict[str, Any]] = []
        for d in (0, 1, 2, 3):
            for t in cl[d]:
                com, mass = _entity_com_and_mass(d, t)
                tags, coords, _ = gmsh.model.mesh.getNodes(d, t, includeBoundary=False)
                pts_donor = (
                    np.asarray(coords, dtype=float).reshape(-1, 3)
                    if len(tags)
                    else np.zeros((0, 3), dtype=float)
                )
                elems: list[tuple[int, np.ndarray]] = []
                if d >= 1:
                    ets, _, ens = gmsh.model.mesh.getElements(d, t)
                    for et, en in zip(ets, ens):
                        elems.append((int(et), np.asarray(en, dtype=np.int64)))
                    if not elems and (d < 3 or not surface_only):
                        raise CopyGroupPreconditionError(
                            f"Donor .msh file {msh_path} has no dim={d} elements on entity {t}. "
                            "Pass save_all=True to generate_mesh() when creating a donor .msh file."
                        )

                records.append(
                    {
                        "dim": int(d),
                        "tag_donor": int(t),
                        "slot": int(slot_by_vol.get(t, -1)) if d == 3 else -1,
                        "com": com.tolist(),
                        "mass": float(mass),
                        "node_tags": np.asarray(tags, dtype=np.int64),
                        "coords": pts_donor,
                        "elems": elems,
                    }
                )
        return records
    finally:
        gmsh.model.remove()
        gmsh.model.setCurrent(prev_model)


def _match_closure_to_donor_records(
    grp: CopyGroup,
    inst_idx: int,
    inst_vols: list[int],
    donor_records: list[dict[str, Any]],
) -> dict[tuple[int, int], int]:
    """Match every entity `(dim, tag_inst)` in `closure_of_volumes(inst_vols)` to its donor record index."""
    inst = grp.instances[inst_idx]
    cl = closure_of_volumes(inst_vols)
    slot_by_vol = {int(v): s for s, v in enumerate(inst_vols)}

    by_dim: dict[int, list[tuple[int, dict[str, Any]]]] = {0: [], 1: [], 2: [], 3: []}
    for rec_idx, rec in enumerate(donor_records):
        by_dim[int(rec["dim"])].append((rec_idx, rec))

    match: dict[tuple[int, int], int] = {}
    for d in (0, 1, 2, 3):
        candidates = by_dim[d]
        if len(cl[d]) != len(candidates):
            raise CopyGroupCongruenceError(
                f"CopyGroup {grp.name!r} instance {inst_idx}: dim={d} closure entity "
                f"count ({len(cl[d])}) does not match donor .msh count ({len(candidates)})."
            )
        used_rec_indices: set[int] = set()
        for t in cl[d]:
            com_world, mass = _entity_com_and_mass(d, t)
            com_donor = inst.inverse_transform_point(com_world)
            slot = slot_by_vol.get(t, -1) if d == 3 else -1

            best_idx: int | None = None
            best_err = float("inf")
            for rec_idx, rec in candidates:
                if rec_idx in used_rec_indices:
                    continue
                if d == 3 and int(rec.get("slot", -1)) != slot:
                    continue
                d_com = float(
                    np.max(np.abs(com_donor - np.asarray(rec["com"], dtype=float)))
                )
                d_mass = abs(mass - float(rec["mass"]))
                err = max(d_com, d_mass)
                if err < best_err:
                    best_err = err
                    best_idx = rec_idx

            if best_idx is None or best_err > grp.role_tolerance:
                raise CopyGroupCongruenceError(
                    f"CopyGroup {grp.name!r} instance {inst_idx}: unmatched entity "
                    f"(dim={d}, tag={t}, com_donor={tuple(np.round(com_donor, 8))}, "
                    f"mass={mass:.8g}, min_error={best_err:.3e} > role_tolerance={grp.role_tolerance})."
                )
            used_rec_indices.add(best_idx)
            match[(d, t)] = best_idx

    return match


# ---------------------------------------------------------------------------
# Discrete Stamping & Periodic Donor Pre-2D Helper
# ---------------------------------------------------------------------------


def _element_orientation_vector(dim: int, tag: int) -> np.ndarray | None:
    _ets, _etags, ens = gmsh.model.mesh.getElements(dim, tag)
    if not ens or not len(ens[0]):
        return None
    if dim == 1:
        n0, n1 = int(ens[0][0]), int(ens[0][1])
        p0 = np.asarray(gmsh.model.mesh.getNode(n0)[0], dtype=float)
        p1 = np.asarray(gmsh.model.mesh.getNode(n1)[0], dtype=float)
        return p1 - p0
    if dim == 2:
        n0, n1, n2 = int(ens[0][0]), int(ens[0][1]), int(ens[0][2])
        p0 = np.asarray(gmsh.model.mesh.getNode(n0)[0], dtype=float)
        p1 = np.asarray(gmsh.model.mesh.getNode(n1)[0], dtype=float)
        p2 = np.asarray(gmsh.model.mesh.getNode(n2)[0], dtype=float)
        return np.cross(p1 - p0, p2 - p0)
    return None


def make_periodic_pitch_pre_2d_hook(
    pitch: tuple[float, float, float],
    tol: float = 1e-6,
):
    """Return a ``pre_2d_hook`` for generating a ``gap = 0`` tileable donor ``.msh``.

    When a donor unit cell will be stamped into abutting (`gap = 0`) instances
    separated by ``pitch = (dx, dy, dz)``, pass ``pre_2d_hook=make_periodic_pitch_pre_2d_hook(pitch)``
    when generating the donor `.msh` file so its ``+pitch`` boundary curves/faces
    receive the shifted 1D/2D mesh of its ``0`` boundary curves/faces before `generate(3)`.
    """
    shift = np.asarray(pitch, dtype=float)
    A_rel = np.eye(4)
    A_rel[:3, 3] = shift

    def _hook() -> None:
        all_vols = [t for _, t in gmsh.model.getEntities(3)]
        cl = closure_of_volumes(all_vols)
        self_pairs: dict[int, list[tuple[int, int, np.ndarray]]] = {1: [], 2: []}
        for d in (1, 2):
            com_mass = {t: _entity_com_and_mass(d, t) for t in cl[d]}
            for t_src, (c_src, m_src) in com_mass.items():
                for t_dst, (c_dst, m_dst) in com_mass.items():
                    if t_src == t_dst:
                        continue
                    if (
                        abs(m_src - m_dst) <= tol
                        and float(np.max(np.abs((c_src + shift) - c_dst))) <= tol
                    ):
                        self_pairs[d].append((t_src, t_dst, A_rel))

        gmsh.option.setNumber("Mesh.MeshOnlyEmpty", 1)
        gmsh.model.mesh.generate(1)
        _stamp_self_congruence_pairs(1, self_pairs[1])
        gmsh.model.mesh.generate(2)
        _stamp_self_congruence_pairs(2, self_pairs[2])

    return _hook


def _stamp_self_congruence_pairs(
    dim: int,
    self_pairs: list[tuple[int, int, np.ndarray]],
) -> None:
    for t_src, t_dst, A_rel in self_pairs:
        vec_src = _element_orientation_vector(dim, t_src)
        vec_dst = _element_orientation_vector(dim, t_dst)
        flip_orient = (
            vec_src is not None
            and vec_dst is not None
            and float(np.dot(A_rel[:3, :3] @ vec_src, vec_dst)) < 0.0
        )
        gmsh.model.mesh.clear([(dim, t_dst)])
        src_b_ents = (
            [(dim, t_src)]
            + [
                (d, abs(bt))
                for d, bt in gmsh.model.getBoundary(
                    [(dim, t_src)], oriented=False, recursive=False
                )
            ]
            + [
                (d, abs(pt))
                for d, pt in gmsh.model.getBoundary(
                    [(dim, t_src)], oriented=False, recursive=True
                )
            ]
        )
        dst_b_ents = (
            [(dim, t_dst)]
            + [
                (d, abs(bt))
                for d, bt in gmsh.model.getBoundary(
                    [(dim, t_dst)], oriented=False, recursive=False
                )
            ]
            + [
                (d, abs(pt))
                for d, pt in gmsh.model.getBoundary(
                    [(dim, t_dst)], oriented=False, recursive=True
                )
            ]
        )
        dst_tags_all: list[int] = []
        dst_coords_all: list[np.ndarray] = []
        for d_b, t_b in set(dst_b_ents):
            ex_t, ex_c, _ = gmsh.model.mesh.getNodes(d_b, t_b, includeBoundary=False)
            if len(ex_t):
                dst_tags_all.extend(int(x) for x in ex_t)
                dst_coords_all.append(np.asarray(ex_c, dtype=float).reshape(-1, 3))
        kdt = KDTree(np.vstack(dst_coords_all)) if dst_coords_all else None

        nmap: dict[int, int] = {}
        for d_b, t_b in set(src_b_ents):
            if d_b == dim:
                continue
            s_t, s_c, _ = gmsh.model.mesh.getNodes(d_b, t_b, includeBoundary=False)
            if not len(s_t):
                continue
            want = (
                np.asarray(s_c, dtype=float).reshape(-1, 3) @ A_rel[:3, :3].T
                + A_rel[:3, 3]
            )
            dist, idx = kdt.query(want, distance_upper_bound=_NODE_SNAP_TOL)
            for k_i, (dd, j_i) in enumerate(zip(dist, idx)):
                if dd < _NODE_SNAP_TOL:
                    nmap[int(s_t[k_i])] = dst_tags_all[int(j_i)]

        s_t, s_c, _ = gmsh.model.mesh.getNodes(dim, t_src, includeBoundary=False)
        if len(s_t):
            want = (
                np.asarray(s_c, dtype=float).reshape(-1, 3) @ A_rel[:3, :3].T
                + A_rel[:3, 3]
            )
            base = gmsh.model.mesh.getMaxNodeTag()
            new_tags = list(range(base + 1, base + 1 + len(s_t)))
            coords_flat = want.reshape(-1).tolist()
            pcoords = gmsh.model.getParametrization(dim, t_dst, coords_flat)
            gmsh.model.mesh.addNodes(dim, t_dst, new_tags, coords_flat, pcoords)
            for st, nt in zip(s_t, new_tags):
                nmap[int(st)] = int(nt)

        max_s = max(nmap.keys(), default=0)
        lut = np.zeros(max_s + 1, dtype=np.int64)
        for k_s, v_s in nmap.items():
            lut[k_s] = v_s

        ets, _, ens = gmsh.model.mesh.getElements(dim, t_src)
        for et, en in zip(ets, ens):
            mapped = lut[np.asarray(en, dtype=np.int64)]
            if flip_orient:
                if dim == 1 and len(mapped) % 2 == 0:
                    mapped = mapped.reshape(-1, 2)[:, [1, 0]].ravel()
                elif dim == 2 and len(mapped) % 3 == 0:
                    mapped = mapped.reshape(-1, 3)[:, [0, 2, 1]].ravel()
            gmsh.model.mesh.addElementsByType(t_dst, int(et), [], mapped)


def _stamp_instance_dims(
    donor_records: list[dict[str, Any]],
    match: dict[tuple[int, int], int],
    inst: CopyInstance,
    inst_vols: list[int],
    dims: tuple[int, ...],
    surface_only: bool = False,
) -> None:
    """Stamp ``dims`` (`(0, 1, 2)` or `(3,)`) from ``donor_records`` onto ``inst_vols``."""
    cl = (
        shell_closure_of_volumes(inst_vols)[0]
        if surface_only
        else closure_of_volumes(inst_vols)
    )
    det_r = float(np.linalg.det(inst.affine_matrix()[:3, :3]))

    src_keys_list: list[np.ndarray] = []
    dst_vals_list: list[np.ndarray] = []

    for d in range(0, max(dims) + 1):
        for t in cl[d]:
            rec = donor_records[match[(d, t)]]
            d_tags: np.ndarray = rec["node_tags"]
            if not len(d_tags):
                continue
            want = inst.transform_points(rec["coords"])
            ex_t, ex_c, _ = gmsh.model.mesh.getNodes(d, t, includeBoundary=False)
            mapped_tags = np.empty(len(want), dtype=np.int64)
            reused_mask = np.zeros(len(want), dtype=bool)
            if len(ex_t):
                dist, idx = KDTree(np.asarray(ex_c, dtype=float).reshape(-1, 3)).query(
                    want, distance_upper_bound=_NODE_SNAP_TOL
                )
                hit = dist < _NODE_SNAP_TOL
                if np.any(hit):
                    reused_mask[hit] = True
                    mapped_tags[hit] = np.asarray(ex_t, dtype=np.int64)[idx[hit]]

            fresh_idx = np.flatnonzero(~reused_mask)
            if len(fresh_idx):
                if d not in dims:
                    raise RuntimeError(
                        f"CopyGroup stamp dim={d} tag={t}: {len(fresh_idx)} unmatched "
                        f"nodes outside stage dims={dims}."
                    )
                base = int(gmsh.model.mesh.getMaxNodeTag())
                nt = np.arange(base + 1, base + 1 + len(fresh_idx), dtype=np.int64)
                coords_flat = want[fresh_idx].reshape(-1)
                pcoords = (
                    gmsh.model.getParametrization(d, t, coords_flat)
                    if d in (1, 2)
                    else []
                )
                gmsh.model.mesh.addNodes(d, t, nt, coords_flat, pcoords)
                mapped_tags[fresh_idx] = nt

            src_keys_list.append(d_tags)
            dst_vals_list.append(mapped_tags)

    if not src_keys_list:
        return
    all_src = np.concatenate(src_keys_list)
    all_dst = np.concatenate(dst_vals_list)
    lut = np.zeros(int(all_src.max()) + 1, dtype=np.int64)
    lut[all_src] = all_dst

    for d in dims:
        if d == 0:
            continue
        for t in cl[d]:
            _, ex, _ = gmsh.model.mesh.getElements(d, t)
            if sum(len(x) for x in ex):
                continue
            rec = donor_records[match[(d, t)]]
            for et, en in rec["elems"]:
                mapped = lut[en]
                if d == 3 and det_r < 0.0 and len(mapped) % 4 == 0:
                    mapped = mapped.reshape(-1, 4)[:, [1, 0, 2, 3]].ravel()
                gmsh.model.mesh.addElementsByType(t, int(et), [], mapped)


# ---------------------------------------------------------------------------
# Streamlined Hook Controller
# ---------------------------------------------------------------------------


class CopyGroupPipeline:
    """Loads donor `.msh` templates and stamps `0D/1D/2D` in `pre_2d_hook` and `3D` in `pre_3d_hook`."""

    def __init__(self, copy_groups: list[CopyGroup]) -> None:
        """Initialize the pipeline with the configured CopyGroups."""
        self.copy_groups = copy_groups
        self.inst_vols_by_group: dict[str, dict[int, list[int]]] = {}
        self.donor_records_by_group: dict[str, list[dict[str, Any]]] = {}
        self.matches_by_group: dict[str, dict[int, dict[tuple[int, int], int]]] = {}

    def pre_2d_hook(self) -> None:
        """Load donor templates, seed 1D filler mesh, and stamp 0D/1D/2D onto target instances."""
        gmsh.option.setNumber("Mesh.Optimize", 0)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 0)

        for grp in self.copy_groups:
            inst_vols_map = _resolve_group_instance_tags(grp)
            self.inst_vols_by_group[grp.name] = inst_vols_map
            records = _load_donor_records_from_msh_file(
                grp.msh_path,
                grp.xao_path,
                grp.canonical_donor_names(),
                surface_only=grp.surface_only,
            )
            self.donor_records_by_group[grp.name] = records
            self.matches_by_group[grp.name] = {
                k: _match_closure_to_donor_records(grp, k, vols, records)
                for k, vols in inst_vols_map.items()
            }

        gmsh.option.setNumber("Mesh.MeshOnlyEmpty", 1)
        gmsh.model.mesh.generate(1)

        target_curves: set[int] = set()
        target_faces: set[int] = set()
        hidden_interior_2d: list[tuple[int, int]] = []
        for grp in self.copy_groups:
            for vols in self.inst_vols_by_group[grp.name].values():
                cl = closure_of_volumes(vols)
                target_curves.update(cl[1])
                target_faces.update(cl[2])
                if grp.surface_only:
                    _shell_cl, int_dimtags = shell_closure_of_volumes(vols)
                    hidden_interior_2d.extend(int_dimtags)

        gmsh.model.mesh.clear(
            [(2, f) for f in sorted(target_faces)]
            + [(1, c) for c in sorted(target_curves)]
        )

        for grp in self.copy_groups:
            records = self.donor_records_by_group[grp.name]
            for inst_idx, vols in self.inst_vols_by_group[grp.name].items():
                _stamp_instance_dims(
                    records,
                    self.matches_by_group[grp.name][inst_idx],
                    grp.instances[inst_idx],
                    vols,
                    dims=(0, 1, 2),
                    surface_only=grp.surface_only,
                )

        if hidden_interior_2d:
            gmsh.model.setVisibility(hidden_interior_2d, 0)
            gmsh.option.setNumber("Mesh.MeshOnlyVisible", 1)
        gmsh.option.setNumber("Mesh.MeshOnlyEmpty", 1)

    def pre_3d_hook(self) -> None:
        """Stamp 3D tetrahedra onto volume groups and hide hollow volumes for surface_only groups."""
        hollow_vol_dimtags: list[tuple[int, int]] = []
        for grp in self.copy_groups:
            if grp.surface_only:
                for vols in self.inst_vols_by_group[grp.name].values():
                    hollow_vol_dimtags.extend((3, v) for v in vols)
                continue
            records = self.donor_records_by_group[grp.name]
            for inst_idx, vols in self.inst_vols_by_group[grp.name].items():
                _stamp_instance_dims(
                    records,
                    self.matches_by_group[grp.name][inst_idx],
                    grp.instances[inst_idx],
                    vols,
                    dims=(3,),
                )
        if hollow_vol_dimtags:
            gmsh.model.setVisibility(hollow_vol_dimtags, 0)
            gmsh.option.setNumber("Mesh.MeshOnlyVisible", 1)
        gmsh.option.setNumber("Mesh.MeshOnlyEmpty", 1)

    def post_3d_hook(self) -> None:
        """Reset MeshOnlyEmpty/MeshOnlyVisible, strip synthetic groups, and verify 3D conformality."""
        gmsh.option.setNumber("Mesh.MeshOnlyEmpty", 0)
        gmsh.option.setNumber("Mesh.MeshOnlyVisible", 0)
        gmsh.model.setVisibility(gmsh.model.getEntities(), 1)
        strip_copy_group_physical_groups()

        # Strip empty 3D physical groups and empty internal 2D interface groups
        # belonging to surface_only hollow cavity instances.
        if any(grp.surface_only for grp in self.copy_groups):
            empty_pgs: list[tuple[int, int]] = []
            empty_names: list[str] = []
            for dim in (2, 3):
                for _d, gtag in gmsh.model.getPhysicalGroups(dim):
                    ents = gmsh.model.getEntitiesForPhysicalGroup(dim, gtag)
                    has_elems = any(
                        len(gmsh.model.mesh.getElements(dim, int(e))[0]) > 0
                        for e in ents
                    )
                    if not has_elems:
                        empty_pgs.append((dim, gtag))
                        empty_names.append(gmsh.model.getPhysicalName(dim, gtag))
            if empty_pgs:
                gmsh.model.removePhysicalGroups(empty_pgs)
                for gname in empty_names:
                    with contextlib.suppress(Exception):
                        gmsh.model.removePhysicalName(gname)

        allt, allc, _ = gmsh.model.mesh.getNodes()
        if not len(allt):
            return
        pos = {
            int(t): np.asarray(allc[3 * k : 3 * k + 3], dtype=float)
            for k, t in enumerate(allt)
        }
        rows = []
        for _, v in gmsh.model.getEntities(3):
            en = gmsh.model.mesh.getElements(3, v)[2]
            if en:
                rows.append(np.asarray(en[0], dtype=int).reshape(-1, 4))
        if not rows:
            return
        T = np.vstack(rows)
        p = np.array([[pos[int(n)] for n in r] for r in T])
        vol = (
            np.einsum(
                "ij,ij->i",
                np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]),
                p[:, 3] - p[:, 0],
            )
            / 6.0
        )
        neg_count = int((vol <= 0.0).sum())
        if neg_count > 0:
            raise RuntimeError(
                f"CopyGroup conformality verification failed: {neg_count} tetrahedra "
                "have non-positive volume / negative Jacobian."
            )
        fc: Counter = Counter()
        for t in T:
            for f in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)):
                fc[tuple(sorted(t[list(f)]))] += 1
        bad_inc = {k: v for k, v in Counter(fc.values()).items() if k not in (1, 2)}
        if bad_inc:
            raise RuntimeError(
                f"CopyGroup conformality verification failed: non-manifold tet-face "
                f"incidences detected: {bad_inc}"
            )
