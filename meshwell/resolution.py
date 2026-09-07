"""Resolution specifications."""
import copy
import warnings
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field

# Mapping from apply_to sub-types to gmsh entity type strings.
# To add more curve subtypes in the future, simply add entries here
# and extend the Literal type below.
CURVE_SUBTYPE_MAP: dict[str, str] = {
    "lines": "Line",
    "circles": "Circle",
}


class ResolutionSpec(BaseModel):
    """A ResolutionSpec is attached to a pre-CAD entity.

    It sets a mesh size field (see child classes) to the resulting post-CAD volumes, surfaces, curves, or points.

    The volumes, surfaces, curves can be filtered based on their mass (volume, area, length). Points can be filtered based on the length of the curve they belong to.
    """

    apply_to: Literal["volumes", "surfaces", "curves", "lines", "circles", "points"]
    min_mass: float = 0
    max_mass: float = np.inf
    sharing: list[str] | None = None
    not_sharing: list[str] | None = None
    restrict_to: list[str] | None = None

    def to_dict(self) -> dict:
        """Convert resolution spec to dictionary representation."""
        import numpy as np

        d = self.model_dump()
        # Handle inf for JSON
        for k, v in d.items():
            if v == np.inf:
                d[k] = "inf"
        d["type"] = "ResolutionSpec"
        d["resolution_type"] = self.__class__.__name__
        return d

    @property
    def entity_str(self):
        """Convenience wrapper."""
        if self.apply_to == "volumes":
            return "RegionsList"
        if self.apply_to == "surfaces":
            return "SurfacesList"
        if self.apply_to == "curves":
            return "CurvesList"
        if self.apply_to in CURVE_SUBTYPE_MAP:
            return "CurvesList"
        if self.apply_to == "points":
            return "PointsList"
        return None

    @property
    def target_dimension(self):
        """Convenience wrapper."""
        if self.apply_to == "volumes":
            return 3
        if self.apply_to == "surfaces":
            return 2
        if self.apply_to == "curves":
            return 1
        if self.apply_to in CURVE_SUBTYPE_MAP:
            return 1
        if self.apply_to == "points":
            return 0
        return None

    def adapt(
        self,
        size_map: "np.ndarray",  # noqa: ARG002
        context: Any = None,  # noqa: ARG002
        *,
        change_max: float = 2.0,  # noqa: ARG002
        max_ratio: float = 1.3,  # noqa: ARG002
    ) -> "ResolutionSpec":
        """Project a pointwise (x, y, z, size) map onto this spec's parameters.

        Base implementation: this spec type does not support adaptation;
        return self unchanged (documented, not silent).
        """
        warnings.warn(
            f"{type(self).__name__} does not support adapt(); returning unchanged.",
            stacklevel=2,
        )
        return self


class ConstantInField(ResolutionSpec):
    """Provides constant resolution within specified entities.

    This class implements a resolution specification that applies a uniform
    mesh size throughout the specified geometric entities.

    Attributes:
        resolution: The constant mesh resolution to apply.
    """

    resolution: float

    def apply(self, model: Any, entities_mass_dict, **kwargs) -> int | None:  # noqa
        """Apply constant resolution field to the model.

        Creates a MathEval field with constant resolution and restricts it
        to the specified entities.

        Args:
            model: The mesh model to apply the field to.
            entities_mass_dict: Dictionary mapping entity tags to their masses.
            **kwargs: Unused kwargs

        Returns:
            int: Index of the created restrict field.
        """
        matheval_field_index = model.mesh.field.add("MathEval")
        model.mesh.field.setString(matheval_field_index, "F", f"{self.resolution}")
        restrict_field_index = model.mesh.field.add("Restrict")
        model.mesh.field.setNumber(
            restrict_field_index, "InField", matheval_field_index
        )
        model.mesh.field.setNumbers(
            restrict_field_index,
            self.entity_str,
            list(entities_mass_dict.keys()),
        )

        return restrict_field_index

    def refine(self, resolution_factor: float):
        """Create a refined copy with adjusted resolution.

        Args:
            resolution_factor: Factor to multiply the resolution by.

        Returns:
            ConstantInField: A new instance with refined resolution.
        """
        result = copy.copy(self)
        if result.resolution is not None:
            result.resolution *= resolution_factor

        return result


class SampledField(ResolutionSpec):
    """Base class for size fields that require sampling entities at points.

    This class provides shared functionality for resolution specifications
    that need to sample geometric entities to determine appropriate mesh sizing.

    Attributes:
        mass_per_sampling: Mass threshold per sampling point.
            If None, defaults to 0.5 * sizemin.
        max_sampling: Maximum number of sampling points allowed (default: 100).
        sizemin: Minimum mesh size.
    """

    mass_per_sampling: float | None = None
    max_sampling: int = 100
    sizemin: float

    def calculate_samplings(self, entities_mass_dict):
        """Calculate optimal sampling distribution based on entity masses.

        Determines the number of sampling points for each entity based on
        its mass and the specified mass_per_sampling ratio.

        Args:
            entities_mass_dict: Dictionary mapping entity tags to their masses.

        Returns:
            dict: Mapping of entity tags to their calculated sampling counts.
        """
        if self.mass_per_sampling is None:
            # Default sampling is half the minimum resolution
            mass_per_sampling = 0.5 * self.sizemin
        else:
            mass_per_sampling = self.mass_per_sampling

        return {
            tag: min(max(2, int(mass / mass_per_sampling)), self.max_sampling)
            if mass is not None
            else 1
            for tag, mass in entities_mass_dict.items()
        }

    def apply_distance(self, model: Any, entities_mass_dict):
        """Create and configure a distance field for the specified entities.

        Args:
            model: The mesh model to apply the field to.
            entities_mass_dict: Dictionary mapping entity tags to their masses.

        Returns:
            int: Index of the created distance field.
        """
        # Compute optimal samplings for each entity
        samplings_dict = self.calculate_samplings(entities_mass_dict)

        # FIXME: It is computationally cheaper to have a large sampling on all the curves rather than one field per curve; but there is probably an optimum somewhere.
        # For instance, the distribution should be very skewed (tiny vertical curves, tiny curves in bends, vs long horizontal ones), so there may be benefits for a small number of optimized fields.
        samplings = max(samplings_dict.values())
        entities = list(entities_mass_dict.keys())

        distance_field_index = model.mesh.field.add("Distance")
        model.mesh.field.setNumbers(distance_field_index, self.entity_str, entities)
        model.mesh.field.setNumber(distance_field_index, "Sampling", samplings)
        return distance_field_index

    def apply_restrict(
        self,
        model: Any,
        target_field_index: int,
        restrict_to_str: str,
        restrict_to_tags=None,
    ):
        """Apply restriction to limit a field to specific entities.

        Creates a restriction field that limits the application of another
        field to specified geometric entities.

        Args:
            model: The mesh model to apply the restriction to.
            target_field_index (int): Index of the field to restrict.
            restrict_to_str (str): String identifier for the restriction type.
            restrict_to_tags: List of entity tags to restrict to.

        Returns:
            int: Index of the created restriction field.
        """
        restrict_field_index = model.mesh.field.add("Restrict")
        model.mesh.field.setNumber(restrict_field_index, "InField", target_field_index)
        model.mesh.field.setNumbers(
            restrict_field_index,
            restrict_to_str,
            restrict_to_tags,
        )

        return restrict_field_index


class ThresholdField(SampledField):
    """Implements linear growth of resolution away from entities.

    This class creates a threshold field that provides fine resolution near
    specified entities and gradually increases the mesh size with distance.

    Attributes:
        sizemax (float): Maximum mesh size at far distances.
        sizemin (float): Minimum mesh size near entities.
        distmin (float): Distance where minimum size applies (default: 0).
        distmax (float): Distance where maximum size applies.
    """

    sizemax: float
    sizemin: float
    distmin: float = 0
    distmax: float

    def apply(
        self,
        model: Any,
        entities_mass_dict,
        restrict_to_str,
        restrict_to_tags=None,
    ) -> int:
        """Apply threshold field with linear resolution growth.

        Creates a distance-based field that transitions from minimum to maximum
        mesh size over the specified distance range.

        Args:
            model: The mesh model to apply the field to.
            entities_mass_dict: Dictionary mapping entity tags to their masses.
            restrict_to_str (str): String identifier for restriction type.
            restrict_to_tags: List of entity tags to restrict the field to.

        Returns:
            int | None: Index of the created field, or None if skipped.

        Warnings:
            UserWarning: If attempting to set distance field on a Volume.
        """
        if self.entity_str == "RegionsList":
            warnings.warn(
                "Cannot set a distance field on a Volume! Skipping", stacklevel=2
            )
        else:
            distance_field_index = self.apply_distance(
                model=model,
                entities_mass_dict=entities_mass_dict,
            )
            threshold_field_index = model.mesh.field.add("Threshold")
            model.mesh.field.setNumber(
                threshold_field_index, "InField", distance_field_index
            )
            model.mesh.field.setNumber(threshold_field_index, "SizeMin", self.sizemin)
            model.mesh.field.setNumber(threshold_field_index, "DistMin", self.distmin)
            if self.sizemax and self.distmax:
                model.mesh.field.setNumber(
                    threshold_field_index, "SizeMax", self.sizemax
                )
                model.mesh.field.setNumber(
                    threshold_field_index, "DistMax", self.distmax
                )
            model.mesh.field.setNumber(threshold_field_index, "StopAtDistMax", 1)

            # Restriction field
            if restrict_to_tags:
                return self.apply_restrict(
                    model, threshold_field_index, restrict_to_str, restrict_to_tags
                )
            return threshold_field_index
        return None

    def refine(self, resolution_factor: float):
        """Create a refined copy with adjusted size parameters.

        Args:
            resolution_factor (float): Factor to multiply size parameters by.

        Returns:
            ThresholdField: A new instance with refined size parameters.
        """
        result = copy.copy(self)

        if result.sizemax is not None:
            result.sizemax *= resolution_factor
        if result.sizemin is not None:
            result.sizemin *= resolution_factor

        return result


class ExponentialField(SampledField):
    """Exponential growth of the characteristic length away from the entity.

    Attributes:
        growth_factor: Factor by which the mesh size grows exponentially
        lengthscale: Characteristic length scale for the exponential growth
    """

    growth_factor: float
    lengthscale: float

    def apply(
        self,
        model: Any,
        entities_mass_dict,
        restrict_to_str,
        restrict_to_tags=None,
    ) -> int | None:
        """Apply the exponential field to the mesh model.

        Args:
            model: The mesh model to apply the field to
            entities_mass_dict: Dictionary mapping entities to their mass properties
            restrict_to_str: String representation of restriction criteria
            restrict_to_tags: Optional list of tags to restrict the field to

        Returns:
            int: Index of the created field, or None if field cannot be applied

        Warnings:
            UserWarning: If attempting to set distance field on a Volume
        """
        if self.entity_str == "RegionsList":
            warnings.warn(
                "Cannot set a distance field on a Volume! Skipping", stacklevel=2
            )
        else:
            distance_field_index = self.apply_distance(
                model=model,
                entities_mass_dict=entities_mass_dict,
            )

            # Math field
            matheval_field_index = model.mesh.field.add("MathEval")
            model.mesh.field.setString(
                matheval_field_index,
                "F",
                f"{self.sizemin} * {self.growth_factor}^(F{distance_field_index} / {self.lengthscale})",
            )

            # Restriction field
            if restrict_to_tags:
                return self.apply_restrict(
                    model, matheval_field_index, restrict_to_str, restrict_to_tags
                )
            return matheval_field_index
        return None

    def refine(self, resolution_factor: float):
        """Create a refined copy of this field with adjusted resolution.

        The minimum size is scaled by the resolution factor to create a finer
        or coarser mesh as needed.

        Args:
            resolution_factor (float): Factor to scale the minimum size by.
                                     Values < 1.0 create finer meshes,
                                     values > 1.0 create coarser meshes

        Returns:
            ExponentialField: A new field instance with adjusted resolution
        """
        result = copy.copy(self)
        if result.sizemin is not None:
            result.sizemin *= resolution_factor

        return result


class DirectSizeSpecification(ResolutionSpec):
    """Dataclass for directly specifying mesh sizes from data points."""

    refinement_data: np.ndarray
    min_size: float | None = None
    max_size: float | None = None

    # Apply to all dimensions by default
    apply_to: Literal[
        "volumes", "surfaces", "curves", "lines", "circles", "points"
    ] | None = None

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True

    def apply(self, model: Any, _entities_mass_dict, **kwargs) -> int:
        """Apply the direct size specification to the mesh model."""
        import gmsh

        # 1. Use Data Directly
        r_data = self.refinement_data

        # Ensure 2D array
        if r_data.ndim == 1:
            r_data = r_data.reshape(1, -1)

        # Parse coords and values (Assume x,y,z,size)
        if r_data.shape[1] != 4:
            raise ValueError(
                f"refinement_data must be (N, 4) [x,y,z,size], got shape {r_data.shape}"
            )

        coords = r_data[:, :3]
        sizes = r_data[:, 3]

        # 2. Clamp sizes
        if self.min_size:
            sizes = np.maximum(sizes, self.min_size)
        if self.max_size:
            sizes = np.minimum(sizes, self.max_size)

        # 3. Create PostView In-Memory
        view_tag = gmsh.view.add("size_field")
        # GMSH expects a flat list of (x, y, z, val) per point for SP (scalar point)
        flattened_data = np.column_stack((coords, sizes)).flatten().tolist()
        gmsh.view.addListData(view_tag, "SP", len(coords), flattened_data)

        # 4. Create Field
        field_index = model.mesh.field.add("PostView")
        model.mesh.field.setNumber(field_index, "ViewTag", view_tag)

        # 5. Apply Restriction
        if kwargs.get("restrict_to_tags"):
            restrict_to_str = kwargs.get(
                "restrict_to_str", "SurfacesList"
            )  # Default to surfaces
            restrict_field = model.mesh.field.add("Restrict")
            model.mesh.field.setNumber(restrict_field, "InField", field_index)
            model.mesh.field.setNumbers(
                restrict_field, restrict_to_str, kwargs["restrict_to_tags"]
            )
            return restrict_field

        return field_index

    def adapt(
        self,
        size_map: "np.ndarray",
        context: Any = None,  # noqa: ARG002
        *,
        change_max: float = 2.0,  # noqa: ARG002
        max_ratio: float = 1.3,  # noqa: ARG002
    ) -> "DirectSizeSpecification":
        """Replace the carried size map with the new one."""
        result = copy.copy(self)
        result.refinement_data = np.asarray(size_map, dtype=float)
        return result

    def refine(self, resolution_factor: float) -> "DirectSizeSpecification":
        """Create a copy with the size column scaled by ``resolution_factor``."""
        result = copy.copy(self)
        data = np.asarray(self.refinement_data, dtype=float).copy()
        data[:, 3] *= resolution_factor
        result.refinement_data = data
        return result


class Graded(BaseModel):
    """Geometric grading for a structured sweep's normal direction.

    Cell k has size ``h0 * ratio**k``; cells are emitted until the sweep
    thickness (known only to the CAD stage) is filled, with the last
    cell adjusted to land exactly on the thickness. Carries NO thickness
    on purpose — see the structured-sweeps design doc.
    """

    h0: float = Field(gt=0)
    ratio: float = Field(ge=1)


class StructuredSweepResolutionSpec(ResolutionSpec):
    """Discretization of a structured sweep (2D band today, 3D later).

    Consumed by the sweep stamping kernel, not by gmsh size fields,
    hence the no-op ``apply()``. Keyed in ``resolution_specs`` by the
    ``StructuredSweep.name`` it discretizes.
    """

    apply_to: Literal["surfaces"] = "surfaces"
    tangential: float | list[float] | None = None
    normal: dict[str, int | Graded | list[float]] = Field(default_factory=dict)
    element_type: Literal["triangle", "quad"] = "triangle"

    class Config:
        """Pydantic model config."""

        arbitrary_types_allowed = True

    def apply(self, **_kwargs) -> None:
        """No-op: consumed by the sweep stamping kernel."""

    def refine(self, resolution_factor: float) -> "StructuredSweepResolutionSpec":
        """Create a copy with all sizes scaled by ``resolution_factor``.

        Follows the existing convention (see ConstantInField.refine):
        sizes are multiplied by the factor, so refine(0.5) is finer.
        """
        import math

        result = copy.copy(self)
        if isinstance(self.tangential, (int, float)):
            result.tangential = float(self.tangential) * resolution_factor
        elif self.tangential is not None:
            off = np.asarray(self.tangential, dtype=float)
            result.tangential = _equidistribute(
                off, np.diff(off) * resolution_factor
            ).tolist()
        new_normal: dict = {}
        for side, n in self.normal.items():
            if isinstance(n, int):
                new_normal[side] = max(1, math.ceil(n / resolution_factor))
            elif isinstance(n, Graded):
                new_normal[side] = Graded(h0=n.h0 * resolution_factor, ratio=n.ratio)
            else:
                off = np.asarray(n, dtype=float)
                new_normal[side] = _equidistribute(
                    off, np.diff(off) * resolution_factor
                ).tolist()
        result.normal = new_normal
        return result

    def adapt(
        self,
        size_map: "np.ndarray",
        context: "SweepAdaptContext | None" = None,
        *,
        change_max: float = 2.0,
        max_ratio: float = 1.3,
    ) -> "StructuredSweepResolutionSpec":
        """Project the size map onto this band's tangential/normal arrays.

        Directional min-collapse at the old cell midpoints, per-iteration
        change clamp, gradation cap, then 1D equidistribution. Returns
        self unchanged when no context is given or no signal covers the band.
        """
        if context is None:
            warnings.warn(
                "StructuredSweepResolutionSpec.adapt needs a SweepAdaptContext "
                "(use meshwell.remesh.remesh_structured); returning unchanged.",
                stacklevel=2,
            )
            return self
        arrays = _adapt_sweep_arrays(self, size_map, context, change_max, max_ratio)
        if arrays is None:
            return self
        new_normal, t_off, h_t = arrays
        t_new = _insert_required(_equidistribute(t_off, h_t), context.required_ts)
        result = copy.copy(self)
        result.tangential = [float(v) for v in t_new]
        result.normal = new_normal
        return result


def resolve_normal_offsets(normal_spec, thickness: float, atol: float) -> "np.ndarray":
    """Resolve a per-side normal spec into offsets [0, ..., thickness].

    ``int n`` -> n uniform layers. ``Graded`` -> geometric cells, last
    cell adjusted to land on thickness. Explicit array -> validated to
    span [0, thickness] within ``atol`` (SweepNormalExtentError).
    """
    from meshwell.structured.exceptions import SweepNormalExtentError

    if isinstance(normal_spec, int):
        if normal_spec < 1:
            raise SweepNormalExtentError(normal_spec, thickness)
        return np.linspace(0.0, thickness, normal_spec + 1)
    if isinstance(normal_spec, Graded):
        offsets = [0.0]
        h = normal_spec.h0
        while offsets[-1] + h < thickness - atol:
            offsets.append(offsets[-1] + h)
            h *= normal_spec.ratio
        offsets.append(thickness)
        # merge a sliver last cell into its neighbour for quality
        if len(offsets) >= 3 and (offsets[-1] - offsets[-2]) < 0.5 * (
            offsets[-2] - offsets[-3]
        ):
            del offsets[-2]
        return np.asarray(offsets)
    offsets = np.asarray(normal_spec, dtype=float)
    if abs(offsets[0]) > atol or abs(offsets[-1] - thickness) > atol:
        raise SweepNormalExtentError(offsets, thickness)
    return offsets


def _gradation_limit(h: "np.ndarray", max_ratio: float) -> "np.ndarray":
    """Cap neighbor cell-size ratios at ``max_ratio`` (standard two-pass sweep)."""
    h = np.asarray(h, dtype=float).copy()
    for i in range(1, len(h)):
        h[i] = min(h[i], h[i - 1] * max_ratio)
    for i in range(len(h) - 2, -1, -1):
        h[i] = min(h[i], h[i + 1] * max_ratio)
    return h


def _equidistribute(
    offsets: "np.ndarray", h_cells: "np.ndarray", min_cells: int = 2
) -> "np.ndarray":
    """Redistribute ``offsets`` so each new cell holds an equal share of ∫ dη / h.

    ``h_cells`` is the piecewise-constant target size on the old cells.
    Endpoints are pinned exactly; at least ``min_cells`` cells are emitted.
    """
    offsets = np.asarray(offsets, dtype=float)
    d = np.diff(offsets)
    density = d / np.asarray(h_cells, dtype=float)
    cum = np.concatenate([[0.0], np.cumsum(density)])
    n = max(min_cells, int(np.ceil(cum[-1] - 1e-9)))
    new = np.interp(np.linspace(0.0, cum[-1], n + 1), cum, offsets)
    new[0], new[-1] = offsets[0], offsets[-1]
    return new


def _insert_required(
    offsets: "np.ndarray", required: tuple, tol: float = 1e-9
) -> "np.ndarray":
    """Snap the nearest interior offset onto each required coordinate.

    An interior offset already pinned to another required coordinate is not
    reused; a new offset is inserted instead so every required point survives.
    """
    off = np.asarray(offsets, dtype=float).copy()
    req = np.asarray(required, dtype=float)
    for r in required:
        if off.size and np.min(np.abs(off - r)) <= tol:
            continue  # already present
        # interior offsets not already pinned to a required coordinate
        free = [i for i in range(1, len(off) - 1) if np.min(np.abs(req - off[i])) > tol]
        if free:
            i = free[int(np.argmin(np.abs(off[free] - r)))]
            off[i] = r
        else:
            # keep sorted so indices 0 / -1 stay the true endpoints
            off = np.sort(np.append(off, r))
    return np.sort(off)


@dataclass
class SweepAdaptContext:
    """Axis-aligned band frame for StructuredSweepResolutionSpec.adapt.

    Built by the driver from the .xao's __sweep|/__sweepsrc| groups
    (see meshwell.remesh._sweep_band_frames).
    """

    thickness: dict
    t_axis: int
    n_axis: int
    t0: float
    t1: float
    n0: float
    normal_sign: dict
    required_ts: tuple = ()


def _size_interpolator(size_map: "np.ndarray"):
    """Callable mapping (M, 2) xy points to sizes; linear with nearest fallback."""
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

    pts = np.asarray(size_map, dtype=float)[:, :2]
    vals = np.asarray(size_map, dtype=float)[:, 3]
    nearest = NearestNDInterpolator(pts, vals)
    if len(pts) < 4:
        return nearest
    try:
        linear = LinearNDInterpolator(pts, vals)
    except Exception:  # degenerate (collinear) point sets
        return nearest

    def interp(p):
        out = linear(p)
        bad = np.isnan(out)
        if bad.any():
            out[bad] = nearest(p[bad])
        return out

    return interp


def _old_tangential_offsets(tangential, length: float) -> "np.ndarray":
    """Old tangential offsets [0, ..., length] from a scalar spacing or array."""
    if isinstance(tangential, (int, float)):
        h = float(tangential)
        return np.unique(np.concatenate([np.arange(0.0, length, h), [length]]))
    return np.asarray(tangential, dtype=float)


def _adapt_sweep_arrays(
    spec: "StructuredSweepResolutionSpec",
    size_map: "np.ndarray",
    ctx: SweepAdaptContext,
    change_max: float,
    max_ratio: float,
):
    """Collapse the size map onto the band's 1D arrays; damp and gradation-limit.

    Returns (new_normal_offsets_per_side, old_tangential_offsets,
    tangential_cell_sizes) with the tangential part left un-equidistributed
    so the driver can min-merge shared-tangential groups first. Returns None
    when no size-map point lies in the band (never silently coarsen).
    """
    size_map = np.asarray(size_map, dtype=float)
    length = ctx.t1 - ctx.t0
    t_off = _old_tangential_offsets(spec.tangential, length)

    n_bounds = [ctx.n0] + [
        ctx.n0 + ctx.normal_sign[side] * t for side, t in ctx.thickness.items()
    ]
    eps = 1e-9
    tc = size_map[:, ctx.t_axis]
    nc = size_map[:, ctx.n_axis]
    inside = (
        (tc >= ctx.t0 - eps)
        & (tc <= ctx.t1 + eps)
        & (nc >= min(n_bounds) - eps)
        & (nc <= max(n_bounds) + eps)
    )
    if not inside.any():
        return None

    interp = _size_interpolator(size_map)
    t_mid = 0.5 * (t_off[1:] + t_off[:-1])
    h_t_target = np.full(len(t_mid), np.inf)
    new_normal: dict = {}
    for side, thick in ctx.thickness.items():
        off = resolve_normal_offsets(spec.normal[side], thick, atol=1e-9)
        n_mid = 0.5 * (off[1:] + off[:-1])
        tt, nn = np.meshgrid(t_mid, n_mid, indexing="ij")
        pts = np.zeros((tt.size, 2))
        pts[:, ctx.t_axis] = (ctx.t0 + tt).ravel()
        pts[:, ctx.n_axis] = (ctx.n0 + ctx.normal_sign[side] * nn).ravel()
        sampled = interp(pts).reshape(len(t_mid), len(n_mid))
        h_old = np.diff(off)
        h_n = np.clip(sampled.min(axis=0), h_old / change_max, h_old * change_max)
        h_n = _gradation_limit(h_n, max_ratio)
        new_normal[side] = _equidistribute(off, h_n).tolist()
        h_t_target = np.minimum(h_t_target, sampled.min(axis=1))

    h_old_t = np.diff(t_off)
    h_t = np.clip(h_t_target, h_old_t / change_max, h_old_t * change_max)
    h_t = _gradation_limit(h_t, max_ratio)
    return new_normal, t_off, h_t


class StructuredExtrusionResolutionSpec(StructuredSweepResolutionSpec):
    """Number of z-layers per structured slab for wedge stamping.

    Deprecated alias of ``StructuredSweepResolutionSpec(normal=n_layers)``;
    kept because the 3D wedge kernel reads ``n_layers`` directly.
    """

    apply_to: Literal["volumes"] = "volumes"  # type: ignore[assignment]
    n_layers: int = Field(default=1, ge=1)


class BoundaryLayerResolutionSpec(ResolutionSpec):
    """Configure a gmsh BoundaryLayer field on a 1D physical name.

    Attach (via ``resolution_specs``) to any dim-1 physical group -- a
    PolyLine name, an interface ``a___b``, or a boundary ``a___None``. Grows
    an anisotropic graded layer off those curves. Registered with gmsh via
    ``setAsBoundaryLayer`` (NOT combined into the Min background size field),
    hence ``apply`` returns ``None``. Full gmsh passthrough; optional params
    are only pushed when set. Fan points are a separate follow-up (they need
    named 0D point support).
    """

    apply_to: Literal["curves"] = "curves"
    size: float = Field(gt=0)  # gmsh Size: first-layer normal size
    thickness: float = Field(gt=0)  # gmsh Thickness: total layer thickness
    ratio: float = Field(default=1.0, ge=1.0)  # gmsh Ratio: geometric growth
    quads: bool = False  # gmsh Quads (0/1)
    size_far: float | None = None  # gmsh SizeFar
    nb_layers: int | None = None  # gmsh NbLayers
    intersect_metrics: bool = False  # gmsh IntersectMetrics
    aniso_max: float | None = None  # gmsh AnisoMax
    beta: float | None = None  # gmsh Beta

    def apply(self, model: Any, entities_mass_dict, **_kwargs) -> None:
        """Create a gmsh BoundaryLayer field on the given curves.

        Registered via setAsBoundaryLayer; returns None so it is not merged
        into the Min background size field.

        Args:
            model: The mesh model to apply the field to.
            entities_mass_dict: Dictionary mapping entity tags to their masses.
            **_kwargs: Unused kwargs (restrict_to_str, restrict_to_tags).

        Returns:
            None: this field is registered as a boundary layer, not merged
            into the Min background field.
        """
        if not entities_mass_dict:
            return
        f = model.mesh.field.add("BoundaryLayer")
        model.mesh.field.setNumbers(f, "CurvesList", list(entities_mass_dict.keys()))
        model.mesh.field.setNumber(f, "Size", self.size)
        model.mesh.field.setNumber(f, "Thickness", self.thickness)
        model.mesh.field.setNumber(f, "Ratio", self.ratio)
        model.mesh.field.setNumber(f, "Quads", 1 if self.quads else 0)
        if self.size_far is not None:
            model.mesh.field.setNumber(f, "SizeFar", self.size_far)
        if self.nb_layers is not None:
            model.mesh.field.setNumber(f, "NbLayers", self.nb_layers)
        model.mesh.field.setNumber(
            f, "IntersectMetrics", 1 if self.intersect_metrics else 0
        )
        if self.aniso_max is not None:
            model.mesh.field.setNumber(f, "AnisoMax", self.aniso_max)
        if self.beta is not None:
            model.mesh.field.setNumber(f, "Beta", self.beta)
        model.mesh.field.setAsBoundaryLayer(f)
        return
