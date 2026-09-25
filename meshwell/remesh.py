"""Remesh module for adaptive mesh refinement."""

from __future__ import annotations

import copy
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from os import cpu_count
from pathlib import Path

import gmsh
import meshio
import numpy as np
from scipy.interpolate import NearestNDInterpolator
from scipy.spatial import cKDTree

from meshwell.cad_settings import CADSettings
from meshwell.model import ModelManager
from meshwell.resolution import (
    DirectSizeSpecification,
    StructuredExtrusionResolutionSpec,
    StructuredSweepResolutionSpec,
    SweepAdaptContext,
    _adapt_sweep_arrays,
    _equidistribute,
    _insert_required,
)


def _identity_threshold_func(
    _metric_values: np.ndarray, current_sizes: np.ndarray
) -> np.ndarray:
    """Default threshold function that returns sizes unchanged."""
    return current_sizes


@dataclass
class RemeshingStrategy:
    """Base dataclass for defining a remeshing strategy.

    Args:
        refinement_data: Optional (N, 4) array of (x, y, z, data) to evaluate strategy on.
        func: Optional callable taking refinement_data and returning a metric array.
              Returns: (N,) array of metric values.
        threshold_func: Callable mapping (metric_values, current_sizes) to new_sizes.
        min_size: Minimum allowed mesh size.
        max_size: Maximum allowed mesh size.
    """

    refinement_data: Path | np.ndarray | None
    func: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None
    threshold_func: Callable[
        [np.ndarray, np.ndarray], np.ndarray
    ] = _identity_threshold_func
    min_size: float | None = None
    max_size: float | None = None

    def apply(self, current_sizes: np.ndarray, metric_values: np.ndarray) -> np.ndarray:
        """Apply the strategy to calculate new sizes based on metric values."""
        new_sizes = self.threshold_func(metric_values, current_sizes)

        # Clamp sizes
        if self.min_size is not None:
            new_sizes = np.maximum(new_sizes, self.min_size)
        if self.max_size is not None:
            new_sizes = np.minimum(new_sizes, self.max_size)

        return new_sizes


@dataclass
class BinaryScalingStrategy(RemeshingStrategy):
    """Strategy that scales mesh size by a factor where metric > threshold."""

    threshold: float = 0.5
    factor: float = 0.5

    def __post_init__(self):
        """Initialize threshold function if not set."""
        if self.threshold_func is _identity_threshold_func:
            self.threshold_func = self._binary_scaling

    def _binary_scaling(
        self, metric_values: np.ndarray, current_sizes: np.ndarray
    ) -> np.ndarray:
        mask = metric_values > self.threshold
        new_sizes = current_sizes.copy()
        new_sizes[mask] *= self.factor
        return new_sizes


@dataclass
class SigmoidScalingStrategy(RemeshingStrategy):
    """Strategy that scales mesh size smoothly using a sigmoid transition."""

    threshold: float = 0.5
    factor: float = 0.5
    steepness: float = 1.0

    def __post_init__(self):
        """Initialize threshold function if not set."""
        if self.threshold_func is _identity_threshold_func:
            self.threshold_func = self._sigmoid_scaling

    def _sigmoid_scaling(
        self, metric_values: np.ndarray, current_sizes: np.ndarray
    ) -> np.ndarray:
        weights = 1.0 / (
            1.0 + np.exp(-self.steepness * (metric_values - self.threshold))
        )
        target_sizes = current_sizes * self.factor
        return current_sizes * (1.0 - weights) + target_sizes * weights


@dataclass
class MMGRemeshingStrategy(BinaryScalingStrategy):
    """Strategy for MMG remeshing with specific parameters."""

    hausd: float | None = None
    hgrad: float | None = None


class Remesher:
    """Base class for adaptive mesh refinement."""

    def __init__(
        self,
        n_threads: int = cpu_count(),
        filename: str = "temp_remesh",
        model: ModelManager | None = None,
        verbosity: int = 0,
    ):
        """Initialize remesh processor.

        Args:
            n_threads: Number of threads for processing
            filename: Base filename for the model
            model: Optional Model instance to use (creates new if None)
            verbosity: Verbosity level
        """
        self.n_threads = n_threads
        self.verbosity = verbosity

        # Use provided model or create new one
        if model is None:
            self.model_manager = ModelManager(
                n_threads=n_threads,
                filename=filename,
            )
            self._owns_model = True
        else:
            self.model_manager = model
            self._owns_model = False

        # Store mesh data
        self.vtags = None
        self.vxyz = None
        self.triangles_tags = None
        self.triangles = None

    def _resolve_intake(
        self, geometry_file: Path | str, cad_settings: CADSettings | None
    ) -> CADSettings:
        """Apply the mesh-stage ``.xao`` intake rules (raise on missing/mismatch).

        A caller-supplied model is checked for consistency; an internally
        created one (``_owns_model``) is not, since its ``point_tolerance``
        was never chosen by the caller.
        """
        from meshwell.mesh import _resolve_mesh_cad_settings

        settings = _resolve_mesh_cad_settings(
            input_file=geometry_file,
            model=None if self._owns_model else self.model_manager,
            cad_settings=cad_settings,
            point_tolerance=None,
        )
        self.model_manager.cad_settings = settings
        return settings

    def _load_mesh_data(self, input_mesh: Path | meshio.Mesh | ModelManager) -> None:
        """Load mesh data from file, meshio object, or ModelManager.

        Args:
            input_mesh: Path to .msh file, meshio.Mesh object, or ModelManager.
        """
        if isinstance(input_mesh, (str, Path)):
            # Ensure gmsh is initialized
            if not gmsh.isInitialized():
                gmsh.initialize()

            # Create a temporary model to load the mesh
            temp_model = "temp_mesh_reader"

            # Store current model if any
            try:
                current_model = gmsh.model.getCurrent()
            except:  # noqa: E722
                current_model = None

            gmsh.model.add(temp_model)
            gmsh.model.setCurrent(temp_model)
            gmsh.merge(str(input_mesh))

            self._extract_gmsh_mesh_data()

            gmsh.model.remove()
            if current_model:
                gmsh.model.setCurrent(current_model)

        elif isinstance(input_mesh, meshio.Mesh):
            self.vxyz = input_mesh.points
            # Handle 2D (triangle) and 3D (tetra) elements
            if "triangle" in input_mesh.cells_dict:
                self.triangles = input_mesh.cells_dict["triangle"]
            elif "tetra" in input_mesh.cells_dict:
                self.triangles = input_mesh.cells_dict["tetra"]
            else:
                # Fallback or empty
                self.triangles = None

        elif isinstance(input_mesh, ModelManager):
            current_model = gmsh.model.getCurrent()
            # Assuming the model is already active or we can switch to it
            # But ModelManager wraps gmsh model, so we might need to be careful about current model
            # For now, let's assume we can extract from it if it's the current one or we make it current
            # This path might be tricky if ModelManager doesn't expose everything,
            # but let's try to use the underlying gmsh model
            if input_mesh.model:
                # If ModelManager has a model object, try to use it
                pass  # It's just a module reference usually

            # Just extract from current state assuming user set it up
            self._extract_gmsh_mesh_data()
            if current_model:
                gmsh.model.setCurrent(current_model)

    def _extract_gmsh_mesh_data(self):
        """Helper to extract nodes and elements from current GMSH model."""
        self.vtags, vxyz, _ = gmsh.model.mesh.getNodes()
        self.vxyz = vxyz.reshape((-1, 3))

        vmap = {j: i for i, j in enumerate(self.vtags)}

        # Try 3D first
        try:
            self.triangles_tags, evtags = gmsh.model.mesh.getElementsByType(4)
            evid = np.array([vmap[j] for j in evtags])
            self.triangles = evid.reshape((self.triangles_tags.shape[-1], -1))
        except:  # noqa: E722
            # Try 2D
            try:
                self.triangles_tags, evtags = gmsh.model.mesh.getElementsByType(2)
                evid = np.array([vmap[j] for j in evtags])
                self.triangles = evid.reshape((self.triangles_tags.shape[-1], -1))
            except:  # noqa: E722
                self.triangles = None

    def _extract_edges(self) -> set[tuple[int, int]]:
        """Extract unique edges from the loaded mesh elements.

        Returns:
            Set of tuples (n1, n2) where n1 < n2.
        """
        edges = set()
        if self.triangles is not None:
            num_nodes_per_elem = self.triangles.shape[1]
            if num_nodes_per_elem == 3:  # Triangles
                for elem in self.triangles:
                    edges.add(tuple(sorted((elem[0], elem[1]))))
                    edges.add(tuple(sorted((elem[1], elem[2]))))
                    edges.add(tuple(sorted((elem[2], elem[0]))))
            elif num_nodes_per_elem == 4:  # Tetrahedra
                for elem in self.triangles:
                    edges.add(tuple(sorted((elem[0], elem[1]))))
                    edges.add(tuple(sorted((elem[0], elem[2]))))
                    edges.add(tuple(sorted((elem[0], elem[3]))))
                    edges.add(tuple(sorted((elem[1], elem[2]))))
                    edges.add(tuple(sorted((elem[1], elem[3]))))
                    edges.add(tuple(sorted((elem[2], elem[3]))))
        return edges

    def get_current_mesh_sizes(self) -> np.ndarray:
        """Calculate current mesh size at each node (average connected edge length).

        Returns:
            Array of size values at mesh nodes
        """
        if self.vxyz is None:
            raise ValueError("Mesh data not loaded. Call _load_mesh_data first.")

        edges = self._extract_edges()

        node_sum_lengths = np.zeros(len(self.vxyz))
        node_counts = np.zeros(len(self.vxyz))

        for n1, n2 in edges:
            p1 = self.vxyz[n1]
            p2 = self.vxyz[n2]
            length = np.linalg.norm(p1 - p2)

            node_sum_lengths[n1] += length
            node_counts[n1] += 1
            node_sum_lengths[n2] += length
            node_counts[n2] += 1

        with np.errstate(divide="ignore", invalid="ignore"):
            node_sizes = node_sum_lengths / node_counts
            node_sizes[node_counts == 0] = 0.0

        return node_sizes

    def compute_size_field(self, strategies: list[RemeshingStrategy]) -> np.ndarray:
        """Compute the target size field based on strategies.

        Args:
            strategies: List of remeshing strategies.

        Returns:
            np.ndarray: Size map (N, 4) containing (x, y, z, size).
        """
        # 1. Calculate Current Sizes (Baseline)
        current_sizes = self.get_current_mesh_sizes()

        # Create interpolator for baseline sizes
        baseline_interpolator = NearestNDInterpolator(self.vxyz, current_sizes)

        # 2. Process each strategy independently
        all_coords = []
        all_sizes = []

        for strategy in strategies:
            # Load strategy-specific refinement data
            if strategy.refinement_data is not None:
                # Load data
                if isinstance(strategy.refinement_data, (str, Path)):
                    try:
                        r_data = np.load(strategy.refinement_data)
                    except:  # noqa: E722
                        r_data = np.loadtxt(strategy.refinement_data)
                else:
                    r_data = strategy.refinement_data

                # Ensure 2D array
                if r_data.ndim == 1:
                    r_data = r_data.reshape(1, -1)

                if r_data.size == 0:
                    # Handle empty data gracefully
                    continue

                # Parse coords and values
                if r_data.shape[1] == 3:
                    eval_coords = r_data
                    eval_values = None
                elif r_data.shape[1] == 4:
                    eval_coords = r_data[:, :3]
                    eval_values = r_data[:, 3]
                else:
                    raise ValueError(
                        f"refinement_data must be (N, 3) or (N, 4), got shape {r_data.shape}"
                    )

                # Interpolate baseline sizes at evaluation points
                eval_sizes = baseline_interpolator(eval_coords)

            else:
                # Use mesh nodes
                eval_coords = self.vxyz
                eval_values = None
                eval_sizes = current_sizes.copy()

            # Apply strategy or direct size specification
            # Calculate metric/size
            if strategy.func is not None:
                if eval_values is None:
                    result = strategy.func(eval_coords, None)
                else:
                    result = strategy.func(eval_coords, eval_values)
            else:
                # If no func provided, use the data values directly as metric
                if eval_values is not None:
                    result = eval_values
                else:
                    # No data and no func? This is likely an error or identity
                    # For now, let's assume it's an error unless we want to support coordinate-based identity
                    raise ValueError(
                        "Strategy must have either 'func' or 'refinement_data' with values."
                    )

            # RemeshingStrategy: apply factor based on metric and interpolation function
            eval_sizes = strategy.apply(eval_sizes, result)

            # Store results
            all_coords.append(eval_coords)
            all_sizes.append(eval_sizes)

        # 3. Combine all strategy results
        if len(all_coords) == 0:
            # No strategies or all empty, use baseline
            size_map = np.column_stack([self.vxyz, current_sizes])
        else:
            # Build combined size map from refinement data only
            # We need to handle overlapping points by taking minimum size
            combined_coords = np.vstack(all_coords)
            combined_sizes = np.hstack(all_sizes)

            # Remove duplicates, keeping minimum size
            # Build KDTree to find duplicates
            tree = cKDTree(combined_coords)

            # Query all points against themselves
            groups = tree.query_ball_point(combined_coords, r=1e-5)

            # Process groups to keep minimum size
            processed = set()
            final_coords = []
            final_sizes = []

            for i, group in enumerate(groups):
                if i in processed:
                    continue

                # Mark all in group as processed
                for idx in group:
                    processed.add(idx)

                # Take minimum size in group
                min_size = min(combined_sizes[idx] for idx in group)
                final_coords.append(combined_coords[i])
                final_sizes.append(min_size)

            size_map = np.column_stack([np.array(final_coords), np.array(final_sizes)])

        return size_map

    def to_msh(self, output_file: Path, format: str = "msh") -> None:
        """Save current mesh to file.

        Args:
            output_file: Output file path
            format: File format (default: "msh")
        """
        self.model_manager.save_to_mesh(output_file, format)

    def finalize(self):
        """Finalize resources."""
        if self._owns_model:
            self.model_manager.finalize()


class RemeshGMSH(Remesher):
    """Remesher using GMSH backend."""

    def remesh(
        self,
        input_mesh: Path | meshio.Mesh | ModelManager,
        geometry_file: Path,
        strategies: list[RemeshingStrategy],
        dim: int = 2,
        global_2D_algorithm: int = 6,
        global_3D_algorithm: int = 1,
        mesh_element_order: int = 1,
        optimization_flags: tuple[tuple[str, int]] | None = None,
        output_mesh: Path | None = None,
        default_characteristic_length: float = 1.0,
        cad_settings: CADSettings | None = None,
    ) -> np.ndarray:
        """Remesh using GMSH.

        Args:
            input_mesh: Input mesh source.
            geometry_file: Path to .xao geometry file.
            strategies: List of remeshing strategies.
            dim: Dimension.
            global_2D_algorithm: GMSH 2D algo.
            global_3D_algorithm: GMSH 3D algo.
            mesh_element_order: Element order.
            optimization_flags: Optimization flags.
            output_mesh: Optional output path (if not provided, result is in model).
            default_characteristic_length: Default characteristic length.
            cad_settings: CAD settings of ``geometry_file``; required when
                the ``.xao`` carries no meshwell metadata, must match it
                otherwise (see :func:`meshwell.mesh.mesh`).

        Returns:
            np.ndarray: The generated size map.
        """
        settings = self._resolve_intake(geometry_file, cad_settings)

        # 1. Load Mesh Data (Baseline)
        self._load_mesh_data(input_mesh)

        # 2. Compute Size Field
        size_map = self.compute_size_field(strategies)

        # 3. Create DirectSizeSpecification
        spec = DirectSizeSpecification(
            refinement_data=size_map,
            apply_to=None,  # Global
        )

        # 4. Run Mesh Generation using meshwell.mesh.Mesh logic
        # We can reuse self.model_manager
        from meshwell.mesh import Mesh

        # Initialize model for remeshing
        self.model_manager.ensure_initialized(str(self.model_manager.filename))

        # Load geometry
        gmsh.open(str(geometry_file))
        gmsh.model.occ.synchronize()

        # Create Mesh instance attached to our model manager
        mesh_gen = Mesh(model=self.model_manager, cad_settings=settings)

        # Process geometry with our resolution spec
        # Note: We pass the spec as a global resolution spec
        resolution_specs = {None: [spec]}

        # We need to adapt the call to process_geometry or similar
        # mesh() utility does this:
        # mesh_generator.process_geometry(...)

        mesh_gen.process_geometry(
            dim=dim,
            resolution_specs=resolution_specs,
            global_2D_algorithm=global_2D_algorithm,
            global_3D_algorithm=global_3D_algorithm,
            mesh_element_order=mesh_element_order,
            optimization_flags=optimization_flags,
            verbosity=self.verbosity,
            default_characteristic_length=default_characteristic_length,
        )

        if output_mesh:
            self.to_msh(output_mesh)

        return size_map


class RemeshMMG(Remesher):
    """Remesher using MMG backend.

    Incompatible with structured sweep bands: MMG would remesh the band and
    destroy its tensor structure, and the ``__sweep|`` marker groups are
    stripped from ``.msh`` files so this cannot be detected here. For models
    with structured bands use :func:`remesh_structured`. (Extension path:
    mark band cells as Required entities in the Medit file so MMG adapts
    only the unstructured complement.)
    """

    def __init__(
        self,
        mmg_executable: str = "mmg2d_O3",
        verbosity: int = 0,
        n_threads: int = cpu_count(),
        filename: str = "temp_remesh_mmg",
        model: ModelManager | None = None,
    ):
        super().__init__(n_threads, filename, model, verbosity)
        self.mmg_executable = mmg_executable

    def _find_executable(self) -> str:
        """Find the MMG executable."""
        # Check if absolute path
        if Path(self.mmg_executable).is_file():
            return self.mmg_executable

        # Check in system PATH
        path = shutil.which(self.mmg_executable)
        if path:
            return path

        # Check in .venv/bin (common pattern)
        venv_bin = Path(__file__).parent.parent / ".venv" / "bin" / self.mmg_executable
        if venv_bin.is_file():
            return str(venv_bin)

        # Check in .venv/bin relative to cwd (fallback)
        cwd_venv_bin = Path.cwd() / ".venv" / "bin" / self.mmg_executable
        if cwd_venv_bin.is_file():
            return str(cwd_venv_bin)

        raise FileNotFoundError(
            f"MMG executable '{self.mmg_executable}' not found. "
            "Please install MMG (e.g. `pip install pymmg`) or provide the full path."
        )

    def _write_sol_file(self, path: Path, sizes: np.ndarray, dim: int = 2) -> None:
        """Write solution file (.sol) for MMG."""
        with path.open("w") as f:
            f.write("MeshVersionFormatted 2\n")
            f.write(f"Dimension {dim}\n")
            f.write("SolAtVertices\n")
            f.write(f"{len(sizes)}\n")
            f.write("1 1\n")  # 1 solution field, type 1 (scalar)
            for s in sizes:
                f.write(f"{s:.16f}\n")
            f.write("End\n")

    def remesh(
        self,
        input_mesh: Path | meshio.Mesh,
        output_mesh: Path,
        strategies: list[RemeshingStrategy],
        dim: int = 2,
        hmin: float | None = None,
        hmax: float | None = None,
        hausd: float | None = None,
        hgrad: float | None = None,
    ) -> np.ndarray:
        """Remesh using MMG.

        Args:
            input_mesh: Input mesh file or object.
            output_mesh: Output mesh file path.
            strategies: List of remeshing strategies.
            dim: Dimension (2 or 3).
            hmin: Minimum edge size (MMG parameter).
            hmax: Maximum edge size (MMG parameter).
            hausd: Hausdorff distance (MMG parameter).
            hgrad: Gradation (MMG parameter).

        Returns:
            np.ndarray: The generated size map.
        """
        # 1. Load Mesh Data (Baseline)
        self._load_mesh_data(input_mesh)

        # 2. Compute Size Field
        # Note: compute_size_field returns (x, y, z, size) at combined points
        # But MMG needs sizes at the INPUT MESH vertices.
        # So we need to interpolate the result of compute_size_field back to self.vxyz

        target_size_map = self.compute_size_field(strategies)

        # Interpolate back to mesh nodes
        interpolator = NearestNDInterpolator(
            target_size_map[:, :3], target_size_map[:, 3]
        )
        final_sizes = interpolator(self.vxyz)

        # Also respect the baseline size at the node itself (already handled in compute_size_field logic?
        # compute_size_field combines strategies. If strategies didn't cover a node, it used baseline.
        # But if we have refinement data points, compute_size_field returns those points.
        # So interpolation is correct.)

        # 3. Run MMG
        executable = self._find_executable()

        # Load meshio object for writing
        if isinstance(input_mesh, (str, Path)):
            mesh = meshio.read(input_mesh)
        elif isinstance(input_mesh, meshio.Mesh):
            mesh = input_mesh
        else:
            # If loaded via ModelManager, we need to export it to meshio
            # This is a bit inefficient but robust
            with tempfile.NamedTemporaryFile(suffix=".msh") as tmp:
                self.model_manager.save_to_mesh(tmp.name)
                mesh = meshio.read(tmp.name)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            mesh_file = tmp_path / "input.mesh"
            sol_file = tmp_path / "input.sol"
            out_file = tmp_path / "output.mesh"

            # Prepare mesh for writing
            points_to_write = mesh.points
            if dim == 2:
                if points_to_write.shape[1] == 3:
                    if not np.allclose(points_to_write[:, 2], 0) and self.verbosity > 0:
                        print(
                            "Warning: 2D remeshing requested but mesh has non-zero Z coordinates. Projecting to Z=0."
                        )
                    points_to_write = points_to_write[:, :2]
            elif dim == 3 and points_to_write.shape[1] == 2:
                # Pad with Z=0 if 2D points provided for 3D
                points_to_write = np.column_stack(
                    [points_to_write, np.zeros(len(points_to_write))]
                )

            # Create temporary mesh object with correct dimension
            # Handle physical tags: Map gmsh:physical -> medit:ref
            cell_data_in = {}
            if "gmsh:physical" in mesh.cell_data:
                cell_data_in["medit:ref"] = mesh.cell_data["gmsh:physical"]
            elif "gmsh:geometrical" in mesh.cell_data:
                cell_data_in["medit:ref"] = mesh.cell_data["gmsh:geometrical"]

            # Merge cell blocks of the same type to avoid tag loss in Medit format
            unique_types = []
            for cell_block in mesh.cells:
                if cell_block.type not in unique_types:
                    unique_types.append(cell_block.type)

            merged_cells = []
            merged_cell_data = {}
            if "medit:ref" in cell_data_in:
                merged_cell_data["medit:ref"] = []

            for ctype in unique_types:
                data_list = [c.data for c in mesh.cells if c.type == ctype]
                merged_cells.append((ctype, np.concatenate(data_list)))
                if "medit:ref" in cell_data_in:
                    tag_list = [
                        cell_data_in["medit:ref"][i]
                        for i, c in enumerate(mesh.cells)
                        if c.type == ctype
                    ]
                    merged_cell_data["medit:ref"].append(np.concatenate(tag_list))

            mesh_to_write = meshio.Mesh(
                points_to_write, merged_cells, cell_data=merged_cell_data
            )

            # Write mesh in Medit format
            meshio.write(mesh_file, mesh_to_write, file_format="medit")

            # Write solution
            self._write_sol_file(sol_file, final_sizes, dim=dim)

            # Build command
            cmd = [
                executable,
                "-in",
                str(mesh_file),
                "-sol",
                str(sol_file),
                "-out",
                str(out_file),
            ]

            # Check for MMGRemeshingStrategy parameters
            for strategy in strategies:
                if isinstance(strategy, MMGRemeshingStrategy):
                    if strategy.hausd is not None:
                        hausd = strategy.hausd
                    if strategy.hgrad is not None:
                        hgrad = strategy.hgrad

            if hmin is not None:
                cmd.extend(["-hmin", str(hmin)])
            if hmax is not None:
                cmd.extend(["-hmax", str(hmax)])
            if hausd is not None:
                cmd.extend(["-hausd", str(hausd)])
            if hgrad is not None:
                cmd.extend(["-hgrad", str(hgrad)])
            if self.verbosity > 0:
                cmd.extend(["-v", str(self.verbosity)])
            else:
                cmd.extend(["-v", "-1"])  # Silent

            # Run
            try:
                subprocess.run(  # noqa: S603
                    cmd, check=True, capture_output=(self.verbosity == 0)
                )
            except subprocess.CalledProcessError as e:
                if self.verbosity == 0 and e.stdout:
                    print(e.stdout.decode())
                    if e.stderr:
                        print(e.stderr.decode())
                raise RuntimeError(f"MMG failed with exit code {e.returncode}") from e

            # Read result
            new_mesh = meshio.read(out_file)

            # Restore physical tags: medit:ref -> gmsh:physical
            if "medit:ref" in new_mesh.cell_data:
                new_mesh.cell_data["gmsh:physical"] = new_mesh.cell_data["medit:ref"]
                del new_mesh.cell_data["medit:ref"]

            # Add dummy geometrical tags to satisfy GMSH 2.2 format expectations
            new_mesh.cell_data["gmsh:geometrical"] = [
                np.zeros(len(c), dtype=int) for c in new_mesh.cells
            ]

            # Restore field_data (names)
            new_mesh.field_data = mesh.field_data

            # Save to requested output
            new_points = new_mesh.points
            if new_points.shape[1] == 2:
                new_points = np.column_stack([new_points, np.zeros(len(new_points))])

            new_mesh.points = new_points

            # Explicitly specify format if .msh to avoid ANSYS confusion
            file_format = None
            if output_mesh.suffix == ".msh":
                file_format = "gmsh22"

            meshio.write(output_mesh, new_mesh, file_format=file_format)

            return target_size_map


def remesh_gmsh(
    input_mesh: Path | meshio.Mesh | ModelManager,
    geometry_file: Path,
    output_mesh: Path,
    strategies: list[RemeshingStrategy],
    dim: int = 2,
    global_2D_algorithm: int = 6,
    global_3D_algorithm: int = 1,
    mesh_element_order: int = 1,
    optimization_flags: tuple[tuple[str, int]] | None = None,
    verbosity: int = 0,
    n_threads: int = cpu_count(),
    filename: str = "temp_remesh",
    model: ModelManager | None = None,
    default_characteristic_length: float = 1.0,
    cad_settings: CADSettings | None = None,
) -> np.ndarray:
    """Utility function for adaptive mesh refinement using GMSH."""
    remesher = RemeshGMSH(
        n_threads=n_threads,
        filename=filename,
        model=model,
        verbosity=verbosity,
    )

    size_map = remesher.remesh(
        input_mesh=input_mesh,
        geometry_file=geometry_file,
        strategies=strategies,
        dim=dim,
        global_2D_algorithm=global_2D_algorithm,
        global_3D_algorithm=global_3D_algorithm,
        mesh_element_order=mesh_element_order,
        optimization_flags=optimization_flags,
        output_mesh=output_mesh,
        default_characteristic_length=default_characteristic_length,
        cad_settings=cad_settings,
    )

    remesher.finalize()

    return size_map


def remesh_mmg(
    input_mesh: Path | meshio.Mesh,
    output_mesh: Path,
    strategies: list[RemeshingStrategy],
    dim: int = 2,
    mmg_executable: str = "mmg2d_O3",
    verbosity: int = 0,
    **kwargs,
) -> np.ndarray:
    """Utility function for adaptive mesh refinement using MMG."""
    remesher = RemeshMMG(mmg_executable=mmg_executable, verbosity=verbosity)
    return remesher.remesh(
        input_mesh=input_mesh,
        output_mesh=output_mesh,
        strategies=strategies,
        dim=dim,
        **kwargs,
    )


def compute_total_size_map(
    input_mesh: Path | meshio.Mesh | ModelManager,
    strategies: list[RemeshingStrategy],
    n_threads: int = cpu_count(),
    verbosity: int = 0,
) -> np.ndarray:
    """Compute the combined size map from multiple strategies without remeshing."""
    remesher = Remesher(n_threads=n_threads, verbosity=verbosity)
    remesher._load_mesh_data(input_mesh)
    return remesher.compute_size_field(strategies)


def gradation_limit_size_map(
    size_map: np.ndarray, max_ratio: float, n_passes: int = 5, k: int = 8
) -> np.ndarray:
    """Cap size growth over the point cloud: h_j <= h_i + (max_ratio - 1) * d_ij.

    gmsh has no built-in gradation control on background fields, so a raw
    size map with a large jump would mesh abrupt unstructured transitions.
    """
    # ponytail: Jacobi passes over a kNN graph, not a full graph-distance
    # closure; raise n_passes or switch to a Dijkstra-style sweep if shocks
    # survive in practice.
    size_map = np.asarray(size_map, dtype=float).copy()
    n = len(size_map)
    if n < 2:
        return size_map
    coords = size_map[:, :3]
    g = max_ratio - 1.0
    tree = cKDTree(coords)
    kk = min(k, n)
    dist, idx = tree.query(coords, k=kk)
    for _ in range(n_passes):
        cap = (size_map[idx[:, 1:], 3] + g * dist[:, 1:]).min(axis=1)
        new = np.minimum(size_map[:, 3], cap)
        if np.allclose(new, size_map[:, 3]):
            break
        size_map[:, 3] = new
    return size_map


def _sweep_band_frames(sweeps, tol: float = 1e-4) -> dict:
    """Recover axis-aligned band frames from the loaded .xao's sweep groups.

    Uses the synthetic ``__sweepsrc|name`` attachment curves for the
    tangential axis/extent and attachment coordinate, and the
    ``__sweep|name|side|i`` faces for per-side signs and interior split
    coordinates (boundary-curve endpoints strictly inside the extent -- the
    same signal ``_stamp_all`` reads). Requires the geometry to be loaded in
    the current gmsh model.

    ``tol`` excludes extent-end endpoints. It must exceed the CAD
    perturbation (if non-zero, e.g. 1e-5), which offsets face corner points from the
    src-curve bounding box by up to that amount; splits closer than ``tol``
    to a band end are dropped.
    """
    from meshwell.structured.sweep2d import discover_sweeps

    # STL-tight bounds: OCC's default getBoundingBox pads by the shape
    # tolerance (~2e-3 here); the STL path tracks the geometry to within the
    # CAD perturbation. Mirrors the orchestrator's own bounds handling.
    gmsh.option.setNumber("Geometry.OCCBoundsUseStl", 1)
    discovered = discover_sweeps()
    frames: dict = {}
    for sweep in sweeps:
        d = discovered.get(sweep.name)
        if d is None or not d["src_curves"]:
            continue
        boxes = [gmsh.model.getBoundingBox(1, t) for t in d["src_curves"]]
        lo = [min(b[i] for b in boxes) for i in range(3)]
        hi = [max(b[i + 3] for b in boxes) for i in range(3)]
        t_axis = 0 if (hi[0] - lo[0]) >= (hi[1] - lo[1]) else 1
        n_axis = 1 - t_axis
        t0, t1 = lo[t_axis], hi[t_axis]
        n0 = 0.5 * (lo[n_axis] + hi[n_axis])
        signs: dict = {}
        splits: set = set()
        for (side, _i), ftag in d["faces"].items():
            b = gmsh.model.getBoundingBox(2, ftag)
            center_n = 0.5 * (b[n_axis] + b[n_axis + 3])
            signs[side] = 1.0 if center_n > n0 else -1.0
            # Interior BOP/seam splits: boundary-curve endpoint coordinates
            # strictly inside the extent (a face's bbox cannot see seam
            # splits on a single-face band).
            for _dc, ct in gmsh.model.getBoundary(
                [(2, ftag)], oriented=False, recursive=False
            ):
                for _dp, pt in gmsh.model.getBoundary([(1, abs(ct))], oriented=False):
                    t = gmsh.model.getValue(0, abs(pt), [])[t_axis]
                    if t0 + tol < t < t1 - tol:
                        splits.add(round(t - t0, 12))
        frames[sweep.name] = SweepAdaptContext(
            thickness=dict(sweep.thickness),
            t_axis=t_axis,
            n_axis=n_axis,
            t0=t0,
            t1=t1,
            n0=n0,
            normal_sign=signs,
            required_ts=tuple(sorted(splits)),
        )
    return frames


def _is_adaptable_sweep_spec(spec) -> bool:
    return isinstance(spec, StructuredSweepResolutionSpec) and not isinstance(
        spec, StructuredExtrusionResolutionSpec
    )


def _snap_offsets(offs, q: float) -> list:
    """Round offsets onto the point_tolerance lattice the stamping kernel uses.

    ``_stamp_all`` snaps every stamped node onto the ``point_tolerance`` grid
    (sweep2d.py: ``grid_xy = round(grid_xy / q) * q``) to undo the CAD
    perturbation, so a returned spec carrying the raw (continuous)
    equidistributed offsets would not describe the mesh that was produced.
    Snap here so the returned arrays round-trip exactly.
    """
    if q <= 0:
        return [float(v) for v in offs]
    return [float(round(float(v) / q) * q) for v in offs]


def _tangential_key(spec, ctx):
    """Group key: sweeps sharing extent + equal tangential share one new array."""
    if isinstance(spec.tangential, (int, float)):
        tkey = ("scalar", float(spec.tangential))
    else:
        tkey = ("array", tuple(round(float(v), 12) for v in spec.tangential))
    return (ctx.t_axis, round(ctx.t0, 9), round(ctx.t1, 9), tkey)


def remesh_structured(
    input_mesh,
    geometry_file,
    sweeps,
    resolution_specs,
    strategies=None,
    *,
    size_map=None,
    change_max: float = 2.0,
    max_ratio: float = 1.3,
    output_mesh=None,
    dim: int = 2,
    global_2D_algorithm: int = 6,
    global_3D_algorithm: int = 1,
    mesh_element_order: int = 1,
    optimization_flags=None,
    default_characteristic_length: float = 1.0,
    verbosity: int = 0,
    n_threads: int = cpu_count(),
    filename: str = "temp_remesh_structured",
    model: ModelManager | None = None,
    cad_settings: CADSettings | None = None,
) -> tuple[meshio.Mesh, dict]:
    """Adapt every ResolutionSpec to a size map and regenerate from the .xao.

    Structured bands are adapted by spec regeneration (tangential/normal
    arrays re-derived from the size map); unstructured regions follow the
    gradation-limited map via a global DirectSizeSpecification. Returns
    ``(new_mesh, adapted_resolution_specs)``; feed the specs back in to
    iterate the adaptive loop.

    ``cad_settings`` follows the :func:`meshwell.mesh.mesh` intake rules:
    required when ``geometry_file`` carries no meshwell metadata, must match
    the embedded settings otherwise.
    """
    if size_map is None:
        if strategies is None:
            raise ValueError("Provide either strategies or size_map.")
        size_map = compute_total_size_map(
            input_mesh, strategies, n_threads=n_threads, verbosity=verbosity
        )
    size_map = gradation_limit_size_map(np.asarray(size_map, float), max_ratio)

    remesher = RemeshGMSH(
        n_threads=n_threads, filename=filename, model=model, verbosity=verbosity
    )
    settings = remesher._resolve_intake(geometry_file, cad_settings)
    try:
        remesher.model_manager.ensure_initialized(str(remesher.model_manager.filename))
        # OCCBoundsUseStl must be ON *before* open: gmsh only builds the
        # STL tessellation (which tracks the CAD-perturbed geometry to ~1e-5)
        # at import time. Setting it afterwards leaves _sweep_band_frames with
        # loose shape-tolerance bounds (~2e-3), which misclassify the band's
        # own end corners as interior splits and corrupt the tangential grid.
        gmsh.option.setNumber("Geometry.OCCBoundsUseStl", 1)
        gmsh.open(str(geometry_file))
        gmsh.model.occ.synchronize()
        frames = _sweep_band_frames(sweeps)
        # Turn STL bounds back off before meshing (they feed the meshing
        # octree otherwise).
        gmsh.option.setNumber("Geometry.OCCBoundsUseStl", 0)
        # The stamping kernel snaps stamped nodes onto this lattice, so the
        # returned spec arrays are snapped to match (see _snap_offsets).
        q = settings.point_tolerance
        # gmsh (4.15) deadlocks in generate() when a PostView background field
        # (the DirectSizeSpecification below) is meshed multi-threaded -- the
        # sweep stamping's generate(1) never returns. The whole point of this
        # driver is the background field, so pin gmsh meshing single-threaded
        # (matches the n_threads=1 the DirectSizeSpecification tests use).
        for _opt in (
            "General.NumThreads",
            "Mesh.MaxNumThreads1D",
            "Mesh.MaxNumThreads2D",
            "Mesh.MaxNumThreads3D",
        ):
            gmsh.option.setNumber(_opt, 1)

        new_specs = {k: list(v) for k, v in resolution_specs.items()}

        # Non-sweep specs adapt independently; sweep specs are collected for
        # shared-tangential grouping.
        sweep_items = []  # (key, index, spec, ctx)
        for key, specs in new_specs.items():
            for i, spec in enumerate(specs):
                if _is_adaptable_sweep_spec(spec) and key in frames:
                    sweep_items.append((key, i, spec, frames[key]))
                else:
                    specs[i] = spec.adapt(
                        size_map, change_max=change_max, max_ratio=max_ratio
                    )

        # Group sweeps by shared tangential (the existing stacking convention),
        # min-merge their tangential targets, emit ONE shared array per group.
        groups: dict = {}
        for item in sweep_items:
            groups.setdefault(_tangential_key(item[2], item[3]), []).append(item)
        for members in groups.values():
            adapted = [
                _adapt_sweep_arrays(spec, size_map, ctx, change_max, max_ratio)
                for (_k, _i, spec, ctx) in members
            ]
            with_signal = [a for a in adapted if a is not None]
            if not with_signal:
                continue  # no signal over any band in the group: all unchanged
            t_off = with_signal[0][1]
            h_t = np.min([a[2] for a in with_signal], axis=0)
            required = tuple(
                sorted({r for (_k, _i, _s, ctx) in members for r in ctx.required_ts})
            )
            t_new = _snap_offsets(
                _insert_required(_equidistribute(t_off, h_t), required), q
            )
            for (key, i, spec, _ctx), arrays in zip(members, adapted):
                result = copy.copy(spec)
                result.tangential = list(t_new)
                # no-signal members keep their normal arrays but must adopt
                # the shared tangential for seam conformity
                if arrays is not None:
                    result.normal = {
                        side: _snap_offsets(off, q) for side, off in arrays[0].items()
                    }
                else:
                    result.normal = spec.normal
                new_specs[key][i] = result

        # Unstructured regions follow the (gradation-limited) map. Reuse an
        # existing global DirectSizeSpecification (already updated via adapt
        # above) instead of stacking a new one each iteration.
        has_direct = any(
            isinstance(s, DirectSizeSpecification)
            for ss in new_specs.values()
            for s in ss
        )
        if not has_direct:
            new_specs.setdefault(None, []).append(
                DirectSizeSpecification(refinement_data=size_map, apply_to=None)
            )

        from meshwell.mesh import Mesh

        mesh_gen = Mesh(model=remesher.model_manager)
        new_mesh = mesh_gen.process_geometry(
            dim=dim,
            resolution_specs=new_specs,
            global_2D_algorithm=global_2D_algorithm,
            global_3D_algorithm=global_3D_algorithm,
            mesh_element_order=mesh_element_order,
            optimization_flags=optimization_flags,
            verbosity=verbosity,
            default_characteristic_length=default_characteristic_length,
        )
        if output_mesh is not None:
            remesher.to_msh(Path(output_mesh))
    finally:
        remesher.finalize()

    return new_mesh, new_specs
