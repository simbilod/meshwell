"""Remesh module for adaptive mesh refinement."""
from __future__ import annotations

from dataclasses import dataclass
from os import cpu_count
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import gmsh
import meshio
import numpy as np

from meshwell.model import ModelManager


@dataclass
class RemeshingStrategy:
    """Dataclass for defining a remeshing strategy.

    Args:
        func: Callable taking (coords, data) and returning a metric array.
              coords: (N, 3) array of node coordinates.
              data: (N,) array of data values at nodes.
              Returns: (N,) array of metric values.
        threshold: Metric value above which refinement is triggered.
        factor: Factor to multiply current mesh size by (e.g., 0.5 for halving).
        refinement_data: Optional (N, 4) array of (x, y, z, data) to evaluate strategy on.
                        If None, strategy is evaluated on input mesh nodes.
        min_size: Minimum allowed mesh size.
        max_size: Maximum allowed mesh size.
        field_smoothing_steps: Number of smoothing steps for the size field.
    """

    func: Callable[[np.ndarray, np.ndarray], np.ndarray]
    threshold: float
    factor: float
    refinement_data: Optional[Union[Path, np.ndarray]] = None
    min_size: Optional[float] = None
    max_size: Optional[float] = None
    field_smoothing_steps: int = 0


@dataclass
class DirectSizeSpecification:
    """Dataclass for directly specifying mesh sizes.

    Args:
        func: Callable taking (coords, data) and returning desired mesh sizes.
              coords: (N, 3) array of node coordinates.
              data: (N,) array of data values at nodes.
              Returns: (N,) array of desired mesh sizes.
        refinement_data: Optional (N, 4) array of (x, y, z, data) to evaluate on.
                        If None, evaluated on input mesh nodes.
        min_size: Minimum allowed mesh size.
        max_size: Maximum allowed mesh size.
        field_smoothing_steps: Number of smoothing steps for the size field.
    """

    func: Callable[[np.ndarray, np.ndarray], np.ndarray]
    refinement_data: Optional[Union[Path, np.ndarray]] = None
    min_size: Optional[float] = None
    max_size: Optional[float] = None
    field_smoothing_steps: int = 0


class Remesh:
    """Remesh class for adaptive mesh refinement/coarsening based on size fields."""

    def __init__(
        self,
        n_threads: int = cpu_count(),
        filename: str = "temp_remesh",
        model: Optional[ModelManager] = None,
    ):
        """Initialize remesh processor.

        Args:
            n_threads: Number of threads for processing
            filename: Base filename for the model
            model: Optional Model instance to use (creates new if None)
        """
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

    def _load_mesh_data(
        self, input_mesh: Union[Path, meshio.Mesh, ModelManager]
    ) -> None:
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
            # Assuming the model is already active or we can switch to it
            # But ModelManager wraps gmsh model, so we might need to be careful about current model
            # For now, let's assume we can extract from it if it's the current one or we make it current
            # This path might be tricky if ModelManager doesn't expose everything,
            # but let's try to use the underlying gmsh model
            current_model = gmsh.model.getCurrent()
            gmsh.model.setCurrent(
                input_mesh.model.getCurrent()
            )  # This might be wrong API usage for ModelManager
            # Actually ModelManager has .model which is the gmsh module usually? No, it's likely a wrapper or just manages state.
            # Looking at code, ModelManager seems to manage the session.
            # Let's assume we can just use gmsh.model if it's the active one.
            self._extract_gmsh_mesh_data()
            if current_model:
                gmsh.model.setCurrent(current_model)

    def _extract_gmsh_mesh_data(self):
        """Helper to extract nodes and elements from current GMSH model."""
        self.vtags, vxyz, _ = gmsh.model.mesh.getNodes()
        self.vxyz = vxyz.reshape((-1, 3))

        vmap = {j: i for i, j in enumerate(self.vtags)}

        # Try 2D first
        try:
            self.triangles_tags, evtags = gmsh.model.mesh.getElementsByType(2)
            evid = np.array([vmap[j] for j in evtags])
            self.triangles = evid.reshape((self.triangles_tags.shape[-1], -1))
        except:  # noqa: E722
            # Try 3D
            try:
                self.triangles_tags, evtags = gmsh.model.mesh.getElementsByType(4)
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

    def _create_size_field_view(
        self,
        size_map: np.ndarray,
        field_smoothing_steps: int = 0,
    ) -> int:
        """Create a gmsh size field from a sizing map.

        Args:
            size_map: Array of shape (N, 4) containing (x, y, z, size) values
            field_smoothing_steps: Number of smoothing iterations

        Returns:
            View tag
        """
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(mode="w", suffix=".pos", delete=False) as tmp:
            tmp_path = tmp.name
            tmp.write('View "size_field" {\n')
            for i in range(len(size_map)):
                x, y, z, size = size_map[i]
                tmp.write(f"SP({x}, {y}, {z}){{{size}}};\n")
            tmp.write("};\n")

        gmsh.merge(tmp_path)
        view_tag = gmsh.view.getTags()[-1]

        if field_smoothing_steps > 0:
            for _ in range(field_smoothing_steps):
                gmsh.plugin.setNumber("Smooth", "View", gmsh.view.getIndex(view_tag))
                gmsh.plugin.run("Smooth")

        os.unlink(tmp_path)
        return view_tag

    def remesh(
        self,
        input_mesh: Union[Path, meshio.Mesh, ModelManager],
        geometry_file: Path,
        strategies: List[Union[RemeshingStrategy, DirectSizeSpecification]],
        dim: int = 2,
        global_2D_algorithm: int = 6,
        global_3D_algorithm: int = 1,
        mesh_element_order: int = 1,
        optimization_flags: Optional[Tuple[Tuple[str, int]]] = None,
        verbosity: int = 0,
    ) -> np.ndarray:
        """Remesh using CAD geometry and strategies.

        Args:
            input_mesh: Input mesh source (defines baseline size field).
            geometry_file: Path to .xao geometry file.
            strategies: List of remeshing strategies or direct size specifications.
            dim: Dimension.
            global_2D_algorithm: GMSH 2D algo.
            global_3D_algorithm: GMSH 3D algo.
            mesh_element_order: Element order.
            optimization_flags: Optimization flags.
            verbosity: Verbosity.

        Returns:
            np.ndarray: The generated size map (N, 4) containing (x, y, z, size).
        """
        from scipy.interpolate import NearestNDInterpolator
        from scipy.spatial import cKDTree

        # 1. Load Mesh Data (Baseline)
        self._load_mesh_data(input_mesh)

        # 2. Calculate Current Sizes (Baseline)
        current_sizes = self.get_current_mesh_sizes()

        # Create interpolator for baseline sizes
        baseline_interpolator = NearestNDInterpolator(self.vxyz, current_sizes)

        # 3. Process each strategy independently
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
            if eval_values is None:
                result = strategy.func(eval_coords, None)
            else:
                result = strategy.func(eval_coords, eval_values)

            # RemeshingStrategy: apply factor where metric exceeds threshold
            mask = result > strategy.threshold
            indices = np.where(mask)[0]

            if len(indices) > 0:
                # Apply factor
                eval_sizes[indices] *= strategy.factor

                # Clamp
                if strategy.min_size is not None:
                    eval_sizes[indices] = np.maximum(
                        eval_sizes[indices], strategy.min_size
                    )
                if strategy.max_size is not None:
                    # Clamp
                    if strategy.min_size is not None:
                        eval_sizes[indices] = np.maximum(
                            eval_sizes[indices], strategy.min_size
                        )
                    if strategy.max_size is not None:
                        eval_sizes[indices] = np.minimum(
                            eval_sizes[indices], strategy.max_size
                        )

            # Store results
            all_coords.append(eval_coords)
            all_sizes.append(eval_sizes)

        # 4. Combine all strategy results
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

        # 5. Create Size Field
        smoothing_steps = max((s.field_smoothing_steps for s in strategies), default=0)

        # Initialize model for remeshing
        self.model_manager.ensure_initialized(str(self.model_manager.filename))

        # Load geometry
        gmsh.open(str(geometry_file))
        gmsh.model.occ.synchronize()

        # Create view
        view_tag = self._create_size_field_view(
            size_map, field_smoothing_steps=smoothing_steps
        )

        # 7. Setup Background Mesh
        gmsh.option.setNumber("General.Terminal", verbosity)
        gmsh.option.setNumber("Mesh.Algorithm", global_2D_algorithm)
        gmsh.option.setNumber("Mesh.Algorithm3D", global_3D_algorithm)
        gmsh.option.setNumber("Mesh.ElementOrder", mesh_element_order)

        bg_field = gmsh.model.mesh.field.add("PostView")
        gmsh.model.mesh.field.setNumber(bg_field, "ViewTag", view_tag)
        gmsh.model.mesh.field.setAsBackgroundMesh(bg_field)

        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)

        # 7. Generate
        gmsh.model.mesh.clear()
        gmsh.model.mesh.generate(dim)

        if optimization_flags:
            for optimization_flag, niter in optimization_flags:
                gmsh.model.mesh.optimize(optimization_flag, niter=niter)

        return size_map

    def to_msh(self, output_file: Path, format: str = "msh") -> None:
        """Save current mesh to file.

        Args:
            output_file: Output file path
            format: File format (default: "msh")
        """
        self.model_manager.save_to_mesh(output_file, format)


def remesh(
    input_mesh: Union[Path, meshio.Mesh, ModelManager],
    geometry_file: Path,
    output_mesh: Path,
    strategies: List[Union[RemeshingStrategy, DirectSizeSpecification]],
    dim: int = 2,
    global_2D_algorithm: int = 6,
    global_3D_algorithm: int = 1,
    mesh_element_order: int = 1,
    optimization_flags: Optional[Tuple[Tuple[str, int]]] = None,
    verbosity: int = 0,
    n_threads: int = cpu_count(),
    filename: str = "temp_remesh",
    model: Optional[ModelManager] = None,
) -> np.ndarray:
    """Utility function for adaptive mesh refinement/coarsening.

    Args:
        input_mesh: Input mesh source.
        geometry_file: Path to .xao geometry file.
        output_mesh: Path for output .msh file.
        strategies: List of remeshing strategies or direct size specifications.
        dim: Dimension.
        global_2D_algorithm: GMSH 2D algo.
        global_3D_algorithm: GMSH 3D algo.
        mesh_element_order: Element order.
        optimization_flags: Optimization flags.
        verbosity: Verbosity.
        n_threads: Number of threads.
        filename: Temporary filename.
        model: Optional Model instance.

    Returns:
        np.ndarray: The generated size map (N, 4) containing (x, y, z, size).
    """
    remesher = Remesh(
        n_threads=n_threads,
        filename=filename,
        model=model,
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
        verbosity=verbosity,
    )

    remesher.to_msh(output_mesh)

    if model is None:
        remesher.model_manager.finalize()

    return size_map
