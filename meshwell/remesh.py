"""Remesh module for adaptive mesh refinement."""
from __future__ import annotations

from os import cpu_count
from pathlib import Path

import gmsh
import numpy as np

from meshwell.model import ModelManager


class Remesh:
    """Remesh class for adaptive mesh refinement/coarsening based on size fields."""

    def __init__(
        self,
        n_threads: int = cpu_count(),
        filename: str = "temp_remesh",
        model: ModelManager | None = None,
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

    def _load_mesh_data(self, input_mesh: Path) -> None:
        """Load mesh data from .msh file for size field interpolation.

        Args:
            input_mesh: Path to input .msh file
        """
        # Ensure gmsh is initialized (mesh() may have finalized it)
        if not gmsh.isInitialized():
            gmsh.initialize()

        # Create a temporary model to load the mesh
        temp_model = "temp_mesh_reader"

        # Store current model if any
        try:
            current_model = gmsh.model.getCurrent()
        except:  # noqa: E722
            current_model = None

        # Add and use temporary model
        gmsh.model.add(temp_model)
        gmsh.model.setCurrent(temp_model)
        gmsh.merge(str(input_mesh))

        # Extract node data
        self.vtags, vxyz, _ = gmsh.model.mesh.getNodes()
        self.vxyz = vxyz.reshape((-1, 3))

        # Create vertex tag to index mapping
        vmap = {j: i for i, j in enumerate(self.vtags)}

        # Extract element data (triangles for 2D, tetrahedra for 3D)
        # Try 2D first
        try:
            self.triangles_tags, evtags = gmsh.model.mesh.getElementsByType(2)
            evid = np.array([vmap[j] for j in evtags])
            self.triangles = evid.reshape((self.triangles_tags.shape[-1], -1))
        except:  # noqa: E722
            # Try 3D if 2D fails
            self.triangles_tags, evtags = gmsh.model.mesh.getElementsByType(4)
            evid = np.array([vmap[j] for j in evtags])
            self.triangles = evid.reshape((self.triangles_tags.shape[-1], -1))

        # Remove the temporary model and restore previous model
        gmsh.model.remove()
        if current_model:
            gmsh.model.setCurrent(current_model)

    def get_current_mesh_sizes(self, input_mesh: Path) -> np.ndarray:
        """Calculate current mesh size at each node (average connected edge length).

        Args:
            input_mesh: Path to input .msh file

        Returns:
            Array of size values at mesh nodes
        """
        # Load mesh data if not already loaded
        if self.vxyz is None:
            self._load_mesh_data(input_mesh)

        # Calculate edge lengths
        # We need to reconstruct edges from elements
        edges = set()

        # Helper to add edges from elements
        def add_edges(elements):
            if elements is None:
                return
            for elem in elements:
                # Create edges (i, j) with i < j
                num_nodes = len(elem)
                for i in range(num_nodes):
                    n1, n2 = elem[i], elem[(i + 1) % num_nodes]
                    if n1 > n2:
                        n1, n2 = n2, n1
                    edges.add((n1, n2))

        add_edges(self.triangles)

        # Also check for 3D elements (tetrahedra) if triangles are empty or we want full connectivity
        # But _load_mesh_data only loads 2D or 3D elements into self.triangles (which is a bit of a misnomer in 3D case in the original code,
        # let's check _load_mesh_data implementation).
        # In _load_mesh_data:
        # self.triangles = evid.reshape((self.triangles_tags.shape[-1], -1))
        # It loads either 2D or 3D elements into self.triangles.
        # If it's 3D (tetrahedra), the logic above (looping over nodes) creates edges on the surface of the tet,
        # but we need all edges.
        # For a tetrahedron (4 nodes), edges are (0,1), (0,2), (0,3), (1,2), (1,3), (2,3).
        # The loop above does (0,1), (1,2), (2,3), (3,0) which misses (0,2) and (1,3).

        # Let's refine the edge extraction based on element type
        # But self.triangles just contains indices. We don't strictly know if it's triangles or tets from just the array
        # without checking the dimension or the source.
        # However, _load_mesh_data tries 2D then 3D.

        # Let's re-implement edge extraction to be more robust
        edges = set()
        if self.triangles is not None:
            num_nodes_per_elem = self.triangles.shape[1]

            if num_nodes_per_elem == 3:  # Triangles
                for elem in self.triangles:
                    e1 = tuple(sorted((elem[0], elem[1])))
                    e2 = tuple(sorted((elem[1], elem[2])))
                    e3 = tuple(sorted((elem[2], elem[0])))
                    edges.add(e1)
                    edges.add(e2)
                    edges.add(e3)
            elif num_nodes_per_elem == 4:  # Tetrahedra
                for elem in self.triangles:
                    # 6 edges
                    edges.add(tuple(sorted((elem[0], elem[1]))))
                    edges.add(tuple(sorted((elem[0], elem[2]))))
                    edges.add(tuple(sorted((elem[0], elem[3]))))
                    edges.add(tuple(sorted((elem[1], elem[2]))))
                    edges.add(tuple(sorted((elem[1], elem[3]))))
                    edges.add(tuple(sorted((elem[2], elem[3]))))

        # Initialize accumulators
        node_sum_lengths = np.zeros(len(self.vxyz))
        node_counts = np.zeros(len(self.vxyz))

        # Compute lengths and accumulate
        for n1, n2 in edges:
            # Indices in self.vxyz are n1, n2 (assuming 0-based from _load_mesh_data)
            # _load_mesh_data maps tags to indices 0..N-1
            # self.triangles contains these indices

            p1 = self.vxyz[n1]
            p2 = self.vxyz[n2]
            length = np.linalg.norm(p1 - p2)

            node_sum_lengths[n1] += length
            node_counts[n1] += 1
            node_sum_lengths[n2] += length
            node_counts[n2] += 1

        # Compute average
        # Avoid division by zero for isolated nodes (though unlikely in a valid mesh)
        with np.errstate(divide="ignore", invalid="ignore"):
            node_sizes = node_sum_lengths / node_counts
            node_sizes[node_counts == 0] = 0.0

        return node_sizes

    def _create_size_field_from_map(
        self,
        size_map: np.ndarray,
        field_smoothing_steps: int = 5,
    ) -> int:
        """Create a gmsh size field from a sizing map.

        Args:
            size_map: Array of shape (N, 4) containing (x, y, z, size) values
            field_smoothing_steps: Number of smoothing iterations for the size field

        Returns:
            View tag for the created size field (list-based)
        """
        # Create a .pos file directly with the size data
        # This creates a list-based view that works with background meshing
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(mode="w", suffix=".pos", delete=False) as tmp:
            tmp_path = tmp.name

            # Write .pos file header
            tmp.write('View "size_field" {\n')

            # Write scalar points (SP) for each point in the size map
            # Format: SP(x, y, z){value};
            for i in range(len(size_map)):
                x, y, z, size = size_map[i]
                tmp.write(f"SP({x}, {y}, {z}){{{size}}};\n")

            tmp.write("};\n")

        # Load the .pos file as a list-based view
        gmsh.merge(tmp_path)

        # Get the tag of the loaded view
        view_tag = gmsh.view.getTags()[-1]

        # Apply smoothing if requested
        if field_smoothing_steps > 0:
            for _ in range(field_smoothing_steps):
                gmsh.plugin.setNumber("Smooth", "View", gmsh.view.getIndex(view_tag))
                gmsh.plugin.run("Smooth")

        # Clean up temporary file
        os.unlink(tmp_path)

        return view_tag

    def _interpolate_size_to_nodes(self, size_map: np.ndarray) -> np.ndarray:
        """Interpolate size values from sizing map to mesh nodes.

        Args:
            size_map: Array of shape (N, 4) containing (x, y, z, size) values

        Returns:
            Array of size values at mesh nodes
        """
        # Extract coordinates and sizes from size map
        map_coords = size_map[:, :3]
        map_sizes = size_map[:, 3]

        # For each node, find nearest points in size map and interpolate
        node_sizes = np.zeros(len(self.vxyz))

        for i, node_coord in enumerate(self.vxyz):
            # Compute distances to all size map points
            distances = np.linalg.norm(map_coords - node_coord, axis=1)

            # Use inverse distance weighting with k nearest neighbors
            k = min(5, len(map_coords))  # Use up to 5 nearest points
            nearest_indices = np.argpartition(distances, k - 1)[:k]
            nearest_distances = distances[nearest_indices]
            nearest_sizes = map_sizes[nearest_indices]

            # Avoid division by zero
            epsilon = 1e-10
            weights = 1.0 / (nearest_distances + epsilon)
            weights /= weights.sum()

            # Weighted average
            node_sizes[i] = np.sum(weights * nearest_sizes)

        return node_sizes

    def _load_geometry(self, geometry_file: Path) -> None:
        """Load CAD geometry from .xao file.

        Args:
            geometry_file: Path to .xao geometry file
        """
        # Load the geometry - use open() instead of merge() for .xao files
        # This properly imports the CAD model
        gmsh.open(str(geometry_file))

        # Synchronize is not needed after open(), it's automatic
        # But we can sync just to be safe
        gmsh.model.occ.synchronize()

    def remesh(
        self,
        input_mesh: Path,
        geometry_file: Path,
        size_map: np.ndarray,
        dim: int = 2,
        global_2D_algorithm: int = 6,
        global_3D_algorithm: int = 1,
        mesh_element_order: int = 1,
        field_smoothing_steps: int = 5,
        optimization_flags: tuple[tuple[str, int]] | None = None,
        verbosity: int = 0,
    ) -> None:
        """Remesh using CAD geometry and a size field from an existing mesh.

        Args:
            input_mesh: Path to input .msh file (used to extract node positions for size field interpolation)
            geometry_file: Path to .xao geometry file (defines the domain to mesh)
            size_map: Array of shape (N, 4) containing (x, y, z, size) values
            dim: Dimension of mesh (2 or 3)
            global_2D_algorithm: GMSH 2D meshing algorithm
            global_3D_algorithm: GMSH 3D meshing algorithm
            mesh_element_order: Element order
            field_smoothing_steps: Number of smoothing iterations for size field
            optimization_flags: Mesh optimization flags
            verbosity: GMSH verbosity level
        """
        # Load mesh data for size field interpolation
        self._load_mesh_data(input_mesh)

        # Initialize model and load CAD geometry
        self.model_manager.ensure_initialized(str(self.model_manager.filename))
        self._load_geometry(geometry_file)

        # IMPORTANT: Create size field AFTER loading geometry
        # because gmsh.open() clears all views
        view_tag = self._create_size_field_from_map(
            size_map,
            field_smoothing_steps=field_smoothing_steps,
        )

        # Set mesh options
        gmsh.option.setNumber("General.Terminal", verbosity)
        gmsh.option.setNumber("Mesh.Algorithm", global_2D_algorithm)
        gmsh.option.setNumber("Mesh.Algorithm3D", global_3D_algorithm)
        gmsh.option.setNumber("Mesh.ElementOrder", mesh_element_order)

        # Create background mesh field from the view
        bg_field = gmsh.model.mesh.field.add("PostView")
        gmsh.model.mesh.field.setNumber(bg_field, "ViewTag", view_tag)
        gmsh.model.mesh.field.setAsBackgroundMesh(bg_field)

        # Turn off default meshing size sources
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)

        # Clear the old mesh
        gmsh.model.mesh.clear()

        # Generate new mesh
        gmsh.model.mesh.generate(dim)

        # Apply optimization if requested
        if optimization_flags:
            for optimization_flag, niter in optimization_flags:
                gmsh.model.mesh.optimize(optimization_flag, niter=niter)

    def to_msh(self, output_file: Path, format: str = "msh") -> None:
        """Save current mesh to file.

        Args:
            output_file: Output file path
            format: File format (default: "msh")
        """
        self.model_manager.save_to_mesh(output_file, format)

    def load_mesh(self, input_mesh: Path) -> None:
        """Load mesh data from .msh file.

        Args:
            input_mesh: Path to input .msh file
        """
        self._load_mesh_data(input_mesh)

    def _interpolate_edges(
        self,
        coords: np.ndarray,
        connectivity: np.ndarray | None,
        sizes: np.ndarray,
    ) -> np.ndarray:
        """Interpolate size field along edges to ensure sufficient density.

        If the edge length is larger than the target size at the nodes,
        intermediate points are added with interpolated sizes.

        Args:
            coords: Array of shape (N, 3) containing node coordinates
            connectivity: Array of shape (M, 2) containing edges (node indices).
                          If None, edges are extracted from loaded mesh.
            sizes: Array of shape (N,) containing target sizes at nodes

        Returns:
            Array of shape (K, 4) containing (x, y, z, size) for all points
            (original + interpolated)
        """
        # If connectivity is not provided, try to extract from loaded mesh
        if connectivity is None:
            if self.triangles is None:
                # No connectivity available, return original points
                return np.column_stack([coords, sizes])

            # Extract edges from self.triangles (similar to get_current_mesh_sizes)
            edges_set = set()
            num_nodes_per_elem = self.triangles.shape[1]
            if num_nodes_per_elem == 3:  # Triangles
                for elem in self.triangles:
                    edges_set.add(tuple(sorted((elem[0], elem[1]))))
                    edges_set.add(tuple(sorted((elem[1], elem[2]))))
                    edges_set.add(tuple(sorted((elem[2], elem[0]))))
            elif num_nodes_per_elem == 4:  # Tetrahedra
                for elem in self.triangles:
                    edges_set.add(tuple(sorted((elem[0], elem[1]))))
                    edges_set.add(tuple(sorted((elem[0], elem[2]))))
                    edges_set.add(tuple(sorted((elem[0], elem[3]))))
                    edges_set.add(tuple(sorted((elem[1], elem[2]))))
                    edges_set.add(tuple(sorted((elem[1], elem[3]))))
                    edges_set.add(tuple(sorted((elem[2], elem[3]))))
            connectivity = np.array(list(edges_set))

        new_points = []

        # Add original points
        for i in range(len(coords)):
            new_points.append([coords[i][0], coords[i][1], coords[i][2], sizes[i]])

        # Iterate over edges and add intermediate points
        for n1, n2 in connectivity:
            p1 = coords[n1]
            p2 = coords[n2]
            s1 = sizes[n1]
            s2 = sizes[n2]

            length = np.linalg.norm(p1 - p2)

            # Determine target size for this edge
            # We can use the average size or the minimum size
            target_size = (s1 + s2) / 2.0

            # If edge is too long, subdivide
            if length > target_size:
                num_segments = int(np.ceil(length / target_size))

                for k in range(1, num_segments):
                    t = k / num_segments
                    p_interp = (1 - t) * p1 + t * p2
                    s_interp = (1 - t) * s1 + t * s2
                    new_points.append([p_interp[0], p_interp[1], p_interp[2], s_interp])

        return np.array(new_points)

    def refine_by_gradient(
        self,
        coords: np.ndarray,
        data: np.ndarray,
        current_sizes: np.ndarray,
        threshold: float,
        factor: float = 0.5,
        min_size: float | None = None,
        max_size: float | None = None,
        k: int = 4,
    ) -> np.ndarray:
        """Refine mesh based on the gradient of the data.

        Args:
            coords: Array of shape (N, 3) containing node coordinates
            data: Array of shape (N,) containing data values at nodes
            current_sizes: Array of shape (N,) containing current mesh sizes
            threshold: Gradient magnitude threshold for refinement
            factor: Factor to multiply current size by (default: 0.5)
            min_size: Minimum allowed mesh size
            max_size: Maximum allowed mesh size
            k: Number of neighbors for gradient estimation (default: 4)

        Returns:
            Array of shape (K, 4) containing (x, y, z, new_size)
        """
        from scipy.spatial import cKDTree

        tree = cKDTree(coords)
        k = max(k, 4)
        distances, indices = tree.query(coords, k=k)

        new_sizes = current_sizes.copy()

        for i in range(len(coords)):
            neighbor_indices = indices[i]
            neighbor_coords = coords[neighbor_indices]
            neighbor_data = data[neighbor_indices]

            local_coords = neighbor_coords - coords[i]

            try:
                grad, _, _, _ = np.linalg.lstsq(
                    local_coords, neighbor_data - data[i], rcond=None
                )
                grad_norm = np.linalg.norm(grad)

                if grad_norm > threshold:
                    new_sizes[i] *= factor
            except:  # noqa: E722
                pass

        if min_size is not None:
            new_sizes = np.maximum(new_sizes, min_size)
        if max_size is not None:
            new_sizes = np.minimum(new_sizes, max_size)

        # Interpolate edges to ensure density
        return self._interpolate_edges(coords, None, new_sizes)

    def refine_by_value_difference(
        self,
        coords: np.ndarray,
        connectivity: np.ndarray,
        data: np.ndarray,
        current_sizes: np.ndarray,
        threshold: float,
        factor: float = 0.5,
        min_size: float | None = None,
        max_size: float | None = None,
    ) -> np.ndarray:
        """Refine mesh based on value difference between connected nodes.

        Args:
            coords: Array of shape (N, 3) containing node coordinates
            connectivity: Array of shape (M, 2) containing edges (node indices)
            data: Array of shape (N,) containing data values at nodes
            current_sizes: Array of shape (N,) containing current mesh sizes
            threshold: Value difference threshold for refinement
            factor: Factor to multiply current size by (default: 0.5)
            min_size: Minimum allowed mesh size
            max_size: Maximum allowed mesh size

        Returns:
            Array of shape (K, 4) containing (x, y, z, new_size)
        """
        new_sizes = current_sizes.copy()

        for n1, n2 in connectivity:
            diff = abs(data[n1] - data[n2])

            if diff > threshold:
                new_sizes[n1] *= factor
                new_sizes[n2] *= factor

        if min_size is not None:
            new_sizes = np.maximum(new_sizes, min_size)
        if max_size is not None:
            new_sizes = np.minimum(new_sizes, max_size)

        return self._interpolate_edges(coords, connectivity, new_sizes)

    def refine_by_error(
        self,
        coords: np.ndarray,
        data: np.ndarray,
        current_sizes: np.ndarray,
        total_error_fraction: float = 0.8,
        factor: float = 0.5,
        min_size: float | None = None,
        max_size: float | None = None,
    ) -> np.ndarray:
        """Refine mesh based on error contribution.

        Args:
            coords: Array of shape (N, 3) containing node coordinates
            data: Array of shape (N,) containing error values at nodes
            current_sizes: Array of shape (N,) containing current mesh sizes
            total_error_fraction: Fraction of total error to target
            factor: Factor to multiply current size by (default: 0.5)
            min_size: Minimum allowed mesh size
            max_size: Maximum allowed mesh size

        Returns:
            Array of shape (K, 4) containing (x, y, z, new_size)
        """
        if np.any(data < 0):
            raise ValueError("Error data must be non-negative.")

        new_sizes = current_sizes.copy()

        sorted_indices = np.argsort(data)[::-1]
        sorted_error = data[sorted_indices]

        total_error = np.sum(data)
        cumulative_error = np.cumsum(sorted_error)

        cutoff_index = np.searchsorted(
            cumulative_error, total_error * total_error_fraction
        )

        nodes_to_refine = sorted_indices[: cutoff_index + 1]
        new_sizes[nodes_to_refine] *= factor

        if min_size is not None:
            new_sizes = np.maximum(new_sizes, min_size)
        if max_size is not None:
            new_sizes = np.minimum(new_sizes, max_size)

        return self._interpolate_edges(coords, None, new_sizes)


def remesh(
    input_mesh: Path,
    geometry_file: Path,
    output_mesh: Path,
    size_map: np.ndarray,
    dim: int = 2,
    global_2D_algorithm: int = 6,
    global_3D_algorithm: int = 1,
    mesh_element_order: int = 1,
    field_smoothing_steps: int = 5,
    optimization_flags: tuple[tuple[str, int]] | None = None,
    verbosity: int = 0,
    n_threads: int = cpu_count(),
    filename: str = "temp_remesh",
    model: ModelManager | None = None,
) -> None:
    """Utility function for adaptive mesh refinement/coarsening.

    Args:
        input_mesh: Path to input .msh file (used for size field interpolation)
        geometry_file: Path to .xao geometry file (defines domain to mesh)
        output_mesh: Path for output .msh file
        size_map: Array of shape (N, 4) containing (x, y, z, size) values
        dim: Dimension of mesh (2 or 3)
        global_2D_algorithm: GMSH 2D meshing algorithm (default: 6)
        global_3D_algorithm: GMSH 3D meshing algorithm (default: 1)
        mesh_element_order: Element order (default: 1)
        field_smoothing_steps: Number of smoothing iterations for size field
        optimization_flags: Mesh optimization flags
        verbosity: GMSH verbosity level
        n_threads: Number of threads for processing
        filename: Temporary filename for GMSH model
        model: Optional Model instance to use (creates new if None)
    """
    remesher = Remesh(
        n_threads=n_threads,
        filename=filename,
        model=model,
    )

    # Perform remeshing
    remesher.remesh(
        input_mesh=input_mesh,
        geometry_file=geometry_file,
        size_map=size_map,
        dim=dim,
        global_2D_algorithm=global_2D_algorithm,
        global_3D_algorithm=global_3D_algorithm,
        mesh_element_order=mesh_element_order,
        field_smoothing_steps=field_smoothing_steps,
        optimization_flags=optimization_flags,
        verbosity=verbosity,
    )

    # Save to file
    remesher.to_msh(output_mesh)

    # Finalize if we created the model
    if model is None:
        remesher.model_manager.finalize()
