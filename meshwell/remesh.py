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
