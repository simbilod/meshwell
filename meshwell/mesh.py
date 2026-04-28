"""Mesh class definition."""
from __future__ import annotations

import contextlib
import tempfile
from collections.abc import Sequence
from os import cpu_count
from pathlib import Path

import meshio
import numpy as np

import gmsh
from meshwell._mesh_entity import _MeshEntity
from meshwell.model import ModelManager


def _normalize_algo(alg: int | Sequence[int]) -> tuple[int, ...]:
    """Coerce a single algorithm id or a sequence of fallbacks into a tuple."""
    if isinstance(alg, Sequence) and not isinstance(alg, (str, bytes)):
        seq = tuple(alg)
        if not seq:
            raise ValueError("algorithm sequence must not be empty")
        return seq
    return (int(alg),)


def _pair_algos(
    algos_2d: tuple[int, ...], algos_3d: tuple[int, ...]
) -> list[tuple[int, int]]:
    """Pair 2D/3D fallback sequences position-wise, padding with the last value."""
    n = max(len(algos_2d), len(algos_3d))
    return [
        (algos_2d[min(i, len(algos_2d) - 1)], algos_3d[min(i, len(algos_3d) - 1)])
        for i in range(n)
    ]


def _filter_msh_to_seam_groups(msh_path):
    """Rewrite msh_path keeping only physical groups whose name starts with '_seam___'.

    Reads via meshio, drops cells / cell_data / field_data not associated with seam
    groups, writes back to msh_path. Used by phase-1 workers in the distributed
    pipeline to emit only the seam surface mesh.

    Notes:
        meshio's ``field_data`` maps physical-group name -> ``[tag_id, dim]``.
        Gmsh tag ids are scoped per-dimension, so the filter keeps a (dim,
        tag_id) pair set rather than a flat tag-id set (different physical
        groups in different dimensions can share a tag id).

        We must write with ``file_format="gmsh22"``: meshio 5.3 .msh inference
        picks gmsh4 which silently drops ``field_data`` on round-trip.
    """
    # Hard-coded type -> topological dim. Covers everything meshwell emits
    # (triangles for surface groups, tetras for volume groups, plus higher-
    # order variants). Unknown types are skipped rather than guessed.
    _TYPE_DIM = {
        "vertex": 0,
        "line": 1,
        "line3": 1,
        "triangle": 2,
        "triangle6": 2,
        "triangle7": 2,
        "quad": 2,
        "quad8": 2,
        "quad9": 2,
        "tetra": 3,
        "tetra10": 3,
        "hexahedron": 3,
        "hexahedron20": 3,
        "hexahedron27": 3,
        "wedge": 3,
        "wedge15": 3,
        "wedge18": 3,
        "pyramid": 3,
        "pyramid13": 3,
        "pyramid14": 3,
    }

    msh_path = Path(msh_path)
    m = meshio.read(msh_path)
    keep_field_data = {
        name: np.asarray(tag_dim)
        for name, tag_dim in (m.field_data or {}).items()
        if name.startswith("_seam___")
    }
    if not keep_field_data:
        # Nothing to keep — write a minimal empty mesh.
        empty = meshio.Mesh(points=m.points[:0], cells=[])
        meshio.write(msh_path, empty, file_format="gmsh22")
        return

    # (dim, tag_id) pairs we want to keep. tag ids are per-dimension in gmsh.
    keep_pairs = {(int(arr[1]), int(arr[0])) for arr in keep_field_data.values()}

    new_cells = []
    new_cell_data = {k: [] for k in (m.cell_data or {})}
    gmsh_phys_arr = (m.cell_data or {}).get("gmsh:physical")
    for i, cellblock in enumerate(m.cells):
        if gmsh_phys_arr is None or i >= len(gmsh_phys_arr):
            continue
        block_dim = _TYPE_DIM.get(cellblock.type)
        if block_dim is None:
            # Unknown cell type — skip rather than guess.
            continue
        gmsh_phys = gmsh_phys_arr[i]
        mask = np.array(
            [(block_dim, int(t)) in keep_pairs for t in gmsh_phys],
            dtype=bool,
        )
        if not mask.any():
            continue
        idx = np.where(mask)[0]
        new_cells.append(meshio.CellBlock(cellblock.type, cellblock.data[idx]))
        for k in new_cell_data:
            new_cell_data[k].append(np.asarray(m.cell_data[k][i])[idx])

    out = meshio.Mesh(
        points=m.points,
        cells=new_cells,
        cell_data=new_cell_data,
        field_data=keep_field_data,
    )
    # Force gmsh22 format: meshio's gmsh4 writer drops field_data on .msh
    # files in our version (5.3.5), but gmsh22 round-trips it correctly.
    meshio.write(msh_path, out, file_format="gmsh22")


class Mesh:
    """Mesh class for generating meshes from cad models."""

    def __init__(
        self,
        n_threads: int = cpu_count(),
        filename: str = "temp",
        model: ModelManager | None = None,
        point_tolerance: float | None = None,
    ):
        """Initialize mesh generator.

        Args:
            n_threads: Number of threads for processing
            filename: Base filename for the model
            model: Optional Model instance to use (creates new if None)
            point_tolerance: Optional point tolerance for the model

        """
        # Use provided model or create new one
        if model is None:
            self.model_manager = ModelManager(
                n_threads=n_threads,
                filename=filename,
                point_tolerance=point_tolerance,
            )
            self._owns_model = True
        else:
            self.model_manager = model
            self._owns_model = False

    def _initialize_model(self, input_file: Path | None = None) -> None:
        """Initialize GMSH model and optionally load .xao file."""
        self.model_manager.ensure_initialized("temp")

        if input_file is not None:
            input_file = Path(input_file)
            gmsh.merge(str(input_file.with_suffix(".xao")))

    def _initialize_mesh_settings(
        self,
        verbosity: int,
        default_characteristic_length: float,
        global_2D_algorithm: int | Sequence[int],
        global_3D_algorithm: int | Sequence[int],
        gmsh_version: float | None,
        mesh_element_order: int = 1,
    ) -> None:
        """Initialize basic mesh settings.

        If either algorithm argument is a sequence, the first value is applied
        here and process_mesh handles fallback to later ones.
        """
        gmsh.option.setNumber("General.Terminal", verbosity)
        gmsh.option.setNumber(
            "Mesh.CharacteristicLengthMax", default_characteristic_length
        )
        gmsh.option.setNumber("Mesh.Algorithm", _normalize_algo(global_2D_algorithm)[0])
        gmsh.option.setNumber(
            "Mesh.Algorithm3D", _normalize_algo(global_3D_algorithm)[0]
        )
        gmsh.option.setNumber("Mesh.ElementOrder", mesh_element_order)
        if gmsh_version is not None:
            gmsh.option.setNumber("Mesh.MshFileVersion", gmsh_version)
        self.model_manager.sync_model()

    def _apply_periodic_boundaries(
        self, periodic_entities: list[tuple[str, str]]
    ) -> None:
        """Apply periodic boundary conditions."""
        mapping = {
            self.model_manager.model.getPhysicalName(dimtag[0], dimtag[1]): dimtag
            for dimtag in self.model_manager.model.getPhysicalGroups()
        }

        for label1, label2 in periodic_entities:
            if label1 not in mapping or label2 not in mapping:
                continue

            self._set_periodic_pair(mapping, label1, label2)

    def _set_periodic_pair(self, mapping: dict, label1: str, label2: str) -> None:
        """Set up periodic boundary pair."""
        tags1 = self.model_manager.model.getEntitiesForPhysicalGroup(*mapping[label1])
        tags2 = self.model_manager.model.getEntitiesForPhysicalGroup(*mapping[label2])

        vector1 = self.model_manager.model.occ.getCenterOfMass(
            mapping[label1][0], tags1[0]
        )
        vector2 = self.model_manager.model.occ.getCenterOfMass(
            mapping[label1][0], tags2[0]
        )
        vector = np.subtract(vector1, vector2)

        self.model_manager.model.mesh.setPeriodic(
            mapping[label1][0],
            tags1,
            tags2,
            [1, 0, 0, vector[0], 0, 1, 0, vector[1], 0, 0, 1, vector[2], 0, 0, 0, 1],
        )

    def _apply_mesh_refinement(
        self,
        background_remeshing_file: Path | None | None,
        boundary_delimiter: str,
        resolution_specs: dict,
        interface_delimiter: str = "___",
        _global_physical_names: list[str] | None = None,
    ) -> None:
        """Apply mesh refinement settings.

        TODO: enable simultaneous background mesh and entity-based refinement
        """
        if background_remeshing_file is None:
            self._apply_entity_refinement(
                boundary_delimiter,
                resolution_specs,
                interface_delimiter,
                _global_physical_names=_global_physical_names,
            )
        else:
            self._apply_background_refinement()

    def get_top_physical_names(self) -> list[str]:
        """Get all physical names of dimension dim from the GMSH model.

        Returns:
            List of physical names as strings

        """
        return self.model_manager.get_top_physical_names()

    def get_all_physical_names(self) -> list[str]:
        """Get all physical names from the GMSH model.

        Returns:
            List of physical names as strings

        """
        return self.model_manager.get_physical_names()

    def get_physical_dimtags(self, physical_name: str) -> list[tuple[int, int]]:
        """Get the dimtags associated with a physical group name.

        Args:
            physical_name: Name of the physical group

        Returns:
            List of (dim, tag) tuples for entities in the physical group

        """
        return self.model_manager.get_physical_dimtags(physical_name)

    def _restore_structured_sweeps(self, blueprint: dict) -> None:
        """Analyze structured sweeps."""
        import logging

        logger = logging.getLogger(__name__)

        top_names = self.get_top_physical_names()
        for p_name in top_names:
            if p_name in blueprint and blueprint[p_name].get("mesh_structured", False):
                logger.warning(
                    f"Physical group '{p_name}' requested mesh_structured=True. "
                    "Note: Native OpenCASCADE structured sweeping via 'removeAllDuplicates' "
                    "cannot guarantee conformality in hybrid Gmsh meshes without stripping ExtrudeParams. "
                    "Proceeding with unstructured or recombined fallback."
                )

    def _recover_labels_from_cad(
        self,
        resolution_specs: dict,
        interface_delimiter: str = "___",
        boundary_delimiter: str = "None",
    ) -> tuple[list, dict]:
        """Recover labeled entities from loaded CAD model.

        Args:
            resolution_specs: Dictionary mapping physical names to resolution specifications
            blueprint: mapping between entity and extrusion type
            interface_delimiter: String used to separate names in an interface
            boundary_delimiter: String used to identify boundary entities

        Returns:
            Tuple of (final_entity_list, final_entity_dict)
        """
        final_entity_list = []
        final_entity_dict = {}

        # We address entities by "named" physicals (not default):
        top_physical_names = self.get_top_physical_names()
        all_physical_names = self.get_all_physical_names()

        # Collect all base names from interfaces to handle removed entities (voids)
        base_names = set(all_physical_names)
        for other_p_name in all_physical_names:
            parts = other_p_name.split(interface_delimiter)
            if len(parts) == 2:
                if parts[0]:
                    base_names.add(parts[0])
                if parts[1] and parts[1] != boundary_delimiter:
                    base_names.add(parts[1])

        for index, physical_name in enumerate(sorted(base_names)):
            resolutions = resolution_specs.get(physical_name, [])

            if not resolutions and physical_name not in top_physical_names:
                continue

            entities = _MeshEntity(
                index=index,
                physical_name=physical_name,
                model=self.model_manager.model,
                dimtags=self.get_physical_dimtags(physical_name=physical_name),
                resolutions=resolutions,
            )
            entities.update_boundaries()

            # Recover interfaces and boundaries from physical groups
            # We look for groups named "A___B" or "B___A" where A is physical_name
            for other_p_name in all_physical_names:
                parts = other_p_name.split(interface_delimiter)
                if len(parts) == 2:
                    if parts[0] == physical_name:
                        suffix = parts[1]
                        dimtags = self.get_physical_dimtags(other_p_name)
                        if dimtags:
                            for i_dim, i_tag in dimtags:
                                if i_dim < entities.dim or entities.dim == -1:
                                    if entities.dim == -1:
                                        entities._explicit_dim = max(
                                            entities._explicit_dim or 0, i_dim + 1
                                        )

                                    if suffix == boundary_delimiter:
                                        entities.mesh_edge_name_interfaces.append(i_tag)
                                    else:
                                        entities.interfaces.append(i_tag)

                                    # Always treat interfaces as boundaries for refinement
                                    entities.boundaries.append(i_tag)

                    elif parts[1] == physical_name:
                        dimtags = self.get_physical_dimtags(other_p_name)
                        if dimtags:
                            for i_dim, i_tag in dimtags:
                                if i_dim < entities.dim or entities.dim == -1:
                                    if entities.dim == -1:
                                        entities._explicit_dim = max(
                                            entities.dim, i_dim + 1
                                        )

                                    entities.interfaces.append(i_tag)

                                    # Always treat interfaces as boundaries for refinement
                                    entities.boundaries.append(i_tag)

            final_entity_list.append(entities)
            final_entity_dict[physical_name] = entities

        return final_entity_list, final_entity_dict

    def _apply_entity_refinement(
        self,
        boundary_delimiter: str,
        resolution_specs: dict,
        interface_delimiter: str = "___",
        _global_physical_names: list[str] | None = None,
    ) -> None:
        """Apply mesh refinement based on entity information.

        Args:
            boundary_delimiter: String used to identify boundary entities
            resolution_specs: Resolution specifications
            interface_delimiter: String used to separate names in an interface
            _global_physical_names: Optional list of physical names that exist
                across the wider distributed model. ResolutionSpec name refs
                (``restrict_to`` / ``sharing`` / ``not_sharing``) that match a
                name in this set but not in the local entity dict are silently
                no-op'd with a warning, instead of being treated as a typo.

        """
        from collections import defaultdict

        # Recover labeled entities from loaded CAD model
        final_entity_list, final_entity_dict = self._recover_labels_from_cad(
            resolution_specs,
            interface_delimiter=interface_delimiter,
            boundary_delimiter=boundary_delimiter,
        )

        # Build reverse indices for performance
        tag_to_entity_names = defaultdict(set)
        for name, entity in final_entity_dict.items():
            # Include tags for all dimensions that this entity covers
            for d in range(entity.dim + 1):
                tags = entity.filter_tags_by_target_dimension(d)
                for tag in tags:
                    tag_to_entity_names[(d, tag)].add(name)

        # Collect all refinement fields
        refinement_field_indices = []

        # Handle Global Specs (key is None)
        if None in resolution_specs:
            for spec in resolution_specs[None]:
                # Apply globally (empty dict means no mass filtering, restrict_to_tags=None means global)
                field_index = spec.apply(
                    self.model_manager.model, {}, restrict_to_tags=None
                )
                refinement_field_indices.append(field_index)

        # Collect constant fields for batching
        constant_collector = defaultdict(lambda: defaultdict(list))

        for entity in final_entity_list:
            refinement_field_indices.extend(
                entity.add_refinement_fields_to_model(
                    final_entity_dict,
                    boundary_delimiter,
                    constant_collector=constant_collector,
                    tag_to_entity_names=tag_to_entity_names,
                    global_physical_names=_global_physical_names,
                )
            )

        # Process constant fields in batches
        for resolution, entity_types in constant_collector.items():
            matheval_field_index = self.model_manager.model.mesh.field.add("MathEval")
            self.model_manager.model.mesh.field.setString(
                matheval_field_index, "F", f"{resolution}"
            )

            restrict_field_index = self.model_manager.model.mesh.field.add("Restrict")
            self.model_manager.model.mesh.field.setNumber(
                restrict_field_index, "InField", matheval_field_index
            )

            for entity_str, tags in entity_types.items():
                self.model_manager.model.mesh.field.setNumbers(
                    restrict_field_index,
                    entity_str,
                    list(set(tags)),
                )
            refinement_field_indices.append(restrict_field_index)

        # If we have refinement fields, create a minimum field
        if refinement_field_indices:
            # Use the smallest element size overall
            min_field_index = self.model_manager.model.mesh.field.add("Min")
            self.model_manager.model.mesh.field.setNumbers(
                min_field_index, "FieldsList", refinement_field_indices
            )
            self.model_manager.model.mesh.field.setAsBackgroundMesh(min_field_index)

        # Turn off default meshing options
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)

    def _apply_background_refinement(self) -> None:
        """Apply mesh refinement based on background mesh."""
        # Create background field from post-processing view
        bg_field = self.model_manager.model.mesh.field.add("PostView")
        self.model_manager.model.mesh.field.setNumber(bg_field, "ViewIndex", 0)
        gmsh.model.mesh.field.setAsBackgroundMesh(bg_field)

        # Turn off default meshing options
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)

    def process_mesh(
        self,
        dim: int,
        global_3D_algorithm: int,
        global_scaling: float,
        verbosity: int,
        optimization_flags: tuple[tuple[str, int]] | None,
    ) -> meshio.Mesh:
        """Generate mesh and return meshio object (no file I/O).

        The caller is expected to have already set Mesh.Algorithm /
        Mesh.Algorithm3D on the gmsh option state (see ``_initialize_mesh_settings``).
        Retry-on-failure logic lives in ``process_geometry`` because a raised
        ``mesh.generate()`` leaves the gmsh runtime in a "busy" state that only
        a full finalize+reinit will clear.
        """
        gmsh.option.setNumber("Mesh.ScalingFactor", global_scaling)
        gmsh.option.setNumber("Mesh.AngleToleranceFacetOverlap", 1e-5)

        if global_3D_algorithm == 1 and verbosity:
            gmsh.logger.start()

        self.model_manager.model.mesh.generate(dim)

        if optimization_flags:
            for optimization_flag, niter in optimization_flags:
                self.model_manager.model.mesh.optimize(optimization_flag, niter=niter)

        # Return mesh object without writing to file
        with contextlib.redirect_stdout(
            None
        ), tempfile.TemporaryDirectory() as tmpdirname:
            temp_mesh_path = f"{tmpdirname}/mesh.msh"
            gmsh.write(temp_mesh_path)
            return meshio.read(temp_mesh_path)

    def save_to_file(self, output_file: Path) -> None:
        """Save current mesh to file.

        Args:
            output_file: Output mesh file path

        """
        self.model_manager.save_to_mesh(output_file)

    def to_msh(self, output_file: Path, format: str = "msh") -> None:
        """Save current mesh to .msh file.

        Args:
            output_file: Output file path (will be suffixed with .format)
            format: File format to use in gmsh

        """
        self.model_manager.save_to_mesh(output_file, format)

    def to_meshio(self) -> meshio.Mesh:
        """Convert current mesh to meshio.Mesh object.

        Returns:
            meshio.Mesh: Current mesh as meshio object

        """
        with contextlib.redirect_stdout(
            None
        ), tempfile.TemporaryDirectory() as tmpdirname:
            temp_mesh_path = f"{tmpdirname}/mesh.msh"
            gmsh.write(temp_mesh_path)
            return meshio.read(temp_mesh_path)

    def load_xao_file(self, input_file: Path) -> None:
        """Load CAD geometry from .xao file.

        Args:
            input_file: Input .xao file path

        """
        self.model_manager.load_from_xao(input_file)

    def process_geometry(
        self,
        dim: int,
        default_characteristic_length: float,
        background_remeshing_file: Path | None = None,
        global_scaling: float = 1.0,
        global_2D_algorithm: int | Sequence[int] = 6,
        global_3D_algorithm: int | Sequence[int] = 1,
        mesh_element_order: int = 1,
        verbosity: int | None = 0,
        periodic_entities: list[tuple[str, str]] | None = None,  # noqa: ARG002
        optimization_flags: tuple[tuple[str, int]] | None = None,
        boundary_delimiter: str = "None",
        resolution_specs: dict = (),
        gmsh_version: float | None = None,
        interface_delimiter: str = "___",
        _global_physical_names: list[str] | None = None,
    ) -> meshio.Mesh:
        """Process loaded geometry into mesh (no file I/O).

        Args:
            dim: Dimension of mesh to generate
            default_characteristic_length: Default mesh size
            background_remeshing_file: Optional background mesh file for refinement
            global_scaling: Global scaling factor
            global_2D_algorithm: GMSH 2D meshing algorithm, or a sequence of
                algorithms to try in order if earlier attempts fail.
            global_3D_algorithm: GMSH 3D meshing algorithm, or a sequence of
                algorithms to try in order if earlier attempts fail.
            mesh_element_order: Element order
            verbosity: GMSH verbosity level
            periodic_entities: List of periodic boundary pairs
            optimization_flags: Mesh optimization flags
            boundary_delimiter: Delimiter for boundary names
            resolution_specs: Mesh resolution specifications
            gmsh_version: GMSH version
            blueprint: mapping between entity and extrusion type
            interface_delimiter: String used to separate names in an interface

        Returns:
            meshio.Mesh: Generated mesh object

        """
        self._initialize_model()

        attempts = _pair_algos(
            _normalize_algo(global_2D_algorithm),
            _normalize_algo(global_3D_algorithm),
        )

        def _run_once(algo2d: int, algo3d: int) -> meshio.Mesh:
            self._initialize_mesh_settings(
                verbosity=verbosity,
                default_characteristic_length=default_characteristic_length,
                global_2D_algorithm=algo2d,
                global_3D_algorithm=algo3d,
                gmsh_version=gmsh_version,
                mesh_element_order=mesh_element_order,
            )
            self._apply_mesh_refinement(
                background_remeshing_file=background_remeshing_file,
                boundary_delimiter=boundary_delimiter,
                resolution_specs=resolution_specs,
                interface_delimiter=interface_delimiter,
                _global_physical_names=_global_physical_names,
            )
            return self.process_mesh(
                dim=dim,
                global_3D_algorithm=algo3d,
                global_scaling=global_scaling,
                verbosity=verbosity,
                optimization_flags=optimization_flags,
            )

        if len(attempts) == 1:
            algo2d, algo3d = attempts[0]
            return _run_once(algo2d, algo3d)

        # Multi-attempt: persist CAD before the first try so we can restore
        # after a failed generate() leaves gmsh in an unrecoverable "busy"
        # state (neither mesh.clear() nor re-setting options releases it).
        with tempfile.TemporaryDirectory() as tmp:
            cad_checkpoint = Path(tmp) / "cad_checkpoint.xao"
            self.model_manager.save_to_xao(cad_checkpoint)

            for attempt_idx, (algo2d, algo3d) in enumerate(attempts):
                try:
                    return _run_once(algo2d, algo3d)
                except Exception as exc:
                    if attempt_idx == len(attempts) - 1:
                        raise
                    print(
                        f"mesh attempt {attempt_idx + 1}/{len(attempts)} "
                        f"(2D={algo2d}, 3D={algo3d}) failed: {exc}. "
                        f"Retrying with next algorithm.",
                        flush=True,
                    )
                    # Full gmsh reset: finalize + reinit + reload CAD.
                    self.model_manager.finalize()
                    self.model_manager.load_from_xao(cad_checkpoint)

        raise RuntimeError("unreachable: retry loop exited without returning")


def mesh(
    dim: int,
    default_characteristic_length: float,
    input_file: Path | None = None,
    output_file: Path | None = None,
    resolution_specs: dict | None = None,
    background_remeshing_file: Path | None = None,
    global_scaling: float = 1.0,
    global_2D_algorithm: int | Sequence[int] = 6,
    global_3D_algorithm: int | Sequence[int] = 1,
    mesh_element_order: int = 1,
    verbosity: int | None = 0,
    periodic_entities: list[tuple[str, str]] | None = None,
    optimization_flags: tuple[tuple[str, int]] | None = None,
    boundary_delimiter: str = "None",
    n_threads: int = cpu_count(),
    filename: str = "temp",
    model: ModelManager | None = None,
    point_tolerance: float | None = None,
    gmsh_version: float | None = None,
    interface_delimiter: str = "___",
    _global_physical_names: list[str] | None = None,
    _emit_only_seam_surfaces: bool = False,
) -> meshio.Mesh | None:
    """Utility function that wraps the Mesh class for easier usage.

    Args:
        dim: Dimension of mesh to generate
        default_characteristic_length: Default mesh size
        input_file: Path to input .xao file
        output_file: Path for output mesh file
        resolution_specs: Mesh resolution specifications
        background_remeshing_file: Optional background mesh file for refinement
        global_scaling: Global scaling factor
        global_2D_algorithm: GMSH 2D meshing algorithm, or a sequence of
            algorithms tried in order with fallback on failure.
        global_3D_algorithm: GMSH 3D meshing algorithm, or a sequence of
            algorithms tried in order with fallback on failure.
        mesh_element_order: Element order
        verbosity: GMSH verbosity level
        periodic_entities: List of periodic boundary pairs
        optimization_flags: Mesh optimization flags
        boundary_delimiter: Delimiter for boundary names
        n_threads: Number of threads to use
        filename: Temporary filename for GMSH model
        model: Optional Model instance to use (creates new if None)
        gmsh_version: GMSH MSH file version (e.g. 2.2 or 4.1)
        point_tolerance: used to set GMSH global variables. Should be similar to used in CAD.
        interface_delimiter: String used to separate names in an interface
        _global_physical_names: Optional list of physical names known across the
            wider distributed model (see :func:`generate_mesh`).
        _emit_only_seam_surfaces: If True, post-process ``output_file`` so that
            only physical groups whose name starts with ``_seam___`` survive.
            Used by phase-1 workers in the distributed-meshing pipeline to emit
            just the seam-surface mesh; everything else is treated as scratch.
            No effect when ``output_file`` is None. Implies a rewrite to gmsh22
            format (meshio's gmsh4 writer drops field_data on round-trip).

    Returns:
        Optional[meshio.Mesh]: Generated mesh object

    """
    mesh_generator = Mesh(
        n_threads=n_threads,
        filename=filename,
        model=model,
        point_tolerance=point_tolerance,
    )

    if resolution_specs is None:
        resolution_specs = {}

    try:
        # Load geometry from file if provided
        if input_file is not None:
            mesh_generator.load_xao_file(input_file)

        # Process geometry into mesh
        mesh_obj = mesh_generator.process_geometry(
            dim=dim,
            background_remeshing_file=background_remeshing_file,
            default_characteristic_length=default_characteristic_length,
            global_scaling=global_scaling,
            global_2D_algorithm=global_2D_algorithm,
            global_3D_algorithm=global_3D_algorithm,
            mesh_element_order=mesh_element_order,
            verbosity=verbosity,
            periodic_entities=periodic_entities,
            optimization_flags=optimization_flags,
            boundary_delimiter=boundary_delimiter,
            resolution_specs=resolution_specs,
            gmsh_version=gmsh_version,
            interface_delimiter=interface_delimiter,
            _global_physical_names=_global_physical_names,
        )

        # Save to file if output file provided
        if output_file is not None:
            mesh_generator.save_to_file(output_file)
            if _emit_only_seam_surfaces:
                _filter_msh_to_seam_groups(output_file)
    finally:
        # Finalize if we created the model -- even on failure, so gmsh
        # state doesn't leak into subsequent test runs / callers.
        if model is None:
            mesh_generator.model_manager.finalize()

    return mesh_obj
