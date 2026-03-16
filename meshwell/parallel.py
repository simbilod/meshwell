"""Parallel meshing utilities for meshwell."""

import copy
import hashlib
import tempfile
import uuid
from typing import Any, Dict, List, Tuple

import gmsh
import meshio
import numpy as np
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon


def _string_to_tag(name: str) -> int:
    """Convert a physical name string to a deterministic positive Gmsh tag ID (int32)."""
    return int(hashlib.md5(name.encode("utf-8")).hexdigest()[:7], 16) + 1  # noqa: S324


def extract_geometries(geom, target_types):
    """Recursively extract geometries of target types from a shapely geometry."""
    if type(geom) in target_types:
        if not geom.is_empty:
            return [geom]
        return []
    if hasattr(geom, "geoms"):
        res = []
        for g in geom.geoms:
            res.extend(extract_geometries(g, target_types))
        return res
    return []


def intersect_entity(entity: Any, mask_poly: Polygon) -> Any:
    """Clones a meshwell entity and intersects its geometries with a mask polygon."""
    from shapely.geometry import LineString, Polygon

    new_ent = copy.copy(entity)
    intersected = False

    if hasattr(entity, "polygons") and hasattr(entity, "extrude"):
        # PolyPrism
        if getattr(entity, "extrude", False):
            geoms = entity.polygons
            if isinstance(
                geoms, list
            ):  # PolyPrism sometimes holds a single polygon, sometimes a MultiPolygon
                geoms = GeometryCollection(geoms)
            res = geoms.intersection(mask_poly)
            valid_polys = extract_geometries(res, (Polygon,))
            if valid_polys:
                new_ent.polygons = (
                    MultiPolygon(valid_polys)
                    if len(valid_polys) > 1
                    else valid_polys[0]
                )
                intersected = True
        else:
            new_buffered = []
            for z, poly in getattr(entity, "buffered_polygons", []):
                res = poly.intersection(mask_poly)
                valid_polys = extract_geometries(res, (Polygon,))
                if valid_polys:
                    new_poly = (
                        MultiPolygon(valid_polys)
                        if len(valid_polys) > 1
                        else valid_polys[0]
                    )
                    new_buffered.append((z, new_poly))
                    intersected = True
            new_ent.buffered_polygons = new_buffered

    elif hasattr(entity, "polygons"):
        # PolySurface
        geoms = entity.polygons
        if isinstance(geoms, list):
            geoms = GeometryCollection(geoms)
        res = geoms.intersection(mask_poly)
        valid_polys = extract_geometries(res, (Polygon,))
        if valid_polys:
            new_ent.polygons = valid_polys  # PolySurface expects a list
            intersected = True

    elif hasattr(entity, "linestrings"):
        # PolyLine
        geoms = entity.linestrings
        if isinstance(geoms, list):
            geoms = GeometryCollection(geoms)
        res = geoms.intersection(mask_poly)
        valid_lines = extract_geometries(res, (LineString,))
        if valid_lines:
            new_ent.linestrings = valid_lines  # PolyLine expects a list
            intersected = True

    if intersected:
        return new_ent
    return None


def decompose_domain(
    entities_list: List, subdomains: List[Polygon], halo_buffer: float
) -> Dict[int, Dict[str, List]]:
    """Decompose a list of meshwell entities into subdomain contents and halos.

    Returns:
        Dict mapping subdomain index to a dict with:
            - 'domain': List of (original_entity, intersected_geometry)
            - 'halo': List of (original_entity, intersected_geometry)
    """
    subdomain_tasks = {}

    for s_idx, subdomain in enumerate(subdomains):
        domain_entities = []
        halo_entities = []

        # Create the halo polygon. cap_style=3 means square buffers for cleaner bounds.
        halo_poly = subdomain.buffer(halo_buffer, cap_style=3)

        for entity in entities_list:
            # Domain intersection
            domain_ent = intersect_entity(entity, subdomain)
            if domain_ent is not None:
                domain_entities.append(domain_ent)

            # Halo intersection (halo only, excluding domain to avoid overlaps if needed)
            halo_only_poly = halo_poly.difference(subdomain)
            if not halo_only_poly.is_empty:
                halo_ent = intersect_entity(entity, halo_only_poly)
                if halo_ent is not None:
                    halo_entities.append(halo_ent)

        subdomain_tasks[s_idx] = {
            "subdomain_poly": subdomain,
            "domain": domain_entities,
            "halo": halo_entities,
        }

    return subdomain_tasks


def _meshing_task(
    task_params: Dict[str, Any],
    default_char_length: float,
    resolution_specs: Dict,
    dim: int,
) -> Tuple[str, Dict[str, np.ndarray]]:
    """The Dask task that runs inside a worker.

    Constructs a local meshwell Model, meshes it, crops out the halo, and returns a filepath to the .msh and its field_data.
    """
    from meshwell.model import ModelManager

    task_id = str(uuid.uuid4())
    model = ModelManager(filename=f"worker_mesh_{task_id}")

    cad_entities = []
    entity_tag_mapping = {}  # To keep track of original physical names
    local_resolutions = dict(resolution_specs) if resolution_specs else {}

    # Add domain entities
    for ent in task_params["domain"]:
        cad_entities.append(ent)
        if hasattr(ent, "physical_name"):
            names = (
                ent.physical_name
                if isinstance(ent.physical_name, tuple)
                else (ent.physical_name,)
            )
            for name in names:
                if name:
                    entity_tag_mapping[name] = True

    # Add halo entities
    for ent in task_params["halo"]:
        orig_tags = getattr(ent, "physical_name", ())
        if not isinstance(orig_tags, tuple):
            orig_tags = (orig_tags,) if orig_tags else ()

        cad_entities.append(ent)

        # Clone the resolution rule so the halo gets the exact same size field as its parent entity
        for orig_tag in orig_tags:
            entity_tag_mapping[orig_tag] = False  # Flag it as having halo presence
            if orig_tag and orig_tag in local_resolutions:
                local_resolutions[orig_tag] = local_resolutions[orig_tag]

    if not cad_entities:
        return ""  # Nothing to mesh

    model.cad.process_entities(cad_entities)

    _ = model.mesh.process_geometry(
        dim=dim,
        default_characteristic_length=default_char_length,
        resolution_specs=local_resolutions,
    )

    tmp_path = f"{tempfile.gettempdir()}/mesh_worker_{task_id}.msh"

    # Native GMSH Halo Cropping
    subdomain_polygon = task_params.get("subdomain_poly")
    if subdomain_polygon is not None:
        from shapely.geometry import Point

        buffered_subdomain = subdomain_polygon.buffer(1e-5)

        # We must filter elements natively before saving to avoid meshio corruption
        entities = gmsh.model.getEntities()
        for dim, tag in entities:
            elem_types, elem_tags, _ = gmsh.model.mesh.getElements(dim, tag)
            if not elem_types:
                continue

            for _i, etype in enumerate(elem_types):
                tags_for_type = elem_tags[_i]

                # Get number of nodes per element of this type
                # To do this safely, we ask gmsh for element properties
                (
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                ) = gmsh.model.mesh.getElementProperties(etype)

                # Get barycenters to check against Polygon
                barycenters = gmsh.model.mesh.getBarycenters(etype, tag, False, True)
                barycenters = barycenters.reshape(-1, 3)

                remove_tags = []
                for idx, bc in enumerate(barycenters):
                    if not buffered_subdomain.contains(Point(bc[0], bc[1])):
                        remove_tags.append(tags_for_type[idx])

                if len(remove_tags) > 0:
                    try:
                        gmsh.model.mesh.removeElements(
                            dim, tag, elementTags=remove_tags
                        )
                    except Exception as e:
                        print(f"Failed to removeElements for dim {dim} tag {tag}: {e}")

    # Generate the physical names mapping BEFORE we save using gmsh
    new_field_data = {}
    physical_groups = gmsh.model.getPhysicalGroups()

    # Collect entities for each physical name to avoid collisions and partial updates
    name_to_entities = {}
    for dim, p_tag in physical_groups:
        name = gmsh.model.getPhysicalName(dim, p_tag)
        if name:
            entities = gmsh.model.getEntitiesForPhysicalGroup(dim, p_tag)

            # Only include entities that still have mesh elements after cropping
            valid_entities = []
            for e_tag in entities:
                _, etags, _ = gmsh.model.mesh.getElements(dim, e_tag)
                if any(len(t) > 0 for t in etags):
                    valid_entities.append(e_tag)

            if valid_entities:
                if name not in name_to_entities:
                    name_to_entities[name] = {"dim": dim, "entities": []}
                name_to_entities[name]["entities"].extend(valid_entities)

    # Remove all old physical groups before adding new ones
    gmsh.model.removePhysicalGroups()

    for name, info in name_to_entities.items():
        dim_ent = info["dim"]
        entities = info["entities"]
        new_tag_id = _string_to_tag(name)
        gmsh.model.addPhysicalGroup(dim_ent, entities, new_tag_id, name)
        new_field_data[name] = np.array([new_tag_id, dim_ent], dtype=np.int32)

    gmsh.write(tmp_path)
    model.finalize()

    return tmp_path, new_field_data


def mesh_parallel(
    entities_list: List,
    subdomains: List[Polygon],
    halo_buffer: float,
    n_jobs: int = 4,
    default_characteristic_length: float = 2.0,
    resolution_specs: Dict | None = None,
    dim: int = 2,
) -> meshio.Mesh:
    """Meshes a list of entities in parallel by decomposing the domain.

    Args:
        entities_list: List of meshwell entities.
        subdomains: List of shapely Polygons defining subdomains.
        halo_buffer: Width of the halo overlap between subdomains.
        n_jobs: Number of parallel processes to use.
        default_characteristic_length: Default mesh size.
        resolution_specs: Mapping of physical names to resolution fields.
        dim: Dimension of the mesh (2 or 3).

    Returns:
        A stitched meshio.Mesh object.
    """
    if resolution_specs is None:
        resolution_specs = {}

    tasks = decompose_domain(entities_list, subdomains, halo_buffer)

    from shapely.ops import unary_union
    union_subdomains = unary_union(subdomains)
    global_boundary = union_subdomains.boundary

    from concurrent.futures import ProcessPoolExecutor, as_completed

    futures = []
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        for task_params in tasks.values():
            if not task_params["domain"]:
                continue  # Skip empty subdomains
            f = executor.submit(
                _meshing_task,
                task_params,
                default_characteristic_length,
                resolution_specs,
                dim,
            )
            futures.append(f)

        msh_results = [future.result() for future in as_completed(futures)]

    # Final stitching
    msh_results = [r for r in msh_results if r[0]]  # Filter empty
    if not msh_results:
        raise ValueError("No meshes were generated.")

    msh_paths = [r[0] for r in msh_results]

    # Aggregate field data from all sub-meshes (names to tags)
    aggregated_field_data = {}
    for _, worker_field_data in msh_results:
        for name, data in worker_field_data.items():
            if name not in aggregated_field_data:
                aggregated_field_data[name] = data


    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)

    for p in msh_paths:
        gmsh.merge(p)

    gmsh.model.mesh.removeDuplicateNodes()

    # Post-process: Identify internal ___None interfaces using multiplicity
    # 1. Group entities by (name, dim) using aggregated_field_data as the source of truth
    # This is more robust than gmsh.model.getPhysicalName which might be empty after merge
    name_dim_to_entities = {}
    for name, data in aggregated_field_data.items():
        p_tag = data[0] if isinstance(data, (list, np.ndarray)) else data
        p_dim = data[1] if isinstance(data, (list, np.ndarray)) else (dim - 1)
        
        entities = gmsh.model.getEntitiesForPhysicalGroup(p_dim, p_tag)
        if len(entities) > 0:
            key = (name, p_dim)
            if key not in name_dim_to_entities:
                name_dim_to_entities[key] = []
            name_dim_to_entities[key].extend(entities)

    # 2. Identify internal ___None interfaces
    internal_entities = set() # (dim, tag)
    none_keys = [k for k in name_dim_to_entities if "___None" in k[0]]
    if none_keys:
        all_bcs = []
        entity_metadata = [] # (d, e_tag, num_elements)
        
        for (g_name, d_g) in none_keys:
            entities = name_dim_to_entities[(g_name, d_g)]
            for e_tag in entities:
                elem_types, elem_tags, _ = gmsh.model.mesh.getElements(d_g, e_tag)
                num_ent_elems = 0
                for i, etype in enumerate(elem_types):
                    if len(elem_tags[i]) > 0:
                        bcs = gmsh.model.mesh.getBarycenters(etype, e_tag, False, True).reshape(-1, 3)
                        all_bcs.append(np.round(bcs, 6))
                        num_ent_elems += bcs.shape[0]
                if num_ent_elems > 0:
                    entity_metadata.append((d_g, e_tag, num_ent_elems))
        
        if all_bcs:
            all_bcs = np.vstack(all_bcs)
            _, inverse_indices, counts = np.unique(all_bcs, axis=0, return_inverse=True, return_counts=True)
            is_internal_elem = counts[inverse_indices] > 1
            
            offset = 0
            for d, e_tag, n in entity_metadata:
                if np.any(is_internal_elem[offset : offset + n]):
                    internal_entities.add((d, e_tag))
                offset += n

    # 3. Re-build physical groups with deterministic tags and filtered ___None
    gmsh.model.removePhysicalGroups()
    final_field_data = {}

    for (g_name, d_g), g_ents in name_dim_to_entities.items():
        if "___None" in g_name:
            # Filter ___None groups: keep only those that were NOT identified as internal duplicates
            entities_to_keep = [e for e in g_ents if (d_g, e) not in internal_entities]
            if entities_to_keep:
                new_tag = _string_to_tag(g_name)
                gmsh.model.addPhysicalGroup(d_g, entities_to_keep, new_tag, g_name)
                final_field_data[g_name] = np.array([new_tag, d_g], dtype=np.int32)
        else:
            new_tag = _string_to_tag(g_name)
            gmsh.model.addPhysicalGroup(d_g, g_ents, new_tag, g_name)
            final_field_data[g_name] = np.array([new_tag, d_g], dtype=np.int32)

    gmsh.model.mesh.removeDuplicateElements()

    final_path = (
        f"{tempfile.gettempdir()}/mesh_parallel_final_{uuid.uuid4().hex[:8]}.msh"
    )
    gmsh.write(final_path)
    gmsh.finalize()

    final_mesh = meshio.read(final_path)

    # Synchronize physical names for the final mesh to match what was actually added
    final_mesh.field_data = final_field_data

    return final_mesh
