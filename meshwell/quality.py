"""Mesh quality analysis utilities for GMSH meshes."""

import math
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")


class MeshQualityAnalyzer:
    """Analyzer for mesh quality metrics and diagnostics."""

    def __init__(self, mesh_file):
        self.mesh_file = mesh_file
        self.nodes = {}
        self.elements = {}
        self.physical_names = {}
        self.physical_groups = set()
        self.tetra_elements = []
        self.surface_elements = []
        self.triangle_elements = []  # For 2D triangular meshes
        self.quality_metrics = {}
        self.mesh_dimension = None  # Will be determined during parsing
        self.gmsh_version = None  # GMSH format version
        self.file_type = 0  # 0=ASCII, 1=binary

    def check_mesh_file(self):
        """Check if mesh file exists and get basic info."""
        print("=== Mesh File Check ===")

        if not Path(self.mesh_file).exists():
            print(f"❌ Mesh file '{self.mesh_file}' does not exist")
            return False

        file_size = Path(self.mesh_file).stat().st_size
        print(f"✓ Mesh file exists: {self.mesh_file}")
        print(f"✓ File size: {file_size} bytes")

        if file_size == 0:
            print("❌ Mesh file is empty")
            return False

        try:
            with Path(self.mesh_file).open() as f:
                lines = f.readlines()[:10]
                print("✓ File is readable, first 10 lines preview:")
                for i, line in enumerate(lines):
                    print(f"  {i+1:2d}: {line.rstrip()}")
        except Exception as e:
            print(f"❌ Error reading file: {e}")
            return False

        return True

    def parse_gmsh_mesh(self):
        """Parse the GMSH mesh file and extract all data."""
        print("\n=== Parsing GMSH Mesh ===")

        try:
            with Path(self.mesh_file).open() as f:
                lines = f.readlines()

            # Parse sections
            self._parse_mesh_format(lines)
            self._parse_physical_names(lines)
            self._parse_nodes(lines)
            self._parse_elements(lines)
            self._determine_mesh_dimension()

            print(f"✓ Parsed {len(self.nodes)} nodes")
            print(f"✓ Parsed {len(self.tetra_elements)} tetrahedra")
            print(f"✓ Parsed {len(self.triangle_elements)} triangular elements")
            print(f"✓ Parsed {len(self.surface_elements)} surface elements")
            print(f"✓ Found {len(self.physical_names)} physical regions")
            print(f"✓ Detected mesh dimension: {self.mesh_dimension}D")

            return True

        except Exception as e:
            print(f"❌ Error parsing GMSH mesh: {e}")
            return False

    def _parse_mesh_format(self, lines):
        """Parse MeshFormat section to determine GMSH version and file type."""
        in_section = False
        for line in lines:
            line = line.strip()
            if line == "$MeshFormat":
                in_section = True
                continue
            if line == "$EndMeshFormat":
                break
            if in_section and line:
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        self.gmsh_version = float(parts[0])
                        self.file_type = int(parts[1])
                        print(
                            f"✓ Detected GMSH format version {self.gmsh_version}, type {self.file_type}"
                        )
                        break
                    except ValueError:
                        pass

        # Default to version 2.2 if not found (legacy format)
        if self.gmsh_version is None:
            self.gmsh_version = 2.2
            print(
                f"⚠️ No MeshFormat section found, assuming GMSH version {self.gmsh_version}"
            )

    def _parse_physical_names(self, lines):
        """Parse PhysicalNames section."""
        in_section = False
        for _, line in enumerate(lines):
            line = line.strip()
            if line == "$PhysicalNames":
                in_section = True
                continue
            if line == "$EndPhysicalNames":
                break
            if in_section and line and not line.startswith("$"):
                parts = line.split()
                if len(parts) >= 3 and parts[0].isdigit():
                    dim, tag, name = int(parts[0]), int(parts[1]), parts[2].strip('"')
                    self.physical_names[tag] = (dim, name)

    def _parse_nodes(self, lines):
        """Parse Nodes section based on GMSH format version."""
        if self.gmsh_version >= 4.0:
            self._parse_nodes_v4(lines)
        else:
            self._parse_nodes_legacy(lines)

    def _parse_nodes_v4(self, lines):
        """Parse Nodes section for GMSH 4.x format."""
        in_section = False
        expecting_node_ids = False
        expecting_coords = False
        current_node_ids = []
        nodes_in_block = 0

        for line in lines:
            line = line.strip()
            if line == "$Nodes":
                in_section = True
                continue
            if line == "$EndNodes":
                break
            if in_section and line:
                parts = line.split()

                # Skip the header line (e.g., "54 2689 1 2689") - large numbers
                if (
                    len(parts) == 4
                    and all(p.isdigit() for p in parts)
                    and int(parts[1]) > 100
                ):
                    continue

                # Entity block header: entityDim entityTag parametric numNodesInBlock
                if len(parts) == 4 and not expecting_node_ids and not expecting_coords:
                    try:
                        _, _, _, num_nodes = map(
                            int, parts
                        )  # entity_dim, entity_tag, parametric, num_nodes
                        current_node_ids = []
                        nodes_in_block = num_nodes
                        expecting_node_ids = True
                        expecting_coords = False
                        continue
                    except ValueError:
                        pass

                # Parse node IDs (one per line)
                if expecting_node_ids and len(parts) == 1 and parts[0].isdigit():
                    current_node_ids.append(int(parts[0]))
                    if len(current_node_ids) == nodes_in_block:
                        expecting_node_ids = False
                        expecting_coords = True
                    continue

                # Parse coordinates (3 per line, one for each node ID)
                if expecting_coords and len(parts) == 3:
                    try:
                        coords = [float(parts[0]), float(parts[1]), float(parts[2])]
                        if current_node_ids:
                            node_id = current_node_ids.pop(0)
                            self.nodes[node_id] = np.array(coords)
                            if not current_node_ids:  # Reset when all coords processed
                                expecting_coords = False
                    except ValueError:
                        pass
                    continue

    def _parse_nodes_legacy(self, lines):
        """Parse Nodes section for GMSH 2.x and earlier formats."""
        in_section = False
        for line in lines:
            line = line.strip()
            if line == "$Nodes":
                in_section = True
                continue
            if line == "$EndNodes":
                break
            if in_section and line:
                parts = line.split()
                # Skip the node count line
                if len(parts) == 1 and parts[0].isdigit():
                    continue
                # Parse node: node_id x y z
                if len(parts) >= 4:
                    try:
                        node_id = int(parts[0])
                        coords = [float(parts[1]), float(parts[2]), float(parts[3])]
                        self.nodes[node_id] = np.array(coords)
                    except ValueError:
                        pass

    def _parse_elements(self, lines):
        """Parse Elements section based on GMSH format version."""
        if self.gmsh_version >= 4.0:
            self._parse_elements_v4(lines)
        else:
            self._parse_elements_legacy(lines)

    def _parse_elements_v4(self, lines):
        """Parse Elements section for GMSH 4.x format."""
        in_section = False
        current_entity_type = None
        current_entity_tag = None

        for line in lines:
            line = line.strip()
            if line == "$Elements":
                in_section = True
                continue
            if line == "$EndElements":
                break
            if in_section and line:
                parts = line.split()

                # Skip the overall section header line (e.g., "10 7221 1 7221")
                if (
                    len(parts) == 4
                    and all(p.isdigit() for p in parts)
                    and int(parts[1]) > 1000
                    and int(parts[3]) > 1000
                ):
                    continue

                # Entity block header: entityDim entityTag elementType numElementsInBlock
                # Valid entity headers have small dim (0-3), reasonable element types (1-15), etc.
                if (
                    len(parts) == 4
                    and all(p.isdigit() for p in parts)
                    and int(parts[0]) <= 3
                    and int(parts[2]) <= 15
                ):
                    _, entity_tag, elm_type, _ = map(int, parts)
                    current_entity_type = elm_type
                    current_entity_tag = entity_tag
                    continue

                # Parse elements in GMSH 4.1 format (elementTag + node IDs)
                if current_entity_type is not None:
                    if current_entity_type == 4:  # 4-node tetrahedron
                        if len(parts) >= 5:  # elementTag + 4 nodes
                            try:
                                elm_id = int(parts[0])
                                node_ids = [int(p) for p in parts[1:5]]

                                # Find physical tag based on entity tag
                                physical_tag = self._find_physical_tag_for_entity(
                                    current_entity_tag, 3
                                )

                                self.tetra_elements.append(
                                    {
                                        "id": elm_id,
                                        "nodes": node_ids,
                                        "physical_tag": physical_tag,
                                    }
                                )
                                if physical_tag:
                                    self.physical_groups.add(physical_tag)
                            except ValueError:
                                pass
                    elif (
                        current_entity_type == 2 and len(parts) >= 4
                    ):  # 3-node triangle
                        try:
                            elm_id = int(parts[0])
                            node_ids = [int(p) for p in parts[1:4]]

                            # Find physical tag based on entity tag
                            physical_tag = self._find_physical_tag_for_entity(
                                current_entity_tag, 2
                            )

                            element = {
                                "id": elm_id,
                                "nodes": node_ids,
                                "physical_tag": physical_tag,
                            }
                            self.surface_elements.append(element)
                            if physical_tag:
                                self.physical_groups.add(physical_tag)
                        except ValueError:
                            pass

    def _parse_elements_legacy(self, lines):
        """Parse Elements section for GMSH 2.x and earlier formats."""
        in_section = False
        for line in lines:
            line = line.strip()
            if line == "$Elements":
                in_section = True
                continue
            if line == "$EndElements":
                break
            if in_section and line:
                parts = line.split()
                # Skip the element count line
                if len(parts) == 1 and parts[0].isdigit():
                    continue

                # Parse element: elm-number elm-type number-of-tags < tag > ... node-number-list
                if len(parts) >= 2 and parts[1].isdigit():
                    elm_type = int(parts[1])
                    elm_id = int(parts[0])
                    physical_tag = None

                    if len(parts) >= 3 and parts[2].isdigit():
                        num_tags = int(parts[2])
                        if num_tags > 0 and len(parts) >= 3 + num_tags:
                            physical_tag = int(parts[3])
                            self.physical_groups.add(physical_tag)

                    if elm_type == 4:  # 4-node tetrahedron
                        if len(parts) >= 7:
                            node_ids = [int(p) for p in parts[-4:]]
                            self.tetra_elements.append(
                                {
                                    "id": elm_id,
                                    "nodes": node_ids,
                                    "physical_tag": physical_tag,
                                }
                            )
                    elif elm_type == 2 and len(parts) >= 6:  # 3-node triangle
                        node_ids = [int(p) for p in parts[-3:]]
                        element = {
                            "id": elm_id,
                            "nodes": node_ids,
                            "physical_tag": physical_tag,
                        }
                        self.surface_elements.append(element)

    def _find_physical_tag_for_entity(self, _entity_tag, target_dim):
        """Find physical tag for a given entity tag and dimension."""
        # In GMSH 4.x, physical tags are often the same as entity tags for their dimension
        for tag, (dim, _) in self.physical_names.items():
            if dim == target_dim:
                return tag
        return None

    def _determine_mesh_dimension(self):
        """Determine if this is a 2D or 3D mesh and classify triangular elements appropriately."""
        # Check if we have any tetrahedral elements
        if self.tetra_elements:
            self.mesh_dimension = 3
            # In 3D mesh, triangular elements remain as surface elements
            return

        # Check if we have triangular elements with 2D physical regions
        has_2d_regions = any(dim == 2 for dim, _ in self.physical_names.values())

        if has_2d_regions and self.surface_elements:
            self.mesh_dimension = 2
            # Move triangular elements from surface_elements to triangle_elements for 2D analysis
            for element in self.surface_elements:
                if (
                    element["physical_tag"]
                    and element["physical_tag"] in self.physical_names
                ):
                    dim = self.physical_names[element["physical_tag"]][0]
                    if dim == 2:
                        self.triangle_elements.append(element)

            # Remove 2D triangular elements from surface_elements
            self.surface_elements = [
                e
                for e in self.surface_elements
                if not (
                    e["physical_tag"]
                    and e["physical_tag"] in self.physical_names
                    and self.physical_names[e["physical_tag"]][0] == 2
                )
            ]
        else:
            # Default to 3D if unclear
            self.mesh_dimension = 3

    def check_mesh_connectivity(self):
        """Check mesh connectivity and topology."""
        print("\n=== Mesh Connectivity Analysis ===")

        if self.mesh_dimension == 3:
            return self._check_3d_connectivity()
        if self.mesh_dimension == 2:
            return self._check_2d_connectivity()
        print("❌ Unknown mesh dimension")
        return False

    def _check_3d_connectivity(self):
        """Check connectivity for 3D tetrahedral meshes."""
        if not self.tetra_elements:
            print("❌ No tetrahedral elements found")
            return False

        # Check node connectivity
        node_usage = defaultdict(int)
        for tetra in self.tetra_elements:
            for node_id in tetra["nodes"]:
                node_usage[node_id] += 1

        orphaned_nodes = [nid for nid in self.nodes if node_usage[nid] == 0]
        if orphaned_nodes:
            print(
                f"⚠️ Found {len(orphaned_nodes)} orphaned nodes (not connected to any tetrahedron)"
            )
            if len(orphaned_nodes) <= 10:
                print(f"   Orphaned node IDs: {orphaned_nodes}")

        # Check element connectivity
        edge_count = defaultdict(int)
        face_count = defaultdict(int)

        for tetra in self.tetra_elements:
            nodes = tetra["nodes"]
            # Count edges (pairs of nodes)
            for i in range(4):
                for j in range(i + 1, 4):
                    edge = tuple(sorted([nodes[i], nodes[j]]))
                    edge_count[edge] += 1

            # Count faces (triplets of nodes)
            faces = [
                tuple(sorted([nodes[0], nodes[1], nodes[2]])),
                tuple(sorted([nodes[0], nodes[1], nodes[3]])),
                tuple(sorted([nodes[0], nodes[2], nodes[3]])),
                tuple(sorted([nodes[1], nodes[2], nodes[3]])),
            ]
            for face in faces:
                face_count[face] += 1

        # Check for non-manifold edges (shared by more than 2 faces)
        non_manifold_edges = [
            (edge, count) for edge, count in edge_count.items() if count > 6
        ]
        if non_manifold_edges:
            print(f"⚠️ Found {len(non_manifold_edges)} non-manifold edges")
            if len(non_manifold_edges) <= 5:
                for edge, count in non_manifold_edges[:5]:
                    print(f"   Edge {edge}: used {count} times")

        # Check for boundary faces (faces belonging to only one tetrahedron)
        boundary_faces = [face for face, count in face_count.items() if count == 1]
        internal_faces = [face for face, count in face_count.items() if count == 2]
        bad_faces = [face for face, count in face_count.items() if count > 2]

        print(f"✓ Boundary faces: {len(boundary_faces)}")
        print(f"✓ Internal faces: {len(internal_faces)}")
        if bad_faces:
            print(
                f"❌ Non-manifold faces: {len(bad_faces)} (faces shared by >2 tetrahedra)"
            )
            return False

        return True

    def _check_2d_connectivity(self):
        """Check connectivity for 2D triangular meshes."""
        if not self.triangle_elements:
            print("❌ No triangular elements found")
            return False

        # Check node connectivity
        node_usage = defaultdict(int)
        for triangle in self.triangle_elements:
            for node_id in triangle["nodes"]:
                node_usage[node_id] += 1

        orphaned_nodes = [nid for nid in self.nodes if node_usage[nid] == 0]
        if orphaned_nodes:
            print(
                f"⚠️ Found {len(orphaned_nodes)} orphaned nodes (not connected to any triangle)"
            )
            if len(orphaned_nodes) <= 10:
                print(f"   Orphaned node IDs: {orphaned_nodes}")

        # Check edge connectivity
        edge_count = defaultdict(int)

        for triangle in self.triangle_elements:
            nodes = triangle["nodes"]
            # Count edges (pairs of nodes)
            for i in range(3):
                for j in range(i + 1, 3):
                    edge = tuple(sorted([nodes[i], nodes[j]]))
                    edge_count[edge] += 1

        # Check for boundary edges (shared by only one triangle) and internal edges (shared by two)
        boundary_edges = [edge for edge, count in edge_count.items() if count == 1]
        internal_edges = [edge for edge, count in edge_count.items() if count == 2]
        non_manifold_edges = [edge for edge, count in edge_count.items() if count > 2]

        print(f"✓ Boundary edges: {len(boundary_edges)}")
        print(f"✓ Internal edges: {len(internal_edges)}")
        if non_manifold_edges:
            print(
                f"❌ Non-manifold edges: {len(non_manifold_edges)} (edges shared by >2 triangles)"
            )
            return False

        return True

    def analyze_geometric_quality(self):
        """Analyze geometric quality metrics."""
        print("\n=== Geometric Quality Analysis ===")

        if self.mesh_dimension == 3:
            return self._analyze_3d_quality()
        if self.mesh_dimension == 2:
            return self._analyze_2d_quality()
        print("❌ Unknown mesh dimension")
        return False

    def _analyze_3d_quality(self):
        """Analyze geometric quality metrics for tetrahedra."""
        if not self.tetra_elements:
            print("❌ No tetrahedral elements to analyze")
            return False

        aspect_ratios = []
        volumes = []
        min_angles = []
        max_angles = []
        edge_lengths = []

        # Per-physical-group metrics
        group_metrics = defaultdict(
            lambda: {"aspect_ratios": [], "volumes": [], "min_angles": [], "count": 0}
        )

        degenerate_count = 0
        negative_volume_count = 0

        for tetra in self.tetra_elements:
            try:
                coords = np.array([self.nodes[nid] for nid in tetra["nodes"]])

                # Calculate volume
                v0, v1, v2, v3 = coords
                vol = (
                    np.abs(np.linalg.det(np.column_stack([v1 - v0, v2 - v0, v3 - v0])))
                    / 6.0
                )
                volumes.append(vol)

                if vol <= 0:
                    negative_volume_count += 1
                    continue

                if vol < 1e-15:  # Very small volume threshold
                    degenerate_count += 1
                    continue

                # Calculate edge lengths
                tetra_edges = []
                for i in range(4):
                    for j in range(i + 1, 4):
                        edge_len = np.linalg.norm(coords[i] - coords[j])
                        tetra_edges.append(edge_len)
                        edge_lengths.append(edge_len)

                # Calculate aspect ratio (longest edge / shortest edge)
                min_edge = min(tetra_edges)
                max_edge = max(tetra_edges)
                if min_edge > 0:
                    aspect_ratio = max_edge / min_edge
                    aspect_ratios.append(aspect_ratio)

                # Calculate angles (simplified - between edges from one vertex)
                angles = []
                for i in range(4):
                    vertex = coords[i]
                    other_vertices = [coords[j] for j in range(4) if j != i]

                    # Calculate angles between edges from this vertex
                    for j in range(3):
                        for k in range(j + 1, 3):
                            v1 = other_vertices[j] - vertex
                            v2 = other_vertices[k] - vertex

                            v1_norm = np.linalg.norm(v1)
                            v2_norm = np.linalg.norm(v2)

                            if v1_norm > 0 and v2_norm > 0:
                                cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
                                cos_angle = np.clip(cos_angle, -1, 1)
                                angle = math.degrees(math.acos(cos_angle))
                                angles.append(angle)

                if angles:
                    min_angle = min(angles)
                    min_angles.append(min_angle)
                    max_angles.append(max(angles))

                    # Track per-group metrics
                    phys_tag = tetra.get("physical_tag")
                    if phys_tag is not None:
                        group_metrics[phys_tag]["aspect_ratios"].append(aspect_ratio)
                        group_metrics[phys_tag]["volumes"].append(vol)
                        group_metrics[phys_tag]["min_angles"].append(min_angle)
                        group_metrics[phys_tag]["count"] += 1

            except Exception as e:
                print(f"⚠️ Error analyzing tetrahedron {tetra['id']}: {e}")
                continue

        # Report statistics
        if volumes:
            print("Volume statistics:")
            print(f"  Min volume: {min(volumes):.2e}")
            print(f"  Max volume: {max(volumes):.2e}")
            print(f"  Mean volume: {np.mean(volumes):.2e}")
            print(f"  Volume ratio (max/min): {max(volumes)/min(volumes):.2e}")

        if aspect_ratios:
            print("Aspect ratio statistics:")
            print(f"  Min aspect ratio: {min(aspect_ratios):.2f}")
            print(f"  Max aspect ratio: {max(aspect_ratios):.2f}")
            print(f"  Mean aspect ratio: {np.mean(aspect_ratios):.2f}")

            # Quality assessment
            excellent = sum(1 for ar in aspect_ratios if ar < 3)
            good = sum(1 for ar in aspect_ratios if 3 <= ar < 10)
            poor = sum(1 for ar in aspect_ratios if 10 <= ar < 100)
            very_poor = sum(1 for ar in aspect_ratios if ar >= 100)

            total = len(aspect_ratios)
            print("  Quality distribution:")
            print(f"    Excellent (AR < 3): {excellent} ({100*excellent/total:.1f}%)")
            print(f"    Good (3 ≤ AR < 10): {good} ({100*good/total:.1f}%)")
            print(f"    Poor (10 ≤ AR < 100): {poor} ({100*poor/total:.1f}%)")
            print(f"    Very poor (AR ≥ 100): {very_poor} ({100*very_poor/total:.1f}%)")

            if very_poor > 0:
                print(f"⚠️ {very_poor} elements have very poor aspect ratios (≥100)")

        if edge_lengths:
            print("Edge length statistics:")
            print(f"  Min edge length: {min(edge_lengths):.2e}")
            print(f"  Max edge length: {max(edge_lengths):.2e}")
            print(f"  Edge length ratio: {max(edge_lengths)/min(edge_lengths):.2e}")

            # Check for extreme edge length ratios
            if max(edge_lengths) / min(edge_lengths) > 1e6:
                print(
                    "⚠️ Very large edge length ratio detected - may cause numerical issues"
                )

        if min_angles:
            print("Angle statistics:")
            print(f"  Min angle: {min(min_angles):.1f}°")
            print(f"  Max angle: {max(max_angles):.1f}°")

            # Check for very small angles
            very_small_angles = sum(1 for angle in min_angles if angle < 1.0)
            if very_small_angles > 0:
                print(f"⚠️ {very_small_angles} elements have very small angles (<1°)")

        if degenerate_count > 0:
            print(f"❌ {degenerate_count} degenerate elements (volume < 1e-15)")

        if negative_volume_count > 0:
            print(f"❌ {negative_volume_count} elements with negative/zero volume")

        # Store metrics for later use
        self.quality_metrics = {
            "aspect_ratios": aspect_ratios,
            "volumes": volumes,
            "min_angles": min_angles,
            "edge_lengths": edge_lengths,
            "degenerate_count": degenerate_count,
            "negative_volume_count": negative_volume_count,
            "group_metrics": dict(group_metrics),  # Convert defaultdict to dict
        }

        return degenerate_count == 0 and negative_volume_count == 0

    def _analyze_2d_quality(self):
        """Analyze geometric quality metrics for triangular elements."""
        if not self.triangle_elements:
            print("❌ No triangular elements to analyze")
            return False

        aspect_ratios = []
        areas = []
        min_angles = []
        max_angles = []
        edge_lengths = []

        # Per-physical-group metrics
        group_metrics = defaultdict(
            lambda: {"aspect_ratios": [], "areas": [], "min_angles": [], "count": 0}
        )

        degenerate_count = 0
        negative_area_count = 0

        for triangle in self.triangle_elements:
            try:
                coords = np.array([self.nodes[nid] for nid in triangle["nodes"]])

                # Calculate area using cross product
                v0, v1, v2 = coords
                edge1 = v1 - v0
                edge2 = v2 - v0

                # For 2D triangles, use z-component of cross product
                if coords.shape[1] == 2:  # 2D coordinates
                    # Add z=0 for cross product calculation
                    edge1_3d = np.append(edge1, 0)
                    edge2_3d = np.append(edge2, 0)
                    cross = np.cross(edge1_3d, edge2_3d)
                    area = abs(cross) / 2.0
                else:  # 3D coordinates (triangle in 3D space)
                    cross = np.cross(edge1, edge2)
                    area = np.linalg.norm(cross) / 2.0

                areas.append(area)

                if area <= 0:
                    negative_area_count += 1
                    continue

                if area < 1e-15:  # Very small area threshold
                    degenerate_count += 1
                    continue

                # Calculate edge lengths
                triangle_edges = []
                for i in range(3):
                    for j in range(i + 1, 3):
                        edge_len = np.linalg.norm(coords[i] - coords[j])
                        triangle_edges.append(edge_len)
                        edge_lengths.append(edge_len)

                # Calculate aspect ratio (longest edge / shortest edge)
                min_edge = min(triangle_edges)
                max_edge = max(triangle_edges)
                if min_edge > 0:
                    aspect_ratio = max_edge / min_edge
                    aspect_ratios.append(aspect_ratio)

                # Calculate angles using law of cosines
                a, b, c = triangle_edges
                angles = []

                # Angle opposite to edge a
                cos_A = (b * b + c * c - a * a) / (2 * b * c)
                cos_A = np.clip(cos_A, -1, 1)
                angles.append(math.degrees(math.acos(cos_A)))

                # Angle opposite to edge b
                cos_B = (a * a + c * c - b * b) / (2 * a * c)
                cos_B = np.clip(cos_B, -1, 1)
                angles.append(math.degrees(math.acos(cos_B)))

                # Angle opposite to edge c
                cos_C = (a * a + b * b - c * c) / (2 * a * b)
                cos_C = np.clip(cos_C, -1, 1)
                angles.append(math.degrees(math.acos(cos_C)))

                min_angle = min(angles)
                min_angles.append(min_angle)
                max_angles.append(max(angles))

                # Track per-group metrics
                phys_tag = triangle.get("physical_tag")
                if phys_tag is not None:
                    group_metrics[phys_tag]["aspect_ratios"].append(aspect_ratio)
                    group_metrics[phys_tag]["areas"].append(area)
                    group_metrics[phys_tag]["min_angles"].append(min_angle)
                    group_metrics[phys_tag]["count"] += 1

            except Exception as e:
                print(f"⚠️ Error analyzing triangle {triangle['id']}: {e}")
                continue

        # Report statistics
        if areas:
            print("Area statistics:")
            print(f"  Min area: {min(areas):.2e}")
            print(f"  Max area: {max(areas):.2e}")
            print(f"  Mean area: {np.mean(areas):.2e}")
            print(f"  Area ratio (max/min): {max(areas)/min(areas):.2e}")

        if aspect_ratios:
            print("Aspect ratio statistics:")
            print(f"  Min aspect ratio: {min(aspect_ratios):.2f}")
            print(f"  Max aspect ratio: {max(aspect_ratios):.2f}")
            print(f"  Mean aspect ratio: {np.mean(aspect_ratios):.2f}")

            # Quality assessment for triangles
            excellent = sum(1 for ar in aspect_ratios if ar < 2)
            good = sum(1 for ar in aspect_ratios if 2 <= ar < 5)
            poor = sum(1 for ar in aspect_ratios if 5 <= ar < 20)
            very_poor = sum(1 for ar in aspect_ratios if ar >= 20)

            total = len(aspect_ratios)
            print("  Quality distribution:")
            print(f"    Excellent (AR < 2): {excellent} ({100*excellent/total:.1f}%)")
            print(f"    Good (2 ≤ AR < 5): {good} ({100*good/total:.1f}%)")
            print(f"    Poor (5 ≤ AR < 20): {poor} ({100*poor/total:.1f}%)")
            print(f"    Very poor (AR ≥ 20): {very_poor} ({100*very_poor/total:.1f}%)")

            if very_poor > 0:
                print(f"⚠️ {very_poor} elements have very poor aspect ratios (≥20)")

        if edge_lengths:
            print("Edge length statistics:")
            print(f"  Min edge length: {min(edge_lengths):.2e}")
            print(f"  Max edge length: {max(edge_lengths):.2e}")
            print(f"  Edge length ratio: {max(edge_lengths)/min(edge_lengths):.2e}")

            # Check for extreme edge length ratios
            if max(edge_lengths) / min(edge_lengths) > 1e6:
                print(
                    "⚠️ Very large edge length ratio detected - may cause numerical issues"
                )

        if min_angles:
            print("Angle statistics:")
            print(f"  Min angle: {min(min_angles):.1f}°")
            print(f"  Max angle: {max(max_angles):.1f}°")

            # Check for very small angles
            very_small_angles = sum(1 for angle in min_angles if angle < 5.0)
            very_large_angles = sum(1 for angle in max_angles if angle > 150.0)
            if very_small_angles > 0:
                print(f"⚠️ {very_small_angles} elements have very small angles (<5°)")
            if very_large_angles > 0:
                print(f"⚠️ {very_large_angles} elements have very large angles (>150°)")

        if degenerate_count > 0:
            print(f"❌ {degenerate_count} degenerate elements (area < 1e-15)")

        if negative_area_count > 0:
            print(f"❌ {negative_area_count} elements with negative/zero area")

        # Store metrics for later use
        self.quality_metrics = {
            "aspect_ratios": aspect_ratios,
            "areas": areas,
            "min_angles": min_angles,
            "edge_lengths": edge_lengths,
            "degenerate_count": degenerate_count,
            "negative_area_count": negative_area_count,
            "group_metrics": dict(group_metrics),  # Convert defaultdict to dict
        }

        return degenerate_count == 0 and negative_area_count == 0

    def check_physical_regions(self):
        """Report element counts for all physical groups across all dimensions."""
        print("\n=== Physical Region Analysis ===")

        # Collect all elements with their physical tags
        all_elements = []
        all_elements.extend([(e, "line") for e in getattr(self, "line_elements", [])])
        all_elements.extend([(e, "triangle") for e in self.triangle_elements])
        all_elements.extend([(e, "tetrahedron") for e in self.tetra_elements])
        all_elements.extend([(e, "surface") for e in self.surface_elements])

        # Count elements by physical tag
        region_counts = defaultdict(lambda: defaultdict(int))
        total_elements = defaultdict(int)

        for element, elem_type in all_elements:
            if element.get("physical_tag") is not None:
                tag = element["physical_tag"]
                region_counts[tag][elem_type] += 1
                total_elements[tag] += 1

        # Report by physical region
        if region_counts:
            print("Physical region element counts:")
            for tag in sorted(region_counts.keys()):
                if tag in self.physical_names:
                    dim, name = self.physical_names[tag]
                    types = region_counts[tag]
                    type_strs = [
                        f"{count} {etype}{'s' if count != 1 else ''}"
                        for etype, count in types.items()
                        if count > 0
                    ]
                    print(
                        f"  {name} (dim={dim}, tag={tag}): {' + '.join(type_strs)} = {total_elements[tag]} total"
                    )
                else:
                    types = region_counts[tag]
                    type_strs = [
                        f"{count} {etype}{'s' if count != 1 else ''}"
                        for etype, count in types.items()
                        if count > 0
                    ]
                    print(
                        f"  Unknown region (tag={tag}): {' + '.join(type_strs)} = {total_elements[tag]} total"
                    )
        else:
            print("No elements with physical tags found")

        return True

    def report_per_group_quality(self):
        """Report quality metrics broken down by physical group."""
        print("\n=== Quality Metrics Per Physical Group ===")

        group_metrics = self.quality_metrics.get("group_metrics", {})
        if not group_metrics:
            print("No per-group metrics available")
            return True

        # Determine if 2D or 3D
        is_3d = (
            "volumes" in next(iter(group_metrics.values())) if group_metrics else False
        )

        for tag in sorted(group_metrics.keys()):
            metrics = group_metrics[tag]
            count = metrics["count"]

            if count == 0:
                continue

            # Get physical name
            if tag in self.physical_names:
                _, name = self.physical_names[tag]
                print(f"\n{name} (tag={tag}, {count} elements):")
            else:
                print(f"\nPhysical group {tag} ({count} elements):")

            # Aspect ratios
            ars = metrics.get("aspect_ratios", [])
            if ars:
                print(
                    f"  Aspect ratio: min={min(ars):.2f}, max={max(ars):.2f}, mean={np.mean(ars):.2f}"
                )

                # Quality breakdown
                if is_3d:
                    poor = sum(1 for ar in ars if ar >= 10)
                    if poor > 0:
                        print(
                            f"  ⚠️ {poor} elements with AR >= 10 ({100*poor/len(ars):.1f}%)"
                        )
                else:
                    poor = sum(1 for ar in ars if ar >= 5)
                    if poor > 0:
                        print(
                            f"  ⚠️ {poor} elements with AR >= 5 ({100*poor/len(ars):.1f}%)"
                        )

            # Volumes or areas
            if is_3d:
                vols = metrics.get("volumes", [])
                if vols:
                    print(
                        f"  Volume: min={min(vols):.2e}, max={max(vols):.2e}, mean={np.mean(vols):.2e}"
                    )
            else:
                areas = metrics.get("areas", [])
                if areas:
                    print(
                        f"  Area: min={min(areas):.2e}, max={max(areas):.2e}, mean={np.mean(areas):.2e}"
                    )

            # Angles
            min_angles = metrics.get("min_angles", [])
            if min_angles:
                print(f"  Min angle: {min(min_angles):.1f}°")
                small_angles = sum(1 for a in min_angles if a < 5.0)
                if small_angles > 0:
                    print(
                        f"  ⚠️ {small_angles} elements with angle < 5° ({100*small_angles/len(min_angles):.1f}%)"
                    )

        return True

    def check_contacts_and_boundaries(self):
        """Simple report of physical groups by dimension."""
        print("\n=== Physical Groups by Dimension ===")

        # Group physical names by dimension
        groups_by_dim = defaultdict(list)
        for tag, (dim, name) in self.physical_names.items():
            groups_by_dim[dim].append((tag, name))

        # Report by dimension
        for dim in sorted(groups_by_dim.keys()):
            dim_name = {0: "Points", 1: "Lines", 2: "Surfaces", 3: "Volumes"}.get(
                dim, f"Dim-{dim}"
            )
            print(f"{dim_name} (dimension {dim}):")
            for tag, name in sorted(groups_by_dim[dim]):
                print(f"  {name} (tag {tag})")

        if not groups_by_dim:
            print("No physical groups defined")

        return True

    def check_mesh_gradation(self):
        """Check mesh gradation and smoothness."""
        print("\n=== Mesh Gradation Analysis ===")

        if self.mesh_dimension == 3:
            return self._check_3d_gradation()
        if self.mesh_dimension == 2:
            return self._check_2d_gradation()
        print("❌ Unknown mesh dimension")
        return False

    def _check_3d_gradation(self):
        """Check gradation for 3D tetrahedral meshes."""
        if not self.tetra_elements or len(self.tetra_elements) < 2:
            print("❌ Insufficient elements for gradation analysis")
            return False

        # Build adjacency information
        face_to_elements = defaultdict(list)
        for i, tetra in enumerate(self.tetra_elements):
            nodes = tetra["nodes"]
            faces = [
                tuple(sorted([nodes[0], nodes[1], nodes[2]])),
                tuple(sorted([nodes[0], nodes[1], nodes[3]])),
                tuple(sorted([nodes[0], nodes[2], nodes[3]])),
                tuple(sorted([nodes[1], nodes[2], nodes[3]])),
            ]
            for face in faces:
                face_to_elements[face].append(i)

        # Calculate size ratios between adjacent elements
        size_ratios = []
        for element_indices in face_to_elements.values():
            if len(element_indices) == 2:  # Internal face
                i, j = element_indices
                vol_i = self.quality_metrics.get("volumes", [1])[
                    min(i, len(self.quality_metrics.get("volumes", [1])) - 1)
                ]
                vol_j = self.quality_metrics.get("volumes", [1])[
                    min(j, len(self.quality_metrics.get("volumes", [1])) - 1)
                ]

                if vol_i > 0 and vol_j > 0:
                    ratio = max(vol_i, vol_j) / min(vol_i, vol_j)
                    size_ratios.append(ratio)

        if size_ratios:
            print("Mesh gradation statistics:")
            print(f"  Mean size ratio: {np.mean(size_ratios):.2f}")
            print(f"  Max size ratio: {max(size_ratios):.2f}")

            # Check for abrupt size changes
            abrupt_changes = sum(1 for ratio in size_ratios if ratio > 5)
            if abrupt_changes > 0:
                print(f"⚠️ {abrupt_changes} abrupt size changes (ratio > 5)")
                print("   This may cause convergence issues in simulations")

        return True

    def _check_2d_gradation(self):
        """Check gradation for 2D triangular meshes."""
        if not self.triangle_elements or len(self.triangle_elements) < 2:
            print("❌ Insufficient elements for gradation analysis")
            return False

        # Build adjacency information via shared edges
        edge_to_elements = defaultdict(list)
        for i, triangle in enumerate(self.triangle_elements):
            nodes = triangle["nodes"]
            edges = [
                tuple(sorted([nodes[0], nodes[1]])),
                tuple(sorted([nodes[1], nodes[2]])),
                tuple(sorted([nodes[2], nodes[0]])),
            ]
            for edge in edges:
                edge_to_elements[edge].append(i)

        # Calculate size ratios between adjacent elements
        size_ratios = []
        for element_indices in edge_to_elements.values():
            if len(element_indices) == 2:  # Internal edge
                i, j = element_indices
                area_i = self.quality_metrics.get("areas", [1])[
                    min(i, len(self.quality_metrics.get("areas", [1])) - 1)
                ]
                area_j = self.quality_metrics.get("areas", [1])[
                    min(j, len(self.quality_metrics.get("areas", [1])) - 1)
                ]

                if area_i > 0 and area_j > 0:
                    ratio = max(area_i, area_j) / min(area_i, area_j)
                    size_ratios.append(ratio)

        if size_ratios:
            print("Mesh gradation statistics:")
            print(f"  Mean size ratio: {np.mean(size_ratios):.2f}")
            print(f"  Max size ratio: {max(size_ratios):.2f}")

            # Check for abrupt size changes (more lenient for 2D)
            abrupt_changes = sum(1 for ratio in size_ratios if ratio > 3)
            if abrupt_changes > 0:
                print(f"⚠️ {abrupt_changes} abrupt size changes (ratio > 3)")
                print("   This may cause convergence issues in simulations")

        return True

    def generate_quality_report(self):
        """Generate a comprehensive quality report."""
        print("\n" + "=" * 60)
        print("  MESH QUALITY SUMMARY REPORT")
        print("=" * 60)

        # Overall assessment
        issues = []
        warnings = []

        if self.quality_metrics.get("degenerate_count", 0) > 0:
            issues.append(
                f"Degenerate elements: {self.quality_metrics['degenerate_count']}"
            )

        # Check for negative volume/area
        if self.mesh_dimension == 3:
            if self.quality_metrics.get("negative_volume_count", 0) > 0:
                issues.append(
                    f"Negative volume elements: {self.quality_metrics['negative_volume_count']}"
                )
        else:
            if self.quality_metrics.get("negative_area_count", 0) > 0:
                issues.append(
                    f"Negative area elements: {self.quality_metrics['negative_area_count']}"
                )

        aspect_ratios = self.quality_metrics.get("aspect_ratios", [])
        if aspect_ratios:
            if self.mesh_dimension == 3:
                very_poor = sum(1 for ar in aspect_ratios if ar >= 100)
                poor_threshold = 10
            else:  # 2D
                very_poor = sum(1 for ar in aspect_ratios if ar >= 20)
                poor_threshold = 5

            if very_poor > 0:
                issues.append(f"Elements with very poor aspect ratio: {very_poor}")

            poor = sum(
                1
                for ar in aspect_ratios
                if poor_threshold <= ar < (100 if self.mesh_dimension == 3 else 20)
            )
            if poor > len(aspect_ratios) * 0.1:  # More than 10% poor elements
                warnings.append(
                    f"High proportion of poor quality elements: {poor}/{len(aspect_ratios)}"
                )

        edge_lengths = self.quality_metrics.get("edge_lengths", [])
        if edge_lengths and max(edge_lengths) / min(edge_lengths) > 1e6:
            issues.append("Extreme edge length ratio detected")

        if self.mesh_dimension == 3:
            print(f"Total elements analyzed: {len(self.tetra_elements)}")
            print(
                f"Physical regions: {len([n for n in self.physical_names.values() if n[0] == 3])}"
            )
            print(
                f"Contact surfaces: {len([n for n in self.physical_names.values() if n[0] == 2 and 'contact' in n[1].lower()])}"
            )
        else:
            print(f"Total elements analyzed: {len(self.triangle_elements)}")
            print(
                f"Physical regions: {len([n for n in self.physical_names.values() if n[0] == 2])}"
            )
            print(
                f"Boundary edges: {len([n for n in self.physical_names.values() if n[0] == 1])}"
            )

        if not issues:
            print("\n✅ MESH QUALITY: EXCELLENT")
            print(
                "   No critical issues detected. Mesh should work well for simulation."
            )
        elif len(issues) == 1 and not warnings:
            print("\n⚠️ MESH QUALITY: GOOD")
            print("   Minor issues detected but mesh should be usable:")
            for issue in issues:
                print(f"   - {issue}")
        else:
            print("\n❌ MESH QUALITY: POOR")
            print("   Critical issues detected that may prevent convergence:")
            for issue in issues:
                print(f"   - {issue}")

        if warnings:
            print("\n⚠️ WARNINGS:")
            for warning in warnings:
                print(f"   - {warning}")

        # Recommendations
        print("\n📋 RECOMMENDATIONS:")
        if issues or warnings:
            print("   1. Refine mesh in problematic regions")
            print("   2. Use gradual size transitions")
            print("   3. Check geometry for sharp angles or thin features")
            print("   4. Consider using mesh optimization tools")
        else:
            print("   1. Mesh quality is excellent!")
            print("   2. Consider this mesh as a reference for future meshes")

        print("=" * 60)


def main(mesh_file):
    """Main diagnostic function."""
    print("=" * 60)
    print("  ENHANCED MESH QUALITY ANALYZER")
    print("=" * 60)
    print(f"Target mesh file: '{mesh_file}'")

    analyzer = MeshQualityAnalyzer(mesh_file)

    # Run all checks
    checks = [
        ("File Check", analyzer.check_mesh_file),
        ("Mesh Parsing", analyzer.parse_gmsh_mesh),
        ("Connectivity", analyzer.check_mesh_connectivity),
        ("Geometric Quality", analyzer.analyze_geometric_quality),
        ("Physical Regions", analyzer.check_physical_regions),
        ("Per-Group Quality", analyzer.report_per_group_quality),
        ("Contacts/Boundaries", analyzer.check_contacts_and_boundaries),
        ("Mesh Gradation", analyzer.check_mesh_gradation),
    ]

    all_passed = True
    for check_name, check_func in checks:
        try:
            result = check_func()
            if not result:
                all_passed = False
                print(f"❌ {check_name} failed")
        except Exception as e:
            print(f"❌ {check_name} failed with error: {e}")
            all_passed = False

    # Generate final report
    analyzer.generate_quality_report()

    if all_passed:
        print("\n🎉 All checks passed! Mesh is ready for simulations.")
        return 0
    print(
        "\n⚠️ Some checks failed. Review the issues above before running simulations."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
