# %% [markdown]
# # Mesh Quality Analysis
#
# After generating a mesh, it's important to assess its quality before using it in simulations. Poor quality meshes can lead to convergence issues, numerical instability, or incorrect results. Meshwell provides a comprehensive `MeshQualityAnalyzer` that checks various aspects of your mesh.

# %%
import numpy as np
import shapely
from meshwell.cad import cad
from meshwell.mesh import mesh
from meshwell.polysurface import PolySurface
from meshwell.quality import MeshQualityAnalyzer
from meshwell.quality import main
from meshwell.resolution import ResolutionSpec
from pathlib import Path

# %% [markdown]
# ## Creating a Mesh to Analyze
#
# First, let's create a simple 2D mesh to demonstrate the quality analyzer:

# %%
# Create geometry
polygon = shapely.box(-5, -5, 5, 5)
entity = PolySurface(polygons=polygon, physical_name="test_region", mesh_order=1)

# Generate CAD and mesh
cad(entities_list=[entity], output_file="quality_test.xao")
test_mesh = mesh(
    dim=2,
    input_file="quality_test.xao",
    output_file="quality_test.msh",
    default_characteristic_length=1.0,
)

print(f"Created mesh with {len(test_mesh.points)} vertices")

# %% [markdown]
# ## Using the Quality Analyzer
#
# The `MeshQualityAnalyzer` provides a comprehensive suite of checks. You can run all checks at once using the `main()` function:

# %%

# Run comprehensive quality analysis
exit_code = main("quality_test.msh")

# %% [markdown]
# The analyzer performs the following checks:
#
# 1. **File Check**: Verifies the mesh file exists and is readable
# 2. **Mesh Parsing**: Loads nodes, elements, and physical groups from GMSH format
# 3. **Connectivity**: Checks for orphaned nodes, non-manifold edges/faces
# 4. **Geometric Quality**: Analyzes aspect ratios, volumes/areas, angles, edge lengths
# 5. **Physical Regions**: Reports element counts per physical group
# 6. **Contacts/Boundaries**: Lists physical groups by dimension
# 7. **Mesh Gradation**: Detects abrupt size changes between adjacent elements

# %% [markdown]
# ## Programmatic Access
#
# You can also use the analyzer programmatically to access specific metrics:

# %%
# Create analyzer instance
analyzer = MeshQualityAnalyzer("quality_test.msh")

# Run parsing
analyzer.check_mesh_file()
analyzer.parse_gmsh_mesh()

# Run specific checks
print("\n=== Manual Analysis ===")
analyzer.check_mesh_connectivity()
analyzer.analyze_geometric_quality()

# Access computed metrics
metrics = analyzer.quality_metrics
print(
    f"\nAspect ratio range: {min(metrics['aspect_ratios']):.2f} - {max(metrics['aspect_ratios']):.2f}"
)
print(f"Mean aspect ratio: {np.mean(metrics['aspect_ratios']):.2f}")

# %% [markdown]
# ## Understanding Quality Metrics
#
# ### Aspect Ratio
# The ratio of longest to shortest edge in an element. Lower is better:
# - **Excellent**: AR < 2 (2D) or AR < 3 (3D)
# - **Good**: 2-5 (2D) or 3-10 (3D)
# - **Poor**: 5-20 (2D) or 10-100 (3D)
# - **Very Poor**: >20 (2D) or >100 (3D)
#
# ### Angles
# Interior angles of triangular/tetrahedral elements:
# - **Ideal**: Close to 60° for equilateral triangles/regular tetrahedra
# - **Warning**: < 5° or > 150°
# - **Critical**: < 1° (can cause numerical instability)
#
# ### Mesh Gradation
# Size ratio between adjacent elements:
# - **Good**: Ratio < 2
# - **Acceptable**: Ratio < 3 (2D) or < 5 (3D)
# - **Poor**: Ratio > 3 (2D) or > 5 (3D) - may cause convergence issues

# %% [markdown]
# ## Per-Group Quality Metrics
#
# The analyzer can also report quality metrics broken down by physical group, helping you identify which specific regions have quality issues.
#
# Let's create a mesh with multiple regions to demonstrate:

# %%
# Create geometry with two regions - one with fine mesh, one with coarse
poly1 = shapely.box(-5, -5, 0, 0)
poly2 = shapely.box(0, 0, 5, 5)

region1 = PolySurface(polygons=poly1, physical_name="fine_region", mesh_order=1)
region2 = PolySurface(polygons=poly2, physical_name="coarse_region", mesh_order=1)

cad(entities_list=[region1, region2], output_file="multi_region.xao")

# Generate mesh with different sizes in each region

fine_spec = ResolutionSpec(resolution=0.3, apply_to="surfaces")
coarse_spec = ResolutionSpec(resolution=1.5, apply_to="surfaces")

multi_mesh = mesh(
    dim=2,
    input_file="multi_region.xao",
    output_file="multi_region.msh",
    default_characteristic_length=1.0,
    resolution_specs={
        "fine_region": [fine_spec],
        "coarse_region": [coarse_spec],
    },
)

print(f"Multi-region mesh: {len(multi_mesh.points)} vertices")

# Analyze with per-group reporting
analyzer_multi = MeshQualityAnalyzer("multi_region.msh")
analyzer_multi.check_mesh_file()
analyzer_multi.parse_gmsh_mesh()
analyzer_multi.analyze_geometric_quality()
analyzer_multi.report_per_group_quality()

# %% [markdown]
# The per-group quality report shows:
# - **Aspect ratios** (min/max/mean) for each physical group
# - **Areas or volumes** statistics per group
# - **Minimum angles** per group
# - **Warnings** for elements with poor quality in specific groups
#
# This helps you:
# - Identify which regions need refinement
# - Compare quality across different mesh zones
# - Focus optimization efforts on problematic areas
# - Understand quality distribution in multi-material or multi-physics simulations


# %% [markdown]
# ## Interpreting the Quality Report
#
# The final quality report categorizes meshes as:
#
# - **✅ EXCELLENT**: No critical issues, mesh ready for simulation
# - **⚠️ GOOD**: Minor issues present but mesh is usable
# - **❌ POOR**: Critical issues that may prevent convergence
#
# Common issues and solutions:
#
# | Issue | Cause | Solution |
# |-------|-------|----------|
# | High aspect ratios | Stretched elements near boundaries | Refine mesh or use boundary layers |
# | Degenerate elements | Overlapping nodes or zero volume | Check geometry for self-intersections |
# | Extreme edge ratios | Mixed coarse/fine regions | Use gradual size transitions |
# | Very small angles | Sharp geometric features | Smooth geometry or use local refinement |
# | Non-manifold edges | T-junctions or inconsistent connectivity | Fix geometry or remesh |


# %%
# Clean up files

for f in [
    "quality_test.xao",
    "quality_test.msh",
    "multi_region.xao",
    "multi_region.msh",
    "poor_quality.xao",
    "poor_quality.msh",
    "quality_3d.xao",
    "quality_3d.msh",
]:
    Path(f).unlink(missing_ok=True)
