import gmsh
import sys

gmsh.initialize()
gmsh.model.add("extruded_surface_occ")

# Create points for base surface (z=0)
p1 = gmsh.model.occ.addPoint(0, 0, 0)
p2 = gmsh.model.occ.addPoint(1.5, 0, 0)
p3 = gmsh.model.occ.addPoint(1.2, 1.3, 0)
p4 = gmsh.model.occ.addPoint(0.3, 0.8, 0)
p5 = gmsh.model.occ.addPoint(0.7, 0.4, 0)

# Create base curves
base_curves = [
    gmsh.model.occ.addLine(p1, p2),  # Straight line
    gmsh.model.occ.addCircleArc(p2, p5, p3),  # Arc using control point
    gmsh.model.occ.addBSpline([p3, p5, p4]),  # BSpline through points
    gmsh.model.occ.addLine(p4, p1),  # Straight line
]

# Create base wire and surface
base_wire = gmsh.model.occ.addWire(base_curves)
base_surface = gmsh.model.occ.addSurfaceFilling(base_wire)
gmsh.model.occ.synchronize()

# Copy and translate the surface to z=1
top_entities = gmsh.model.occ.copy([(2, base_surface)])
gmsh.model.occ.translate(top_entities, 0, 0, 1.0)
# Scale the top surface to be 1.5x larger
com = gmsh.model.occ.getCenterOfMass(2, base_surface)
gmsh.model.occ.dilate(top_entities, com[0], com[1], 1.0, 1.5, 1.5, 1.0)
top_surface = top_entities[0][1]

# Get boundary curves of both surfaces
gmsh.model.occ.synchronize()
base_bounds = gmsh.model.getBoundary([(2, base_surface)])
top_bounds = gmsh.model.getBoundary([(2, top_surface)])

# Create vertical lines connecting corresponding points
side_curves = []
for (_, base_curve), (_, top_curve) in zip(base_bounds, top_bounds):
    # Get start points of each curve
    base_points = gmsh.model.getBoundary([(1, base_curve)])
    top_points = gmsh.model.getBoundary([(1, top_curve)])

    # Create vertical lines
    for (_, base_pt), (_, top_pt) in zip(base_points, top_points):
        side_curves.append(gmsh.model.occ.addLine(base_pt, top_pt))

# Create side surfaces
side_surfaces = []
for i in range(len(base_curves)):
    # Create wire for each side face
    side_wire = gmsh.model.occ.addWire(
        [
            base_curves[i],  # Bottom edge
            side_curves[i * 2],  # First vertical edge
            side_curves[i * 2 + 1],  # Second vertical edge
            -top_bounds[i][1],  # Top edge (negative for orientation)
        ]
    )
    # Create surface from wire
    side_surfaces.append(gmsh.model.occ.addSurfaceFilling(side_wire))

gmsh.model.occ.synchronize()

# Heal the geometry
gmsh.model.occ.removeAllDuplicates()
gmsh.model.occ.synchronize()


# Create surface loop from all surfaces
surface_loop = gmsh.model.occ.addSurfaceLoop(
    [
        base_surface,  # Bottom surface
        top_surface,  # Top surface
        *side_surfaces,  # Side surfaces
    ]
)

# Create volume from surface loop
volume = gmsh.model.occ.addVolume([surface_loop])
gmsh.model.occ.synchronize()

# Set mesh order to 2 for curvilinear elements
gmsh.option.setNumber("Mesh.ElementOrder", 2)

# Enable high-order optimization to improve element quality
gmsh.option.setNumber("Mesh.HighOrderOptimize", 2)

# Generate mesh
gmsh.model.mesh.generate(3)

if "-nopopup" not in sys.argv:
    gmsh.fltk.run()

gmsh.finalize()
