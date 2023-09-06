from meshwell.prism import Prism

from shapely import Polygon, box
from shapely.geometry import Point, MultiPolygon

from meshwell.model import Model
from collections import OrderedDict

if __name__ == "__main__":
    model = Model()

    """
    DEFINE ENTITIES
    """

    core_thickness = 0.22
    core_width = 0.5
    core_length = 5
    box_thickness = 1
    cladding_thickness = 1

    simulation_width = 10
    simulation_length = 10
    pml_thickness = 1
    simulation_height = box_thickness + core_thickness + cladding_thickness

    # Holes
    num_holes = 5
    hole_radius = 0.1  # Define the radius of the holes
    hole_separation = 0.3  # Define the separation between the holes
    hole_offset = (
        -(num_holes - 1) * (2 * hole_radius + hole_separation) / 2 - hole_radius
    )

    # Define the centers of the num_holes holes
    hole_centers = [(i * hole_separation + hole_offset, 0) for i in range(num_holes)]

    # Create the holes as circles
    holes = [Point(center).buffer(hole_radius) for center in hole_centers]

    # Create a multipolygon from the holes
    holes_multipolygon = MultiPolygon(holes)

    # Core
    core_polygon = Polygon(
        shell=(
            (-core_length / 2, -core_width / 2),
            (core_length / 2, -core_width / 2),
            (core_length / 2, core_width / 2),
            (-core_length / 2, core_width / 2),
        ),
    )
    core_buffers = {
        0: 0.0,
        core_thickness: 0.0,
    }
    core = Prism(
        polygons=core_polygon - holes_multipolygon,
        buffers=core_buffers,
        model=model,
    )

    cladding_polygon = box(
        xmin=-simulation_length / 2,
        ymin=-simulation_width / 2,
        xmax=simulation_length / 2,
        ymax=simulation_width / 2,
    )

    box_prism = Prism(
        polygons=cladding_polygon,
        buffers={
            -box_thickness: 0.0,
            0: 0.0,
        },
        model=model,
    )

    cladding_prism = Prism(
        polygons=cladding_polygon,
        buffers={
            0: 0.0,
            cladding_thickness: 0.0,
        },
        model=model,
    )

    pml_polygon = box(
        xmin=-simulation_length / 2 - pml_thickness,
        ymin=-simulation_width / 2 - pml_thickness,
        xmax=simulation_length / 2 + pml_thickness,
        ymax=simulation_width / 2 + pml_thickness,
    )

    PML = Prism(
        polygons=pml_polygon,
        buffers={
            -box_thickness - pml_thickness: 0.0,
            cladding_thickness + pml_thickness: 0.0,
        },
        model=model,
    )

    """
    ASSEMBLE AND NAME ENTITIES
    """
    entities = OrderedDict()
    entities["core"] = core
    entities["cladding"] = cladding_prism
    entities["box"] = box_prism
    entities["PML"] = PML

    mesh = model.mesh(
        entities_dict=entities,
        verbosity=0,
        filename="mesh.msh",
    )
