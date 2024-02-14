from meshwell.prism import Prism

from shapely import Polygon, box

from meshwell.model import Model

if __name__ == "__main__":
    model = Model()

    """
    DEFINE ENTITIES
    """

    simulation_width = 1
    simulation_length = 1

    substrate_thickness = 1
    cladding_thickness = 1
    pml_thickness = 1

    device_thickness = 0.1
    device_polygon = Polygon(
        shell=((0.25, 0.25), (0.25, 0.75), (0.75, 0.75), (0.25, 0.25))
    )
    device_buffers = {
        0: 0.0,
        device_thickness: 0.0,
    }

    device = Prism(
        polygons=device_polygon,
        buffers={
            0: 0.0,
            device_thickness: 0.0,
        },
        model=model,
        physical_name="device",
        mesh_order=1,
        resolution={"resolution": 0.05, "SizeMax": 1.0, "DistMax": 1.0},
    )

    background_polygon = box(
        xmin=0, ymin=0, xmax=simulation_length, ymax=simulation_width
    )

    substrate = Prism(
        polygons=background_polygon,
        buffers={
            -substrate_thickness: 0.0,
            0: 0.0,
        },
        model=model,
        physical_name="substrate",
        mesh_order=2,
    )

    cladding = Prism(
        polygons=background_polygon,
        buffers={
            0: 0.0,
            cladding_thickness: 0.0,
        },
        model=model,
        physical_name="cladding",
        mesh_order=3,
        resolution={"resolution": 0.5, "SizeMax": 1.0, "DistMax": 1.0},
    )

    PML_cladding = Prism(
        polygons=background_polygon,
        buffers={
            cladding_thickness: 0.0,
            cladding_thickness + pml_thickness: 0.0,
        },
        model=model,
        physical_name="PML_cladding",
        mesh_order=1,
    )

    PML_substrate = Prism(
        polygons=background_polygon,
        buffers={
            -substrate_thickness: 0.0,
            -substrate_thickness - pml_thickness: 0.0,
        },
        model=model,
        physical_name="PML_substrate",
        mesh_order=1,
    )

    """
    BOUNDARIES
    """

    right_polygon = box(
        xmin=simulation_length,
        ymin=0,
        xmax=simulation_length + 1,
        ymax=simulation_width,
    )
    right = Prism(
        polygons=right_polygon,
        buffers={
            -substrate_thickness - pml_thickness: 0.0,
            cladding_thickness + pml_thickness: 0.0,
        },
        model=model,
        physical_name="right",
        mesh_bool=False,
        mesh_order=0,
    )

    left_polygon = box(xmin=-1, ymin=0, xmax=0, ymax=simulation_width)
    left = Prism(
        polygons=left_polygon,
        buffers={
            -substrate_thickness - pml_thickness: 0.0,
            cladding_thickness + pml_thickness: 0.0,
        },
        model=model,
        physical_name="left",
        mesh_bool=False,
        mesh_order=0,
    )

    up_polygon = box(
        xmin=0, ymin=simulation_width, xmax=simulation_length, ymax=simulation_width + 1
    )
    up = Prism(
        polygons=up_polygon,
        buffers={
            -substrate_thickness - pml_thickness: 0.0,
            cladding_thickness + pml_thickness: 0.0,
        },
        model=model,
        physical_name="up",
        mesh_bool=False,
        mesh_order=0,
    )

    down_polygon = box(xmin=0, ymin=-1, xmax=simulation_length, ymax=0)
    down = Prism(
        polygons=down_polygon,
        buffers={
            -substrate_thickness - pml_thickness: 0.0,
            cladding_thickness + pml_thickness: 0.0,
        },
        model=model,
        physical_name="down",
        mesh_bool=False,
        mesh_order=0,
    )

    """
    ASSEMBLE AND NAME ENTITIES
    """
    entities = [
        substrate,
        device,
        cladding,
        PML_cladding,
        PML_substrate,
        up,
        down,
        left,
        right,
    ]

    mesh = model.mesh(
        entities_list=entities,
        verbosity=0,
        filename="mesh.msh",
        periodic_entities=[
            (x + "___" + s1, x + "___" + s2)
            for x in ("cladding", "substrate", "PML_cladding", "PML_substrate")
            for (s1, s2) in (("left", "right"), ("up", "down"))
        ],
    )
