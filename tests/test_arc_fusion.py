import numpy as np
from shapely.geometry import Polygon

from meshwell.cad_occ import cad_occ
from meshwell.occ_xao_writer import occ_to_xao
from meshwell.polyprism import PolyPrism


def create_bend_polygon(r_inner, r_outer, angle_deg, num_points=50):
    angle_rad = np.deg2rad(angle_deg)
    angles = np.linspace(0, angle_rad, num_points)
    inner_pts = [(r_inner * np.cos(a), r_inner * np.sin(a)) for a in angles]
    outer_pts = [(r_outer * np.cos(a), r_outer * np.sin(a)) for a in reversed(angles)]
    return Polygon(inner_pts + outer_pts + [inner_pts[0]])


def test_arc_cad_fusion():
    # CPW bend parameters
    r_center = 20.0
    trace_width = 5.0
    gap = 2.0
    ground_width = 10.0

    # Trace
    poly_trace = create_bend_polygon(
        r_inner=r_center - trace_width / 2,
        r_outer=r_center + trace_width / 2,
        angle_deg=90.0,
    )

    # Inner ground
    poly_gnd_in = create_bend_polygon(
        r_inner=r_center - trace_width / 2 - gap - ground_width,
        r_outer=r_center - trace_width / 2 - gap,
        angle_deg=90.0,
    )

    # Outer ground
    poly_gnd_out = create_bend_polygon(
        r_inner=r_center + trace_width / 2 + gap,
        r_outer=r_center + trace_width / 2 + gap + ground_width,
        angle_deg=90.0,
    )

    # Create an overlapping straight waveguide block to trigger fragmentation and fusing
    poly_straight = Polygon([(-5.0, 15.0), (5.0, 15.0), (5.0, 25.0), (-5.0, 25.0)])

    prism_trace = PolyPrism(
        polygons=poly_trace,
        buffers={0.0: 0.0, 5.0: 0.0},
        physical_name="trace",
        identify_arcs=True,
        arc_tolerance=1e-3,
        mesh_order=1,
    )

    prism_gnd_in = PolyPrism(
        polygons=poly_gnd_in,
        buffers={0.0: 0.0, 5.0: 0.0},
        physical_name="gnd_in",
        identify_arcs=True,
        arc_tolerance=1e-3,
        mesh_order=1,
    )

    prism_gnd_out = PolyPrism(
        polygons=poly_gnd_out,
        buffers={0.0: 0.0, 5.0: 0.0},
        physical_name="gnd_out",
        identify_arcs=True,
        arc_tolerance=1e-3,
        mesh_order=1,
    )

    prism_straight = PolyPrism(
        polygons=poly_straight,
        buffers={0.0: 0.0, 5.0: 0.0},
        physical_name="straight",
        identify_arcs=False,  # straight block
        mesh_order=2,
    )

    # Process entities with OCC fragmenter, then serialize to XAO via gmsh
    occ_entities = cad_occ([prism_trace, prism_gnd_in, prism_gnd_out, prism_straight])
    occ_to_xao(occ_entities, "output.xao")


if __name__ == "__main__":
    test_arc_cad_fusion()
    print("Test passed!")
