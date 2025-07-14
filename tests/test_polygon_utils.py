import numpy as np
from shapely.geometry import Polygon
from meshwell.polygon_utils import decompose


def test_decompose():
    # Parameters
    r_outer = 10
    w = 2
    r_inner = r_outer - w
    theta_start = 0
    theta_end = np.pi / 2  # 90 degrees

    # Outer arc points (counter-clockwise)
    outer_arc = [
        (r_outer * np.cos(theta), r_outer * np.sin(theta))
        for theta in np.linspace(theta_start, theta_end, 30)
    ]

    # Inner arc points (clockwise)
    inner_arc = [
        (r_inner * np.cos(theta), r_inner * np.sin(theta))
        for theta in np.linspace(theta_end, theta_start, 30)
    ]

    # Combine to form the quarter ring
    quarter_ring_coords = outer_arc + inner_arc

    quarter_ring = Polygon(quarter_ring_coords)
    decomposed_quarter_ring = decompose(quarter_ring)

    import matplotlib.pyplot as plt

    x, y = quarter_ring.exterior.xy

    for obj, points in decomposed_quarter_ring:
        if obj == "line":
            plt.plot(*zip(*points), color="blue")
        elif obj == "arc":
            plt.plot(*zip(*points), color="red")

    plt.scatter(x, y, "ko")
    plt.gca().set_aspect("equal")
    plt.title("Quarter Ring Polygon")
    plt.show()


def test_decompose_complex():
    # Parameters for arcs
    r1 = 8
    r2 = 5
    theta1 = np.pi / 4
    theta2 = 3 * np.pi / 4

    # Arc 1 (counter-clockwise)
    arc1 = [
        (r1 * np.cos(theta), r1 * np.sin(theta))
        for theta in np.linspace(theta1, theta2, 20)
    ]

    # Line segment
    line1 = [
        (r1 * np.cos(theta2), r1 * np.sin(theta2)),
        (r2 * np.cos(theta2), r2 * np.sin(theta2)),
    ]

    # Arc 2 (clockwise)
    arc2 = [
        (r2 * np.cos(theta), r2 * np.sin(theta))
        for theta in np.linspace(theta2, theta1, 20)
    ]

    # Wavy section: add slight waviness around the straight line from arc2 end to arc1 start
    start = (r2 * np.cos(theta1), r2 * np.sin(theta1))
    end = (r1 * np.cos(theta1), r1 * np.sin(theta1))
    t = np.linspace(0, 1, 40)
    # Compute direction vector
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = np.hypot(dx, dy)
    # Perpendicular direction for waviness
    perp = (-dy / length, dx / length)
    amplitude = 0.3  # small amplitude for slight waviness
    frequency = 4  # number of waves along the segment
    wavy = [
        (
            start[0]
            + dx * ti
            + amplitude * np.sin(2 * np.pi * frequency * ti) * perp[0],
            start[1]
            + dy * ti
            + amplitude * np.sin(2 * np.pi * frequency * ti) * perp[1],
        )
        for ti in t
    ]

    # Combine all sections
    complex_coords = arc1 + line1 + arc2 + wavy

    complex_poly = Polygon(complex_coords)
    decomposed = decompose(complex_poly)

    print("Decomposed parts:")
    for obj, points in decomposed:
        print(f"{obj}: {len(points)} points")

    import matplotlib.pyplot as plt

    x, y = complex_poly.exterior.xy

    for obj, points in decomposed:
        if obj == "long_line":
            plt.plot(*zip(*points), color="blue")
        elif obj == "arc":
            plt.plot(*zip(*points), color="red")
        else:
            plt.plot(*zip(*points), color="green")

    plt.scatter(x, y, color="k", marker=".")
    plt.gca().set_aspect("equal")
    plt.title("Complex Polygon with Lines, Arcs, and Wavy Section")
    plt.show()


if __name__ == "__main__":
    # test_decompose()
    test_decompose_complex()
