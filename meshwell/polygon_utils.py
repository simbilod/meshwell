import shapely
from math import isclose
import numpy as np


def decompose(polygon: shapely.geometry.Polygon) -> list:
    """
    Decomposes a polygon into its constituent parts, including arcs.

    Args:
        polygon (shapely.geometry.Polygon): The polygon to decompose.

    Returns:
        list: A list of decomposed parts, which can include arcs.
    """

    def is_colinear(p1, p2, p3, tol=1e-6):
        # Area of triangle method
        area = abs(
            (
                p1[0] * (p2[1] - p3[1])
                + p2[0] * (p3[1] - p1[1])
                + p3[0] * (p1[1] - p2[1])
            )
            / 2.0
        )
        return area < tol

    def fit_circle(points):
        # Fit a circle to 3 points
        A = np.array(
            [
                [2 * (points[1][0] - points[0][0]), 2 * (points[1][1] - points[0][1])],
                [2 * (points[2][0] - points[0][0]), 2 * (points[2][1] - points[0][1])],
            ]
        )
        b = np.array(
            [
                points[1][0] ** 2
                + points[1][1] ** 2
                - points[0][0] ** 2
                - points[0][1] ** 2,
                points[2][0] ** 2
                + points[2][1] ** 2
                - points[0][0] ** 2
                - points[0][1] ** 2,
            ]
        )
        try:
            center = np.linalg.solve(A, b)
            r = np.linalg.norm(np.array(points[0]) - center)
            return center, r
        except np.linalg.LinAlgError:
            return None, None

    def points_on_circle(points, center, r, dist_tol):
        for pt in points:
            if not isclose(np.linalg.norm(np.array(pt) - center), r, abs_tol=dist_tol):
                return False
        return True

    def decompose_segments(coords, dist_tol=1e-2, min_arc_points=5):
        n = len(coords)
        i = 0
        result = []
        while i < n - 2:
            # Try to find a straight segment
            j = i + 2
            while j < n and is_colinear(coords[i], coords[j - 1], coords[j]):
                j += 1
            if j > i + 2:
                result.append(("line", coords[i:j]))
                i = j - 1
                continue
            # Try to find a circular arc
            if i + min_arc_points <= n:
                arc_found = False
                for k in range(i + min_arc_points, n + 1):
                    center, r = fit_circle([coords[i], coords[i + 1], coords[k - 1]])
                    if center is not None and points_on_circle(
                        coords[i:k], center, r, dist_tol
                    ):
                        arc_found = True
                    else:
                        if arc_found:
                            result.append(("arc", coords[i : k - 1]))
                            i = k - 2
                        break
                if not arc_found:
                    result.append(("line", coords[i : i + 2]))
                    i += 1
            else:
                result.append(("line", coords[i : i + 2]))
                i += 1
        if i < n - 1:
            result.append(("line", coords[i:n]))
        return result

    coords = list(polygon.exterior.coords)
    decomposed = decompose_segments(coords)

    return decomposed
