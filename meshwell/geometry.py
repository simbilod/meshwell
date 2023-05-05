class Geometry:
    """
    Base Geometry class inherited by Prism and PolySurface.

    Attributes:
        polygons: list of shapely (Multi)Polygon
        buffers: dict of {z: buffer} used to shrink/grow base polygons at specified z-values
        model: GMSH model to synchronize
    """

    def __init__(
        self,
        model,
        polygons,
        buffers,
    ):
        self.model = model

        # Parse buffers
        self.buffered_polygons = self._get_buffered_polygons(polygons, buffers)

        # Track gmsh entities for bottom-up volume definition
        self.points = {}
        self.segments = {}

    def _add_get_point(self, x, y, z):
        """Add a point to the model, or reuse a previously-defined point.
        Args:
            x: float, x-coordinate
            y: float, y-coordinate
            z: float, z-coordinate
        Returns:
            ID of the added or retrieved point
        """
        if (x, y, z) not in self.points.keys():
            self.points[(x, y, z)] = self.model.add_point(x, y, z)
        return self.points[(x, y, z)]

    def _add_get_segment(self, xyz1, xyz2):
        """Add a segment (2-point line) to the gmsh model, or retrieve a previously-defined segment.
        The OCC kernel does not care about orientation.
        Args:
            xyz1: first [x,y,z] coordinate
            xyz2: second [x,y,z] coordinate
        Returns:
            ID of the added or retrieved line segment
        """
        if (xyz1, xyz2) in self.segments.keys():
            return self.segments[(xyz1, xyz2)]
        elif (xyz2, xyz1) in self.segments.keys():
            return self.segments[(xyz2, xyz1)]
        else:
            self.segments[(xyz1, xyz2)] = self.model.add_line(
                self._add_get_point(xyz1[0], xyz1[1], xyz1[2]),
                self._add_get_point(xyz2[0], xyz2[1], xyz2[2]),
            )
            return self.segments[(xyz1, xyz2)]

    def _channel_loop_from_vertices(self, vertices):
        """Add a curve loop from the list of vertices.
        Args:
            model: GMSH model
            vertices: list of [x,y,z] coordinates
        Returns:
            ID of the added curve loop
        """
        edges = []
        for vertex1, vertex2 in [
            (vertices[i], vertices[i + 1]) for i in range(len(vertices) - 1)
        ]:
            gmsh_line = self._add_get_segment(vertex1, vertex2)
            edges.append(gmsh_line)
        return self.model.add_curve_loop(edges)

    def _add_surface(self, vertices):
        """Add a surface composed of the segments formed by vertices.

        Args:
            vertices: List of xyz coordinates, whose subsequent entries define a closed loop.
        Returns:
            ID of the added surface
        """
        channel_loop = self._channel_loop_from_vertices(vertices)
        return self.model.add_plane_surface([channel_loop])
