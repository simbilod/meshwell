from dataclasses import dataclass
from shapely import Polygon, MultiPolygon


@dataclass
class ProcessStep:
    """Generic process step."""

    name: str


@dataclass
class Grow(ProcessStep):
    """Growth or deposition operation."""

    thickness: float
    axis: tuple[float, float, float] = (1, 1, 1)
    mask: Polygon | MultiPolygon | None = None
    isotropic: bool = True
    fillet_fraction: float = 1.0

    @property
    def a(self):
        # return self.axis[0] / abs(sum(self.axis))**2 * self.thickness
        return self.axis[0]

    @property
    def b(self):
        # return self.axis[1] / abs(sum(self.axis))**2 * self.thickness
        return self.axis[1]

    @property
    def c(self):
        # return self.axis[2] / abs(sum(self.axis))**2 * self.thickness
        return self.axis[2]

    def apply(self, hull_dimtags, model):
        dim, tag = hull_dimtags[0]
        initial_hull = model.occ.copy(hull_dimtags)

        # Get bounding box to calculate scale factors
        xmin, ymin, zmin, xmax, ymax, zmax = model.occ.getBoundingBox(dim, tag)

        # Calculate center for dilation
        x = (xmin + xmax) / 2
        y = (ymin + ymax) / 2
        z = (zmin + zmax) / 2

        # Calculate dimensions
        width = xmax - xmin
        height = ymax - ymin
        depth = zmax - zmin

        # Calculate scale factors to achieve desired growth
        # For each axis: new_size = size * scale_factor
        # Therefore: size * scale_factor - size = 2 * thickness
        # So: scale_factor = (size + 2*thickness) / size
        scale_x = (width + 2 * self.thickness * self.a) / width if width != 0 else 1
        scale_y = (height + 2 * self.thickness * self.b) / height if height != 0 else 1
        scale_z = (depth + 2 * self.thickness * self.c) / depth if depth != 0 else 1

        # Perform dilation with calculated scale factors
        model.occ.dilate(hull_dimtags, x, y, z, scale_x, scale_y, scale_z)
        model.occ.synchronize()

        # if self.isotropic:
        #     # Rest of filleting code remains the same...
        #     if hull_dimtags[0] == 3:  # 3D volume
        #         faces = model.getBoundary([hull_dimtags])
        #         edges = model.getBoundary(faces, combined=False, recursive=False)
        #     elif hull_dimtags[0] == 2:  # 2D surface
        #         edges = model.getBoundary([hull_dimtags])
        #     else:
        #         raise ValueError(f"Unsupported dimension for filleting: {hull_dimtags[0]}")

        #     if not edges:
        #         raise RuntimeError("No edges found for filleting")

        #     radius = self.thickness * self.fillet_fraction

        #     try:
        #         hull_dimtags = model.occ.fillet([tag],
        #                     [e[1] for e in edges],
        #                     [radius])
        #     except Exception as e:
        #         print(f"Warning: Fillet operation failed with radius {radius}. "
        #             f"Trying with half radius...")
        #         try:
        #             hull_dimtags = model.occ.fillet([tag],
        #                         [e[1] for e in edges],
        #                         [radius/2])
        #         except Exception as e:
        #             print(f"Warning: Fillet operation failed even with reduced radius. "
        #                 f"Continuing without filleting.")

        # Cut with initial hull
        model.occ.synchronize()
        growth = model.occ.cut(
            hull_dimtags, initial_hull, removeTool=True, removeObject=True
        )

        model.occ.synchronize()
        return growth[0]


class Etch(Grow):
    """Etch operation."""
