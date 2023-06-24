import gmsh


class GMSH_entity:
    """
    Delayed evaluation of a gmsh occ kernel entity.

    Attributes:
        gmsh_function: entity-defining function from model.occ
        gmsh_function_kwargs: dict of keyword arguments for gmsh_function
        dim: dimension of the object (to properly generate dimtag)
        model: meshwell model
    """

    def __init__(
        self,
        gmsh_function,
        gmsh_function_kwargs,
        dim,
        model,
    ):
        self.gmsh_function = gmsh_function
        self.gmsh_function_kwargs = gmsh_function_kwargs
        self.dim = dim
        self.model = model

    def instanciate(self):
        """Returns dim tag from entity."""
        entity = self.gmsh_function(**self.gmsh_function_kwargs)
        self.model.occ.synchronize()
        return [(self.dim, entity)]


if __name__ == "__main__":
    from meshwell.model import Model

    model = Model()

    gmsh_entity1 = GMSH_entity(
        gmsh_function=model.occ.add_box,
        gmsh_function_kwargs={"x": 0, "y": 0, "z": 0, "dx": 1, "dy": 1, "dz": 1},
        dim=3,
        model=model,
    )

    gmsh_entity2 = GMSH_entity(
        gmsh_function=model.occ.addCylinder,
        gmsh_function_kwargs={
            "x": 2,
            "y": 2,
            "z": 2,
            "dx": 1,
            "dy": 0,
            "dz": 0,
            "r": 1,
            "angle": 1.0,
        },
        dim=3,
        model=model,
    )

    tag1 = gmsh_entity1.instanciate()
    tag2 = gmsh_entity2.instanciate()
    print(tag1, tag2)

    model.model.mesh.generate(3)
    gmsh.write("mesh.msh")
