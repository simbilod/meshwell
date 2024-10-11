import gmsh

from typing import Any, Dict, Optional, List
from pydantic import BaseModel, validator
from meshwell.resolution import ResolutionSpec


class GMSH_entity(BaseModel):
    """
    Delayed evaluation of a gmsh occ kernel entity.

    Attributes:
        gmsh_function: entity-defining function from model.occ
        gmsh_function_kwargs: dict of keyword arguments for gmsh_function
        dim: dimension of the object (to properly generate dimtag)
        model: meshwell model
        physical_name: name of the physical this entity wil belong to
        mesh_order: priority of the entity if it overlaps with others (lower numbers override higher numbers)
        mesh_bool: if True, entity will be meshed; if not, will not be meshed (useful to tag boundaries)
    """

    gmsh_function: Any
    gmsh_function_kwargs: Dict[str, Any]
    dimension: int
    model: Any
    physical_name: Optional[str | tuple[str]] = None
    mesh_order: float | None = None
    mesh_bool: bool = True
    additive: bool = False
    resolutions: List[ResolutionSpec] | None = None

    @validator("physical_name", pre=True, always=True)
    def _set_physical_name(cls, physical_name: Optional[str | tuple[str]]):
        if isinstance(physical_name, str):
            return [physical_name]
        else:
            return physical_name

    def instanciate(self):
        """Returns dim tag from entity."""
        entity = self.gmsh_function(**self.gmsh_function_kwargs)
        self.model.occ.synchronize()
        return [(self.dimension, entity)]


if __name__ == "__main__":
    from meshwell.model import Model

    model = Model()

    gmsh_entity1 = GMSH_entity(
        gmsh_function=model.occ.add_box,
        gmsh_function_kwargs={"x": 0, "y": 0, "z": 0, "dx": 1, "dy": 1, "dz": 1},
        dimension=3,
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
        dimension=3,
        model=model,
    )

    tag1 = gmsh_entity1.instanciate()
    tag2 = gmsh_entity2.instanciate()
    print(tag1, tag2)

    model.model.mesh.generate(3)
    gmsh.write("mesh.msh")
