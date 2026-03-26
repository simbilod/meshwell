"""OCC entity definitions for wrapping arbitrary OCP shape-creating functions."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from meshwell.validation import format_physical_name

if TYPE_CHECKING:
    from OCP.TopoDS import TopoDS_Shape


class OCC_entity:
    """Delayed evaluation of an OCC shape-creating function.

    Attributes:
        occ_function: callable that returns a TopoDS_Shape
        physical_name: name(s) of the physical this entity will belong to
        mesh_order: priority of the entity if it overlaps with others (lower numbers override higher numbers)
        mesh_bool: if True, entity will be meshed; if not, will not be meshed
        additive: if True, entity will be added (fused) rather than subtracted (cut)
        dimension: dimension of the created entity
    """

    def __init__(
        self,
        occ_function: callable,
        physical_name: str | tuple[str, ...] | None = None,
        mesh_order: float | None = None,
        mesh_bool: bool = True,
        additive: bool = False,
        dimension: int | None = None,
    ):
        self.occ_function = occ_function
        self.physical_name = format_physical_name(physical_name)
        self.mesh_order = mesh_order
        self.mesh_bool = mesh_bool
        self.additive = additive
        self.dimension = dimension

    def instanciate_occ(self) -> TopoDS_Shape:
        """Execute the OCC function and return the shape."""
        return self.occ_function()

    def instanciate(self, cad_model: Any) -> Any:
        """GMSH instantiation is not supported for OCC_entity."""
        raise NotImplementedError(
            "OCC_entity only supports the OCC CAD backend. "
            "Use GMSH_entity for the GMSH backend."
        )

    def to_dict(self) -> dict:
        """Convert entity to dictionary representation.

        Returns:
            Dictionary containing serializable entity data
        """
        from functools import partial

        if isinstance(self.occ_function, partial):
            func_name = self.occ_function.func.__name__
            args = self.occ_function.args
            keywords = self.occ_function.keywords
        else:
            func_name = self.occ_function.__name__
            args = ()
            keywords = {}

        return {
            "type": "OCC_entity",
            "function_name": func_name,
            "args": args,
            "keywords": keywords,
            "physical_name": self.physical_name,
            "mesh_order": self.mesh_order,
            "mesh_bool": self.mesh_bool,
            "additive": self.additive,
            "dimension": self.dimension,
        }

    @classmethod
    def from_dict(cls, data: dict, registry: dict[str, callable] | None = None) -> "OCC_entity":
        """Reconstruct entity from dictionary representation.

        Args:
            data: Dictionary containing entity data
            registry: Optional dictionary mapping function names to callables.
                     Used to resolve the occ_function.

        Returns:
            OCC_entity instance
        """
        from functools import partial

        func_name = data["function_name"]

        if registry and func_name in registry:
            func = registry[func_name]
        else:
            # Try to resolve from common OCP locations if not in registry
            # This is a bit complex for OCP as it has many modules.
            # For now we rely on the registry or it being available in globals.
            raise ValueError(
                f"Function {func_name} not found in registry. "
                "Please provide a registry to from_dict for OCC_entity."
            )

        p_func = partial(func, *data["args"], **data["keywords"])

        return cls(
            occ_function=p_func,
            physical_name=data["physical_name"],
            mesh_order=data["mesh_order"],
            mesh_bool=data["mesh_bool"],
            additive=data["additive"],
            dimension=data["dimension"],
        )
