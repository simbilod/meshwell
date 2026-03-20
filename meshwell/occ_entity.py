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
