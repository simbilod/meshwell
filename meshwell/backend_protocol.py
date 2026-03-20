"""CAD backend protocol definition."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class CADBackend(Protocol):
    """Protocol defining the interface for CAD backends."""

    def process_entities(
        self,
        entities: list[Any],
        **kwargs,
    ) -> list[Any]:
        """Process a list of entities into CAD shapes."""
        ...

    def save_checkpoint(self, path: Path) -> None:
        """Save the current CAD state to a file."""
        ...

    def to_gmsh_model(self, model_manager: Any) -> None:
        """Inject the processed CAD shapes into a GMSH model."""
        ...
