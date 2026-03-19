"""GMSH CAD backend implementation."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from meshwell.cad_gmsh import CAD


class GmshBackend:
    """CAD backend using GMSH directly."""

    def __init__(self, **kwargs):
        self.processor = CAD(**kwargs)
        self.results = []

    def process_entities(self, entities: list[Any], **kwargs) -> list[Any]:
        """Process entities using GMSH backend."""
        self.results = self.processor.process_entities(entities, **kwargs)
        return self.results

    def save_checkpoint(self, path: Path) -> None:
        """Save GMSH model to XAO."""
        self.processor.to_xao(path)

    def to_gmsh_model(self, model_manager: Any) -> None:  # noqa: ARG002
        """This is a no-op as the GMSH backend already populates the model.

        Wait, if model_manager is DIFFERENT from self.processor.model_manager,
        we might need to copy. But usually they share the same singleton gmsh model.
        """
        # Ensure synchronization
        self.processor.model_manager.sync_model()
