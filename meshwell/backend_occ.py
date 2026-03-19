from __future__ import annotations

from pathlib import Path
from typing import Any

from meshwell.cad_occ import CAD_OCC
from meshwell.occ_to_gmsh import inject_occ_entities_into_gmsh, occ_to_xao


class OccBackend:
    """CAD backend using OpenCASCADE (via OCP)."""

    def __init__(self, **kwargs):
        self.processor = CAD_OCC(**kwargs)
        self.results = []

    def process_entities(self, entities: list[Any], **kwargs) -> list[Any]:
        """Process entities using OCC backend."""
        self.results = self.processor.process_entities(entities, **kwargs)
        return self.results

    def save_checkpoint(self, path: Path) -> None:
        """Save OCC results to XAO."""
        occ_to_xao(self.results, path)

    def to_gmsh_model(self, model_manager: Any) -> None:
        """Inject OCC shapes into GMSH model."""
        inject_occ_entities_into_gmsh(self.results, model_manager)
