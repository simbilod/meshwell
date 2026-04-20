"""OCC CAD backend implementation."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from meshwell.cad_occ import CAD_OCC
from meshwell.occ_to_gmsh import inject_occ_entities_into_gmsh, occ_to_xao


class OccBackend:
    """CAD backend using OpenCASCADE (via OCP)."""

    def __init__(
        self,
        progress_bars: bool = False,
        remove_all_duplicates: bool = False,
        **kwargs,
    ):
        self.processor = CAD_OCC(**kwargs)
        self.results = []
        self.progress_bars = progress_bars
        self.remove_all_duplicates = remove_all_duplicates

    def process_entities(self, entities: list[Any], **kwargs) -> list[Any]:
        """Process entities using OCC backend."""
        kwargs.setdefault("progress_bars", self.progress_bars)
        self.results = self.processor.process_entities(entities, **kwargs)
        return self.results

    def save_checkpoint(self, path: Path) -> None:
        """Save OCC results to XAO."""
        occ_to_xao(
            self.results,
            path,
            progress_bars=self.progress_bars,
            remove_all_duplicates=self.remove_all_duplicates,
        )

    def to_gmsh_model(self, model_manager: Any) -> None:
        """Inject OCC shapes into GMSH model."""
        inject_occ_entities_into_gmsh(
            self.results,
            model_manager,
            progress_bars=self.progress_bars,
            remove_all_duplicates=self.remove_all_duplicates,
        )
