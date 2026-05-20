"""BREP bytes round-trip via tempfile.

OCP's BRepTools only exposes file-path Write/Read. We need a bytes-level
hand-off for the process-pool path in cad_occ's parallel pipeline; round-
trip through ``tempfile.NamedTemporaryFile`` so callers don't have to
manage paths. Future work: replace with ``BinTools`` if/when OCP exposes
a stream API.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from OCP.BRep import BRep_Builder
from OCP.BRepTools import BRepTools
from OCP.TopoDS import TopoDS_Shape


def brep_to_bytes(shape: TopoDS_Shape) -> bytes:
    """Serialize ``shape`` to BREP-format bytes."""
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".brep", delete=False) as tf:
        path = Path(tf.name)
    try:
        BRepTools.Write_s(shape, str(path))
        return path.read_bytes()
    finally:
        path.unlink(missing_ok=True)


def brep_from_bytes(data: bytes) -> TopoDS_Shape:
    """Deserialize BREP-format bytes back into a ``TopoDS_Shape``."""
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".brep", delete=False) as tf:
        tf.write(data)
        path = Path(tf.name)
    try:
        shape = TopoDS_Shape()
        builder = BRep_Builder()
        BRepTools.Read_s(shape, str(path), builder)
        return shape
    finally:
        path.unlink(missing_ok=True)
