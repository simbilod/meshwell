"""Batched element insertion via a single MSH4 merge.

Every ``gmsh.model.mesh.addElementsByType`` call invokes
``GModel::destroyMeshCaches()``, invalidating Gmsh's global node cache and
forcing a full rebuild on the subsequent call (O(#nodes)). Calling it
repeatedly inside loops over thousands of faces or slabs introduces
significant overhead.

``ElementBatcher`` buffers element blocks and inserts them in a single
``gmsh.merge`` of a temporary binary MSH 4.1 file containing an ``$Elements``
section: Gmsh's MSH4 reader attaches elements to existing entities by tag and
resolves node tags through the node cache built in a single pass.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import gmsh
import numpy as np

_NODES_PER_TYPE = {1: 2, 2: 3, 3: 4, 4: 4, 5: 8, 6: 6, 7: 5}


def enabled() -> bool:
    # Enabled by default; set MESHWELL_BATCH_ELEMENTS=0 to force direct calls
    return os.environ.get("MESHWELL_BATCH_ELEMENTS", "1") not in ("0", "false", "False")


class ElementBatcher:
    def __init__(self) -> None:
        self._blocks: list[tuple[int, int, int, np.ndarray]] = []
        self._direct = not enabled()

    def add(self, dim: int, tag: int, etype: int, node_tags) -> None:
        if len(node_tags) == 0:
            return
        if self._direct:
            gmsh.model.mesh.addElementsByType(tag, etype, [], list(node_tags))
            return
        arr = np.asarray(node_tags, dtype=np.uint64)
        npe = _NODES_PER_TYPE[etype]
        if arr.size % npe:
            raise ValueError(f"node count {arr.size} not a multiple of {npe}")
        self._blocks.append((dim, int(tag), int(etype), arr.reshape(-1, npe)))

    def flush(self) -> int:
        """Insert all pending elements; returns the number inserted."""
        if self._direct or not self._blocks:
            self._blocks.clear()
            return 0
        n_total = sum(b[3].shape[0] for b in self._blocks)
        first = int(gmsh.model.mesh.getMaxElementTag()) + 1
        next_tag = first
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "batch.msh"
            with path.open("wb") as f:
                f.write(b"$MeshFormat\n4.1 1 8\n")
                f.write(np.array([1], dtype=np.int32).tobytes())
                f.write(b"\n$EndMeshFormat\n$Elements\n")
                f.write(
                    np.array(
                        [len(self._blocks), n_total, first, first + n_total - 1],
                        dtype=np.uint64,
                    ).tobytes()
                )
                for dim, tag, etype, conn in self._blocks:
                    n = conn.shape[0]
                    f.write(np.array([dim, tag, etype], dtype=np.int32).tobytes())
                    f.write(np.array([n], dtype=np.uint64).tobytes())
                    data = np.empty((n, conn.shape[1] + 1), dtype=np.uint64)
                    data[:, 0] = np.arange(next_tag, next_tag + n, dtype=np.uint64)
                    data[:, 1:] = conn
                    f.write(data.tobytes())
                    next_tag += n
                f.write(b"\n$EndElements\n")
            gmsh.merge(str(path))
        # The MSH4 reader leaves the element cache holding only the merged
        # elements; a no-op addElementsByType forces destroyMeshCaches().
        _d0, t0, e0, _ = self._blocks[0]
        gmsh.model.mesh.addElementsByType(t0, e0, [], [])
        self._blocks.clear()
        return n_total
