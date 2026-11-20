"""Benchmark label recovery on a grid of abutting prisms.

Recovery re-queries the model once per (entity, interface) pair if the
physical groups are not indexed, so its cost grows with the square of the
group count while the work it actually needs is linear in the groups.
"""
import time

import shapely

from meshwell.mesh import Mesh
from meshwell.model import ModelManager
from meshwell.polyprism import PolyPrism
from meshwell.resolution import ConstantInField


def _grid(n: int) -> list[PolyPrism]:
    return [
        PolyPrism(
            polygons=shapely.box(i * 2.0, j * 2.0, (i + 1) * 2.0, (j + 1) * 2.0),
            buffers={0.0: 0.0, 1.0: 0.0},
            physical_name=f"cell_{i}_{j}",
            mesh_order=1,
        )
        for i in range(n)
        for j in range(n)
    ]


def _specs(n: int) -> dict:
    specs: dict = {
        None: [ConstantInField(resolution=0.5, apply_to="volumes", entity_str="global")]
    }
    for i in range(n):
        for j in range(n):
            name = f"cell_{i}_{j}"
            specs[name] = [
                ConstantInField(resolution=0.2, apply_to="volumes", entity_str=name)
            ]
    return specs


def test_benchmark_label_recovery(tmp_path):
    from meshwell.orchestrator import cad

    for n in (5, 10, 20):
        model_manager = ModelManager()
        try:
            xao = tmp_path / f"grid_{n}.xao"
            cad(_grid(n), output_file=xao)
            model_manager.load_from_xao(xao)
            mesh_obj = Mesh(model=model_manager)

            start = time.perf_counter()
            recovered_list, _ = mesh_obj._recover_labels_from_cad(_specs(n))
            elapsed = time.perf_counter() - start

            groups = model_manager.model.getPhysicalGroups()
            assert len(recovered_list) == n * n
            assert len(groups) > 0
            print(
                f"\nLabel recovery {n}x{n}: {len(recovered_list)} entities, "
                f"{len(groups)} physical groups, {elapsed:.4f}s"
            )
        finally:
            model_manager.finalize()
