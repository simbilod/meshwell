from __future__ import annotations


from meshwell.model import Model
from meshwell.process import Grow, process_steps_to_entities


def test_process():
    model = Model(
        n_threads=1, filename="test_process"
    )  # 1 thread for deterministic mesh

    process_steps = [
        Grow(
            name="oxidation",
            thickness=1,
        ),
        Grow(
            name="oxidation2",
            thickness=0.5,
        ),
    ]

    model.process_3D(
        process_steps=process_steps, domain=[20, 20, 10], substrate_thickness=10
    )

    entities = process_steps_to_entities(process_steps, model)

    model.mesh(from_cad=True, cad_filename="test_process.xao", entities_list=entities)


if __name__ == "__main__":
    test_process()
