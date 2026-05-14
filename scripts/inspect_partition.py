"""Dump face_partition geometry for every slab in the bench scene.

Goal: verify whether struct_ring_0's partition pieces include any
polygon that extends into stack_0's xy bounds (1, 14)-(2.5, 15.5).
That would explain why face 609 (geometrically at stack_0's location)
gets tagged 'struct_ring_0___None'.
"""
import sys

sys.path.insert(0, ".")

from meshwell import structured_polyprism as sp
from scripts import diagnose_arc_periodic as diag


def main():
    entities = diag.scene_bench()
    slabs = sp.resolve_structured_slabs(entities)
    print(f"Resolved {len(slabs)} slabs")
    for slab in slabs:
        partition = getattr(slab, "face_partition", None)
        if partition is None:
            continue
        print(
            f"\nSLAB {slab.physical_name} z=[{slab.zlo}, {slab.zhi}] "
            f"footprint_area={slab.footprint.area:.3f}"
        )
        print(
            f"  footprint bounds: {tuple(round(v, 3) for v in slab.footprint.bounds)}"
        )
        for i, piece in enumerate(partition):
            print(
                f"  piece[{i}]: bounds={tuple(round(v, 3) for v in piece.bounds)} "
                f"area={piece.area:.3f}"
            )


if __name__ == "__main__":
    main()
