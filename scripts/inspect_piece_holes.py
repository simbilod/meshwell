"""Check whether partition pieces have interior holes (would cause BOP issue)."""
import sys

sys.path.insert(0, ".")

from meshwell import structured_polyprism as sp
from scripts import diagnose_arc_periodic as diag


def main():
    entities = diag.scene_bench()
    slabs = sp.resolve_structured_slabs(entities)
    for slab in slabs:
        partition = getattr(slab, "face_partition", None)
        if partition is None:
            continue
        if "struct_ring" not in slab.physical_name[0]:
            continue
        print(f"\n{slab.physical_name} ({len(partition)} pieces):")
        for i, piece in enumerate(partition):
            n_int = len(piece.interiors)
            print(
                f"  piece[{i}]: bounds={tuple(round(v, 3) for v in piece.bounds)} "
                f"area={piece.area:.3f} exterior_pts={len(piece.exterior.coords)} "
                f"interior_rings={n_int}"
            )


if __name__ == "__main__":
    main()
