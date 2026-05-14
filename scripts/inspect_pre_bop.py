"""Inspect struct_ring_0's shapes BEFORE the global BOPAlgo_Builder runs.

If pre-BOP shapes are clean (3 sub-prisms in struct_ring's bounds) but
post-BOP shapes span the whole scene, then BOPAlgo_Builder.Modified()
is the culprit (mis-attributing fragmented pieces).
"""
import sys

sys.path.insert(0, ".")

from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib

from meshwell.cad_occ import CAD_OCC
from scripts import diagnose_arc_periodic as diag


def bbox(shape):
    box = Bnd_Box()
    BRepBndLib.Add_s(shape, box)
    if box.IsVoid():
        return None
    return box.Get()


def main():
    entities = diag.scene_bench()
    cad = CAD_OCC()
    # Run only up through cut, NOT fragment.
    occ_entities = cad.process_entities_cut_only(entities)
    print(f"After cut (BEFORE fragment): {len(occ_entities)} entities")
    for ent in occ_entities:
        if "struct_ring_0" not in ent.physical_name:
            continue
        print(f"\n{ent.physical_name} idx={ent.index} n_shapes={len(ent.shapes)}")
        for i, s in enumerate(ent.shapes):
            bb = bbox(s)
            bb_str = (
                f"({bb[0]:.3f}, {bb[1]:.3f}, {bb[2]:.3f}, "
                f"{bb[3]:.3f}, {bb[4]:.3f}, {bb[5]:.3f})"
                if bb
                else "<empty>"
            )
            print(f"  shape[{i}]: bbox={bb_str}")


if __name__ == "__main__":
    main()
