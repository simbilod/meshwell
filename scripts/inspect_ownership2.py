"""Verify the actual entity index for 'struct_ring_0' and dump all entities' first shape bbox."""
import sys

sys.path.insert(0, ".")

from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib

from meshwell.cad_occ import cad_occ
from scripts import diagnose_arc_periodic as diag


def bbox(shape):
    box = Bnd_Box()
    BRepBndLib.Add_s(shape, box)
    if box.IsVoid():
        return None
    return box.Get()


def main():
    entities = diag.scene_bench()
    occ_entities, _slabs = cad_occ(entities, return_slabs=True, progress_bars=False)

    print(f"All {len(occ_entities)} OCC entities:")
    for ent in occ_entities:
        nshapes = len(ent.shapes)
        first_bb = bbox(ent.shapes[0]) if ent.shapes else None
        bb_str = (
            f"({first_bb[0]:.2f}, {first_bb[1]:.2f}, {first_bb[2]:.2f}, "
            f"{first_bb[3]:.2f}, {first_bb[4]:.2f}, {first_bb[5]:.2f})"
            if first_bb
            else "<empty>"
        )
        print(
            f"  idx={ent.index:3d} dim={ent.dim} keep={int(ent.keep)} "
            f"phantom={int(ent.is_structured_phantom)} mo={ent.mesh_order} "
            f"n_shapes={nshapes:3d} name={ent.physical_name} first_bb={bb_str}"
        )


if __name__ == "__main__":
    main()
