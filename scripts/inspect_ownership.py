"""Dump each entity's owned OCC shape bboxes after cad_occ.

For the bench scene, verify whether struct_ring_0 ends up owning any
piece that geometrically sits at stack_0's xy bounds. If yes, the bug
is in _resolve_piece_ownership or BOPAlgo.Modified() attribution.
"""
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
    print(f"Calling cad_occ on {len(entities)} entities...")
    occ_entities, _slabs = cad_occ(entities, return_slabs=True, progress_bars=False)
    print(f"Got {len(occ_entities)} OCC-labelled entities; {len(_slabs)} slabs")
    for ent in occ_entities:
        if (
            "struct_ring_0" not in ent.physical_name
            and "stack" not in ent.physical_name
        ):
            continue
        if not ent.shapes:
            continue
        print(
            f"\nentity name={ent.physical_name} idx={ent.index} dim={ent.dim} "
            f"keep={ent.keep} phantom={ent.is_structured_phantom} "
            f"n_shapes={len(ent.shapes)}"
        )
        for i, s in enumerate(ent.shapes):
            bb = bbox(s)
            if bb is None:
                print(f"  shape[{i}]: bbox=<void>")
                continue
            print(
                f"  shape[{i}]: bbox=({bb[0]:.3f}, {bb[1]:.3f}, {bb[2]:.3f}, "
                f"{bb[3]:.3f}, {bb[4]:.3f}, {bb[5]:.3f})"
            )


if __name__ == "__main__":
    main()
