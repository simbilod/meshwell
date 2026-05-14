"""Count how many pieces each entity's originals Modify into during the global BOP."""
import sys

sys.path.insert(0, ".")

from OCP.Bnd import Bnd_Box
from OCP.BOPAlgo import BOPAlgo_Builder
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
    occ_entities = cad.process_entities_cut_only(entities)

    builder = BOPAlgo_Builder()
    builder.SetRunParallel(False)
    builder.SetFuzzyValue(cad.fragment_fuzzy_value)
    builder.SetNonDestructive(False)
    originals_per_entity = []
    for ent in occ_entities:
        originals_per_entity.append(list(ent.shapes))
        for s in ent.shapes:
            builder.AddArgument(s)
    builder.Perform()

    print(f"{'idx':>4} {'name':25s} {'mo':>4} {'orig':>5} {'mod_pieces':>11}")
    for i, ent in enumerate(occ_entities):
        if not occ_entities[i].shapes:
            continue
        total_pieces = 0
        empty_origs = 0
        for original in originals_per_entity[i]:
            modified = builder.Modified(original)
            if modified.IsEmpty():
                if not builder.IsDeleted(original):
                    total_pieces += 1
                    empty_origs += 1
            else:
                total_pieces += modified.Size()
        name = "/".join(ent.physical_name)
        print(
            f"{ent.index:>4} {name[:25]:25s} {ent.mesh_order!s:>4} "
            f"{len(originals_per_entity[i]):>5} "
            f"{total_pieces:>11} (empty_origs={empty_origs})"
        )


if __name__ == "__main__":
    main()
