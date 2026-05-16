"""Phase-4 spike: can we map OCP TopoDS_Face -> gmsh face tag without coords?

Hypothesis: gmsh's XAO loader assigns face tags in a deterministic order
related to the TopExp::MapShapes traversal order of the BREP compound.
If yes, we can compute the mapping purely from OCC TShape identity — no
bbox matching.

Test:
  1. Build a small OCC scene with multiple solids (a box + a stick that
     fragments against it).
  2. Run BOP via OCP's BOPAlgo_Builder, collect post-BOP TopoDS_Face
     objects.
  3. Write the result to a BREP file (similar to what cad_occ does
     internally — XAO is a wrapper around BREP).
  4. Import the BREP into gmsh; collect gmsh face tags.
  5. For each TopoDS_Face, compute its TopExp::MapShapes index in the
     loaded BREP.
  6. Assert: there's a deterministic (and ideally simple) mapping
     index -> gmsh tag.

If (6) holds, Phase 4's OCP->gmsh bridge is just an index lookup.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any


def _list_to_python(lst: Any) -> list[Any]:
    """Convert OCP TopTools_ListOfShape to plain list."""
    return list(lst)


def make_scene() -> tuple[Any, list[Any], Any]:
    """Build a 1x1x1 box and a stick that fragments it.

    Returns (post_bop_compound, tracked_input_faces, builder).
    tracked_input_faces are the 6 faces of the original box (pre-BOP).
    """
    from OCP.BOPAlgo import BOPAlgo_Builder
    from OCP.BRep import BRep_Tool  # noqa: F401
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.gp import gp_Pnt
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    box = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), gp_Pnt(1, 1, 1)).Solid()
    stick = BRepPrimAPI_MakeBox(gp_Pnt(0.3, 0.3, -0.1), gp_Pnt(0.7, 0.7, 1.1)).Solid()

    box_faces: list[Any] = []
    exp = TopExp_Explorer(box, TopAbs_FACE)
    while exp.More():
        box_faces.append(TopoDS.Face_s(exp.Current()))
        exp.Next()

    builder = BOPAlgo_Builder()
    builder.AddArgument(box)
    builder.AddArgument(stick)
    builder.Perform()

    return builder.Shape(), box_faces, builder


def write_brep_and_load_in_gmsh(shape: Any) -> tuple[Path, dict[int, tuple]]:
    """Write the compound to a BREP file, load in gmsh, return gmsh face bboxes by tag."""
    import gmsh
    from OCP.BRepTools import BRepTools

    tmp = Path(tempfile.gettempdir()) / "phase4_spike.brep"
    BRepTools.Write_s(shape, str(tmp))

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("test")
    gmsh.model.occ.importShapes(str(tmp))
    gmsh.model.occ.synchronize()

    face_bboxes: dict[int, tuple] = {}
    for dim, tag in gmsh.model.getEntities(2):
        bb = gmsh.model.getBoundingBox(dim, tag)
        face_bboxes[tag] = tuple(bb)

    return tmp, face_bboxes


def collect_post_bop_faces_with_topexp_indices(
    shape: Any,
) -> list[tuple[int, Any, tuple]]:
    """For each face in the post-BOP compound, return (topexp_index, face, bbox)."""
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    out: list[tuple[int, Any, tuple]] = []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    idx = 0
    while exp.More():
        f = TopoDS.Face_s(exp.Current())
        b = Bnd_Box()
        BRepBndLib.Add_s(f, b)
        xmin, ymin, zmin, xmax, ymax, zmax = b.Get()
        out.append((idx, f, (xmin, ymin, zmin, xmax, ymax, zmax)))
        idx += 1
        exp.Next()
    return out


def collect_via_topexp_mapshapes(shape: Any) -> list[tuple[int, Any, tuple]]:
    """For each face in the BREP compound via TopExp::MapShapes, return (index, face, bbox)."""
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp
    from OCP.TopoDS import TopoDS
    from OCP.TopTools import TopTools_IndexedMapOfShape

    fmap = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(shape, TopAbs_FACE, fmap)
    out: list[tuple[int, Any, tuple]] = []
    n = fmap.Extent()
    for i in range(1, n + 1):  # OCP indices are 1-based
        s = fmap.FindKey(i)
        f = TopoDS.Face_s(s)
        b = Bnd_Box()
        BRepBndLib.Add_s(f, b)
        xmin, ymin, zmin, xmax, ymax, zmax = b.Get()
        out.append((i, f, (xmin, ymin, zmin, xmax, ymax, zmax)))
    return out


def main() -> int:
    print("=" * 78)
    print("Phase 4 spike: gmsh XAO/BREP load tag determinism")
    print("=" * 78)

    compound, _box_faces, _builder = make_scene()

    # Two ways to index the post-BOP faces:
    #   - TopExp_Explorer order (depth-first traversal)
    #   - TopExp::MapShapes order (post-BOP global indexing — what XAO uses)
    explorer_faces = collect_post_bop_faces_with_topexp_indices(compound)
    mapshapes_faces = collect_via_topexp_mapshapes(compound)

    print(f"\nExplorer order: {len(explorer_faces)} faces")
    print(f"MapShapes order: {len(mapshapes_faces)} faces (1-based)")

    _, gmsh_bboxes = write_brep_and_load_in_gmsh(compound)
    print(
        f"\nGmsh-loaded faces: {len(gmsh_bboxes)} (tags: {sorted(gmsh_bboxes.keys())})"
    )

    # Match by bbox to figure out what gmsh tag corresponds to each TopExp index.
    def _bbox_match(target_bb: tuple, tol: float = 1e-7) -> int | None:
        for tag, gbb in gmsh_bboxes.items():
            if all(abs(target_bb[i] - gbb[i]) < tol for i in range(6)):
                return tag
        return None

    print("\nExplorer index -> gmsh tag:")
    explorer_mapping: list[tuple[int, int | None]] = []
    for idx, _f, bb in explorer_faces:
        tag = _bbox_match(bb)
        explorer_mapping.append((idx, tag))
        print(f"  {idx:3d} -> {tag}")

    print("\nMapShapes index -> gmsh tag:")
    mapshapes_mapping: list[tuple[int, int | None]] = []
    for idx, _f, bb in mapshapes_faces:
        tag = _bbox_match(bb)
        mapshapes_mapping.append((idx, tag))
        print(f"  {idx:3d} -> {tag}")

    # Determinism check: do these orderings give predictable gmsh tag sequences?
    print("\n" + "-" * 78)
    print("Determinism diagnostics:")
    print("-" * 78)
    expl_tags = [t for _, t in explorer_mapping if t is not None]
    map_tags = [t for _, t in mapshapes_mapping if t is not None]
    print(f"Explorer-order tags: {expl_tags}")
    print(f"Is explorer order monotonic? {expl_tags == sorted(expl_tags)}")
    print(f"MapShapes-order tags: {map_tags}")
    print(f"Is MapShapes order monotonic? {map_tags == sorted(map_tags)}")

    # Reload the SAME BREP file in a fresh gmsh session — are tags identical?
    import gmsh

    gmsh.finalize()
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("reload")
    tmp = Path(tempfile.gettempdir()) / "phase4_spike.brep"
    gmsh.model.occ.importShapes(str(tmp))
    gmsh.model.occ.synchronize()

    reload_bboxes: dict[int, tuple] = {}
    for dim, tag in gmsh.model.getEntities(2):
        bb = gmsh.model.getBoundingBox(dim, tag)
        reload_bboxes[tag] = tuple(bb)
    gmsh.finalize()

    print("\nReload tags: {sorted(reload_bboxes.keys())}")
    print(f"First-load tags: {sorted(gmsh_bboxes.keys())}")
    print(f"Same set? {set(gmsh_bboxes) == set(reload_bboxes)}")

    # Verify that each tag corresponds to the same bbox across loads:
    bbox_stable = True
    for tag in gmsh_bboxes:
        if tag in reload_bboxes and any(
            abs(gmsh_bboxes[tag][i] - reload_bboxes[tag][i]) > 1e-9 for i in range(6)
        ):
            bbox_stable = False
            print(
                f"  Tag {tag} bbox differs: {gmsh_bboxes[tag]} vs {reload_bboxes[tag]}"
            )
    print(f"Tag->bbox stable across reloads? {bbox_stable}")

    # If both explorer-order tags AND MapShapes-order tags are monotonic AND
    # tag<->bbox is stable, then we have a deterministic OCP-index -> gmsh-tag
    # mapping. We can predict gmsh tags purely from TopExp indices.
    deterministic = (
        all(t is not None for _, t in mapshapes_mapping)
        and map_tags == sorted(map_tags)
        and bbox_stable
        and set(gmsh_bboxes) == set(reload_bboxes)
    )
    print("\n" + "=" * 78)
    print(f"VERDICT: TopExp-index -> gmsh-tag mapping deterministic? {deterministic}")
    if deterministic:
        # The mapping is: gmsh_tag = sorted_gmsh_tags[topexp_index - 1]
        sorted_tags = sorted(gmsh_bboxes.keys())
        first_tag = sorted_tags[0]
        print(f"First gmsh tag: {first_tag}")
        print(f"Last gmsh tag: {sorted_tags[-1]}")
        # Verify the simple formula:
        ok = True
        for idx, expected_tag in mapshapes_mapping:
            predicted = sorted_tags[idx - 1]
            if predicted != expected_tag:
                ok = False
                print(f"  Index {idx} predicted {predicted}, actual {expected_tag}")
        print(f"Simple formula (gmsh_tag = sorted[idx-1])? {ok}")
    print("=" * 78)
    return 0 if deterministic else 1


if __name__ == "__main__":
    sys.exit(main())
