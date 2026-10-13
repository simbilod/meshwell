"""Fast direct XAO loader.

When loading .xao files, standard ``gmsh.open()`` assigns physical group tags
by scanning all model entities to locate the next free tag (O(entities) per
group). On models with tens of thousands of groups, this auto-tagging introduces
massive loading overhead.

This loader reads the BREP geometry with Gmsh, maps XAO topology indices
(global OpenCASCADE ``TopExp::MapShapes``) to Gmsh entity tags, and assigns
physical groups with explicit numeric tags. Yields physical groups identical
in name, dimension, and entity membership to ``gmsh.open()`` in O(1) per group.
"""

from __future__ import annotations

import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import gmsh

_DIM_NAMES = {"vertex": 0, "edge": 1, "face": 2, "solid": 3}
_TOPO_SECTIONS = {"vertices": 0, "edges": 1, "faces": 2, "solids": 3}


def enabled() -> bool:
    # Enabled by default; set MESHWELL_FAST_XAO=0 to fall back to gmsh.open()
    return os.environ.get("MESHWELL_FAST_XAO", "1") not in ("0", "false", "False")


def _build_dmaps(shape):
    """Map shapes by dimension in Gmsh's OCC exploration order.

    Gmsh registers entities top-down: solids and their boundary faces/edges/vertices
    first, followed by standalone faces, edges, and vertices. Exploring in this
    exact order ensures OpenCASCADE map indices match Gmsh entity tags 1-to-1.
    """
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_SOLID, TopAbs_VERTEX
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopTools import TopTools_IndexedMapOfShape

    dmaps = {
        0: TopTools_IndexedMapOfShape(),
        1: TopTools_IndexedMapOfShape(),
        2: TopTools_IndexedMapOfShape(),
        3: TopTools_IndexedMapOfShape(),
    }

    # 3D: Solids and their sub-shapes
    exp_solid = TopExp_Explorer(shape, TopAbs_SOLID)
    while exp_solid.More():
        solid = exp_solid.Current()
        dmaps[3].Add(solid)
        exp_face = TopExp_Explorer(solid, TopAbs_FACE)
        while exp_face.More():
            face = exp_face.Current()
            dmaps[2].Add(face)
            exp_edge = TopExp_Explorer(face, TopAbs_EDGE)
            while exp_edge.More():
                edge = exp_edge.Current()
                dmaps[1].Add(edge)
                exp_v = TopExp_Explorer(edge, TopAbs_VERTEX)
                while exp_v.More():
                    dmaps[0].Add(exp_v.Current())
                    exp_v.Next()
                exp_edge.Next()
            exp_face.Next()
        exp_solid.Next()

    # Standalone faces (not in any solid)
    exp_face = TopExp_Explorer(shape, TopAbs_FACE)
    while exp_face.More():
        face = exp_face.Current()
        if not dmaps[2].Contains(face):
            dmaps[2].Add(face)
            exp_edge = TopExp_Explorer(face, TopAbs_EDGE)
            while exp_edge.More():
                edge = exp_edge.Current()
                dmaps[1].Add(edge)
                exp_v = TopExp_Explorer(edge, TopAbs_VERTEX)
                while exp_v.More():
                    dmaps[0].Add(exp_v.Current())
                    exp_v.Next()
                exp_edge.Next()
        exp_face.Next()

    # Standalone edges
    exp_edge = TopExp_Explorer(shape, TopAbs_EDGE)
    while exp_edge.More():
        edge = exp_edge.Current()
        if not dmaps[1].Contains(edge):
            dmaps[1].Add(edge)
            exp_v = TopExp_Explorer(edge, TopAbs_VERTEX)
            while exp_v.More():
                dmaps[0].Add(exp_v.Current())
                exp_v.Next()
        exp_edge.Next()

    # Standalone vertices
    exp_v = TopExp_Explorer(shape, TopAbs_VERTEX)
    while exp_v.More():
        v = exp_v.Current()
        if not dmaps[0].Contains(v):
            dmaps[0].Add(v)
        exp_v.Next()

    return dmaps


def load(xao_path: Path) -> None:
    from OCP.BRep import BRep_Builder
    from OCP.BRepTools import BRepTools
    from OCP.TopExp import TopExp
    from OCP.TopoDS import TopoDS_Shape
    from OCP.TopTools import TopTools_IndexedMapOfShape

    root = ET.parse(str(xao_path)).getroot()  # noqa: S314
    with tempfile.TemporaryDirectory() as tmp:
        brep = Path(tmp) / "shape.brep"
        with brep.open("w") as f:
            f.write(root.find("geometry/shape").text)

        shape = TopoDS_Shape()
        BRepTools.Read_s(shape, str(brep), BRep_Builder())
        gmap = TopTools_IndexedMapOfShape()
        TopExp.MapShapes_s(shape, gmap)
        dmaps = _build_dmaps(shape)

        local_to_tag = {d: {} for d in range(4)}
        topo = root.find("geometry/topology")
        for sec, d in _TOPO_SECTIONS.items():
            sec_el = topo.find(sec) if topo is not None else None
            if sec_el is None:
                continue
            for el in sec_el:
                local_to_tag[d][int(el.get("index"))] = dmaps[d].FindIndex(
                    gmap.FindKey(int(el.get("reference")))
                )

        gmsh.open(str(brep))

    n_ent = {d: len(gmsh.model.getEntities(d)) for d in range(4)}
    for d in range(4):
        if n_ent[d] != dmaps[d].Extent():
            raise RuntimeError(
                f"fast XAO load: dim {d} gmsh has {n_ent[d]} entities but OCC "
                f"map has {dmaps[d].Extent()}; tag mapping would be invalid"
            )

    groups = root.find("groups")
    next_tag = dict.fromkeys(range(4), 1)
    for g in groups if groups is not None else []:
        d = _DIM_NAMES[g.get("dimension")]
        tags = [local_to_tag[d][int(e.get("index"))] for e in g]
        gmsh.model.addPhysicalGroup(d, tags, tag=next_tag[d], name=g.get("name"))
        next_tag[d] += 1
