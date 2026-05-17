"""End-to-end test: PolyPrism(structured=True) -> mesh -> wedge elements."""
from __future__ import annotations

from shapely.geometry import Polygon


def _square(x=0, y=0, w=1, h=1) -> Polygon:
    return Polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])


def test_single_structured_slab_produces_wedge_mesh(tmp_path):
    """A single structured PolyPrism with n_layers=2 produces wedges in the mesh."""
    import meshio

    from meshwell.orchestrator import generate_mesh
    from meshwell.polyprism import PolyPrism
    from meshwell.structured import StructuredExtrusionResolutionSpec

    p = PolyPrism(
        polygons=_square(0, 0, 2, 2),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
        physical_name="block",
    )

    out_msh = tmp_path / "structured.msh"
    generate_mesh([p], dim=3, output_mesh=out_msh, default_characteristic_length=0.5)

    m = meshio.read(out_msh)
    # Wedge cells should be present. meshio may name them "wedge" or "wedge15"
    # or similar; check for any prismatic cell type.
    cell_types = {cb.type for cb in m.cells}
    wedge_like = any(ct in cell_types for ct in ("wedge", "wedge6", "wedge15"))
    assert wedge_like, f"Expected wedge-like cells, got {cell_types}"
    # Physical "block" should be present.
    assert "block" in m.field_data


def test_simple_slab_lateral_mesh_is_conformal(tmp_path):
    """Lateral OCC face triangles should be conformal with the wedge mesh.

    Each boundary triangle must be either a wedge bot/top triangle or a
    sub-triangle of a wedge lateral quad (no mid-height interior nodes).
    """
    import meshio
    import numpy as np
    from shapely.geometry import Polygon

    from meshwell.orchestrator import generate_mesh
    from meshwell.polyprism import PolyPrism
    from meshwell.structured import StructuredExtrusionResolutionSpec

    sq = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
    p = PolyPrism(
        polygons=sq,
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
        physical_name="slab",
    )
    out = tmp_path / "slab.msh"
    generate_mesh([p], dim=3, output_mesh=out, default_characteristic_length=0.5)
    m = meshio.read(out)

    wedges = np.concatenate([cb.data for cb in m.cells if cb.type == "wedge"], axis=0)
    wedge_tris = {frozenset((int(w[0]), int(w[1]), int(w[2]))) for w in wedges}
    wedge_tris |= {frozenset((int(w[3]), int(w[4]), int(w[5]))) for w in wedges}
    wedge_lateral_quads = set()
    for w in wedges:
        for i, j in [(0, 1), (1, 2), (2, 0)]:
            wedge_lateral_quads.add(
                frozenset((int(w[i]), int(w[j]), int(w[j + 3]), int(w[i + 3])))
            )

    orphans = 0
    for cb in m.cells:
        if cb.type != "triangle":
            continue
        for t in cb.data:
            ts = frozenset((int(t[0]), int(t[1]), int(t[2])))
            if ts in wedge_tris:
                continue
            if any(ts.issubset(q) for q in wedge_lateral_quads):
                continue
            orphans += 1
    assert orphans == 0, f"{orphans} boundary triangles not conformal with wedge mesh"


def test_single_structured_slab_default_characteristic_length(tmp_path):
    """Smoke: structured pipeline runs without exceptions."""
    from meshwell.orchestrator import generate_mesh
    from meshwell.polyprism import PolyPrism
    from meshwell.structured import StructuredExtrusionResolutionSpec

    p = PolyPrism(
        polygons=_square(),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        physical_name="b",
    )
    out_msh = tmp_path / "smoke.msh"
    generate_mesh([p], dim=3, output_mesh=out_msh, default_characteristic_length=1.0)
    assert out_msh.exists()


def test_multipiece_slab_with_top_cap_no_true_orphans(tmp_path):
    """Regression: multi-piece slab+cap produces zero TRUE_ORPHAN triangles.

    Reproduces the slab_with_top_cap demo scene (structured slab 4x4 at
    z=[0,1], non-structured cap 2x4 at z=[1,2]) that historically had 47
    orphan triangles due to LateralKey collision (Bug A) and 32 spurious
    interior-seam 2D mesh triangles (Bug B).

    A TRUE_ORPHAN is a boundary triangle that is neither a wedge bot/top
    triangle nor a sub-triangle of a wedge lateral quad nor a tet-only
    cap triangle nor an interface triangle shared by wedge and tet regions.
    """
    import meshio
    import numpy as np
    from shapely.geometry import Polygon

    from meshwell.orchestrator import generate_mesh
    from meshwell.polyprism import PolyPrism
    from meshwell.structured import StructuredExtrusionResolutionSpec

    slab = PolyPrism(
        polygons=Polygon([(0, 0), (4, 0), (4, 4), (0, 4)]),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
        physical_name="slab",
    )
    cap = PolyPrism(
        polygons=Polygon([(0, 0), (2, 0), (2, 4), (0, 4)]),
        buffers={1.0: 0.0, 2.0: 0.0},
        physical_name="cap",
    )

    out_msh = tmp_path / "slab_with_top_cap.msh"
    generate_mesh(
        [slab, cap], dim=3, output_mesh=out_msh, default_characteristic_length=0.5
    )

    m = meshio.read(out_msh)

    # Collect all wedge cells.
    wedges = np.concatenate([cb.data for cb in m.cells if cb.type == "wedge"], axis=0)
    # Wedge bot/top triangles.
    wedge_tris = {frozenset((int(w[0]), int(w[1]), int(w[2]))) for w in wedges}
    wedge_tris |= {frozenset((int(w[3]), int(w[4]), int(w[5]))) for w in wedges}
    # Wedge lateral quads (as 4-node frozen sets).
    wedge_quads: set[frozenset] = set()
    for w in wedges:
        for i, j in [(0, 1), (1, 2), (2, 0)]:
            wedge_quads.add(
                frozenset((int(w[i]), int(w[j]), int(w[j + 3]), int(w[i + 3])))
            )

    # Collect all tet cells.
    tet_cells = np.concatenate(
        [cb.data for cb in m.cells if cb.type == "tetra"], axis=0
    )
    tet_nodes: set[int] = set(tet_cells.ravel().tolist())
    wedge_nodes: set[int] = set(wedges.ravel().tolist())

    true_orphans = 0
    for cb in m.cells:
        if cb.type != "triangle":
            continue
        for t in cb.data:
            ts = frozenset((int(t[0]), int(t[1]), int(t[2])))
            # Not a wedge bot/top triangle.
            if ts in wedge_tris:
                continue
            # Not a sub-triangle of a wedge lateral quad.
            if any(ts.issubset(q) for q in wedge_quads):
                continue
            # Pure tet (cap-only): all nodes belong only to tets.
            if ts.issubset(tet_nodes) and ts.isdisjoint(wedge_nodes):
                continue
            # Interface: touches both tet and wedge regions.
            if ts & tet_nodes and ts & wedge_nodes:
                continue
            true_orphans += 1

    assert true_orphans == 0, (
        f"{true_orphans} TRUE_ORPHAN triangles found in multi-piece slab+cap mesh; "
        "expected 0 after LateralKey collision fix + interior seam suppression."
    )
