"""Minimal demo: three stacked structured slabs with unstructured cladding."""
import tempfile
from pathlib import Path

from shapely.geometry import Polygon

from meshwell.orchestrator import generate_mesh
from meshwell.polyprism import PolyPrism
from meshwell.resolution import StructuredExtrusionResolutionSpec

SQa1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
SQa2 = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
SQb = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])

a1 = PolyPrism(
    SQa1, {0.0: 0.0, 1.0: 0.0}, physical_name="a1", structured=True, mesh_order=1
)
a2 = PolyPrism(
    SQa2, {0.0: 0.0, 1.0: 0.0}, physical_name="a2", structured=True, mesh_order=2
)
b = PolyPrism(SQb, {1.0: 0.0, 2.0: 0.0}, physical_name="b", structured=True)
# Cladding above and below to satisfy the cohort wrapping invariant.
base = PolyPrism(SQb, {-1.0: 0.0, 0.0: 0.0}, physical_name="base")
cap = PolyPrism(SQb, {2.0: 0.0, 3.0: 0.0}, physical_name="cap")

out_path = Path(tempfile.gettempdir()) / "demo.msh"
generate_mesh(
    entities=[a1, a2, b, base, cap],
    dim=3,
    output_mesh=str(out_path),
    default_characteristic_length=0.4,
    resolution_specs={
        "a1": [StructuredExtrusionResolutionSpec(n_layers=2)],
        "a2": [StructuredExtrusionResolutionSpec(n_layers=2)],
        "b": [StructuredExtrusionResolutionSpec(n_layers=2)],
    },
)
print(f"Wrote {out_path} — open with: gmsh {out_path}")
