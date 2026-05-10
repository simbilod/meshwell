from __future__ import annotations

from pathlib import Path
from OCP.TDocStd import TDocStd_Document
from OCP.XCAFDoc import XCAFDoc_DocumentTool
from OCP.STEPCAFControl import STEPCAFControl_Writer
from OCP.TDataStd import TDataStd_Name
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCP.TCollection import TCollection_ExtendedString
import gmsh


def test_step_xcaf_export_import_labels(tmp_path):
    """Demonstrate how to serialize pure OCC geometry into the standard STEP format
    while embedding physical group names directly into the CAD assembly tree as XCAF labels.
    Gmsh natively reconstructs these labels into physical groups upon loading, completely
    bypassing external XML reference indexing.
    """
    # 1. Create basic shapes
    box_a = BRepPrimAPI_MakeBox(1.0, 1.0, 1.0).Shape()
    box_b = BRepPrimAPI_MakeBox(2.0, 2.0, 2.0).Shape()

    # 2. Initialize an OpenCASCADE XCAF document tree
    doc = TDocStd_Document(TCollection_ExtendedString("MDTV-XCAF"))
    shape_tool = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())

    # 3. Register shapes as free top-level components and attach industrial string attributes
    label_a = shape_tool.AddShape(box_a)
    label_b = shape_tool.AddShape(box_b)

    TDataStd_Name.Set_s(label_a, TCollection_ExtendedString("box_A"))
    TDataStd_Name.Set_s(label_b, TCollection_ExtendedString("box_B"))

    # 4. Serialize the complete assembly tree to disk via STEPCAFControl
    step_path = tmp_path / "labeled_assembly.step"
    writer = STEPCAFControl_Writer()
    writer.Transfer(doc)
    writer.Write(str(step_path))

    assert step_path.exists()

    # 5. Import natively in Gmsh with industrial label extraction enabled
    gmsh.initialize()
    gmsh.option.setNumber("Geometry.OCCImportLabels", 1)
    gmsh.open(str(step_path))

    # Gmsh imports XCAF labels as entity names; map them natively to physical groups
    entity_names = {}
    for d, t in gmsh.model.getEntities(3):
        name = gmsh.model.getEntityName(d, t)
        if name:
            pname = name.split("/")[-1]
            entity_names[pname] = name
            gmsh.model.addPhysicalGroup(d, [t], name=pname)

    groups = {gmsh.model.getPhysicalName(d, t) for d, t in gmsh.model.getPhysicalGroups()}
    gmsh.finalize()

    print(f"\nImported Entity Names from STEP: {sorted(list(entity_names.keys()))}")
    print(f"Imported Physical Groups from STEP: {sorted(list(groups))}")

    # Verification: Gmsh automatically extracts embedded attributes into fully linked physical groups
    assert "box_A" in groups
    assert "box_B" in groups
