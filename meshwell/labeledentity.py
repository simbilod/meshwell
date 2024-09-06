from pydantic import BaseModel, ConfigDict
import gmsh
from typing import List, Union, Any, Tuple
import math


class LabeledEntities(BaseModel):
    """General class to track the gmsh entities that result from the geometry definition."""

    index: int
    model: Any
    dimtags: List[Tuple[int, int]]
    physical_name: str
    resolution: Any
    keep: bool
    boundaries: List[int] = []
    interfaces: List = []

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _fuse_self(self, dimtags: List[Union[int, str]]) -> List[Union[int, str]]:
        if len(dimtags) == 0:
            return []
        elif len(dimtags) != 1:
            dimtags = gmsh.model.occ.fuse(
                [dimtags[0]],
                dimtags[1:],
                removeObject=True,
                removeTool=True,
            )[0]
            self.model.occ.synchronize()
        return dimtags

    def get_tags(self) -> List[int]:
        tags = [tag for dim, tag in self.dimtags]
        if any(isinstance(el, list) for el in tags):
            tags = [item for sublist in tags for item in sublist]
        return tags

    def get_dim(self) -> int:
        return [dim for dim, tag in self.dimtags][0]

    def update_boundaries(self) -> List[int]:
        self.boundaries = [
            tag for dim, tag in gmsh.model.getBoundary(self.dimtags, True, False, False)
        ]
        return self.boundaries

    def add_refinement_fields_to_model(
        self,
        refinement_field_indices: List,
        refinement_max_index: int,
        default_resolution: float,
    ):
        """
        Adds refinement fields to the model based on base_resolution and resolution info.
        """
        n = refinement_max_index
        if self.resolution is not None:
            base_resolution = self.resolution.get("resolution", default_resolution)

        if self.get_dim() == 3:
            entity_str = "RegionsList"
            boundary_str = "SurfacesList"
            curves_str = "CurvesList"
        elif self.get_dim() == 2:
            entity_str = "SurfacesList"
            boundary_str = "CurvesList"
            curves_str = "PointList"
        elif self.get_dim() == 1:
            entity_str = "CurvesList"
            boundary_str = "PointList"
        else:
            entity_str = "PointList"
            boundary_str = None

        if self.resolution and self.resolution.keys() >= {"resolution"}:
            self.model.mesh.field.add("MathEval", n)
            self.model.mesh.field.setString(n, "F", f"{base_resolution}")
            self.model.mesh.field.add("Restrict", n + 1)
            self.model.mesh.field.setNumber(n + 1, "InField", n)
            self.model.mesh.field.setNumbers(
                n + 1,
                entity_str,
                self.get_tags(),
            )
            refinement_field_indices.extend((n + 1,))
            n += 2

        if (
            self.resolution
            and self.resolution.keys()
            >= {
                "DistMax",
                "SizeMax",
            }
            and boundary_str
        ):
            self.model.mesh.field.add("Distance", n)
            self.model.mesh.field.setNumbers(n, boundary_str, self.boundaries)
            self.model.mesh.field.setNumber(n, "Sampling", 100)
            self.model.mesh.field.add("Threshold", n + 1)
            self.model.mesh.field.setNumber(n + 1, "InField", n)
            self.model.mesh.field.setNumber(
                n + 1,
                "SizeMin",
                self.resolution.get("SizeMin", base_resolution),
            )
            self.model.mesh.field.setNumber(
                n + 1, "SizeMax", self.resolution["SizeMax"]
            )
            self.model.mesh.field.setNumber(
                n + 1, "DistMin", self.resolution.get("DistMin", 0)
            )
            self.model.mesh.field.setNumber(
                n + 1, "DistMax", self.resolution["DistMax"]
            )
            self.model.mesh.field.setNumber(n + 1, "StopAtDistMax", 1)
            refinement_field_indices.extend((n + 1,))
            n += 2

        if self.resolution and self.resolution.keys() >= {"Curves"} and curves_str:
            for curveconfig in self.resolution["Curves"]:
                curve_resolution = curveconfig.get("CurveResolution")

                self.model.mesh.field.add("Distance", n)
                # print([c for b in self.boundaries for cs in self.model.occ.getCurveLoops(b)[1] for c in cs])
                # print(self.model.occ.getNodes([c for b in self.boundaries for cs in self.model.occ.getCurveLoops(b)[1] for c in cs][0]))

                self.model.mesh.field.setNumbers(
                    n,
                    curves_str,
                    [
                        c
                        for b in self.boundaries
                        for cs in self.model.occ.getCurveLoops(b)[1]
                        for c in cs
                        if self.model.occ.getMass(1, c)
                        < curveconfig.get("CurveLengthMax", math.inf)
                    ],
                )
                self.model.mesh.field.setNumber(n, "Sampling", 100)
                self.model.mesh.field.add("Threshold", n + 1)
                self.model.mesh.field.setNumber(n + 1, "InField", n)
                self.model.mesh.field.setNumber(
                    n + 1,
                    "SizeMin",
                    self.resolution.get("CurveSizeMin", curve_resolution),
                )
                self.model.mesh.field.setNumber(
                    n + 1, "SizeMax", curveconfig["CurveSizeMax"]
                )
                self.model.mesh.field.setNumber(
                    n + 1, "DistMin", curveconfig.get("CurveDistMin", 0)
                )
                self.model.mesh.field.setNumber(
                    n + 1, "DistMax", curveconfig["CurveDistMax"]
                )
                self.model.mesh.field.setNumber(n + 1, "StopAtDistMax", 1)
                refinement_field_indices.extend((n + 1,))
                n += 2

        return refinement_field_indices, n
