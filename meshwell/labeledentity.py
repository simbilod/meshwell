from pydantic import BaseModel, ConfigDict
import gmsh
from typing import List, Union, Any, Tuple


class LabeledEntities(BaseModel):
    """Class to track entities, boundaries, and physical labels during meshing."""

    index: int
    model: Any
    dimtags: List[Tuple[int, int]]
    physical_name: str
    base_resolution: float
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
            tag
            for dim, tag in gmsh.model.getBoundary(self.dimtags, False, False, False)
        ]
        return self.boundaries
