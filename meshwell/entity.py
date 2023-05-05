import gmsh


class LabeledEntities:
    """Class to track entities, boundaries, and physical labels."""

    def __init__(self, model, index, dimtags, label, base_resolution):
        self.model = model
        self.index = index
        self.dimtags = self._fuse_self(dimtags)
        self.label = label
        self.base_resolution = base_resolution
        self.boundaries = self.update_boundaries()
        self.interfaces = []

    def _fuse_self(self, dimtags):
        if len(dimtags) != 1:
            dimtags = self.model.fuse(
                [dimtags[0]],
                dimtags[1:],
                removeObject=True,
                removeTool=True,
            )[0]
            self.model.synchronize()
        return dimtags

    def get_tags(self):
        tags = [tag for dim, tag in self.dimtags]
        if any(isinstance(el, list) for el in tags):
            tags = [item for sublist in tags for item in sublist]
        return tags

    def get_dim(self):
        return [dim for dim, tag in self.dimtags][0]

    def update_boundaries(self):
        self.boundaries = [
            tag
            for dim, tag in gmsh.model.getBoundary(self.dimtags, False, False, False)
        ]
