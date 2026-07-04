"""Pinning test for PolyPrism.subdivide's bounding-box fold.

subdivide had no prior test coverage. This test isolates the bbox
min/max fold (the part being simplified to idiomatic numpy) by faking
the gmsh ``model.occ`` surface so the assertions are deterministic and
independent of the real OCC geometry kernel. It captures, from the
CURRENT (pre-simplification) implementation, the exact global bounds
and the resulting ``add_box`` call arguments, and must continue to
pass unchanged after the fold is rewritten with numpy.
"""

from shapely.geometry import Polygon

from meshwell.polyprism import PolyPrism


class _FakeOCC:
    def __init__(self, bboxes):
        self._bboxes = bboxes
        self.add_box_calls = []
        self.intersect_calls = []
        self.removed = []
        self._next_tag = 1000

    def getBoundingBox(self, dim, tag):
        assert dim == 3
        return self._bboxes[tag]

    def add_box(self, x, y, z, dx, dy, dz):
        tag = self._next_tag
        self._next_tag += 1
        self.add_box_calls.append((x, y, z, dx, dy, dz))
        return tag

    def intersect(self, objects, tools, removeObject, removeTool):
        self.intersect_calls.append(
            (list(objects), list(tools), removeObject, removeTool)
        )
        # No fake intersection geometry: nothing intersects, nothing is
        # reported consumed. This isolates the test to the bbox fold and
        # the add_box call arguments it drives, without needing real OCC.
        return [], [[]]

    def remove(self, dimtags):
        self.removed.append(list(dimtags))


class _FakeModel:
    def __init__(self, bboxes):
        self.occ = _FakeOCC(bboxes)


def _make_prism():
    return PolyPrism(
        polygons=Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        buffers={0.0: 0.0},
        physical_name="pinning_prism",
    )


def test_subdivide_bbox_fold_and_tool_boxes():
    bboxes = {
        1: (0.0, 0.0, 0.0, 2.0, 4.0, 6.0),
        2: (-1.0, 1.0, 2.0, 3.0, 5.0, 8.0),
    }
    model = _FakeModel(bboxes)
    subdivision = (2, 1, 1)

    result = _make_prism().subdivide(model, [1, 2], subdivision)

    # Manually folded global bounds over the two fake prisms' bboxes.
    expected_xmin, expected_ymin, expected_zmin = -1.0, 0.0, 0.0
    expected_xmax, expected_ymax, expected_zmax = 3.0, 5.0, 8.0
    dx = (expected_xmax - expected_xmin) / subdivision[0]
    dy = (expected_ymax - expected_ymin) / subdivision[1]
    dz = (expected_zmax - expected_zmin) / subdivision[2]

    expected_add_box_calls = [
        (expected_xmin + x_index * dx, expected_ymin, expected_zmin, dx, dy, dz)
        for x_index in range(subdivision[0])
    ]
    assert model.occ.add_box_calls == expected_add_box_calls

    # No fake intersection geometry -> nothing consumed, both seed
    # prisms get removed, and subdivide returns no subprisms.
    assert result == []
    assert sorted(model.occ.removed[0]) == [(3, 1), (3, 2)]


def test_subdivide_bbox_fold_single_prism():
    bboxes = {7: (1.5, -2.0, 0.5, 4.5, 2.0, 3.5)}
    model = _FakeModel(bboxes)
    subdivision = (1, 2, 1)

    _make_prism().subdivide(model, [7], subdivision)

    expected_xmin, expected_ymin, expected_zmin = 1.5, -2.0, 0.5
    expected_xmax, expected_ymax, expected_zmax = 4.5, 2.0, 3.5
    dx = (expected_xmax - expected_xmin) / subdivision[0]
    dy = (expected_ymax - expected_ymin) / subdivision[1]
    dz = (expected_zmax - expected_zmin) / subdivision[2]

    expected_add_box_calls = [
        (expected_xmin, expected_ymin + y_index * dy, expected_zmin, dx, dy, dz)
        for y_index in range(subdivision[1])
    ]
    assert model.occ.add_box_calls == expected_add_box_calls
