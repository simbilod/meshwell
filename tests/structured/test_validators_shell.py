import pytest
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox

from meshwell.structured.exceptions import CohortShellModifiedError
from meshwell.structured.types import ShapeKey, SlabMeta
from meshwell.structured.validators import validate_cohort_shells


def _key_of(shape):
    from OCP.TopTools import TopTools_ShapeMapHasher

    h = TopTools_ShapeMapHasher()
    return ShapeKey(h(shape), int(shape.Orientation()))


def test_no_changes_passes():
    box = BRepPrimAPI_MakeBox(1.0, 1.0, 1.0).Solid()
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer

    exp = TopExp_Explorer(box, TopAbs_FACE)
    faces = []
    while exp.More():
        faces.append(exp.Current())
        exp.Next()
    bot, top, *laterals = faces
    meta = SlabMeta(
        slab_index=0,
        physical_name=("x",),
        bot_face_key=_key_of(bot),
        top_face_key=_key_of(top),
        lateral_face_keys=tuple(_key_of(f) for f in laterals),
    )
    slab_meta = {_key_of(box): meta}

    # Stub builder whose Modified() returns empty + IsDeleted returns False.
    class _StubBuilder:
        def Modified(self, _shape):
            from OCP.TopTools import TopTools_ListOfShape

            return TopTools_ListOfShape()

        def IsDeleted(self, _shape):
            return False

    # Should not raise.
    validate_cohort_shells(
        slab_meta,
        faces_by_key=dict(
            zip(
                [meta.bot_face_key, meta.top_face_key, *meta.lateral_face_keys],
                [bot, top, *laterals],
            )
        ),
        builder=_StubBuilder(),
    )


def test_split_face_raises():
    box = BRepPrimAPI_MakeBox(1.0, 1.0, 1.0).Solid()
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer

    exp = TopExp_Explorer(box, TopAbs_FACE)
    bot = exp.Current()
    meta = SlabMeta(
        slab_index=5,
        physical_name=("x",),
        bot_face_key=_key_of(bot),
        top_face_key=_key_of(bot),
        lateral_face_keys=(),
    )
    slab_meta = {_key_of(box): meta}

    class _SplitBuilder:
        def Modified(self, shape):
            from OCP.TopTools import TopTools_ListOfShape

            lst = TopTools_ListOfShape()
            # Simulate BOP splitting the face into two new faces.
            lst.Append(shape)
            lst.Append(shape)
            return lst

        def IsDeleted(self, _shape):
            return False

    with pytest.raises(CohortShellModifiedError) as exc:
        validate_cohort_shells(
            slab_meta,
            faces_by_key={meta.bot_face_key: bot},
            builder=_SplitBuilder(),
        )
    assert exc.value.slab_index == 5
    assert exc.value.fragment_count == 2
