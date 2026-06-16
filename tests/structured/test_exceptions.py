import meshwell.structured as structured
from meshwell.structured.exceptions import StructuredZStackError


def test_all_exceptions_are_structured_errors():
    """Every exported error inherits a common base for easy catching."""
    exported = [getattr(structured, name) for name in structured.__all__]
    errors = [cls for cls in exported if isinstance(cls, type)]
    assert errors, "structured.__all__ should export the exception family"
    for cls in errors:
        assert issubclass(cls, structured.StructuredError), cls.__name__


def test_zstack_error_carries_context():
    err = StructuredZStackError(entity_index=2, z=1.5, cohort_index=0)
    assert err.entity_index == 2
    assert err.z == 1.5
    assert err.cohort_index == 0
    assert "1.5" in str(err)


def test_canonical_arrangement_error_message():
    from meshwell.structured.exceptions import (
        CanonicalArrangementError,
        StructuredError,
    )

    err = CanonicalArrangementError(
        cohort_index=3,
        reason="vertex pair ((0,0,0),(1,1,0)) not in canonical edge lookup",
    )
    assert isinstance(err, StructuredError)
    assert err.cohort_index == 3
    assert "cohort 3" in str(err)
    assert "vertex pair" in str(err)
