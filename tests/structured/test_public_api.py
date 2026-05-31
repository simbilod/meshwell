def test_public_imports():
    from meshwell.structured import (
        ArcIdentifyConflictError,
        StructuredError,
        StructuredZStackError,
        WedgeCountMismatchError,
    )

    assert issubclass(StructuredZStackError, StructuredError)
    assert issubclass(WedgeCountMismatchError, StructuredError)
    assert issubclass(ArcIdentifyConflictError, StructuredError)
