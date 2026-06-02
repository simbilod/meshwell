def test_public_imports():
    from meshwell.structured import (
        MixedIdentifyArcsError,
        StructuredError,
        StructuredZStackError,
        WedgeCountMismatchError,
    )

    assert issubclass(StructuredZStackError, StructuredError)
    assert issubclass(WedgeCountMismatchError, StructuredError)
    assert issubclass(MixedIdentifyArcsError, StructuredError)
