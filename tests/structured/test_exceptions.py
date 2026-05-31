from meshwell.structured.exceptions import (
    CohortNonManifoldError,
    CohortShellModifiedError,
    StructuredEntityTypeError,
    StructuredExtrudeRequiredError,
    StructuredLateralNLayersMismatchError,
    StructuredTransfiniteRejectedError,
    StructuredZStackError,
    SubPolygonAssemblyError,
    UnstructuredImprintRequiresPolyPrismError,
    WedgeBotNodeMismatchError,
    WedgeCountMismatchError,
)


def test_all_exceptions_are_structured_errors():
    """Every custom error inherits a common base for easy catching."""
    from meshwell.structured.exceptions import StructuredError

    for cls in [
        StructuredExtrudeRequiredError,
        StructuredEntityTypeError,
        StructuredZStackError,
        UnstructuredImprintRequiresPolyPrismError,
        SubPolygonAssemblyError,
        CohortNonManifoldError,
        CohortShellModifiedError,
        StructuredLateralNLayersMismatchError,
        StructuredTransfiniteRejectedError,
        WedgeCountMismatchError,
        WedgeBotNodeMismatchError,
    ]:
        assert issubclass(cls, StructuredError), cls.__name__


def test_zstack_error_carries_context():
    err = StructuredZStackError(entity_index=2, z=1.5, cohort_index=0)
    assert err.entity_index == 2
    assert err.z == 1.5
    assert err.cohort_index == 0
    assert "1.5" in str(err)
