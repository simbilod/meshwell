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


def test_all_matches_live_raised_exceptions():
    """``__all__`` must list exactly the exceptions raised in production code.

    Guards against re-introducing dead exceptions (never raised) or
    forgetting to export a newly raised one.
    """
    import pathlib

    import meshwell
    import meshwell.structured as structured
    from meshwell.structured import exceptions as exc_mod

    # Scan the whole meshwell package: some structured exceptions are
    # raised from core modules (e.g. polyprism.py), not only structured/.
    pkg_dir = pathlib.Path(meshwell.__file__).parent
    source = "\n".join(
        p.read_text()
        for p in pkg_dir.rglob("*.py")
        if p.name != "exceptions.py"
    )

    defined = {
        name
        for name in dir(exc_mod)
        if isinstance(getattr(exc_mod, name), type)
        and issubclass(getattr(exc_mod, name), exc_mod.StructuredError)
    }
    raised = {name for name in defined if f"raise {name}" in source}
    # The base class is exported for catching but never raised directly.
    expected = raised | {"StructuredError"}

    assert set(structured.__all__) == expected, (
        f"__all__ drifted from live exceptions. "
        f"missing={expected - set(structured.__all__)}, "
        f"extra (dead)={set(structured.__all__) - expected}"
    )
