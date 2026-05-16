"""Smoke test: the structured package exists and is importable."""
from __future__ import annotations


def test_structured_package_importable():
    import meshwell.structured  # noqa: F401


def test_structured_public_exports_present():
    """__init__ re-exports the public spec dataclass.

    Intentional TDD red-bar: fails at Task 2 (spec.py absent), passes when Task 3 lands it.
    Not marked xfail to avoid XPASS noise breaking the planned TDD progression.
    """
    from meshwell.structured import StructuredExtrusionResolutionSpec

    assert StructuredExtrusionResolutionSpec is not None
