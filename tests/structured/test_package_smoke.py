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


def test_phase2_public_exports():
    """Phase 2 adds build_phantom_shapes / extract_phantom_map / PhantomMap."""
    from meshwell.structured import (
        PhantomMap,
        build_phantom_shapes,
        extract_phantom_map,
    )

    assert PhantomMap is not None
    assert build_phantom_shapes is not None
    assert extract_phantom_map is not None
