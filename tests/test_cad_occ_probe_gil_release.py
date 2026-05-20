"""GIL-release probe: cached, returns a boolean."""
from __future__ import annotations

import meshwell.cad_occ as cad_occ_mod
from meshwell.cad_occ import _probe_gil_release


def test_probe_returns_bool():
    cad_occ_mod._GIL_PROBE_CACHE = None
    result = _probe_gil_release()
    assert isinstance(result, bool)


def test_probe_is_cached_between_calls():
    cad_occ_mod._GIL_PROBE_CACHE = None
    first = _probe_gil_release()
    # Force the cache to a deliberately-wrong value; second call must use it.
    cad_occ_mod._GIL_PROBE_CACHE = not first
    second = _probe_gil_release()
    assert second == (not first)
