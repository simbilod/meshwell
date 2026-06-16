# Git-tag-driven versioning for meshwell

**Date:** 2026-06-15
**Status:** Approved design, pending implementation plan
**Scope:** Release-process simplification only. No library code behavior changes.

## Goal

Make the git tag the single source of truth for meshwell's version, so the
entire release becomes `git tag vX.Y.Z && git push --tags`. Today the version is
duplicated across four files that have already drifted; this design makes that
drift structurally impossible.

## Background / current state

The version lived (inconsistently) in four places before this work:

| Location | Value before | |
|---|---|---|
| `pyproject.toml` `[project] version` | `2.3.6` | the real one |
| `meshwell/__init__.py` `__version__` | `0.0.1` | stale |
| `.bumpversion.cfg` `current_version` | `0.0.1` | stale |
| `pyproject.toml` `[tool.commitizen] version` | `0.1.0` | stale |

Three release tools were half-configured at once (bump2version, commitizen,
hand-editing) and none agreed. `.bumpversion.cfg` also listed `README.md` as a
version file, but the README has no version string, so `bumpversion` would error
on first run.

A preparatory pass (already applied, outside this spec) reconciled the four
strings to `2.3.6` and fixed the `classifiers` (they advertised 3.11/3.12 while
`requires-python` and the only available `cadquery-ocp` wheels in `uv.lock` are
`>=3.13` / `cp313`). That left a consistent-but-still-manual state; this spec
removes the manual machinery.

## Decisions (locked)

- **Versioning model:** git tag is the version (dynamic, derived at build time).
- **PyPI auth:** keep the existing `twine` + `PYPI_API_TOKEN` flow unchanged.
- **Changelog:** keep `docs/90_changelog.md` hand-curated; no tooling touches it.

## Why the build backend must change

The current build backend is `flit_core`. flit can only take its version from a
literal or from a module `__version__` attribute — it has **no** mechanism to
read a git tag. Tag-based versioning therefore requires switching backends. The
standard, low-friction choice is **hatchling + hatch-vcs**.

## Design

### 1. Build backend → hatchling + hatch-vcs (`pyproject.toml`)

```toml
[build-system]
requires = ["hatchling", "hatch-vcs"]
build-backend = "hatchling.build"

[project]
# version = "2.3.6"   ->  removed
dynamic = ["version"]

[tool.hatch.version]
source = "vcs"

[tool.hatch.build.targets.wheel]
packages = ["meshwell"]
```

hatch-vcs reads the latest tag (`v2.3.7`), normalizes it to PEP 440 (`2.3.7`),
and bakes the resolved version into the wheel and sdist metadata — so PyPI
installs work even though the published artifacts contain no `.git`.

### 2. Runtime version → `importlib.metadata` (`meshwell/__init__.py`)

Replace the hardcoded literal with a metadata lookup so `meshwell.__version__`
can never drift from the installed distribution:

```python
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("meshwell")
except PackageNotFoundError:  # running from a source tree without install
    __version__ = "0.0.0"
```

### 3. Delete dead version machinery

- Delete `.bumpversion.cfg` entirely.
- Remove the `[tool.commitizen]` block from `pyproject.toml`. It existed only to
  track a version number; with tag-based versioning and a manual changelog it has
  no remaining job.

### 4. Release workflow (`.github/workflows/release.yml`)

Minimal change — auth stays as-is:

- Add `fetch-depth: 0` to the `actions/checkout` step so the tag is present in
  the checkout for hatch-vcs to read. Without it, the build resolves to a
  fallback version (e.g. `0.0.0`).
- Keep the `twine upload` step and `PYPI_API_TOKEN` secret unchanged.
- Tidy: drop the vestigial `pip install setuptools wheel` (not needed with
  `python -m build` + hatchling). Keep `build` and `twine`.

## Out of scope

- No switch to Trusted Publishing / OIDC (explicitly declined).
- No automated changelog generation (explicitly declined).
- No changes to `test_code.yml` (it reads Python from `requires-python`, which is
  unaffected) or to `pages.yml`.
- No library/runtime behavior changes beyond how `__version__` is sourced.

## Testing / verification

- In a checkout that has a tag, `python -m build` produces
  `meshwell-<tag>.tar.gz` / `meshwell-<tag>-*.whl` whose version equals the tag
  minus the leading `v`.
- After `pip install` of that artifact,
  `python -c "import meshwell; print(meshwell.__version__)"` prints the tag
  version.
- A dry-run of the release workflow logic (build step only) on a tagged ref
  yields correctly-named artifacts; the upload step is unchanged and need not be
  re-validated.
- `uv` workflows still resolve: `uv.lock` `requires-python` is untouched.

## Net result

The whole release is `git tag vX.Y.Z && git push --tags`. No version string is
edited in any file, ever. The changelog remains a single hand-edited file, and
PyPI authentication is unchanged.
