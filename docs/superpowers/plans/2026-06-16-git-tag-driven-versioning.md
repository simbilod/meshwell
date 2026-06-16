# Git-Tag-Driven Versioning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the git tag the single source of truth for meshwell's version, so a release is `git tag vX.Y.Z && git push --tags` with no version string edited anywhere.

**Architecture:** Switch the build backend from `flit_core` to `hatchling` + `hatch-vcs` so the version is derived from the latest git tag at build time. The runtime `meshwell.__version__` reads installed package metadata via `importlib.metadata`. Delete the redundant version-tracking tools (`.bumpversion.cfg`, `[tool.commitizen]`). The release workflow keeps its existing `twine` + `PYPI_API_TOKEN` upload; it only gains `fetch-depth: 0` so the tag is visible to hatch-vcs.

**Tech Stack:** Python packaging (hatchling, hatch-vcs), `python -m build`, GitHub Actions, uv.

**Spec:** `docs/superpowers/specs/2026-06-15-git-tag-driven-versioning-design.md`

**Working directory:** All paths are relative to the git repo root `meshwell/` (i.e. `/home/simbil/Github/meshwell_structured_manual/meshwell`). The git repo is this directory, not its parent.

---

## Task 1: Switch build backend to hatchling + hatch-vcs

**Files:**
- Modify: `pyproject.toml` (build-system, project.version → dynamic, add hatch config, remove commitizen)

- [ ] **Step 1: Confirm the current build-system and version lines**

Run: `sed -n '1,6p;19,21p' pyproject.toml`
Expected output includes:
```
build-backend = "flit_core.buildapi"
requires = ["flit_core >=3.2,<4"]
...
requires-python = ">=3.13"
version = "2.3.6"
```

- [ ] **Step 2: Replace the `[build-system]` table**

Change:
```toml
[build-system]
build-backend = "flit_core.buildapi"
requires = ["flit_core >=3.2,<4"]
```
to:
```toml
[build-system]
build-backend = "hatchling.build"
requires = ["hatchling", "hatch-vcs"]
```

- [ ] **Step 3: Make `[project].version` dynamic**

In the `[project]` table, replace the line `version = "2.3.6"` with:
```toml
dynamic = ["version"]
```
(Delete the `version = "2.3.6"` line entirely; do not leave both.)

- [ ] **Step 4: Add hatch version + wheel-target config**

Add these two tables. Put them immediately after the `[build-system]` table (top of file, so they are easy to find):
```toml
[tool.hatch.version]
source = "vcs"

[tool.hatch.build.targets.wheel]
packages = ["meshwell"]
```

- [ ] **Step 5: Remove the `[tool.commitizen]` block**

Delete these four lines from `pyproject.toml`:
```toml
[tool.commitizen]
name = "cz_conventional_commits"
version = "2.3.6"
version_files = ["pyproject.toml:version"]
```

- [ ] **Step 6: Verify TOML is still valid and version resolves from the tag**

Run: `python -c "import tomllib; tomllib.load(open('pyproject.toml','rb')); print('toml ok')"`
Expected: `toml ok`

Run: `python -m pip install --quiet hatchling hatch-vcs hatch && python -c "from hatchling.metadata.core import ProjectMetadata" 2>/dev/null; git describe --tags --abbrev=0`
Expected: prints the latest tag, e.g. `v2.3.6` (confirms a tag exists for hatch-vcs to read).

- [ ] **Step 7: Build and confirm the artifact version equals the tag**

Run:
```bash
rm -rf dist && python -m pip install --quiet build && python -m build
ls dist
```
Expected: `dist/` contains `meshwell-2.3.6.tar.gz` and `meshwell-2.3.6-py3-none-any.whl` (version `2.3.6`, taken from tag `v2.3.6` — NOT `0.0.0` and NOT a dev suffix like `2.3.6.post1.devN` on a clean tagged checkout).

> If the version shows a `.devN+g<sha>` suffix, the checkout is ahead of the tag (expected on a dirty/ahead working tree). On the exact tagged commit it is clean. This is acceptable; the CI release runs on the exact tag.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml
git commit -m "build: derive version from git tag via hatch-vcs"
```

---

## Task 2: Source runtime `__version__` from package metadata

**Files:**
- Modify: `meshwell/__init__.py:12`

- [ ] **Step 1: Confirm the current hardcoded version line**

Run: `sed -n '12p' meshwell/__init__.py`
Expected: `__version__ = "2.3.6"`

- [ ] **Step 2: Replace the hardcoded literal with a metadata lookup**

Replace the line:
```python
__version__ = "2.3.6"
```
with:
```python
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("meshwell")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0"
```

- [ ] **Step 3: Verify the installed version is reported**

Run: `python -m pip install --quiet -e . && python -c "import meshwell; print(meshwell.__version__)"`
Expected: prints `2.3.6` (the tag version; an editable install still resolves real metadata).

- [ ] **Step 4: Commit**

```bash
git add meshwell/__init__.py
git commit -m "build: read __version__ from importlib.metadata"
```

---

## Task 3: Delete the dead bump2version config

**Files:**
- Delete: `.bumpversion.cfg`

- [ ] **Step 1: Confirm the file exists and is the bump2version config**

Run: `cat .bumpversion.cfg`
Expected: shows `[bumpversion]` with `current_version = 2.3.6` and the file targets.

- [ ] **Step 2: Delete it**

Run: `git rm .bumpversion.cfg`
Expected: `rm '.bumpversion.cfg'`

- [ ] **Step 3: Confirm no other file references it**

Run: `grep -rn "bumpversion" . --exclude-dir=.git --exclude-dir=.venv || echo "no references"`
Expected: `no references`

- [ ] **Step 4: Commit**

```bash
git commit -m "build: remove unused bump2version config"
```

---

## Task 4: Make the release workflow tag-aware

**Files:**
- Modify: `.github/workflows/release.yml`

- [ ] **Step 1: Confirm current checkout and install steps**

Run: `sed -n '1,40p' .github/workflows/release.yml`
Expected: shows `uses: actions/checkout@v6` (no `with:` block) and `pip install setuptools wheel twine`.

- [ ] **Step 2: Add `fetch-depth: 0` to the checkout step**

Replace:
```yaml
      - uses: actions/checkout@v6
```
with:
```yaml
      - uses: actions/checkout@v6
        with:
          fetch-depth: 0
```
(This makes the tag history available so hatch-vcs resolves the real version instead of a fallback.)

- [ ] **Step 3: Drop the vestigial setuptools/wheel install**

Replace:
```yaml
          python -m pip install --upgrade pip
          pip install setuptools wheel twine
```
with:
```yaml
          python -m pip install --upgrade pip
          pip install build twine
```
(Leave the `Build and publish` step's `pip install build` if present — it is now redundant but harmless; if it exists, remove the duplicate `pip install build` line from the publish step so `build` is installed once in the dependencies step.)

- [ ] **Step 4: Verify the YAML parses**

Run: `python -c "import yaml; yaml.safe_load(open('.github/workflows/release.yml')); print('yaml ok')"`
Expected: `yaml ok`
(If PyYAML is not installed: `python -m pip install --quiet pyyaml` first.)

- [ ] **Step 5: Confirm the upload step and token are unchanged**

Run: `grep -n "TWINE_PASSWORD\|PYPI_API_TOKEN\|twine upload" .github/workflows/release.yml`
Expected: the `TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}` and `twine upload dist/*` lines are still present (auth intentionally unchanged).

- [ ] **Step 6: Commit**

```bash
git add .github/workflows/release.yml
git commit -m "ci: fetch full history in release so hatch-vcs sees the tag"
```

---

## Task 5: End-to-end verification on a clean tagged build

**Files:** none (verification only)

- [ ] **Step 1: Clean-build from the current tag and inspect the wheel metadata**

Run:
```bash
rm -rf dist && python -m build
python - <<'PY'
import pathlib, re
whl = next(pathlib.Path("dist").glob("meshwell-*.whl"))
print("artifact:", whl.name)
assert re.match(r"meshwell-2\.3\.6(\.|-)", whl.name), f"unexpected version in {whl.name}"
print("version OK")
PY
```
Expected: prints the wheel name and `version OK` (artifact version matches tag `v2.3.6`; tolerates a dev suffix if the working tree is ahead of the tag).

- [ ] **Step 2: Install the built wheel into a throwaway venv and check `__version__`**

Run:
```bash
python -m venv /tmp/mw_verify && /tmp/mw_verify/bin/pip install --quiet dist/meshwell-*.whl
/tmp/mw_verify/bin/python -c "import meshwell; print(meshwell.__version__)"
rm -rf /tmp/mw_verify
```
Expected: prints `2.3.6` (matches the tag; confirms metadata-based `__version__` round-trips through a real install).

- [ ] **Step 3: Confirm no stale version literals remain in the repo**

Run: `grep -rn "0\.0\.1\|0\.1\.0" pyproject.toml meshwell/__init__.py 2>/dev/null || echo "no stale literals"`
Expected: `no stale literals`

- [ ] **Step 4: Final sanity — only the tag defines the version now**

Run: `grep -rn "^version = \|current_version\|\[tool.commitizen\]" pyproject.toml .bumpversion.cfg 2>/dev/null || echo "single source: git tag"`
Expected: `single source: git tag` (no literal `version =` in `[project]`, no `.bumpversion.cfg`, no commitizen block).

---

## Self-Review Notes

- **Spec coverage:** Task 1 ⇒ backend switch + dynamic version + commitizen removal (spec §1, §3). Task 2 ⇒ importlib.metadata runtime version (spec §2). Task 3 ⇒ delete `.bumpversion.cfg` (spec §3). Task 4 ⇒ release workflow `fetch-depth: 0` + tidy, auth unchanged (spec §4). Task 5 ⇒ verification (spec "Testing / verification"). Out-of-scope items (OIDC, auto-changelog, test_code.yml, pages.yml) are untouched.
- **Type/name consistency:** package target `["meshwell"]`, distribution name `meshwell`, tag prefix `v`, PEP 440 result `2.3.6` used consistently across tasks.
- **No placeholders:** every code/config edit shows exact before/after text and an explicit verification command with expected output.
