#!/bin/bash
# Generate reference .msh / .xao files using a *previous* commit of meshwell,
# so that the local test suite's byte-by-byte `compare_gmsh_files` checks
# detect divergence introduced by changes in the working tree.
#
# Reference commit selection (in priority order):
#   1. $MESHWELL_REF_COMMIT environment variable, if set
#   2. HEAD~1 (the commit immediately before the current HEAD)
#
# The reference repository (where to install meshwell from):
#   1. $MESHWELL_REF_REPO environment variable, if set
#   2. Local checkout (file://...) -- works in CI and locally
#
# Usage:
#   bash generate_references.sh                              # uses HEAD~1 of local repo
#   MESHWELL_REF_COMMIT=abc123 bash generate_references.sh   # uses a specific commit
#
# CI must use `fetch-depth: 2` (or higher) so HEAD~1 is available.

set -euo pipefail

# Resolve reference commit
if [ -n "${MESHWELL_REF_COMMIT:-}" ]; then
    REF_COMMIT="$MESHWELL_REF_COMMIT"
else
    if ! git rev-parse --verify HEAD~1 >/dev/null 2>&1; then
        echo "ERROR: HEAD~1 is not available in this repository (shallow clone?)." >&2
        echo "       Set MESHWELL_REF_COMMIT explicitly or fetch more history." >&2
        exit 1
    fi
    REF_COMMIT="$(git rev-parse HEAD~1)"
fi

# Resolve reference repo (default: the local repository root, served via file://)
if [ -n "${MESHWELL_REF_REPO:-}" ]; then
    REF_REPO="$MESHWELL_REF_REPO"
else
    REPO_ROOT="$(git rev-parse --show-toplevel)"
    REF_REPO="git+file://${REPO_ROOT}"
fi

echo "[generate_references] Generating references from commit: $REF_COMMIT"
echo "[generate_references] Reference source: $REF_REPO"

# Create and activate venv
uv venv --python=3.13
# shellcheck source=/dev/null
source .venv/bin/activate

# System dependency for gmsh
sudo apt-get install -y libglu1-mesa

# Isolate from any host PYTHONPATH so we install cleanly
export PYTHONPATH=""

# Install meshwell at the reference commit
uv pip install "${REF_REPO}@${REF_COMMIT}[dev]"

# Run the reference generator (writes to ./references/)
python generate_references.py --references-path ./references/
