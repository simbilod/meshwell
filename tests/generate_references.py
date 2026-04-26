"""Regenerate the reference .msh / .xao files used by byte-by-byte tests.

Sets ``MESHWELL_GENERATING_REFERENCES=1`` and runs the test suite. In that
mode, ``meshwell.utils.compare_gmsh_files`` (and ``compare_mesh_headers``)
copy each test's generated output to the reference location instead of
asserting equality. So the test suite itself produces the new fixtures --
no separate write step, no dependence on tests writing to CWD.

Intended invocation: from ``generate_references.sh`` (which installs
meshwell at a previous commit so the resulting references reflect that
commit's behaviour, against which the working tree is then compared).

Usage:
    python generate_references.py [--references-path <dir>]
"""
import argparse
import os
import pathlib
import sys

import pytest

from meshwell.config import PATH

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--references-path",
        type=str,
        help="Path to references directory",
        default=PATH.references,
    )
    args, remaining_argv = parser.parse_known_args()

    PATH.references = pathlib.Path(args.references_path)
    PATH.references.mkdir(parents=True, exist_ok=True)

    # Forward unrecognised CLI flags to pytest.
    sys.argv = [sys.argv[0], *remaining_argv]

    # Wipe stale references so removed tests don't leave dangling files.
    for item in os.listdir(PATH.references):
        if item.endswith((".msh", ".xao")):
            os.remove(os.path.join(PATH.references, item))

    # Switch compare_* helpers to snapshot mode.
    os.environ["MESHWELL_GENERATING_REFERENCES"] = "1"

    # Run the suite. Tests are expected to PASS in this mode (the
    # comparison step becomes a no-op write). Use -n auto for parallelism.
    sys.exit(pytest.main(["-n", "auto"]))
