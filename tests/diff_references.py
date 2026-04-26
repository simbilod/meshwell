#!/usr/bin/env python3
"""Diff current reference output against the previous commit's references.

Usage:
    # Show diffs for all reference files that changed between HEAD~1 and now:
    python tests/diff_references.py

    # Show diff for one specific reference:
    python tests/diff_references.py --name test_2D_resolution

    # Compare against a different base ref:
    python tests/diff_references.py --base origin/main

After regenerating references locally (via tests/generate_references.sh),
run this to inspect what changed before committing the new references.
"""
from __future__ import annotations

import argparse
import difflib
import shutil
import subprocess
import sys
from pathlib import Path

REFS_DIR = Path(__file__).parent / "references"

_GIT = shutil.which("git") or "git"


def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        [_GIT, *args], capture_output=True, text=True, check=check
    )


def list_changed_refs(base: str) -> list[Path]:
    """Reference files that exist locally and differ from `base`."""
    out = (
        _git("diff", "--name-only", base, "--", "tests/references/")
        .stdout.strip()
        .splitlines()
    )
    return [Path(p) for p in out if p]


def show_old(path: Path, base: str) -> str | None:
    """Return the file content at `base`, or None if not present there."""
    res = _git("show", f"{base}:{path}", check=False)
    if res.returncode != 0:
        return None
    return res.stdout


def diff_one(path: Path, base: str) -> None:
    print(f"\n=== {path} ===")
    old = show_old(path, base)
    new = path.read_text() if path.exists() else None
    if old is None and new is None:
        print(f"  (deleted in working tree, not present at {base})")
        return
    if old is None:
        print(f"  (new file; not present at {base})")
        return
    if new is None:
        print(f"  (deleted in working tree; was present at {base})")
        return
    diff = difflib.unified_diff(
        old.splitlines(keepends=True),
        new.splitlines(keepends=True),
        fromfile=f"{base}:{path}",
        tofile=f"local:{path}",
        n=3,
    )
    sys.stdout.writelines(diff)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--base",
        default="HEAD~1",
        help="Git revision to compare against (default: HEAD~1)",
    )
    p.add_argument(
        "--name", help="Compare only this reference (basename, no extension)"
    )
    args = p.parse_args()

    if args.name:
        targets = sorted(REFS_DIR.glob(f"{args.name}.*"))
        if not targets:
            print(
                f"No reference files found matching name '{args.name}'", file=sys.stderr
            )
            return 1
        # Translate to repo-relative paths
        repo = _git("rev-parse", "--show-toplevel").stdout.strip()
        rel = [t.relative_to(repo) for t in targets]
    else:
        rel = list_changed_refs(args.base)
        if not rel:
            print(f"No reference-file changes between {args.base} and working tree.")
            return 0

    for path in rel:
        diff_one(path, args.base)
    return 0


if __name__ == "__main__":
    sys.exit(main())
