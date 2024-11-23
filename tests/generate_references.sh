#!/bin/bash

# Create and activate virtual environment
uv venv --python=3.11
source .venv/bin/activate

# Install published version of the package in editable mode
uv pip install meshwell[dev]

# Execute all Python files in the current directory
python generate_references.py --references-path ./references/
