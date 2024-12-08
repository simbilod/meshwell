#!/bin/bash

# Create and activate virtual environment
uv venv --python=3.11
source .venv/bin/activate

# Install published version of the package in editable mode
sudo apt-get install libglu1-mesa
uv pip install git+https://github.com/simbilod/meshwell.git@4e7c483307d76768de2f45f92698911cb8f8ae54[dev]

# Execute all Python files in the current directory
python generate_references.py --references-path ./references/
