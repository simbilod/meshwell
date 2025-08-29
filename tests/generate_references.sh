#!/bin/bash

# Create and activate virtual environment
uv venv --python=3.11
source .venv/bin/activate

# Install reference version of the package
sudo apt-get install libglu1-mesa
uv pip install git+https://github.com/simbilod/meshwell.git@0c76dc2d12316aa488746f2054c2123c29f32f0d[dev]

# Execute all Python files in the current directory
python generate_references.py --references-path ./references/
