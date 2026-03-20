#!/bin/bash

# Create and activate virtual environment
uv venv --python=3.11
source .venv/bin/activate

# Install reference version of the package
sudo apt-get install libglu1-mesa
export PYTHONPATH=""
uv pip install git+https://github.com/simbilod/meshwell.git@0a224f1083eb0bd62d9ed15c780e3d78a754b81a[dev]

# Execute all Python files in the current directory
python generate_references.py --references-path ./references/
