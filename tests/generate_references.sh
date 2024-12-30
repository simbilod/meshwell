#!/bin/bash

# Create and activate virtual environment
uv venv --python=3.11
source .venv/bin/activate

# Install reference version of the package in editable mode
sudo apt-get install libglu1-mesa
uv pip install git+https://github.com/simbilod/meshwell.git@2c198ca76e04ef1bd752828c6fded2c8771d49c2[dev]

# Execute all Python files in the current directory
python generate_references.py --references-path ./references/
