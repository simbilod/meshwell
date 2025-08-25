#!/bin/bash

# Create and activate virtual environment
uv venv --python=3.11
source .venv/bin/activate

# Install reference version of the package in editable mode
sudo apt-get install libglu1-mesa
uv pip install git+https://github.com/simbilod/meshwell.git@2e72f3bc5a3061c4326d4094c30f2fe47bd15432[dev]

# Execute all Python files in the current directory
python generate_references.py --references-path ./references/
