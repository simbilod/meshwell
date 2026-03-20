#!/bin/bash

# Create and activate virtual environment
uv venv --python=3.11
source .venv/bin/activate

# Install reference version of the package
sudo apt-get install libglu1-mesa
export PYTHONPATH=""
uv pip install git+https://github.com/simbilod/meshwell.git@21f279cebb69f3fa0d7e6e1f442776ea0388c702[dev]

# Execute all Python files in the current directory
python generate_references.py --references-path ./references/
