#!/bin/bash

# Create and activate virtual environment
uv venv --python=3.11
source .venv/bin/activate

# Install reference version of the package in editable mode
sudo apt-get install libglu1-mesa
uv pip install git+https://github.com/simbilod/meshwell.git@1f6880659710b4e1f057673f3ce9ad68a6e6b7ac[dev]

# Execute all Python files in the current directory
python generate_references.py --references-path ./references/
