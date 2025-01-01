#!/bin/bash

# Create and activate virtual environment
uv venv --python=3.11
source .venv/bin/activate

# Install reference version of the package in editable mode
sudo apt-get install libglu1-mesa
uv pip install git+https://github.com/simbilod/meshwell.git@af140b69f2d563beed2c5389d27724d77a3429bb[dev]

# Execute all Python files in the current directory
python generate_references.py --references-path ./references/
