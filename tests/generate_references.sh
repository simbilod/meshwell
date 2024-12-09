#!/bin/bash

# Create and activate virtual environment
uv venv --python=3.11
source .venv/bin/activate

# Install published version of the package in editable mode
sudo apt-get install libglu1-mesa
uv pip install git+https://github.com/simbilod/meshwell.git@21d171d530f89af2f17f14478e95cbc40488a2d5[dev]

# Execute all Python files in the current directory
python generate_references.py --references-path ./references/
