#!/bin/bash

# Create and activate virtual environment
uv venv --python=3.11
source .venv/bin/activate

# Install reference version of the package in editable mode
sudo apt-get install libglu1-mesa
uv pip install git+https://github.com/simbilod/meshwell.git@1d648ce8512a36a3a0b738466cb0c5a2d8f6d987[dev]

# Execute all Python files in the current directory
python generate_references.py --references-path ./references/
