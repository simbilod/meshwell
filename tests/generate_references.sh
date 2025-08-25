#!/bin/bash

# Create and activate virtual environment
uv venv --python=3.11
source .venv/bin/activate

# Install reference version of the package in editable mode
sudo apt-get install libglu1-mesa
uv pip install git+https://github.com/simbilod/meshwell.git@2a17188d2d0cc603eea39a4888e9a805bb8e205f[dev]

# Execute all Python files in the current directory
python generate_references.py --references-path ./references/
