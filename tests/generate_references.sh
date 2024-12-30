#!/bin/bash

# Create and activate virtual environment
uv venv --python=3.11
source .venv/bin/activate

# Install reference version of the package in editable mode
sudo apt-get install libglu1-mesa
uv pip install git+https://github.com/simbilod/meshwell.git@ee10c5ab5da9c1311f5925907e08e494157c702d[dev]

# Execute all Python files in the current directory
python generate_references.py --references-path ./references/
