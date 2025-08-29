#!/bin/bash

# Create and activate virtual environment
uv venv --python=3.11
source .venv/bin/activate

# Install reference version of the package
sudo apt-get install libglu1-mesa
uv pip install git+https://github.com/simbilod/meshwell.git@eaff76bb6a2549fe6e6c10e4939aa625c5c1b56a[dev]

# Execute all Python files in the current directory
python generate_references.py --references-path ./references/
