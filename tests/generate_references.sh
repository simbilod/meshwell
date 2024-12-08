#!/bin/bash

# Create and activate virtual environment
uv venv --python=3.11
source .venv/bin/activate

# Install published version of the package in editable mode
sudo apt-get install libglu1-mesa
uv pip install git+https://github.com/simbilod/meshwell.git@45571bf13b73a5a11bf7a5e4dbda99cb4a5dc91e[dev]

# Execute all Python files in the current directory
python generate_references.py --references-path ./references/
