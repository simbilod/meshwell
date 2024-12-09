#!/bin/bash

# Create and activate virtual environment
uv venv --python=3.11
source .venv/bin/activate

# Install published version of the package in editable mode
sudo apt-get install libglu1-mesa
uv pip install git+https://github.com/simbilod/meshwell.git@a72f6dfaf6b3b89d1ff7b6dd848524e803fddda2[dev]

# Execute all Python files in the current directory
python generate_references.py --references-path ./references/
