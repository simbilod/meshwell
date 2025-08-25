#!/bin/bash

# Create and activate virtual environment
uv venv --python=3.11
source .venv/bin/activate

# Install reference version of the package in editable mode
sudo apt-get install libglu1-mesa
uv pip install git+https://github.com/simbilod/meshwell.git@4bc8d3e7979859e40aba212e9aeb419018e03197[dev]

# Execute all Python files in the current directory
python generate_references.py --references-path ./references/
