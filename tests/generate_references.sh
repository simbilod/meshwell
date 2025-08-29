#!/bin/bash

# Create and activate virtual environment
uv venv --python=3.11
source .venv/bin/activate

# Install reference version of the package
sudo apt-get install libglu1-mesa
uv pip install git+https://github.com/simbilod/meshwell.git@02c5ec4b4ea7cf6321c20aa5eef3d2123a3a39c7[dev]

# Execute all Python files in the current directory
python generate_references.py --references-path ./references/
