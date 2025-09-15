#!/bin/bash

# Create and activate virtual environment
uv venv --python=3.11
source .venv/bin/activate

# Install reference version of the package
sudo apt-get install libglu1-mesa
uv pip install git+https://github.com/simbilod/meshwell.git@ea8190d0776865a788a244c0dc09e3f94b544e24[dev]

# Execute all Python files in the current directory
python generate_references.py --references-path ./references/
