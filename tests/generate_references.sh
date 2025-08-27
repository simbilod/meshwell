#!/bin/bash

# Create and activate virtual environment
uv venv --python=3.11
source .venv/bin/activate

# Install reference version of the package
sudo apt-get install libglu1-mesa
uv pip install git+https://github.com/simbilod/meshwell.git@ef41f021881b07eb9a22a971d4055ab50d8a1866[dev]

# Execute all Python files in the current directory
python generate_references.py --references-path ./references/
