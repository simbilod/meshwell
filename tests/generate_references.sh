#!/bin/bash

# Create and activate virtual environment
uv venv --python=3.11
source .venv/bin/activate

# Install reference version of the package
sudo apt-get install libglu1-mesa
export PYTHONPATH=""
uv pip install git+https://github.com/simbilod/meshwell.git@7d1bdb7844ae85855cdfa5d7ac5a742964e89cfa[dev]

# Execute all Python files in the current directory
python generate_references.py --references-path ./references/
