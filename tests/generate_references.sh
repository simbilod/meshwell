#!/bin/bash

# Create and activate virtual environment
uv venv --python=3.11
source .venv/bin/activate

# Install reference version of the package
sudo apt-get install libglu1-mesa
export PYTHONPATH=""
uv pip install git+https://github.com/simbilod/meshwell.git@9f8d8e7f748a8457b5525f57e935e5c2dd0c0ba8[dev]

# Execute all Python files in the current directory
python generate_references.py --references-path ./references/
