#!/bin/bash

# Create and activate virtual environment
uv venv --python=3.11
source .venv/bin/activate

# Install reference version of the package
sudo apt-get install libglu1-mesa
uv pip install git+https://github.com/simbilod/meshwell.git@d887f3acdd6749c5c7780f47888b748d98559756[dev]

# Execute all Python files in the current directory
python generate_references.py --references-path ./references/
