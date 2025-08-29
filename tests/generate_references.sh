#!/bin/bash

# Create and activate virtual environment
uv venv --python=3.11
source .venv/bin/activate

# Install reference version of the package
sudo apt-get install libglu1-mesa
uv pip install git+https://github.com/simbilod/meshwell.git@f71bb31c5fdcb15137d6165c961a9e5f74bb6f99[dev]

# Execute all Python files in the current directory
python generate_references.py --references-path ./references/
