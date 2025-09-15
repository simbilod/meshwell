#!/bin/bash

# Create and activate virtual environment
uv venv --python=3.11
source .venv/bin/activate

# Install reference version of the package
sudo apt-get install libglu1-mesa
uv pip install git+https://github.com/simbilod/meshwell.git@533bde6e7f867fea087176c02f05f777b1cd2825[dev]

# Execute all Python files in the current directory
python generate_references.py --references-path ./references/
