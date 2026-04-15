#!/bin/bash

# Create and activate virtual environment
uv venv --python=3.11
source .venv/bin/activate

# Install reference version of the package
sudo apt-get install libglu1-mesa
export PYTHONPATH=""
uv pip install git+https://github.com/simbilod/meshwell.git@0a4f7b59825a91bdabf35ad6de258dbbbc4ea3d68[dev]

# Execute all Python files in the current directory
python generate_references.py --references-path ./references/
