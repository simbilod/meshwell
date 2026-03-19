#!/bin/bash

# Create and activate virtual environment
uv venv --python=3.11
source .venv/bin/activate

# Install reference version of the package
sudo apt-get install libglu1-mesa
export PYTHONPATH=""
uv pip install git+https://github.com/simbilod/meshwell.git@05daeb1daaea416e7b019ee7d93a586ee1eac1cf[dev]

# Execute all Python files in the current directory
python generate_references.py --references-path ./references/
