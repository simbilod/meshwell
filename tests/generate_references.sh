#!/bin/bash

# Create and activate virtual environment
uv venv --python=3.11
source .venv/bin/activate

# Install reference version of the package
sudo apt-get install libglu1-mesa
export PYTHONPATH=""
uv pip install git+https://github.com/simbilod/meshwell.git@0a37a9bf03c9bd1fd5a1cce61791b9e945dc68e71[dev]

# Execute all Python files in the current directory
python generate_references.py --references-path ./references/
