import os
import pytest
from meshwell.config import PATH
from contextlib import redirect_stdout
import argparse
import sys
import pathlib

if __name__ == "__main__":

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Generate reference files for tests')
    parser.add_argument('--references-path', type=str, help='Path to references directory', default=PATH.references)
    args, remaining_argv = parser.parse_known_args()

    # Override PATH.references with command line argument if provided
    PATH.references = pathlib.Path(args.references_path)

    # Remove custom arguments from sys.argv
    sys.argv = [sys.argv[0]] + remaining_argv

    # Delete existing references
    exec_dir_name = "./"
    os.makedirs(PATH.references, exist_ok=True)
    test = os.listdir(PATH.references)
    for item in test:
        if item.endswith(".msh"):
            os.remove(os.path.join(PATH.references, item))

    # Run the tests to generate the files (tests will fail)
    pytest.main(["-n", "auto"])

    # Place the references
    test = os.listdir(exec_dir_name)
    for item in test:
        if item.endswith(".msh"):
            os.rename(item, PATH.references / f"{item[:-4]}.msh")
        if item.endswith(".xao"):
            os.rename(item, PATH.references / f"{item[:-4]}.xao")
