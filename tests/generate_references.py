import os
import pytest
from meshwell.config import PATH

if __name__ == "__main__":
    # Delete existing references
    exec_dir_name = "./"
    test = os.listdir(PATH.references)

    for item in test:
        if item.endswith(".msh"):
            os.remove(os.path.join(PATH.references, item))

    # Run the tests to generate the files
    pytest.main()

    # # Rename the references
    test = os.listdir(exec_dir_name)
    for item in test:
        if item.endswith(".msh"):
            os.rename(item, PATH.references / f"{item[:-4]}.reference.msh")
