import os
import pytest

if __name__ == "__main__":
    # Delete existing references
    exec_dir_name = "./"
    save_dir_name = "./references/"
    test = os.listdir(save_dir_name)

    for item in test:
        if item.endswith(".msh"):
            os.remove(os.path.join(save_dir_name, item))

    # Run the tests to generate the files
    pytest.main()

    # # Rename the references
    test = os.listdir(exec_dir_name)
    for item in test:
        if item.endswith(".msh"):
            os.rename(item, f"{save_dir_name}/{item[:-4]}.reference.msh")
