# Meshwell
<p align="center">
  <img src=https://raw.githubusercontent.com/simbilod/meshwell/main/meshwell.png
 width="300" height="300">
</p>

---

[![Docs](https://github.com/simbilod/meshwell/actions/workflows/pages.yml/badge.svg)](https://github.com/simbilod/meshwell/actions/workflows/pages.yml)
[![Tests](https://github.com/simbilod/meshwell/actions/workflows/test_code.yml/badge.svg)](https://github.com/simbilod/meshwell/actions/workflows/test_code.yml)
[![PiPy](https://img.shields.io/pypi/v/meshwell)](https://pypi.org/project/meshwell/)

**Project is under active development, stay tuned for improved features, documentation, and releases!**

Meshwell is a Python wrapper around [GMSH](https://gmsh.info/) and [MMG](https://github.com/MmgTools/MMG) that streamlines the CAD definition, meshing, and remeshing of geometries parametrized by [Shapely](https://shapely.readthedocs.io/en/stable/) polygons (and more).

## Installation of Dependencies (MMG / ParMMG)

Meshwell relies on [MMG](https://github.com/MmgTools/mmg) and [ParMMG](https://github.com/MmgTools/ParMmg) for adaptive remeshing capabilities. We provide a Docker-based build process to simplify the installation of these tools.

To build the Docker image and extract the compiled binaries to `meshwell/bin`, run the following command from the `meshwell` root directory:

```bash
./docker/build.sh
```

After the build completes, you can add the binaries to your PATH:

```bash
export PATH=$PWD/bin:$PATH
```

