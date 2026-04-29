# Meshwell
<p align="center">
  <img src=https://raw.githubusercontent.com/simbilod/meshwell/main/meshwell.png
 width="300" height="300">
</p>

---

Robust, efficient Python-based 2.5D meshing.

## 🧱 Key Features

* Preprocessing of [Shapely](https://shapely.readthedocs.io/en/stable/)-based inputs:
  * vertex snapping + sub-tolerance buffering for robustness against arbitrary inputs with minimal distortion
  * automated curve fitting for seamless curvilinear meshing
* Shared-memory parallel CAD with [OpenCASCADE](https://dev.opencascade.org/)
* Shared-memory parallel meshing with [GMSH](https://gmsh.info/)
* Simplified remeshing with [MMG](https://github.com/MmgTools/MMG)
* Distributed memory processing with domain decomposition
  ([guide](docs/distributed.md))
* (Planned) Hybrid structured / unstructured meshes

## 📦 Installation

```bash
pip install meshwell
```

## 📖 Documentation

For more information and examples, visit the [Documentation Site](https://github.com/simbilod/meshwell/actions/workflows/pages.yml).
