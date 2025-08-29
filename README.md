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

Meshwell is a Python wrapper around [GMSH](https://gmsh.info/) that provides:

(1) a PolyPrism class that simplifies, to the point of automating, the definition of solids from arbitrary (multi)polygons with "buffered" extrusions;

(2) a simple API where such Prisms and regular GMSH OCC objects are specified in an ordered dictionary of mesh priority, and whose keys are then used to label the mesh entities and their interfaces unambiguously;

See the documentation (under construction) for more information and examples. If you encounter a bug, you can make an issue so we can improve the software over time. Contributions are also welcome, see open issues for current bugs and requested features.
