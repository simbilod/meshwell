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

(1) a Prism class that simplifies, to the point of automating, the definition of solids from arbitrary (multi)polygons with "buffered" extrusions;

(2) a simple API where such Prisms and regular GMSH OCC objects are specified in an ordered dictionary of mesh priority, and whose keys are then used to label the mesh entities and their interfaces unambiguously;

For instance:

<ADD EXAMPLE>

See the documentation for more information and examples. If you encounter a big, you can make an issue so we can improve the software over time. Contributions are also welcome, see open issues for current bugs and requested features.

### Background

This code was originally developed to define meshes out of the GDSII descriptions of integrated photonic circuits. A particularity of such devices is rich 2.5D topology, featuring multiple layers of smooth curves in the plane and etching profiles vertically.  Maxwell's equations (hence the name) are solved on these geometries to study how light propagates. It is also of critical interest to simulate how this is affected under other physical effects that can be resolved through finite-element or finite-volume analysis.

### Related projects

* [gdsfactory](https://github.com/gdsfactory/gdsfactory): open-source plugin-rich layout software; meshwell is the backend to [gplugins](https://github.com/gdsfactory/gplugins)' gmsh module
* [femwell](https://github.com/HelgeGehring/femwell): open-source scikit-fem based finite-element simulations, with emphasis on photonics
* [DEVSIM](https://github.com/devsim/devsim): open-source finite-volume simulator, with emphasis on semiconductor TCAD

### Other notable GMSH Python interfaces:

* [gmsh](https://gitlab.onelab.info/gmsh/gmsh): the gmsh Python API itself has significantly improved over the years
* [pygmsh](https://github.com/meshpro/pygmsh): manipulate Python objects instead of gmsh entity tags
* [objectgmsh](https://github.com/nemocrys/objectgmsh): class wrappers around entities
* [gyptis](https://gyptis.gitlab.io/): uses basic gmsh for photonic geometries

### Acknowledgements

* Simon Bilodeau (Princeton): maintainer
* Helge Gehring (Google X): beta testing, use cases, bug fixes, improvements
* Joaquin Matres Abril (Google X): code improvements
* Niko Savola (Google): beta testing, use cases, bug fixes, improvements
