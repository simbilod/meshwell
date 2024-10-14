# [Changelog](https://keepachangelog.com/en/1.0.0/)

## 1.2.1

- bug fixes

## 1.2.0

- "additive" entities [PR#71](https://github.com/simbilod/meshwell/pull/71)
- ability for entities to hold multiple physical_names [PR#72](https://github.com/simbilod/meshwell/pull/72)

## 1.1.3

- resolution_factor method in ResolutionSpec [PR#70](https://github.com/simbilod/meshwell/pull/70)

## 1.1.2

- ResolutionSpec instead of resolution dict [PR#69](https://github.com/simbilod/meshwell/pull/69)
- Skipping version Git refactoring

## 1.0.9

- curves resolution [PR#67](https://github.com/simbilod/meshwell/pull/67) [PR#68](https://github.com/simbilod/meshwell/pull/68)

## 1.0.7

- filename can be pathlib.Path [commit a755de1](https://github.com/simbilod/meshwell/commit/a755de191140d267f4012ffd9f7b149762281091)

## 1.0.6

Speed improvements:
- surface extrusion for buffer-free prisms [PR#59](https://github.com/simbilod/meshwell/pull/59)
- faster interior definition [PR#58](https://github.com/simbilod/meshwell/pull/58)

## 1.0.5

- smoothing options [PR#51](https://github.com/simbilod/meshwell/pull/51)

## 1.0.4

- background remeshing [PR#50](https://github.com/simbilod/meshwell/pull/50)
- minor fixes

## 1.0.3

- step output [PR#43](https://github.com/simbilod/meshwell/pull/43)
- fuse entities by name [PR#44](https://github.com/simbilod/meshwell/pull/44)

## 1.0.2

- default mesh ordering [PR#42](https://github.com/simbilod/meshwell/pull/42)

## 1.0.1

- resolution dict --> resolution attributes [PR#38](https://github.com/simbilod/meshwell/pull/38)

## 1.0.0

Refactoring release, with more to come.

- change from dict to attributes [PR#36](https://github.com/simbilod/meshwell/pull/36)
- more tests [PR#36](https://github.com/simbilod/meshwell/pull/36)
- validation of buffers causing CAD failure [PR#36](https://github.com/simbilod/meshwell/pull/36)
- pydantic coverage [PR#36](https://github.com/simbilod/meshwell/pull/35)
- speedup the CAD operations [PR#31, see also #29, #30](https://github.com/simbilod/meshwell/pull/31)
- photonic crystal sample [PR#35](https://github.com/simbilod/meshwell/pull/35)

## 0.1.0

- periodic boundaries [PR#25](https://github.com/simbilod/meshwell/pull/25)
- default resolution [PR#24](https://github.com/simbilod/meshwell/pull/24/files)
- parallelize CAD operations [PR#22](https://github.com/simbilod/meshwell/pull/22)
- more resolution flags [PR#21](https://github.com/simbilod/meshwell/pull/21)

## 0.0.9

- tqdm requirement
## 0.0.8

- tqdm [PR#16](https://github.com/simbilod/meshwell/pull/16)
- choose gmsh version [PR#15](https://github.com/simbilod/meshwell/pull/15)
## 0.0.7

- add samples
## 0.0.6

- fix boundaries tagging
- check for empty entities to reduce errors

## 0.0.4

- choose meshing algorithm
- number of threads [PR#9](https://github.com/simbilod/meshwell/pull/9)

## 0.0.3

- global mesh scaling option

## 0.0.2

- stabilizing API
- boundary elements

## 0.0.1

- initial
