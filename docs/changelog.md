# [Changelog](https://keepachangelog.com/en/1.0.0/)

## Unreleased

- fix mmg physicals [#PR150](https://github.com/simbilod/meshwell/pull/150)

## 2.2.0s

- remeshing + improved docs [#PR148](https://github.com/simbilod/meshwell/pull/148)

## 2.1.1

- bug fixes [#PR123](https://github.com/simbilod/meshwell/pull/123)
- arbitrary custom gmsh entities [PR#125](https://github.com/simbilod/meshwell/pull/125)
- refactor to expose some Model [#PR127](https://github.com/simbilod/meshwell/pull/127)
- more refactoring [#PR126](https://github.com/simbilod/meshwell/pull/130)[#PR126](https://github.com/simbilod/meshwell/pull/130)
- more bug fixes [#PR142](https://github.com/simbilod/meshwell/pull/142)[#PR143](https://github.com/simbilod/meshwell/pull/143)

## 2.1.0

- support lower-dimensional objects being passed, will be fragmented with higher-dimensional objects [PR#117](https://github.com/simbilod/meshwell/pull/117)
- polyline [PR#119](https://github.com/simbilod/meshwell/pull/119)

## 2.0.0

This is a major breaking release which separates the CAD and meshing operations. See the new syntax in the updates documentation.

- importing gds function and example [PR#102](https://github.com/simbilod/meshwell/pull/102)
- refactor meshing into more manageable chunks, fixing [issue#87](https://github.com/simbilod/meshwell/issues/87) [PR#105](https://github.com/simbilod/meshwell/pull/105) [PR#106](https://github.com/simbilod/meshwell/pull/106)
- refactor meshing into CAD + meshing to checkpoint [PR#107](https://github.com/simbilod/meshwell/pull/107)
- major refactor of the API to better implement CAD + meshing separation without intermediate JSON, fixing tests, fixing docs [PR#115](https://github.com/simbilod/meshwell/pull/115)

## 1.3.8

- fixed issue with application of resolutionspecs [PR#92](https://github.com/simbilod/meshwell/pull/98)
- allow regressions to target specific commit [PR#99](https://github.com/simbilod/meshwell/pull/99)
- parallel pytest
- restrict field [PR#100](https://github.com/simbilod/meshwell/pull/100)

## 1.3.7

- minor fixes

## 1.3.5

- handle boundary in shared/not shared [PR#86]https://github.com/simbilod/meshwell/pull/86)

## 1.3.4

- filter resolutionspec by shared/not shared [PR#85](https://github.com/simbilod/meshwell/pull/85)

## 1.3.2

- fixes to last updates [PR#84](https://github.com/simbilod/meshwell/pull/84)

## 1.3.1

- better exponential size field [PR#82](https://github.com/simbilod/meshwell/pull/82)

## 1.3.0

- major refactoring or ResolutionSpec; split into ConstantInField, ThresholdField, and new ExponentialField [PR#80](https://github.com/simbilod/meshwell/pull/80)

## 1.2.6

- apply resolution fields to groups of entities, keep some sampling calculation [PR#78](https://github.com/simbilod/meshwell/pull/78)

## 1.2.5

- resolution on points based on curve length [PR#77](https://github.com/simbilod/meshwell/pull/77)

## 1.2.4

- subdivide prisms [PR#76](https://github.com/simbilod/meshwell/pull/76)
- don't define resolution fields by default [PR#75](https://github.com/simbilod/meshwell/pull/75)

## 1.2.3

- expose Sampling parameter [PR#74](https://github.com/simbilod/meshwell/pull/74)

## 1.2.2

- sigmoid resolution tapering [PR#73](https://github.com/simbilod/meshwell/pull/73)

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
