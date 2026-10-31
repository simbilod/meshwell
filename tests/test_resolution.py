from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import shapely

from meshwell.cad_occ import cad_occ
from meshwell.mesh import mesh
from meshwell.occ_xao_writer import write_xao
from meshwell.polyprism import PolyPrism
from meshwell.polysurface import PolySurface
from meshwell.resolution import (
    BoxField,
    ConstantInField,
    ExponentialField,
    RegionThresholdField,
    ThresholdField,
)
from meshwell.utils import compare_gmsh_files


@pytest.mark.slow
def test_2D_resolution(tmp_path):
    large_rect = 20
    small_rect = 5

    polygon1 = shapely.Polygon(
        [[0, 0], [large_rect, 0], [large_rect, large_rect], [0, large_rect], [0, 0]],
    )
    polygon2 = shapely.Polygon(
        [
            [small_rect, small_rect],
            [large_rect - small_rect, small_rect],
            [large_rect - small_rect, large_rect - small_rect],
            [small_rect, 6],
            [small_rect, small_rect],
        ],
    )

    poly_obj1 = PolySurface(
        polygons=polygon1,
        mesh_order=2,
        physical_name="outer",
    )
    poly_obj2 = PolySurface(
        polygons=polygon2,
        mesh_order=1,
        physical_name="inner",
    )

    entities_list = [poly_obj1, poly_obj2]

    write_xao(
        cad_occ(
            entities_list,
        ),
        str(tmp_path / "test_2D_resolution.xao"),
    )

    mesh(
        dim=2,
        input_file=str(tmp_path / "test_2D_resolution.xao"),
        output_file=str(tmp_path / "test_2D_resolution.msh"),
        resolution_specs={
            "inner": [
                ThresholdField(sizemin=0.1, distmax=2, sizemax=1, apply_to="curves")
            ],
            "outer": [ConstantInField(apply_to="surfaces", resolution=0.5)],
        },
        n_threads=1,
        default_characteristic_length=100,
    )

    compare_gmsh_files(tmp_path / "test_2D_resolution.msh")


@pytest.mark.slow
def test_3D_resolution(tmp_path):
    polygon1 = shapely.Polygon(
        [[0, 0], [9, 0], [9, 9], [0, 9], [0, 0]],
    )
    polygon2 = shapely.Polygon(
        [[3, 3], [6, 3], [6, 6], [3, 6], [3, 3]],
    )

    buffers = {0.0: 0.0, 3: 0.0}

    prism_obj1 = PolyPrism(
        polygons=polygon1,
        buffers=buffers,
        mesh_order=2,
        physical_name="outer",
    )
    prism_obj2 = PolyPrism(
        polygons=polygon2,
        buffers=buffers,
        mesh_order=1,
        physical_name="inner",
    )

    entities_list = [prism_obj1, prism_obj2]

    write_xao(
        cad_occ(
            entities_list,
        ),
        str(tmp_path / "test_3D_resolution.xao"),
    )

    mesh(
        dim=3,
        input_file=str(tmp_path / "test_3D_resolution.xao"),
        output_file=str(tmp_path / "test_3D_resolution.msh"),
        resolution_specs={
            "inner": [
                ConstantInField(resolution=1, apply_to="volumes"),
                ThresholdField(sizemin=0.2, distmax=1, sizemax=1, apply_to="surfaces"),
            ],
            "outer": [ConstantInField(resolution=1, apply_to="volumes")],
        },
        n_threads=1,
        default_characteristic_length=100,
    )

    compare_gmsh_files(tmp_path / "test_3D_resolution.msh")


@pytest.mark.slow
def test_exponential_field():
    large_rect = 40
    small_rect = 5

    polygon1 = shapely.Polygon(
        [[0, 0], [large_rect, 0], [large_rect, large_rect], [0, large_rect], [0, 0]],
    )
    polygon2 = shapely.Polygon(
        [
            [small_rect, small_rect],
            [large_rect - small_rect, small_rect],
            [large_rect - small_rect, large_rect - small_rect],
            [small_rect, 6],
            [small_rect, small_rect],
        ],
    )

    poly_obj1 = PolySurface(
        polygons=polygon1,
        mesh_order=2,
        physical_name="outer",
    )
    poly_obj2 = PolySurface(
        polygons=polygon2,
        mesh_order=1,
        physical_name="inner",
    )

    entities_list = [poly_obj1, poly_obj2]

    write_xao(
        cad_occ(
            entities_list,
        ),
        "test_exponential_field.xao",
    )

    mesh(
        dim=3,
        input_file="test_exponential_field.xao",
        output_file="test_exponential_field.msh",
        resolution_specs={
            "inner": [
                ExponentialField(
                    growth_factor=2,
                    sizemin=0.3,
                    max_samplings=200,
                    apply_to="curves",
                    lengthscale=2,
                )
            ],
            "outer": [ConstantInField(apply_to="surfaces", resolution=5)],
        },
        n_threads=1,
        default_characteristic_length=100,
    )

    compare_gmsh_files(Path("test_exponential_field.msh"))


@pytest.mark.slow
@pytest.mark.parametrize(
    ("field", "label"),
    [
        (ConstantInField(apply_to="surfaces", resolution=1), "constant"),
        (
            ExponentialField(
                growth_factor=2,
                sizemin=0.3,
                max_samplings=200,
                apply_to="curves",
                lengthscale=2,
            ),
            "exponential",
        ),
        (
            ThresholdField(
                sizemin=1,
                sizemax=5,
                distmin=0,
                distmax=5,
                apply_to="curves",
            ),
            "threshold",
        ),
    ],
)
def test_refine(field, label):
    large_rect = 10

    polygon1 = shapely.Polygon(
        [[0, 0], [large_rect, 0], [large_rect, large_rect], [0, large_rect], [0, 0]],
    )

    poly_obj2 = PolySurface(
        polygons=polygon1,
        mesh_order=1,
        physical_name="inner",
    )

    write_xao(
        cad_occ(
            [poly_obj2],
        ),
        f"test_refine_{label}.xao",
    )

    points = []

    for factor in [0.5, 1, 2]:
        poly_obj2 = PolySurface(
            polygons=polygon1,
            mesh_order=1,
            physical_name="inner",
        )
        output = mesh(
            dim=3,
            input_file=f"test_refine_{label}.xao",
            output_file=f"test_refine_{label}.msh",
            resolution_specs={
                "inner": [field.refine(factor)],
            },
            n_threads=1,
            default_characteristic_length=100,
        )

        points.append(len(output.points))

    print(points)

    assert points[0] > points[1] > points[2]


# FIXME: add regression
@pytest.mark.slow
@pytest.mark.parametrize(
    ("apply_to", "min_mass", "max_mass"),
    [
        ("volumes", 5**3, np.inf),
        ("volumes", 0, 5**3),
        ("surfaces", 5**2, np.inf),
        ("surfaces", 0, 5**2),
        ("curves", 5, np.inf),
        ("curves", 0, 5),
        ("points", 5, np.inf),
        ("points", 0, 5),
    ],
)
def test_filter(apply_to, min_mass, max_mass):
    label = f"{apply_to}_{min_mass}_{max_mass}"

    large_rect = 6
    small_rect = 4

    buffers1 = {-6: 0, 6: 0}
    buffers2 = {-4: 0, 4: 0}

    polygon1 = shapely.Polygon(
        [
            [-large_rect / 2, -large_rect / 2],
            [large_rect / 2, -large_rect / 2],
            [large_rect / 2, large_rect / 2],
            [-large_rect / 2, large_rect / 2],
            [-large_rect / 2, -large_rect / 2],
        ],
    )
    polygon2 = shapely.Polygon(
        [
            [-small_rect / 2, -small_rect / 2],
            [small_rect / 2, -small_rect / 2],
            [small_rect / 2, small_rect / 2],
            [-small_rect / 2, small_rect / 2],
            [-small_rect / 2, -small_rect / 2],
        ],
    )

    prism1 = PolyPrism(
        polygons=polygon1,
        buffers=buffers1,
        mesh_order=2,
        physical_name="outer",
    )
    prism2 = PolyPrism(
        polygons=polygon2,
        buffers=buffers2,
        mesh_order=1,
        physical_name="inner",
    )

    entities_list = [prism1, prism2]

    write_xao(
        cad_occ(
            entities_list,
        ),
        f"test_filter_{label}.xao",
    )

    mesh(
        dim=3,
        input_file=f"test_filter_{label}.xao",
        output_file=f"test_filter_{label}.msh",
        resolution_specs={
            "inner": [
                ThresholdField(
                    apply_to=apply_to,
                    sizemin=0.5,
                    min_mass=min_mass,
                    max_mass=max_mass,
                    sizemax=10,
                    distmin=0,
                    distmax=10,
                )
            ],
            "outer": [
                ConstantInField(
                    apply_to=apply_to,
                    resolution=0.5,
                    min_mass=min_mass,
                    max_mass=max_mass,
                )
            ],
        },
        n_threads=1,
        default_characteristic_length=100,
    )

    compare_gmsh_files(Path(f"test_filter_{label}.msh"))


@pytest.mark.slow
@pytest.mark.parametrize(
    "restrict_to",
    [
        None,
        ["inner_left"],
        ["inner_right"],
        ["outer", "inner_right"],
    ],
)
def test_restrict(restrict_to):
    large_rect = 20
    small_rect = 5

    polygon1 = shapely.Polygon(
        [[0, 0], [large_rect, 0], [large_rect, large_rect], [0, large_rect], [0, 0]],
    )
    polygon2 = shapely.Polygon(
        [
            [large_rect / 2 - small_rect / 2, large_rect / 2 - small_rect / 2],
            [large_rect / 2 + small_rect / 2, large_rect / 2 - small_rect / 2],
            [large_rect / 2 + small_rect / 2, large_rect / 2 + small_rect / 2],
            [large_rect / 2 - small_rect / 2, large_rect / 2 + small_rect / 2],
            [large_rect / 2 - small_rect / 2, large_rect / 2 - small_rect / 2],
        ],
    )

    poly_outer = PolySurface(
        polygons=polygon1,
        mesh_order=2,
        physical_name="outer",
    )

    restrict_to = None if restrict_to is None else restrict_to
    poly_left = PolySurface(
        polygons=shapely.affinity.translate(polygon2, xoff=-3.1),
        mesh_order=1,
        physical_name="inner_left",
    )
    poly_right = PolySurface(
        polygons=shapely.affinity.translate(polygon2, xoff=3.1),
        mesh_order=1,
        physical_name="inner_right",
    )
    entities_list = [poly_outer, poly_left, poly_right]

    write_xao(
        cad_occ(
            entities_list,
        ),
        f"test_restrict_{restrict_to}.xao",
    )

    mesh(
        dim=3,
        input_file=f"test_restrict_{restrict_to}.xao",
        output_file=f"test_restrict_{restrict_to}.msh",
        resolution_specs={
            "inner_left": [
                ThresholdField(
                    sizemin=0.05,
                    distmax=10,
                    sizemax=1,
                    apply_to="curves",
                    restrict_to=restrict_to,
                )
            ],
        },
        n_threads=1,
        default_characteristic_length=100,
    )

    compare_gmsh_files(Path(f"test_restrict_{restrict_to}.msh"))


@pytest.mark.slow
def test_interface_thresholds():
    """Test different threshold fields on each side of an interface."""
    # Create outer square
    outer_square = shapely.Polygon([[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]])

    # Create inner square
    inner_square = shapely.Polygon([[4, 4], [6, 4], [6, 6], [4, 6], [4, 4]])

    # Create outer polysurface with coarser exponential field
    poly_outer = PolySurface(
        polygons=outer_square,
        mesh_order=2,
        physical_name="outer",
    )

    # Create inner polysurface with finer exponential field
    poly_inner = PolySurface(
        polygons=inner_square,
        mesh_order=1,
        physical_name="inner",
    )

    entities_list = [poly_outer, poly_inner]

    write_xao(
        cad_occ(
            entities_list,
        ),
        "test_interface_thresholds.xao",
    )

    mesh(
        dim=3,
        input_file="test_interface_thresholds.xao",
        output_file="test_interface_thresholds.msh",
        resolution_specs={
            "inner": [
                ExponentialField(
                    sizemin=0.05,
                    lengthscale=0.5,  # Shorter transition distance
                    growth_factor=3.0,
                    apply_to="curves",
                    restrict_to=["inner"],
                )
            ],
            "outer": [
                ExponentialField(
                    sizemin=0.05,
                    lengthscale=2.0,
                    growth_factor=3,
                    apply_to="curves",
                    not_sharing=["None"],
                    restrict_to=["outer"],
                )
            ],
        },
        n_threads=1,
        default_characteristic_length=100,
    )

    compare_gmsh_files(Path("test_interface_thresholds.msh"))


def _count_physical_lines(m, physical_name: str) -> int:
    """Count 1D mesh elements belonging to ``physical_name``.

    meshio stores per-physical-group cell indices in ``cell_sets``; one
    sub-array per cell block. We only sum entries where the block type
    is ``line`` so the counts are comparable across refinements.
    """
    total = 0
    for block_idx, arr in enumerate(m.cell_sets.get(physical_name, [])):
        if arr is None:
            continue
        if m.cells[block_idx].type == "line":
            total += len(arr)
    return total


def _count_surface_triangles(m, physical_name: str) -> int:
    total = 0
    for block_idx, arr in enumerate(m.cell_sets.get(physical_name, [])):
        if arr is None:
            continue
        if m.cells[block_idx].type == "triangle":
            total += len(arr)
    return total


def _sharing_scene(tmp_path: Path, resolution_specs=None):
    """Build an outer annulus around two disjoint inner squares and mesh it.

    Resolution is applied only to ``outer`` so we can read off refinement
    effects through the ``outer___None`` (domain boundary), ``outer___A``
    and ``outer___B`` physical groups.
    """
    outer_sq = shapely.Polygon([[0, 0], [20, 0], [20, 10], [0, 10]])
    a_sq = shapely.Polygon([[3, 3], [7, 3], [7, 7], [3, 7]])
    b_sq = shapely.Polygon([[13, 3], [17, 3], [17, 7], [13, 7]])

    xao = tmp_path / "scene.xao"
    write_xao(
        cad_occ(
            [
                PolySurface(polygons=outer_sq, mesh_order=2, physical_name="outer"),
                PolySurface(polygons=a_sq, mesh_order=1, physical_name="A"),
                PolySurface(polygons=b_sq, mesh_order=1, physical_name="B"),
            ]
        ),
        xao,
    )
    return mesh(
        dim=2,
        input_file=xao,
        output_file=tmp_path / "scene.msh",
        n_threads=1,
        default_characteristic_length=5.0,
        resolution_specs=resolution_specs or {},
        verbosity=0,
    )


def _sharing_scene_outer(tmp_path: Path, **field_kwargs):
    """Run :func:`_sharing_scene` with a :class:`ThresholdField` on ``outer``."""
    field = ThresholdField(
        apply_to="curves",
        sizemin=0.1,
        sizemax=5,
        distmin=0,
        distmax=0.1,
        **field_kwargs,
    )
    return _sharing_scene(tmp_path, resolution_specs={"outer": [field]})


@pytest.mark.slow
def test_sharing_default_refines_everything(tmp_path):
    """``sharing=None`` should refine all curves touching the target entity."""
    baseline = _sharing_scene(tmp_path)
    refined = _sharing_scene_outer(tmp_path)

    assert _count_physical_lines(refined, "outer___None") > 5 * _count_physical_lines(
        baseline, "outer___None"
    )
    assert _count_physical_lines(refined, "outer___A") > 5 * _count_physical_lines(
        baseline, "outer___A"
    )
    assert _count_physical_lines(refined, "outer___B") > 5 * _count_physical_lines(
        baseline, "outer___B"
    )


@pytest.mark.slow
def test_sharing_includes_none_keeps_boundary(tmp_path):
    """Explicitly listing ``'None'`` in ``sharing`` keeps the domain boundary."""
    m = _sharing_scene_outer(tmp_path, sharing=["A", "B", "None"])
    assert _count_physical_lines(m, "outer___None") > 100
    assert _count_physical_lines(m, "outer___A") > 50
    assert _count_physical_lines(m, "outer___B") > 50


@pytest.mark.slow
def test_sharing_specific_entity_drops_domain_boundary(tmp_path):
    """Listing only inner entities in ``sharing`` leaves the domain boundary coarse.

    Interface curves (with any inner entity) still refine because the
    include_boundary filter only targets the domain-boundary curves —
    that is the current behavior and the assertion we want to lock in.
    """
    m = _sharing_scene_outer(tmp_path, sharing=["A"])
    assert _count_physical_lines(m, "outer___None") < 20
    assert _count_physical_lines(m, "outer___A") > 50
    assert _count_physical_lines(m, "outer___B") > 50


@pytest.mark.slow
def test_not_sharing_none_drops_domain_boundary(tmp_path):
    """``not_sharing=['None']`` should match ``sharing=[interior...]`` above."""
    m = _sharing_scene_outer(tmp_path, not_sharing=["None"])
    assert _count_physical_lines(m, "outer___None") < 20
    assert _count_physical_lines(m, "outer___A") > 50
    assert _count_physical_lines(m, "outer___B") > 50


@pytest.mark.slow
def test_not_sharing_entity_keeps_boundary(tmp_path):
    """``not_sharing`` without ``'None'`` still refines the domain boundary."""
    m = _sharing_scene_outer(tmp_path, not_sharing=["A"])
    assert _count_physical_lines(m, "outer___None") > 100
    assert _count_physical_lines(m, "outer___A") > 50
    assert _count_physical_lines(m, "outer___B") > 50


@pytest.mark.slow
def test_mass_filter_only_large_surfaces_refined(tmp_path):
    """``min_mass`` should gate refinement by the target dimension's mass.

    Two disjoint PolySurface regions (area 4 and area 64) receive the
    same :class:`ConstantInField` with a mass filter. The filter lets
    the field fire on one region at a time; the unrefined region stays
    at the default element size.
    """
    small_poly = shapely.Polygon([[0, 0], [2, 0], [2, 2], [0, 2]])
    big_poly = shapely.Polygon([[10, 0], [18, 0], [18, 8], [10, 8]])

    xao = tmp_path / "mass.xao"
    write_xao(
        cad_occ(
            [
                PolySurface(polygons=small_poly, mesh_order=1, physical_name="small"),
                PolySurface(polygons=big_poly, mesh_order=1, physical_name="big"),
            ]
        ),
        xao,
    )

    def run(min_mass, max_mass):
        return mesh(
            dim=2,
            input_file=xao,
            output_file=tmp_path / "mass.msh",
            n_threads=1,
            default_characteristic_length=5.0,
            resolution_specs={
                name: [
                    ConstantInField(
                        apply_to="surfaces",
                        resolution=0.3,
                        min_mass=min_mass,
                        max_mass=max_mass,
                    )
                ]
                for name in ("small", "big")
            },
            verbosity=0,
        )

    # "big" area=64 passes min_mass=10; "small" area=4 does not.
    only_big = run(min_mass=10, max_mass=np.inf)
    # "small" passes max_mass=10; "big" does not.
    only_small = run(min_mass=0, max_mass=10)

    big_refined = _count_surface_triangles(only_big, "big")
    big_coarse = _count_surface_triangles(only_small, "big")
    small_refined = _count_surface_triangles(only_small, "small")
    small_coarse = _count_surface_triangles(only_big, "small")

    assert big_refined > 5 * big_coarse
    assert small_refined > 5 * small_coarse


@pytest.mark.slow
def test_mass_filter_curves_excludes_short_edges(tmp_path):
    """``min_mass`` on curves should leave short edges at the default size."""
    rect = shapely.Polygon([[0, 0], [20, 0], [20, 2], [0, 2]])
    xao = tmp_path / "curve_mass.xao"
    write_xao(
        cad_occ(
            [PolySurface(polygons=rect, mesh_order=1, physical_name="slab")],
        ),
        xao,
    )

    def run(min_mass):
        return mesh(
            dim=2,
            input_file=xao,
            output_file=tmp_path / "curve_mass.msh",
            n_threads=1,
            default_characteristic_length=2.0,
            resolution_specs={
                "slab": [
                    ThresholdField(
                        apply_to="curves",
                        sizemin=0.1,
                        sizemax=2.0,
                        distmin=0,
                        distmax=0.05,
                        min_mass=min_mass,
                        max_mass=np.inf,
                    )
                ]
            },
            verbosity=0,
        )

    # ``slab___None`` holds all four perimeter edges (2 long at length 20,
    # 2 short at length 2). With min_mass=10 only the long edges emit a
    # refinement; total 1D count stays below the unfiltered case.
    filtered = _count_physical_lines(run(min_mass=10), "slab___None")
    unfiltered = _count_physical_lines(run(min_mass=0), "slab___None")

    assert unfiltered > filtered


@pytest.mark.parametrize(
    ("within", "outside", "mode", "expected_counts"),
    [
        ([(-1.0, -1.0, 12.0, 5.0)], None, "overlap", (True, True, False)),
        ([(-1.0, -1.0, 5.0, 5.0)], None, "inside", (True, False, False)),
        (
            [(-1.0, -1.0, 25.0, 5.0)],
            [(8.0, -1.0, 16.0, 5.0)],
            "overlap",
            (True, False, True),
        ),
    ],
)
def test_region_threshold_field_modes(tmp_path, within, outside, mode, expected_counts):
    """Test RegionThresholdField in overlap vs inside modes and with outside exclusion."""
    from meshwell.utils import deserialize

    polys = [
        PolySurface(
            polygons=shapely.Polygon([[x, 0], [x + 4, 0], [x + 4, 4], [x, 4]]),
            mesh_order=1,
            physical_name=n,
        )
        for x, n in [(0, "A"), (10, "B"), (20, "C")]
    ]
    xao = tmp_path / "regions.xao"
    write_xao(cad_occ(polys), xao)

    spec = RegionThresholdField(
        apply_to="curves",
        within=within,
        outside=outside,
        mode=mode,
        sizemin=0.2,
        sizemax=5.0,
        distmin=0.0,
        distmax=0.5,
    )
    m = mesh(
        dim=2,
        input_file=xao,
        output_file=tmp_path / "out.msh",
        n_threads=1,
        default_characteristic_length=5.0,
        resolution_specs={n: [spec] for n in "ABC"},
        verbosity=0,
    )
    for n, fine in zip("ABC", expected_counts):
        lines = _count_physical_lines(m, f"{n}___None")
        assert (lines > 30) if fine else (lines < 15), f"Failed for {n}: {lines}"

    d = spec.to_dict()
    assert d["resolution_type"] == "RegionThresholdField"
    assert deserialize(d).mode == spec.mode


def test_region_threshold_clip_to_within(tmp_path):
    """A long strip crossing the box is refined only inside it when clipped."""
    strip = PolySurface(
        polygons=shapely.box(0, 0, 100, 2), mesh_order=1, physical_name="S"
    )
    xao = tmp_path / "strip.xao"
    write_xao(cad_occ([strip]), xao)

    def far_and_near_lines(clip):
        spec = RegionThresholdField(
            apply_to="curves",
            within=[(0.0, -5.0, 20.0, 7.0)],
            clip_to_within=clip,
            sizemin=0.2,
            sizemax=5.0,
            distmax=1.0,
            max_sampling=1000,
        )
        m = mesh(
            dim=2,
            input_file=xao,
            output_file=tmp_path / f"strip_{clip}.msh",
            n_threads=1,
            default_characteristic_length=5.0,
            resolution_specs={"S": [spec]},
            verbosity=0,
        )
        lines = np.concatenate([c.data for c in m.cells if c.type == "line"])
        x_mid = m.points[lines].mean(axis=1)[:, 0]
        return (x_mid > 40).sum(), (x_mid < 15).sum()

    far_full, near_full = far_and_near_lines(clip=False)
    far_clip, near_clip = far_and_near_lines(clip=True)
    assert near_clip > 0.8 * near_full  # still refined inside the box
    assert far_clip < far_full / 5  # but not along the rest of the strip


def test_box_field_refinement_and_serialization(tmp_path):
    """Test BoxField volumetric refinement and serialization round-trip."""
    from meshwell.utils import deserialize

    prism = PolyPrism(
        polygons=shapely.Polygon([[0, 0], [10, 0], [10, 10], [0, 10]]),
        buffers={0.0: 0.0, 5.0: 0.0},
        mesh_order=1,
        physical_name="sub",
    )
    xao = tmp_path / "box.xao"
    write_xao(cad_occ([prism]), xao)

    spec = BoxField(
        vin=1.0,
        vout=5.0,
        xmin=-1.0,
        xmax=5.0,
        ymin=-1.0,
        ymax=5.0,
        zmin=-1.0,
        zmax=6.0,
        thickness=0.5,
    )
    kw = dict(
        dim=3,
        input_file=xao,
        n_threads=1,
        default_characteristic_length=5.0,
        verbosity=0,
    )
    m = mesh(output_file=tmp_path / "box.msh", resolution_specs={"sub": [spec]}, **kw)
    m_c = mesh(output_file=tmp_path / "coarse.msh", resolution_specs={}, **kw)
    assert len(m.points) > 5 * len(m_c.points)

    d = spec.to_dict()
    assert d["resolution_type"] == "BoxField"
    assert deserialize(d).vin == spec.vin


def test_region_threshold_surface_filters_and_aliases():
    """Test 2D surface containment, exclusion, empty within, and exclude alias."""
    import gmsh

    gmsh.initialize()
    try:
        gmsh.model.add("test_surf")
        s1 = gmsh.model.occ.addRectangle(1, 1, 0, 8, 8)
        s2 = gmsh.model.occ.addRectangle(5, 5, 0, 100, 100)
        s_far = gmsh.model.occ.addRectangle(200, 200, 0, 50, 50)
        gmsh.model.occ.synchronize()

        f_in = RegionThresholdField(
            apply_to="surfaces",
            within=[(0.0, 0.0, 10.0, 10.0), (100.0, 100.0, 110.0, 110.0)],
            mode="inside",
            sizemin=1.0,
            sizemax=10.0,
            distmax=10.0,
        )
        kept_in = f_in._filter_entities(gmsh.model, {s1: 64.0, s2: 10000.0})
        assert s1 in kept_in
        assert s2 not in kept_in

        f_out = RegionThresholdField(
            apply_to="surfaces",
            within=[(0.0, 0.0, 300.0, 300.0)],
            exclude=[(40.0, 40.0, 60.0, 60.0)],
            mode="overlap",
            sizemin=1.0,
            sizemax=10.0,
            distmax=10.0,
        )
        assert f_out.outside == [(40.0, 40.0, 60.0, 60.0)]
        kept_out = f_out._filter_entities(gmsh.model, {s2: 10000.0, s_far: 2500.0})
        assert s2 not in kept_out
        assert s_far in kept_out

        f_empty = RegionThresholdField(
            apply_to="surfaces",
            within=[],
            mode="overlap",
            sizemin=1.0,
            sizemax=10.0,
            distmax=10.0,
        )
        assert len(f_empty._filter_entities(gmsh.model, {s1: 100.0})) == 0
    finally:
        gmsh.finalize()


if __name__ == "__main__":
    test_2D_resolution()
    # test_refine(ConstantInField(apply_to="surfaces", resolution=1))
