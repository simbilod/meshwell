"""CADSettings: single-source defaults, XAO metadata round-trip, mesh intake rules."""

from __future__ import annotations

import gmsh
import pytest
import shapely

from meshwell import (
    CADSettings,
    CADSettingsError,
    CADSettingsMismatchError,
    MissingCADSettingsError,
    cad,
    generate_mesh,
    mesh,
)
from meshwell.cad_occ import cad_occ
from meshwell.cad_settings import (
    DEFAULT_ARC_TOLERANCE,
    DEFAULT_MIN_ARC_POINTS,
    DEFAULT_PERTURBATION,
    DEFAULT_POINT_TOLERANCE,
)
from meshwell.occ_xao_writer import write_xao
from meshwell.polyprism import PolyPrism

_BUFFERS = {0.0: 0.0, 1.0: 0.0}


def _scene():
    a = shapely.box(0, 0, 2, 5)
    b = shapely.box(2, 0, 5, 5)
    disc = shapely.Point(3.5, 2.5).buffer(1.0)
    return [
        PolyPrism(polygons=disc, buffers=_BUFFERS, physical_name="disc", mesh_order=1),
        PolyPrism(polygons=a, buffers=_BUFFERS, physical_name="A", mesh_order=2),
        PolyPrism(polygons=b, buffers=_BUFFERS, physical_name="B", mesh_order=3),
    ]


def _gmsh_summary(xao) -> dict:
    """Entity counts, named physical groups and a coarse 3D mesh size."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.open(str(xao))
        ents = {d: len(gmsh.model.getEntities(d)) for d in range(4)}
        groups = {
            (d, gmsh.model.getPhysicalName(d, t)): tuple(
                sorted(gmsh.model.getEntitiesForPhysicalGroup(d, t))
            )
            for d, t in gmsh.model.getPhysicalGroups()
        }
        gmsh.option.setNumber("Mesh.MeshSizeMax", 0.75)
        gmsh.model.mesh.generate(3)
        n_nodes = len(gmsh.model.mesh.getNodes()[0])
        return {"ents": ents, "groups": groups, "nodes": n_nodes}
    finally:
        gmsh.finalize()


def _without_provenance(entities):
    """Drop the cad_occ provenance so write_xao emits no metadata block."""
    for ent in entities:
        ent.cad_settings = None
    return entities


def _mesh_kwargs():
    return {"dim": 3, "default_characteristic_length": 1.0, "n_threads": 1}


# ---------------------------------------------------------------------------
# The dataclass itself
# ---------------------------------------------------------------------------


def test_defaults_come_from_module_constants():
    s = CADSettings()
    assert s.point_tolerance == DEFAULT_POINT_TOLERANCE
    assert s.perturbation == DEFAULT_PERTURBATION == 0.0
    assert s.min_arc_points == DEFAULT_MIN_ARC_POINTS
    assert s.arc_tolerance == DEFAULT_ARC_TOLERANCE
    assert s.identify_arcs is False


def test_derived_fuzzy_values_are_resolved_and_compare_equal():
    implicit = CADSettings(point_tolerance=1e-3)
    explicit = CADSettings(
        point_tolerance=1e-3, fragment_fuzzy_value=1e-3, cut_fuzzy_value=0.5e-3
    )
    assert implicit == explicit
    assert implicit.fragment_fuzzy_value == pytest.approx(1e-3)
    assert implicit.cut_fuzzy_value == pytest.approx(0.5e-3)
    perturbed = CADSettings(point_tolerance=1e-3, perturbation=1e-5)
    assert perturbed.cut_fuzzy_value == pytest.approx(0.8e-5)


def test_dict_and_json_round_trip():
    s = CADSettings(point_tolerance=1e-4, identify_arcs=True, min_arc_points=7)
    assert CADSettings.from_dict(s.to_dict()) == s
    assert CADSettings.from_json(s.to_json()) == s


def test_from_dict_rejects_unknown_keys():
    with pytest.raises(CADSettingsError, match="Unknown"):
        CADSettings.from_dict({**CADSettings().to_dict(), "bogus": 1})


@pytest.mark.parametrize(
    ("kwargs", "exc"),
    [
        ({"point_tolerance": 0.0}, ValueError),
        ({"point_tolerance": -1e-3}, ValueError),
        ({"perturbation": -1e-5}, ValueError),
        ({"arc_tolerance": 0.0}, ValueError),
        ({"min_arc_points": 2}, ValueError),
        ({"min_arc_points": 5.5}, TypeError),
        ({"identify_arcs": 1}, TypeError),
        ({"point_tolerance": None}, TypeError),
        ({"cut_fuzzy_value": 5e-3, "fragment_fuzzy_value": 1e-3}, ValueError),
    ],
)
def test_invalid_settings_raise(kwargs, exc):
    with pytest.raises(exc):
        CADSettings(**kwargs)


def test_from_kwargs_rejects_object_plus_loose_kwargs():
    with pytest.raises(TypeError, match="not both"):
        CADSettings.from_kwargs(CADSettings(), point_tolerance=1e-4)
    assert CADSettings.from_kwargs(None, point_tolerance=None) == CADSettings()


# ---------------------------------------------------------------------------
# XAO embedding
# ---------------------------------------------------------------------------


def test_cad_embeds_settings_and_gmsh_ignores_block(tmp_path):
    settings = CADSettings(point_tolerance=1e-3, identify_arcs=True)
    with_meta = tmp_path / "with.xao"
    cad(_scene(), output_file=with_meta, cad_settings=settings)
    assert CADSettings.from_xao(with_meta) == settings

    # Same geometry written without metadata: gmsh must see an identical
    # model (entities, physical groups, mesh).
    bare = tmp_path / "bare.xao"
    ents = cad(_scene(), cad_settings=settings)
    write_xao(_without_provenance(ents), bare)
    assert CADSettings.from_xao(bare) is None
    assert _gmsh_summary(with_meta) == _gmsh_summary(bare)

    text = with_meta.read_text()
    # Placement contract: after </geometry> (gmsh rejects content before it).
    assert text.index("<meshwell ") > text.index("</geometry>")


def test_cad_loose_kwargs_are_recorded(tmp_path):
    xao = tmp_path / "loose.xao"
    cad(_scene(), output_file=xao, point_tolerance=1e-4, min_arc_points=7)
    assert CADSettings.from_xao(xao) == CADSettings(
        point_tolerance=1e-4, min_arc_points=7
    )


def test_append_to_xao_is_idempotent_and_replaces(tmp_path):
    xao = tmp_path / "m.xao"
    write_xao(_without_provenance(cad_occ(_scene())), xao)
    CADSettings().append_to_xao(xao)
    CADSettings(point_tolerance=2e-3).append_to_xao(xao)
    assert xao.read_text().count("<meshwell ") == 1
    assert CADSettings.from_xao(xao) == CADSettings(point_tolerance=2e-3)


def test_unsupported_metadata_version_raises(tmp_path):
    xao = tmp_path / "m.xao"
    write_xao(cad_occ(_scene()), xao, cad_settings=CADSettings())
    xao.write_text(xao.read_text().replace('<meshwell version="1"', '<meshwell version="99"'))
    with pytest.raises(CADSettingsError, match="version"):
        CADSettings.from_xao(xao)


def test_cad_occ_provenance_is_embedded_by_write_xao(tmp_path):
    xao = tmp_path / "occ.xao"
    write_xao(cad_occ(_scene(), point_tolerance=1e-4), xao)
    assert CADSettings.from_xao(xao) == CADSettings(point_tolerance=1e-4)
    # cad() restamps the pipeline-level settings (incl. arc parameters).
    ents = cad(_scene(), min_arc_points=7)
    assert {e.cad_settings for e in ents} == {CADSettings(min_arc_points=7)}


def test_write_xao_rejects_mixed_provenance(tmp_path):
    ents = cad_occ(_scene())
    ents[0].cad_settings = CADSettings(point_tolerance=2e-3)
    with pytest.raises(CADSettingsMismatchError, match="different settings"):
        write_xao(ents, tmp_path / "mixed.xao")


def test_generate_mesh_checkpoint_carries_settings(tmp_path):
    ckpt = tmp_path / "ckpt.xao"
    generate_mesh(
        _scene(),
        checkpoint_cad=ckpt,
        point_tolerance=1e-3,
        min_arc_points=6,
        **_mesh_kwargs(),
    )
    # gmsh.write()-produced XAO + append_to_xao: readable by both sides.
    assert CADSettings.from_xao(ckpt) == CADSettings(min_arc_points=6)
    mesh(input_file=ckpt, **_mesh_kwargs())


# ---------------------------------------------------------------------------
# Mesh-stage intake rules
# ---------------------------------------------------------------------------


@pytest.fixture
def xao_with_meta(tmp_path):
    xao = tmp_path / "meta.xao"
    cad(_scene(), output_file=xao, point_tolerance=1e-3)
    return xao


@pytest.fixture
def xao_without_meta(tmp_path):
    xao = tmp_path / "bare.xao"
    write_xao(_without_provenance(cad_occ(_scene())), xao)
    return xao


def test_mesh_uses_embedded_settings(xao_with_meta):
    out = mesh(input_file=xao_with_meta, **_mesh_kwargs())
    assert len(out.points) > 0


def test_mesh_accepts_matching_explicit_settings(xao_with_meta):
    mesh(
        input_file=xao_with_meta,
        cad_settings=CADSettings(point_tolerance=1e-3),
        point_tolerance=1e-3,
        **_mesh_kwargs(),
    )


def test_mesh_raises_on_settings_mismatch(xao_with_meta):
    with pytest.raises(CADSettingsMismatchError, match="min_arc_points"):
        mesh(
            input_file=xao_with_meta,
            cad_settings=CADSettings(min_arc_points=9),
            **_mesh_kwargs(),
        )


def test_mesh_raises_on_point_tolerance_mismatch(xao_with_meta):
    with pytest.raises(CADSettingsMismatchError, match="point_tolerance"):
        mesh(input_file=xao_with_meta, point_tolerance=1e-4, **_mesh_kwargs())


def test_mesh_raises_without_metadata_or_settings(xao_without_meta):
    with pytest.raises(MissingCADSettingsError):
        mesh(input_file=xao_without_meta, **_mesh_kwargs())
    # A bare point_tolerance is not a proper CADSettings.
    with pytest.raises(MissingCADSettingsError):
        mesh(input_file=xao_without_meta, point_tolerance=1e-3, **_mesh_kwargs())


def test_mesh_rejects_non_cadsettings(xao_without_meta):
    with pytest.raises(TypeError):
        mesh(
            input_file=xao_without_meta,
            cad_settings={"point_tolerance": 1e-3},
            **_mesh_kwargs(),
        )


def test_mesh_accepts_supplied_settings_for_bare_xao(xao_without_meta):
    out = mesh(
        input_file=xao_without_meta, cad_settings=CADSettings(), **_mesh_kwargs()
    )
    assert len(out.points) > 0


def test_remesh_intake_raises_without_metadata(xao_without_meta):
    from meshwell.remesh import RemeshGMSH

    remesher = RemeshGMSH(n_threads=1)
    with pytest.raises(MissingCADSettingsError):
        remesher._resolve_intake(xao_without_meta, None)
    assert remesher._resolve_intake(xao_without_meta, CADSettings()) == CADSettings()
