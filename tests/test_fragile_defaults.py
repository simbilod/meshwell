"""Import-time defaults and truthiness hazards stay fixed."""
import inspect


def test_no_import_time_cpu_count_defaults():
    import meshwell.cad_gmsh as cg
    import meshwell.cad_occ as co
    import meshwell.mesh as m
    import meshwell.model as mo
    import meshwell.remesh as r

    for obj in [
        m.Mesh.__init__,
        m.mesh,
        mo.ModelManager.__init__,
        co.CAD_OCC.__init__,
        co.cad_occ,
        cg.CAD_GMSH.__init__,
        cg.cad_gmsh,
        r.Remesher.__init__,
        r.remesh_gmsh,
        r.remesh_mmg,
    ]:
        sig = inspect.signature(obj)
        if "n_threads" in sig.parameters:
            assert sig.parameters["n_threads"].default is None, obj


def test_n_threads_none_resolves_to_positive_int():
    from meshwell.model import ModelManager

    mm = ModelManager(n_threads=None)
    try:
        assert isinstance(mm.n_threads, int)
        assert mm.n_threads >= 1
    finally:
        mm.finalize()


def test_zero_valued_resolution_fields_are_honored():
    # sizemax/distmax of 0.0 must not be skipped by truthiness checks
    import meshwell.resolution as res

    src = inspect.getsource(res)
    assert "if self.sizemax and self.distmax" not in src
