import numpy as np
import pytest
import shapely

from meshwell.structured.sweep_cad import clip_sweep_side, sweep_rectangles

P0 = np.array([0.0, 1.0])
P1 = np.array([4.0, 1.0])
UP = np.array([0.0, 1.0])


def test_clean_band_keeps_full_interval():
    region = shapely.box(0, 1, 4, 2)
    assert clip_sweep_side(P0, P1, UP, 0.5, region, 1e-6) == [(0.0, pytest.approx(4.0))]


def test_corner_notch_blocks_its_shadow():
    # region loses a notch x in [3, 4], y in [1, 1.3]: band (thickness .5)
    # can't reach full extent there -> entire [3,4] shadow dropped
    region = shapely.box(0, 1, 4, 2).difference(shapely.box(3, 1, 4, 1.3))
    kept = clip_sweep_side(P0, P1, UP, 0.5, region, 1e-6)
    assert len(kept) == 1
    lo, hi = kept[0]
    assert lo == pytest.approx(0.0)
    assert hi == pytest.approx(3.0)


def test_mid_obstacle_splits_interval():
    region = shapely.box(0, 1, 4, 2).difference(shapely.box(1.5, 1, 2.5, 2))
    kept = clip_sweep_side(P0, P1, UP, 0.5, region, 1e-6)
    assert [pytest.approx(v) for iv in kept for v in iv] == [0.0, 1.5, 2.5, 4.0]


def test_band_thicker_than_region_drops_everything():
    region = shapely.box(0, 1, 4, 1.2)
    assert clip_sweep_side(P0, P1, UP, 0.5, region, 1e-6) == []


def test_sweep_rectangles():
    rects = sweep_rectangles(P0, P1, UP, 0.5, [(0.0, 1.5), (2.5, 4.0)])
    assert len(rects) == 2
    assert rects[0].bounds == (0.0, 1.0, 1.5, 1.5)
    assert rects[1].bounds == (2.5, 1.0, 4.0, 1.5)
