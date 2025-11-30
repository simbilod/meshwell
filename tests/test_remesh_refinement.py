"""Tests for the refinement module."""
import numpy as np
from meshwell.remesh import Remesh


def test_refine_by_gradient():
    """Test gradient-based refinement."""
    # Create a 1D grid of points (as 3D coordinates)
    x = np.linspace(0, 10, 11)
    coords = np.column_stack([x, np.zeros_like(x), np.zeros_like(x)])

    # Data with a step change (high gradient) at x=5
    data = np.zeros_like(x)
    data[x >= 5] = 1.0

    # Current sizes
    current_sizes = np.ones_like(x)

    remesher = Remesh()

    # Refine
    # Threshold 0.1 should catch the step
    new_sizes_map = remesher.refine_by_gradient(
        coords=coords,
        data=data,
        current_sizes=current_sizes,
        threshold=0.1,
        factor=0.5,
    )

    # Check that we have at least as many points as input (interpolation might add more)
    assert len(new_sizes_map) >= len(coords)

    # Find the original points in the output
    # Since interpolation appends, the first N points should be the original ones
    new_sizes = new_sizes_map[: len(coords), 3]

    # Check that sizes near x=5 are reduced
    # Note: Gradient estimation might affect neighbors
    assert np.any(new_sizes < 1.0)
    assert new_sizes[5] == 0.5  # The point at the step should definitely be refined

    # Check that far away points are not refined (gradient is 0)
    assert new_sizes[0] == 1.0
    assert new_sizes[-1] == 1.0


def test_refine_by_value_difference():
    """Test value difference-based refinement."""
    # Create 3 points connected in a line: 0-1-2
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ]
    )

    connectivity = np.array(
        [
            [0, 1],
            [1, 2],
        ]
    )

    # Data: large difference between 0 and 1, small between 1 and 2
    data = np.array([0.0, 1.0, 1.1])

    current_sizes = np.array([1.0, 1.0, 1.0])

    remesher = Remesh()

    new_sizes_map = remesher.refine_by_value_difference(
        coords=coords,
        connectivity=connectivity,
        data=data,
        current_sizes=current_sizes,
        threshold=0.5,
        factor=0.5,
    )

    # Check for interpolation
    # Edge 0-1 (length 1.0) has new sizes 0.5 and 0.5. Target size ~0.5.
    # Length > target size (1.0 > 0.5), so it should be subdivided.
    assert len(new_sizes_map) > len(coords)

    new_sizes = new_sizes_map[: len(coords), 3]

    # 0 and 1 should be refined
    assert new_sizes[0] == 0.5
    assert new_sizes[1] == 0.5

    # 2 should not be refined (diff with 1 is 0.1 < 0.5)
    assert new_sizes[2] == 1.0


def test_refine_by_error():
    """Test error-based refinement."""
    coords = np.zeros((5, 3))
    # Error values: [10, 8, 5, 2, 1]
    # Total error = 26
    data = np.array([10.0, 8.0, 5.0, 2.0, 1.0])

    current_sizes = np.ones(5)

    remesher = Remesh()

    # Target 50% of error (13.0)
    # Top node (10.0) is 38%
    # Top 2 nodes (10+8=18) is 69% > 50%
    # So should refine top 2 nodes

    new_sizes_map = remesher.refine_by_error(
        coords=coords,
        data=data,
        current_sizes=current_sizes,
        total_error_fraction=0.5,
        factor=0.5,
    )

    new_sizes = new_sizes_map[: len(coords), 3]

    assert new_sizes[0] == 0.5  # 10.0
    assert new_sizes[1] == 0.5  # 8.0
    assert new_sizes[2] == 1.0  # 5.0
    assert new_sizes[3] == 1.0  # 2.0
    assert new_sizes[4] == 1.0  # 1.0
